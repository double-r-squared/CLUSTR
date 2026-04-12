# CLUSTR MPI Protocol Extension

Custom MPI implementation layered on top of the existing CLUSTR binary TCP protocol. Designed for LAN/VPN clusters of 2–64 nodes. No external MPI dependency — transport is ASIO async TCP, the same library already bundled for the worker binary.

---

## Design Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Transport | ASIO TCP | Already shipped to workers. Matches existing protocol style. |
| API style | C++20 coroutines (`co_await`) | Non-blocking without callback hell; clean sequential-looking job code |
| Peer port | OS-assigned (bind port 0) | Zero config; allows multiple workers on one machine |
| Barrier | Centralized (rank 0 coordinator) | Simple, correct for ≤16 nodes. Dissemination stubbed for future use |
| Fault model | Fail-stop | Standard MPI behavior. Checkpoint/restart stubbed under `CLUSTR_FAULT_TOLERANT` |
| Wire format | Fixed 24-byte `PeerHeader` + raw payload | Zero-copy on hot path; same `#pragma pack` pattern as scheduler protocol |
| Roster format | INI (same as `system.conf`) | No new parser needed |
| Collectives | send/recv, barrier, bcast, reduce, scatter, gather, allreduce | Covers all common patterns; allreduce falls back to reduce+bcast (ring-allreduce stubbed) |

---

## Architecture Overview

```
Scheduler
  │
  ├─ PEER_ROSTER ──► Worker A (rank 0)   ┐
  ├─ PEER_ROSTER ──► Worker B (rank 1)   │  Phase 1: roster + peer listener setup
  └─ PEER_ROSTER ──► Worker C (rank 2)   ┘
                          │
  ◄── PEER_READY (port) ──┤              Phase 2: each worker reports OS port
  ◄── PEER_READY (port) ──┤
  ◄── PEER_READY (port) ──┘
                          │
  ├─ PEER_ROSTER (complete) ──► all      Phase 3: complete roster with all ports
  ├─ FILE_DATA ──► all
  └─ EXEC_CMD  ──► all                   Phase 4: simultaneous execution

  Workers connect to each other (TCP mesh):
    A ◄──────────────────────────────► B
    A ◄──────────────────────────────► C
    B ◄──────────────────────────────► C

  Job completion:
    rank 0 ──► TASK_RESULT ──► Scheduler
    rank 1 ──► RANK_DONE   ──► Scheduler
    rank 2 ──► RANK_DONE   ──► Scheduler
```

---

## Protocol Messages

Three new message types added to `MessageType` in `protocol.h`:

| Type | Value | Direction | Description |
| --- | --- | --- | --- |
| `PEER_ROSTER` | `0x20` | Scheduler → Worker | Rank assignment + peer addresses. Sent twice: once with partial ports (before PEER_READY), once complete (after all PEER_READY received) |
| `PEER_READY` | `0x21` | Worker → Scheduler | Worker's peer listener is up. Carries the OS-assigned peer port |
| `RANK_DONE` | `0x22` | Worker → Scheduler | Non-root rank finished. Root rank sends `TASK_RESULT` as before |

### `PeerRosterPayload`

```
struct PeerRosterPayload {
    char     job_id[64];              // Job this roster belongs to
    uint32_t my_rank;                 // This worker's rank
    uint32_t num_ranks;               // Total group size
    PeerEntry peers[MAX_MPI_RANKS];   // All peers (skip self)
};

struct PeerEntry {
    uint32_t rank;
    char     ip[64];                  // IPv4/IPv6 string
    uint16_t peer_port;               // 0 in first send; filled in second send
};
```

`MAX_MPI_RANKS = 64`. Total size: 64 + 4 + 4 + 64 × (4 + 64 + 2) = ~4.5 KB. Sent twice per job group.

### `PeerReadyPayload`

```
struct PeerReadyPayload {
    char     job_id[64];
    uint32_t my_rank;
    uint16_t peer_port;   // OS-assigned port from bind(0)
};
```

### `RankDonePayload`

```
struct RankDonePayload {
    char     job_id[64];
    uint32_t rank;
    int32_t  exit_code;
};
```

---

## Peer Wire Format

All peer-to-peer messages use a 24-byte fixed header followed by a variable-length payload. This is separate from the scheduler ↔ worker protocol — peers connect directly, bypassing the scheduler.

```
Offset  Size  Field        Notes
------  ----  -----        -----
0       2     magic        0xC1 0x52  — detects misaligned reads
2       1     type         PeerMsgType enum
3       1     reduce_op    ReduceOp (REDUCE/ALLREDUCE only; 0 otherwise)
4       4     src_rank     uint32_t little-endian
8       4     dst_rank     uint32_t LE; 0xFFFFFFFF = broadcast (reserved)
12      4     tag          int32_t LE; negative values reserved for collectives
16      4     payload_len  uint32_t LE; bytes following this header
20      4     reserved     0; future use (sequence numbers, per-message checksum)
```

Tag conventions:

| Range | Use |
| --- | --- |
| `>= 0` | User-defined point-to-point tags |
| `-1` | barrier tokens |
| `-2` | bcast |
| `-3` | reduce |
| `-4` | scatter |
| `-5` | gather |
| `-6` | alltoallw (FFT redistribute) |

---

## Roster File

Written by the worker daemon to `/var/tmp/clustr/mpi_roster.conf` before exec. Read by `clustr_mpi.h` at startup.

```ini
rank=0
size=3
peer_port=10000
peer.0=100.92.163.46:10000
peer.1=100.100.147.7:10001
peer.2=100.92.163.50:10002
```

Same INI parser as `system_conf.h`. No JSON, no protobuf.

---

## Scheduler Lifecycle (MPI Job)

```
submit_job(num_ranks=3)
  └─ job_queue_.push(job)

try_schedule()
  └─ 3 idle workers available?
       └─ dispatch_group(job)
            ├─ assign rank_workers[0..2]
            ├─ mark all three unavailable
            └─ send partial PEER_ROSTER to each (peer_port=0)

Worker receives PEER_ROSTER
  └─ bind peer listener on port 0 → OS assigns port
  └─ send PEER_READY(job_id, rank, peer_port)

on_peer_ready() [called for each arriving PEER_READY]
  └─ store peer_port on WorkerEntry
  └─ job->ranks_ready++
  └─ all N ready?
       ├─ send complete PEER_ROSTER to each (all ports now known)
       ├─ send FILE_DATA to each
       └─ send EXEC_CMD to each

Workers execute — peer connections established via complete roster
  └─ rank 0:   exits → TASK_RESULT → scheduler
  └─ rank 1,2: exit  → RANK_DONE   → scheduler

on_task_result() / RANK_DONE handler
  └─ mark worker IDLE, rebuild work_queue_
  └─ try_schedule()
```

---

## `clustr_mpi.h` API

Header-only. Include in job `.cpp` files. Requires C++20 and ASIO (both available in the worker compile environment).

### Entry point

```cpp
CLUSTR_MPI_MAIN(mpi) {
    // mpi is a clustr::MPI& — fully connected before this body runs
    co_return 0;
}
```

The macro expands to `int main()` + ASIO `io_context` setup + `mpi.connect()` + coroutine dispatch.

### Point-to-point

```cpp
co_await mpi.send(data, dst_rank, tag);          // std::vector<T>
auto v = co_await mpi.recv<T>(src_rank, tag);    // returns std::vector<T>
```

`T` must be trivially copyable. `tag` defaults to 0.

### Barrier

```cpp
co_await mpi.barrier();
```

Centralized: all ranks send a token to rank 0, rank 0 releases when all N have arrived. O(2N) messages.

### Broadcast

```cpp
// On root:
std::vector<double> data = { ... };
co_await mpi.bcast(data, /*root=*/0);

// On non-root:
std::vector<double> data;   // empty
co_await mpi.bcast(data, 0);  // filled in-place
```

### Reduce

```cpp
auto result = co_await mpi.reduce(data, clustr::ReduceOp::SUM, /*root=*/0);
// result is non-empty only on root
```

Supported ops: `SUM`, `MIN`, `MAX`, `PROD`. All ranks must pass equal-length vectors.

### Scatter / Gather

```cpp
// Root scatters: data.size() must be divisible by mpi.size()
auto chunk = co_await mpi.scatter(data, /*root=*/0);

// All ranks gather back to root
auto full = co_await mpi.gather(chunk, /*root=*/0);
// full is non-empty only on root
```

### Allreduce

```cpp
auto result = co_await mpi.allreduce(data, clustr::ReduceOp::SUM);
// result available on all ranks
```

Current implementation: reduce to rank 0 + bcast. Correct, not bandwidth-optimal. Ring-allreduce (O(1) rounds, bandwidth-optimal) is stubbed in `clustr_mpi.h` for future implementation.

---

## Example Job

```cpp
#include "clustr_mpi.h"
#include <numeric>
#include <iostream>

CLUSTR_MPI_MAIN(mpi) {
    int rank = mpi.rank();
    int size = mpi.size();

    // Each rank computes a local sum
    std::vector<double> local = { static_cast<double>(rank * 10) };

    // Reduce all local sums to rank 0
    auto total = co_await mpi.reduce(local, clustr::ReduceOp::SUM, 0);

    if (rank == 0)
        std::cout << "Total: " << total[0] << std::endl;  // 0+10+20 = 30 for 3 ranks

    co_await mpi.barrier();
    co_return 0;
}
```

Compile command (auto-generated by TUI):

```
g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED \
    -I./include -I./asio_include \
    myjob.cpp -o myjob -lpthread
```

Submit with Ranks = 3 in the Jobs page submit dialog.

---

## Performance Notes

On a gigabit LAN or Tailscale VPN, ASIO TCP performance is indistinguishable from OpenMPI's TCP transport (`ob1` BTL):

| Metric | OpenMPI (TCP) | CLUSTR MPI (ASIO) |
| --- | --- | --- |
| Point-to-point latency | ~10–40 µs | ~10–40 µs |
| Bandwidth | ~90–95% wire | ~90–95% wire |
| InfiniBand / RDMA | Yes (UCX) | No |
| Collective algorithms | Topology-aware | Centralized / naive |

The gap opens on InfiniBand clusters (OpenMPI uses RDMA, dropping latency to ~1–2 µs) and for large allreduce on many nodes (ring-allreduce vs. naive reduce+bcast). For CLUSTR's target scale (2–16 nodes, LAN/VPN), these differences are not observable in practice.

---

## Future Work

| Item | Location | Notes |
| --- | --- | --- |
| Ring-allreduce | `clustr_mpi.h::allreduce()` | Stub present. O(2(N-1)) messages, bandwidth-optimal. Priority for ML workloads |
| Dissemination barrier | `clustr_mpi.h::dissemination_barrier()` | Stub present. O(2 log N) messages. Replace centralized when N > 16 |
| Checkpoint/restart | `clustr_mpi.h` fault tolerance stub | Enable with `-DCLUSTR_FAULT_TOLERANT`. Requires RANK_CHECKPOINT protocol message |
| Heterogeneous byte order | `PeerHeader` fields | Add `htonl`/`ntohl` guards if ARM and x86 nodes are mixed |
| `MPI_ANY_SOURCE` / `MPI_ANY_TAG` | `recv_raw()` mailbox | Wildcard matching not yet supported |
| Sub-communicators | `PeerHeader::comm_id` | World is `comm_id=0`. Phase 5 of the FFT roadmap will populate nonzero ids via `Comm::cart_sub`. |
| Transport selector | `-DCLUSTR_TRANSPORT=CLUSTR_TRANSPORT_ZERO_COPY \| CLUSTR_TRANSPORT_PACK_UNPACK` | Default is zero-copy (scatter-gather writev). Pack/unpack is the reference baseline retained for benchmarking; both implementations live in `include/clustr/transport_*.hpp`. TUI submit dialog exposes them as `MPI-zc` / `MPI-pu`. |

# CLUSTR Roadmap - Parallel Multidimensional FFT

Implementation plan for the parallel FFT described in [docs/FAST-FFT.md](docs/FAST-FFT.md), plus the supporting MPI protocol extensions and transport work it depends on.

Target: a working distributed multidimensional FFT over CLUSTR's existing worker mesh, with zero-copy scatter-gather transport throughout the hot path. Correctness first, then benchmarks.

---

## Design decisions (settled)

| Decision | Choice | Rationale |
| --- | --- | --- |
| Sub-communicators | **First-class `Comm` abstraction** (Option A) | Matches real MPI mental model. `MPI::world()` returns a `Comm`, `Comm::cart_sub()` produces row/column sub-groups for pencil decomposition. Mechanical refactor of existing single-group `MPI`. |
| Non-contiguous transfer | **Both a zero-copy scatter-gather path AND a pack/unpack path, selectable at compile time** | Ship both implementations so we can benchmark them head-to-head on real hardware before committing. Zero-copy is the theoretical winner but pack/unpack sometimes wins on systems where `writev` has high per-entry overhead or cache effects dominate. Mature whichever wins the benchmark; keep the other as reference. See "Dual transport strategy" below. |
| Serial FFT library | **PocketFFT** (header-only, MIT, single `.h`) | Same library NumPy uses. Drop-in, no build changes. Plan-based API so the eventual migration to FFTW is straightforward. |
| Array layout | Row-major, C-style | Matches our existing `std::vector<T>` backing stores and the paper's assumptions. |
| Precomputation | Compute all subarray descriptors + buffer sequences once, reuse across every transform | Matches the paper's measurement methodology (50 repeated transforms with same plan) and our existing "roster loaded once at startup" pattern. |
| Dimensionality | Start with 2D slab, then 3D pencil, then generalize | Each phase is independently testable against a serial reference. |

---

## Dual transport strategy (benchmark-first)

We build **two** transport implementations of `Comm::alltoallw` and the underlying send/recv primitives, and select between them at compile time. Neither is a "stub" - both must be correct, tested, and production-quality. We then benchmark the parallel FFT end-to-end on each and commit to the winner.

### Compile-time selector

A single preprocessor macro in the build system, defaulting to the zero-copy path:

```text
-DCLUSTR_TRANSPORT=ZERO_COPY   (default)
-DCLUSTR_TRANSPORT=PACK_UNPACK
```

Surfaced in `include/clustr_mpi.h` via:

```cpp
#define CLUSTR_TRANSPORT_ZERO_COPY  1
#define CLUSTR_TRANSPORT_PACK_UNPACK 2

#ifndef CLUSTR_TRANSPORT
#  define CLUSTR_TRANSPORT CLUSTR_TRANSPORT_ZERO_COPY
#endif
```

All call sites reference the same `Comm::alltoallw(...)` public API. The fork lives **inside** the transport layer - nothing above it (FFT code, tests, user jobs) knows which implementation is active. This is non-negotiable: if the split leaks upward, benchmarks are contaminated by API differences.

### Implementation A: Zero-copy scatter-gather

Lives in `include/clustr/transport_zero_copy.hpp`. This is the implementation described in Phase 2/3/4 above:

- `Subarray::build_const_buffers()` emits a coalesced `std::vector<asio::const_buffer>` pointing directly into the user's array.
- Transport's `write_seq_chunked`/`read_seq_chunked` feeds the sequence through `writev`/`readv` with runtime `IOV_MAX` chunking.
- No intermediate allocations beyond the buffer-sequence metadata vector (sized by number of iovec entries, not total bytes).

### Implementation B: Pack/unpack

Lives in `include/clustr/transport_pack_unpack.hpp`. This is the traditional MPI approach that `alltoallv`-based implementations use:

- `Subarray::pack(const void* base, uint8_t* out)` walks the descriptor and memcpy's elements into a contiguous scratch buffer owned by the `Comm`.
- A single large `async_write` of the packed buffer.
- On recv, `Subarray::unpack(const uint8_t* in, void* base)` memcpy's from the receive scratch buffer into the final destination at the correct strides.
- Scratch buffers are per-peer, sized once at plan time, reused across every transform (same precomputation discipline as Implementation A).

### What MUST stay common

Both implementations share:

- `Subarray` descriptor type and its construction logic (only the *consumer* methods differ: `build_*_buffers` vs `pack`/`unpack`)
- `Comm::alltoallw` public signature and semantics
- The `RedistributePlan` cache layer
- The FFT classes (`ParallelFFT2D`, `ParallelFFT3D`)
- All unit tests (tests run against the public `Comm` API and must pass on both implementations)

### Benchmark harness (Phase 9.5)

A new phase inserted after the FFTs work on both transports:

- `benchmarks/fft_bench.cpp`: runs 2D and 3D forward+inverse FFTs at several sizes (256^2, 1024^2, 64^3, 128^3, 256^3) and rank counts (2, 4, 8, as many as our test cluster allows), measuring median wall time across 50 repetitions.
- Build twice, once per transport, log results to `benchmarks/results_{zero_copy,pack_unpack}.csv`.
- Decision gate: the winner becomes the default. The loser stays in-tree as a reference implementation for correctness cross-checks, but we stop optimizing it.

### Rules while both implementations exist

1. **No API drift.** If you touch the public `Comm` surface for one transport, touch it identically for the other in the same commit.
2. **Every correctness test runs on both.** CI builds twice (once per transport) and runs the full test suite against each. This catches silent divergence early.
3. **Keep the split narrow.** The `#if CLUSTR_TRANSPORT == ...` fork should exist in exactly one place per primitive (`send_subarray`, `recv_subarray`, `alltoallw`). If we find ourselves sprinkling the macro across FFT code, we have failed at the abstraction.

---

## Phase 0 - Foundation (FFT library + layout)

**Goal:** Have a working serial FFT on a single node before touching MPI.

1. Vendor PocketFFT: copy `pocketfft_hdronly.h` into `include/pocketfft_hdronly.h`. License header preserved.
2. Add a `DistArray<T>` wrapper struct that stores:
   - `global_shape` (vector of sizes)
   - `local_shape` (local portion)
   - `distributed_axis` (which axis is split)
   - `storage` (`std::vector<T>`, row-major contiguous)
3. Serial sanity test: `jobs/fft_serial_test.cpp` - build a small 3D array, run PocketFFT forward+inverse, check round-trip error < 1e-10. Runs on any single worker.

**Exit criteria:** `jobs/fft_serial_test.cpp` passes when submitted with Ranks=1.

---

## Phase 1 - `Comm` abstraction refactor

**Goal:** Replace the monolithic `MPI` class with a `Comm`-based model, without changing any user-visible behavior for existing jobs.

1. Rename the current `clustr::MPI` class to `clustr::Comm`. All member functions (`send`, `recv`, `barrier`, `bcast`, `reduce`, `scatter`, `gather`, `allreduce`) move to `Comm`.
2. Add a thin `clustr::MPI` facade:
   ```cpp
   class MPI {
     Comm world_;
   public:
     explicit MPI(asio::io_context& io, ...);
     Comm& world() { return world_; }
     asio::awaitable<void> connect() { co_await world_.connect(); }
   };
   ```
3. Update `CLUSTR_MPI_MAIN(mpi)` to hand user code the `MPI` object. User code can do `mpi.world().rank()` or add `auto& c = mpi.world();` for brevity.
4. Update existing example jobs (`mpi_hello.cpp`, `mpi_reduce.cpp`, `mpi_scatter_gather.cpp`) - mechanical search/replace.
5. `Comm` owns its own `sockets_`, `mailbox_`, rank, size. Sub-comms share the parent's io_context.

**Non-trivial bit:** connections are established at the world level during `connect()`. Sub-comms do not open new TCP connections - they reuse the world sockets and add a `comm_id` field to `PeerHeader` so messages can be routed to the right mailbox.

**Protocol extension:** `PeerHeader::reserved` field (currently unused 4 bytes) becomes `comm_id`. World is comm_id=0; sub-comms get incrementing IDs assigned collectively via `cart_sub`.

**Exit criteria:** All existing MPI example jobs pass unchanged behaviorally after refactor.

---

## Phase 2 - Zero-copy transport in `clustr_mpi.h` [DONE 2026-04-11]

**Status:** Complete. Both transports (`CLUSTR_TRANSPORT_ZERO_COPY` and `CLUSTR_TRANSPORT_PACK_UNPACK`) live under `include/clustr/`, selected at compile time. TUI submit dialog exposes `MPI-zc` / `MPI-pu` presets. Multi-rank tests (`mpi_hello`, `mpi_reduce`, `mpi_scatter_gather` at Ranks=3) pass on both transports against deployed workers.

**Goal:** Rewrite `send_raw`/`recv_raw` to use ASIO buffer sequences so header and payload travel in one `writev` call, and payloads land directly in the destination buffer with no intermediate packing.

1. `send_raw` takes a `ConstBufferSequence` instead of `(void*, len)`:
   ```cpp
   template <typename BufferSeq>
   asio::awaitable<void> send_raw(int dst, const PeerHeader& hdr, BufferSeq&& payload_bufs);
   ```
   Internally: prepends a `const_buffer` over the header and forwards the combined sequence to the chunked writer below.
2. Typed `send<T>(vector<T>&)` wraps the vector in a single `const_buffer` and forwards to `send_raw`.
3. `recv_raw` splits into two phases: read the header first, then dispatch to `recv_into(MutableBufferSequence)` which reads directly into caller-provided storage. No intermediate `vector<uint8_t>`.
4. Mailbox behavior: if a message arrives with mismatched `(src, tag)`, the only way to buffer it is to allocate. Accept this - mailbox hits are the slow path, matched recvs are the fast path.

### 2a. Layered iovec strategy

The transport is the single point that guarantees "any buffer sequence, any size, any shape, correctly delivered." Three layers, in order of what fires first:

**Runtime IOV_MAX query (foundational):** The per-call iovec limit is kernel- and version-dependent and must NOT be hardcoded. POSIX explicitly warns against assuming a constant (`IOV_MAX` from `<limits.h>` is a compile-time hint that may diverge from the true runtime limit). At `Comm` initialization:

```cpp
long iov_max = ::sysconf(_SC_IOV_MAX);
if (iov_max <= 0) iov_max = 16;  // conservative POSIX minimum floor
transport_iov_cap_ = static_cast<size_t>(iov_max);
```

Cached on the `Comm` (or a process-wide singleton). If `sysconf` is unavailable or returns an error, fall back to the POSIX minimum guarantee of 16, never to a guess like 1024.

**Layer 1 - Coalesce (done by `Subarray` in Phase 3).** Adjacent iovec entries are merged before they reach the transport. For the common FFT case (full-extent inner axis, full-face sends) this alone keeps us well under any reasonable `IOV_MAX`. The transport trusts the caller to have coalesced.

**Layer 2 - Chunked writev/readv.** Internal helpers:

```cpp
template <typename BufferSeq>
asio::awaitable<void> write_seq_chunked(tcp::socket& s, BufferSeq bufs);

template <typename BufferSeq>
asio::awaitable<void> read_seq_chunked(tcp::socket& s, BufferSeq bufs);
```

Each loops over the sequence in batches of at most `transport_iov_cap_ - 1` entries (leave one slot for the header on sends), calling `asio::async_write`/`async_read` once per batch. TCP guarantees in-order delivery on the connection, so splitting a logical message into K batches is transparent to the peer: its own `read_seq_chunked` consumes the same byte stream into its own iovec list. The transport already has to handle short `writev`/`readv` returns - this is the same loop with "resume from entry K, offset O" tracking. **This loop is mandatory, not optional.** Skipping it and exceeding the runtime cap returns `-1 EINVAL` from the kernel - exactly the failure mode POSIX warns about.

**Layer 3 - Staging fallback (safety net, never hit in practice).** If a buffer sequence is so pathological that Layer 1 + 2 cannot make progress (e.g. a single logical element is spread across > `iov_max` entries), allocate a contiguous temp buffer, memcpy the sequence into it, send, and log a warning with the descriptor shape. The log line is loud on purpose - if it ever fires, it means a descriptor construction upstream is broken or an unusual shape needs a dedicated path.

**Why this combination does not bite us at scale:**

- No cap on process grid size. Whatever worker count we bring up, the transport handles it.
- Runtime-queried cap means the code is portable: same binary works on a Linux box with `IOV_MAX=1024` and a tuned kernel with a larger value.
- Coalescing is standard practice in real MPI implementations, so we inherit their headroom characteristics for free.
- The public send/recv API does not change with rank count - callers always see the same `send_iovec(conn, seq)` signature whether N=4 or N=4096.
- Migration to a real MPI library later: Layer 2 becomes a thin shim over `MPI_Alltoallw`; nothing upstream needs changing.

**Constraint:** the typed `recv<T>` must know the expected length to pre-size the destination. For the FFT, all transfers are planned and sizes are known at plan time. For the existing `recv<T>` API we keep the current "read header first, then allocate vector of the right size" behavior - one allocation, no intermediate copy.

**Exit criteria:** existing mpi example jobs still pass. `send_raw` no longer constructs a combined buffer vector. Ping-pong latency equal or better than pre-refactor. Stress test: send a sequence of 4096 small buffers in one call and verify correct delivery (exercises the Layer 2 chunking loop).

---

## Phase 3 - `Subarray` descriptor and scatter-gather buffer sequences [DONE 2026-04-11]

**Status:** Complete. `clustr::Subarray<T>` lives in `include/clustr/subarray.h` as a non-owning view over `DistArray<T>` with recursive contiguous-prefix coalesce, lazy const + mutable buffer caches, and copy/move deleted. Validated by `jobs/subarray_coalesce_test.cpp` (9 geometry cases, including the 64^3-full-face exit criterion) and `jobs/subarray_send_test.cpp` (3 box shapes round-tripped through `Comm::send_raw` between two ranks). Both tests pass on the cluster under `MPI-zc` and `MPI-pu`. `Comm::send_raw`/`recv_raw` are now part of the public Comm API so Phase 3+ callers can hand the transport arbitrary buffer sequences.

**Goal:** Given a `(global_shape, subshape, starts)` triple over a row-major multidim array, produce an ASIO buffer sequence that points at exactly the described elements with no copy.

1. New type `Subarray`:
   ```cpp
   struct Subarray {
     std::vector<size_t> global_shape;
     std::vector<size_t> subshape;
     std::vector<size_t> starts;
     size_t element_size;
   };
   ```
2. Method `Subarray::build_const_buffers(const void* base) const -> std::vector<asio::const_buffer>`: walks the subarray recursively, emitting one `const_buffer` per row of contiguous elements along the innermost axis. A 3D subarray with `subshape={A,B,C}` produces `A*B` buffers of `C*element_size` bytes each.
3. Mutable variant `build_mutable_buffers(void* base)` for receives.
4. **Coalesce pass:** `build_*_buffers` runs a single linear pass that merges any adjacent entries where `base[i] + len[i] == base[i+1]`. For the common case (innermost axis spans its full extent), a subarray with `A*B` raw row-chunks collapses to a single buffer of `A*B*C*element_size` bytes. This is what production MPI implementations do for free on contiguous descriptors and is the primary headroom against `IOV_MAX`.
5. Unit test in `tests/test_subarray.cpp`: construct a known 3D array, extract a known subarray via scatter-gather, verify byte-for-byte equality with a reference loop. Additionally verify that a full-face subarray coalesces to exactly 1 buffer entry (not `A*B`).

**Design note - iovec limits (handled in transport, not here):** `writev`/`readv` cap at the kernel's runtime `IOV_MAX`. The value is not a constant - it varies between kernels and must be queried at runtime, never hardcoded (POSIX explicitly warns against this). Exceeding it produces `EINVAL`. The `Subarray` layer does NOT deal with this - it emits its coalesced sequence and hands off. The transport (Phase 2) handles chunking transparently.

**Exit criteria:** `tests/test_subarray.cpp` passes. Subarray extraction of a 64x64x64 slice is correct across all axis choices. Coalesce verified: a full face of a 64x64x64 array produces exactly 1 buffer entry, not 4096.

---

## Phase 4 - `Comm::alltoallw` [DONE 2026-04-11]

**Status:** Complete. `clustr::alltoallw<T>(Comm&, span<Subarray<T>* const>, span<Subarray<T>* const>)` lives in `include/clustr/alltoallw.h`. Concurrency via `asio::experimental::ranged_parallel_group` over `co_spawn(asio::deferred)` operations, single `wait_for_all()` await, first exception wins. Self-loopback uses a two-cursor stream memcpy that survives mismatched send/recv fragment shapes (no self-socket needed). Tag `-6` reserved in `docs/MPI.md`. Validated by `jobs/alltoallw_test.cpp` at Ranks=3 under both transports: 9/9 (rank, src) pairs pass with three of the four kinds exercised (self loopback, full single-fragment, 64-fragment partial inner). Empty-extent path stays covered by the Phase 3 unit test.

**Goal:** Generalized all-to-all where each `(src, dst)` pair has its own `Subarray` descriptor for send and receive.

1. Signature:
   ```cpp
   asio::awaitable<void> Comm::alltoallw(
       const void* sendbuf,
       const std::vector<Subarray>& send_types,   // indexed by dst rank
       void* recvbuf,
       const std::vector<Subarray>& recv_types);  // indexed by src rank
   ```
2. Implementation: `co_await` N concurrent sends (one per dst) and N concurrent receives (one per src), using `asio::experimental::parallel_group` or manual co_spawn + awaitable_operators. Ordering: all sends posted first, then awaited; all receives posted first, then awaited. Deadlock-safe because TCP has per-direction buffering and each sockets is dedicated to one peer.
3. Tag reservation: `-6` for alltoallw (extend the existing tag table in `docs/MPI.md`).
4. Loopback case (send to self): short-circuit to a direct `memcpy` using the two descriptors. No TCP round-trip.

**Non-trivial bit:** correctness under mailbox interleaving. If `alltoallw` posts receives with tag=-6 while a stray SEND from user code arrives first, it goes into the mailbox. Fine - the mailbox handles this already, and our collectives have always worked this way.

**Exit criteria:** unit test in `tests/test_alltoallw.cpp` runs a 4-rank scatter-gather round trip: rank r sends `r` to rank `(r+1)%4` and `2r` to rank `(r+2)%4`, verifies receives.

---

## Phase 5 - `Comm::cart_create` and `Comm::cart_sub`

**Stage A** (recv path lift, no behavior change) DONE 2026-04-11. **Stage B** (`recv_central.hpp` central dispatch loop, per-(comm_id, src, tag, type) unbounded `asio::experimental::channel`, `Comm::shutdown()` wired into `CLUSTR_MPI_MAIN`, 4-way preset matrix `MPI-{zc,pu}-{i,c}`) DONE 2026-04-11 — full test matrix passes on cluster against `mpi_hello`, `mpi_scatter_gather`, `subarray_send_test`, `alltoallw_test`. Stage C (`cart_create`/`cart_sub`) and Stage D (`cart_sub_test`) still pending.

**Goal:** Produce row/column sub-communicators from a 2D Cartesian process grid.

1. `Comm::cart_create(std::vector<int> dims)` - returns a new `Comm` (or same Comm re-tagged with Cartesian metadata) that knows the grid shape. Validates `product(dims) == size()`.
2. `Comm::cart_sub(int axis)` - returns a new `Comm` containing only the peers along `axis` in the grid. Collective call - all ranks call it together, returns the same `comm_id` assignment on all ranks.
3. New sub-comm: same sockets as parent, new rank numbering (position along `axis`), new size (length of `axis`), new mailbox namespace via `comm_id`.
4. Protocol: `comm_id` assignment is deterministic from `(parent_comm_id, axis)` so no extra handshake is needed. Formula TBD - simplest is a monotonic counter synced across ranks at plan time.

**Non-trivial bit:** how sub-comm messages are routed at the `Comm` level. Proposal: `Comm` holds a reference to the parent's `sockets_` but uses `comm_id` in the header to demultiplex on receive. The world `Comm` owns a `recv_loop` coroutine that reads messages from each socket and routes them to the correct sub-comm's mailbox by `comm_id`. This is a change from the current "each recv reads from the socket inline" model - I want to discuss it before implementing.

**Open question for you:** The receive loop refactor is the biggest architectural change in this roadmap. The alternative is to keep inline reads and have each sub-comm poll its own mailbox, but then two sub-comms recv-ing concurrently on different axes of the same grid will step on each other. I think the central receive loop is necessary - flag for discussion before Phase 5 starts.

**Exit criteria:** `tests/test_cart_sub.cpp` creates a 2x3 grid from 6 ranks, produces row and column subcomms, verifies that `row_comm.size() == 3`, `col_comm.size() == 2`, and a bcast on `row_comm` reaches only the 3 peers in that row.

---

## Phase 6 - `redistribute()` - the core FFT primitive

**Pre-work — scheduler bugs surfaced during Phase 5B cluster testing (HIGH PRIORITY, fix before any redistribute work):**

1. **BUG-1: Queue drain stalls under "First available".** When the running job finishes, the next queued job is not picked up automatically — only force-start succeeds. The queue entry is intact (force-start works), so the missing piece is the dispatch trigger on `TASK_RESULT`. Investigate the job-completion handler / queue tick.
2. **BUG-2: Rank parameter not persisted on queued/force-started jobs.** A force-started `alltoallw_test` ran with `Ranks=1` instead of the configured value (`alltoallw_test: requires Ranks >= 2 (got 1)`). End-to-end trace required: submit dialog -> queue entry struct -> dispatcher -> roster generation. Likely the rank field isn't being copied into the queue entry, or is lost on queue -> dispatch.

**Goal:** Implement the paper's Section 3 routine: swap distributed axis `v` for axis `w` on a `DistArray` using a single `alltoallw` call.

1. Input/output: two `DistArray<T>` with the same `global_shape`, different `distributed_axis`.
2. Computes send/recv sizes via the balanced block decomposition from the paper (Section 3, Step 1).
3. Builds `Subarray` descriptors for send and recv (Step 2) - these depend only on shape and grid, so they should be cached on a `RedistributePlan` object for reuse.
4. Calls `Comm::alltoallw` (Step 3).
5. No explicit cleanup needed - descriptors are cached in the plan and freed on plan destruction.

**API sketch:**
```cpp
class RedistributePlan {
public:
  RedistributePlan(Comm& comm, std::vector<size_t> global_shape,
                   size_t element_size, int axis_v, int axis_w);
  asio::awaitable<void> execute(const void* input, void* output);
private:
  Comm& comm_;
  std::vector<Subarray> send_types_, recv_types_;
};
```

**Exit criteria:** `tests/test_redistribute.cpp`: build a 2D array distributed along axis 0, redistribute to axis 1, verify that element `[i][j]` on the original equals element `[i][j]` on the redistributed array (i.e. the data is logically the same, only the distribution changed).

---

## Phase 7 - 2D slab FFT

**Goal:** First working parallel FFT. 2D array, 1D process group, slab decomposition.

1. `ParallelFFT2D` class holding:
   - Serial FFT plans (PocketFFT) for each axis
   - `RedistributePlan` for the axis swap
   - Scratch buffers
2. `forward()`:
   - Local FFT along axis 1 (the non-distributed, contiguous axis)
   - Redistribute axis 0 <-> axis 1
   - Local FFT along what is now the local axis
3. `inverse()` - mirror sequence with inverse FFT plans.
4. Correctness test: build a 256x256 complex array with known frequency content, forward-transform, check peaks at expected bins, inverse-transform, check round-trip error.

**Exit criteria:** `jobs/fft_2d_test.cpp` passes with Ranks=2, Ranks=4, Ranks=8. Round-trip error < 1e-10.

---

## Phase 8 - 3D pencil FFT

**Goal:** 3D array, 2D process grid, pencil decomposition - the realistic HPC workload.

1. Requires `cart_create([P0, P1])` from Phase 5.
2. Two `RedistributePlan` instances: one on the row subcomm, one on the column subcomm.
3. Five-step forward sequence from `docs/FAST-FFT.md` section 4.2.
4. Correctness test against a serial reference on a 64x64x64 array.

**Exit criteria:** `jobs/fft_3d_test.cpp` passes with Ranks=4 (2x2 grid), Ranks=6 (2x3 grid). Round-trip error < 1e-10.

---

## Phase 9 - Documentation + example job

1. New file `docs/FFT.md`: high-level API reference for `ParallelFFT2D` / `ParallelFFT3D`, usage example, performance notes.
2. Update `docs/MPI.md` to document `Comm`, `cart_sub`, `alltoallw`, and the `comm_id` header field.
3. Update `README.md` MPI section to mention `Comm` and link to the FFT doc.
4. New example `jobs/fft_demo.cpp`: load a small 3D array, transform it, print peak frequencies. User-facing demonstration.

---

## Future optimization backlog (deferred)

Items flagged during roadmap planning that are not on the FFT critical path. Each is a real win but none block Phase 7/8. Revisit once the FFT is working and benchmarks identify bottlenecks.

| # | Location | Current behavior | Zero-copy approach |
| --- | --- | --- | --- |
| 1 | [clustr_mpi.h](include/clustr_mpi.h) `gather`/`scatter` | Temporary `vector<vector<T>>` then `insert` into result | Pre-compute layout, read each chunk straight into its final position in the result vector |
| 2 | [clustr_mpi.h](include/clustr_mpi.h) `reduce` | Receives each rank's vector into a temp, reduces, discards | Read into a scratch buffer per socket; reduce in place as chunks arrive |
| 3 | [client/client.cpp](client/client.cpp) `send_task_result`, `send_rank_done` | Allocate `vector<uint8_t>` of `sizeof(payload) + output.size()`, two memcpys | Scatter-gather write of header + output string via a buffer sequence |
| 4 | [src/scheduler.cpp](src/scheduler.cpp) `on_task_result`, `on_rank_done` | `memcpy` struct out, construct `std::string` from payload bytes | Use `string_view` / `span` over `msg.payload` directly |
| 5 | [src/file_transfer.cpp](src/file_transfer.cpp) | Reads file into a buffer, copies into message payload, sends | `sendfile(2)` on Linux / mmap + scatter-write elsewhere. Biggest win after the FFT path - large binaries currently do a full read-into-memory then write-to-socket. |
| 6 | [src/tcp_server.cpp](src/tcp_server.cpp) `Connection::send_message` | Calls `Message::serialize()` which allocates and copies header+payload into a single `vector<uint8_t>`; lambda captures `data` by value to keep it alive during the async write | Scatter-gather write of serialized header buffer + payload directly. Needs a send queue since you can't `async_write` the same connection from two call sites concurrently. |
| 7 | [src/tcp_server.cpp](src/tcp_server.cpp) `Connection::async_read_payload` | Reads payload into `read_buffer_`, then constructs a `full_msg` vector by *re-prepending* the size bytes just so `deserialize` can parse them back out | Parse the header fields directly from the already-read buffer - no reconstruction step. Pure waste currently. |
| 8 | [src/protocol.cpp](src/protocol.cpp) `Message::serialize` | Builds `vector<uint8_t>` via repeated `push_back` of individual bytes, then `insert` of payload | Write header fields into a fixed 15-byte stack buffer, return `pair<array<uint8_t,15>, span<uint8_t>>` to let caller do scatter-gather |

**Future library migration:** Replace PocketFFT with FFTW when performance requirements justify the build-system cost. FFTW wins for very large transforms and when SIMD matters; PocketFFT is within ~10-20% of FFTW in most measured benchmarks. Plan-based API is similar, so the migration should touch one file (`ParallelFFTnD` implementation).

---

## Open questions to resolve before starting

1. **Phase 5 receive loop refactor** - central receive loop that routes by `comm_id`, vs. per-Comm inline reads. I think central is required once sub-comms exist. Flag for discussion when Phase 5 starts.
2. **IOV_MAX batching** - at what array size does a 3D subarray hit the 1024-iovec limit, and do we need to handle it in Phase 3 or defer to Phase 7? Answer: a 3D subarray with shape `(A,B,C)` produces `A*B` iovecs. `A*B > 1024` means any dim pair past ~32 triggers batching. We hit this immediately on realistic array sizes, so handle in Phase 3.
3. **Mailbox growth under alltoallw** - N concurrent receives over N sockets with tag=-6 should all be matched on arrival without mailbox buffering. Verify with a stress test in Phase 4.

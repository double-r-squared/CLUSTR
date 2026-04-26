# AGENTS.md — orientation for anyone (human or agent) modifying this repo

This is a working brief, not a spec. The goal is to get you productive fast:
where things live, what shapes the design, and the small set of rules that
will save you from breaking something subtle. Read this once; keep it open
the first few times.

For end-user docs (how to run the cluster, what each TUI page does), read
[README.md](README.md). This file is for **developers and agents** who are
about to change code.

---

## What this project is

`clustr` is a distributed HPC job scheduler with **two parts that ship as
one repo**:

1. **A scheduler/worker system** — central scheduler with a live ncurses
   TUI, dispatches compile-and-run jobs to remote worker nodes over a
   custom binary TCP protocol. Workers are deployed inline from the TUI
   (no separate terminal).

2. **A header-only MPI-style runtime** (`include/clustr_mpi.h`) — C++20
   coroutines + ASIO async TCP, no external MPI library, no `mpirun`.
   Job binaries link nothing — they `#include` and use coroutine
   collectives.

Both halves co-evolve. The scheduler dispatches MPI jobs by setting up a
peer roster file; the runtime reads that file and connects the mesh.

Physically it's a single repo with one CMake target tree:
- `clustr` static library (protocol, tcp_server, file_transfer, etc.)
- `scheduler` (server side, links curses)
- `worker` (client side)
- `test_protocol` (tiny gtest-free unit test)

Everything else (job binaries, FFT tests, benchmark binaries) is built
ad-hoc by shell scripts in `tests/` and `scripts/` — see "Why some things
don't go through CMake" below.

---

## Repo map

```
include/                     <- public API surface
  clustr_mpi.h               <- the MPI runtime (header-only). The big one.
  protocol.h                 <- wire protocol (scheduler↔worker)
  scheduler.h, worker_*.h    <- server-side classes
  tui.h                      <- TUI public surface
  dist_array.h               <- distributed array primitive (FFT building block)
  pocketfft_hdronly.h        <- vendored PocketFFT (local 1D engine)
  clustr/                    <- MPI internals (see "Custom MPI internals")
    transport_common.hpp     <- PeerHeader, MailboxKey, shared types
    transport_zero_copy.hpp  <- writev-based send_raw (default)
    transport_pack_unpack.hpp<- single-buffer send_raw (fallback)
    recv_inline.hpp          <- per-call recv (world-comm only)
    recv_central.hpp         <- dispatcher recv (sub-comm capable, Phase 5+)
    cart.hpp                 <- Cartesian topology (cart_create / cart_sub)
    subarray.h               <- box-descriptor over DistArray, coalesces fragments
    alltoallw.h              <- generalized all-to-all over Subarrays
    redistribute.h           <- RedistributePlan (axis swap)
    parallel_fft_2d.h        <- 2D slab FFT
    parallel_fft_3d.h        <- 3D pencil FFT (templated on BenchHook)
    bench_hook.h             <- compile-time-polymorphic instrumentation hook

src/                         <- non-header sources
  protocol.cpp               <- (de)serialization, crc32 (scalar — see OPT items)
  tcp_server.cpp             <- accept loop + per-conn dispatch
  scheduler.cpp              <- the scheduler. Big and mostly state machines.
  file_transfer.cpp          <- FILE_DATA / FILE_ACK + tarball bundling
  remote_exec.cpp            <- exec_sync (compile), spawn_async (run)
  process_monitor.cpp        <- per-process CPU/mem sampling
  capability_detector.cpp    <- HW probe sent to scheduler at HELLO
  tui/
    tui_state.cpp            <- state struct + construction
    tui_draw.cpp             <- per-page render functions
    tui_input.cpp            <- key handlers
    tui_dialogs.cpp          <- modal dialogs (submit, deploy, strategy)
    tui_impl.h               <- private helpers (color pairs, wrap, log_color)

server/main.cpp              <- scheduler entry
client/client.cpp            <- worker binary (fully self-contained)

jobs/                        <- example MPI / FFT job source files
                               These are submitted *via the TUI*, not built by CMake.

tests/                       <- *integration tests*: shell scripts that
                               localhost-oversubscribe N ranks and verify
                               the MPI runtime end to end.
                               Plus one C++ unit test (test_protocol).

bench/                       <- benchmark suite (Phase B). Self-contained;
                               does NOT touch production paths.
  include/bench/             <- harness (timer, stats, json_writer, runner)
  mpi/, fft/, transfer/      <- clustr benches
  openmpi/                   <- OpenMPI baseline (skips if mpicxx missing)
  results/, plots/           <- output (gitignored)

scripts/
  setup_worker.sh            <- one-time worker setup (SSH keys, build tools)
  deploy.sh                  <- bootstrap a worker over SSH
  bench_all.sh               <- run the full bench matrix locally
  plot_bench.py              <- matplotlib from JSONL

docs/                        <- design docs (see "When to read which doc")
  MPI.md                     <- protocol + API reference
  MD-FFT.md                  <- distributed FFT architecture
  FAST-FFT.md                <- algorithm reference (pencil decomposition)
  PYTHON_BRIDGE.md           <- the Python bridge pipeline
  BENCHMARKS.md              <- bench philosophy + how to read plots
  PHASE9_*.md                <- FFT optimization rollup
  archive/                   <- old phase docs + the FFT paper PDF
```

---

## Build & test cheat-sheet

```bash
# Build the cluster binaries (scheduler + worker)
cmake -B build && cmake --build build

# Run the unit test
./build/test_protocol

# Run integration tests on localhost (each spawns N ranks)
tests/run_fft_2d_local.sh             # ranks 2/3/4/8, both transports
tests/run_fft_3d_local.sh             # ranks 3/4/6, both transports
BENCH=1 tests/run_fft_3d_local.sh     # + per-step timing breakdown
tests/run_redistribute_local.sh
tests/run_cart_sub_local.sh
tests/run_python_bridge.sh

# Run the benchmark suite (slow; 20-40 min on a laptop)
bash scripts/bench_all.sh
python3 scripts/plot_bench.py

# Skip parts of the bench:
BENCH_SKIP=openmpi,fft bash scripts/bench_all.sh
```

The `tests/` shell scripts are the **first thing to run** after any change
to `include/clustr_mpi.h` or anything under `include/clustr/`. They cover
both transports and both recv modes.

---

## Two big concepts to internalise before editing

### 1. The transport × recv compile-time matrix

Every MPI build picks one transport and one recv mode at compile time:

| Flag | Values | Effect |
|------|--------|--------|
| `CLUSTR_TRANSPORT` | `1` (default zero-copy) / `2` (pack-unpack) | which `send_raw` you get |
| `CLUSTR_RECV`      | `1` (default inline) / `2` (central) | which `recv_raw` you get |

This gives 4 build configurations: `MPI-zc-i`, `MPI-zc-c`, `MPI-pu-i`,
`MPI-pu-c`. The TUI submit dialog cycles through them. **All four must
work** — when you change MPI internals, build and run the integration
tests against all four.

Practical implications:
- `recv_inline.hpp` does not support sub-communicators. There is a
  `static_assert` in `clustr_mpi.h` if you call `cart_sub` under
  `CLUSTR_RECV_INLINE`. Sub-comms (and therefore the 3D pencil FFT)
  require `CLUSTR_RECV_CENTRAL`.
- Both transport headers must define `Comm::send_raw<BufferSeq>` and
  the `(ptr, len)` overload. Both recv headers must define
  `Comm::recv_raw`. Drift between them is a bug; the test scripts
  build the matrix to catch it.

### 2. Coroutine + ASIO discipline

`clustr_mpi.h` is **header-only** because every job's main is generated
by the `CLUSTR_MPI_MAIN(mpi)` macro, which expands to:
- `int main()` boilerplate
- ASIO `io_context` setup
- Roster load + peer mesh connect
- `co_spawn` of the user body

User code is `asio::awaitable<int>` returning. Every collective is
`co_await`. The runtime is single-threaded per rank — there is no
`std::thread` in the MPI path. If you add anything that wants to be
"in the background", do it via `asio::co_spawn(..., deferred)` and
join via `parallel_group`.

**Rules of thumb:**
- Never call a blocking syscall on the io_context thread. If you
  absolutely must (`std::system("tar")`), accept that all peers
  stall during the call. (See `OPT` items about libarchive.)
- Don't catch and swallow exceptions in awaitables — propagate. The
  scheduler's fault model is fail-stop: an exception is how a rank
  signals "I died, kill the job".
- TCP preserves order on a connection. Both transports rely on this;
  both can split a logical message across syscalls (chunking) and
  trust the receiver to reassemble.

---

## Custom MPI internals — file responsibilities

This is the part you'll touch most when extending the MPI runtime.

```
clustr_mpi.h
├── PeerHeader, ReduceOp, MailboxKey  (in transport_common.hpp)
├── Comm class
│     ├── world-level: rank(), size(), send/recv, collectives
│     └── sub-comm: cart_create, cart_sub, comm_id_-tagged frames
├── MPI<T> wrapper (legacy API: mpi.barrier(), mpi.bcast(), ...)
└── CLUSTR_MPI_MAIN macro

clustr/
├── transport_common.hpp     PeerHeader + cross-cutting types
├── transport_zero_copy.hpp  iov-based send_raw (writev/readv chunked)
├── transport_pack_unpack.hpp single-buffer send_raw (fallback path)
├── recv_inline.hpp          per-call: read header, allocate vec, read payload
├── recv_central.hpp         registered handler dispatch (required for sub-comms)
├── cart.hpp                 Cartesian topology (cart_create, cart_sub)
├── subarray.h               box descriptor with fragment coalescing
├── alltoallw.h              generalized all-to-all over Subarrays
├── redistribute.h           RedistributePlan: cached send/recv subarrays for an axis swap
├── parallel_fft_2d.h        2D slab FFT (1D process grid)
└── parallel_fft_3d.h        3D pencil FFT (2D process grid, templated on BenchHook)
```

### Tag conventions
Negative tags are reserved for collectives so user code (which uses
non-negative tags) can never collide:

```
-1  barrier        -2  bcast        -3  reduce
-4  scatter        -5  gather       -6  alltoallw
```

Documented in `docs/MPI.md`. If you add a new collective, take the
next negative number and update both this file and `docs/MPI.md`.

### comm_id
World comm has `comm_id == 0`. Every `cart_sub` mints a new id (handed
out from the parent); peer headers carry it so the central receiver
can route frames to the right sub-comm dispatcher. `recv_inline`
rejects frames whose comm_id ≠ 0 because it has no routing layer.

---

## Wire protocol (scheduler ↔ worker)

Defined in `include/protocol.h` (types) and `src/protocol.cpp`
(serialization + crc32). The full envelope is fixed-size header +
variable payload; multi-byte ints are little-endian; payload is
crc32'd end-to-end.

Key message types you'll see in the scheduler/worker code:

| Type | Direction | Purpose |
|------|-----------|---------|
| `HELLO` / `HELLO_ACK` | both | handshake |
| `CAPABILITY_REPORT` | W→S | hardware probe (cores, GFLOPS, mem BW) |
| `HEARTBEAT` | W→S | every 5s, "still alive" |
| `FILE_DATA` / `FILE_ACK` | S↔W | source + companions; tarball auto-extracted |
| `EXEC_CMD` / `EXEC_RESULT` | S↔W | sync compile, async run |
| `PROCESS_SPAWNED` | W→S | "your job's PID is …" |
| `TASK_RESULT` | W→S | rank 0 only — final stdout + exit code |
| `RANK_DONE` | W→S | non-rank-0 ranks — done signal |
| `PEER_ROSTER` | S→W | partial first (collect ports) then full (connect mesh) |
| `PEER_READY` | W→S | "I bound my listener on port X" |
| `STATUS_REQ` / `STATUS_REP` | S↔W | live CPU/mem snapshot for the TUI |

Two-phase peer setup is the bit that confuses everyone. Sequence:

```
Scheduler                       Worker (×N, in parallel)
  ──PEER_ROSTER (ports=0)────►
                                bind 0.0.0.0:0 (kernel picks port)
  ◄──PEER_READY (port=…)──────
  [scheduler aggregates ports]
  ──PEER_ROSTER (full)───────►
                                write /var/tmp/clustr/mpi_roster.conf
  ──FILE_DATA + EXEC_CMD─────►
                                compile, exec the binary
                                clustr_mpi reads the roster, connects mesh
```

---

## Conventions and gotchas

A non-exhaustive list of things that will bite you if you don't know
them. Add to this list when you discover the next one.

### Don't break either compile-time matrix
- The MPI matrix: 2 transports × 2 recv modes (above).
- The `CLUSTR_BENCHMARK` flag for FFT: must still produce the
  historical `[bench] <label> = <us>us` printf output. The
  `DefaultBenchHook` alias in `clustr/bench_hook.h` handles this; if
  you template a new class on `BenchHook`, do the same.

### Never narrow the public clustr_mpi API without a search
The TUI auto-fills compile commands assuming specific paths and
flags (`-DASIO_STANDALONE -DASIO_NO_DEPRECATED -DCLUSTR_TRANSPORT=…
-DCLUSTR_RECV=…`). Renaming a header or introducing a new required
flag means editing the TUI dialog *and* `scripts/setup_worker.sh` *and*
the bench scripts. Grep before renaming.

### Keep heap allocations off the FFT redistribute hot path
The FFT pipeline pre-allocates: `RedistributePlan` caches send/recv
Subarray descriptors, `ParallelFFT3D` owns its scratch buffer. Don't
add fresh `std::vector` constructions inside `forward()` / `inverse()`
without a strong reason — the cost shows up immediately in the FFT
bench's per-step breakdown.

### Drag-and-drop UX is non-negotiable
New job types must use the existing pipeline (FILE_DATA, EXEC_CMD,
TASK_RESULT) and produce the same output schema. The TUI does not
have special cases per job type — adding one is a smell. The Python
bridge is a precedent: it ships as a regular `.cpp` that happens to
shell out to `python3` per rank.

### Compile time matters
This repo deliberately avoids template-heavy unification across
`recv_inline` / `recv_central` and across the two transports — they
are separate files with the same API, not one templated thing. If you
catch yourself "DRY-ing" them with a CRTP or std::variant, stop and
discuss first.

### Tests run as localhost oversubscription
`tests/run_*.sh` spin up N peer processes on the same machine, each
binding its own port, and use `CLUSTR_MPI_ROSTER=…` to point them at a
local roster file. They never SSH, never deploy, never touch any
remote node. Don't add SSH calls to a test script — that's what the
deploy script is for.

### CMake doesn't build job files
`jobs/*.cpp` are compiled by the **worker** (or by a test script for
local runs), not by CMake. This is intentional: jobs are end-user
artefacts, the cluster scheduler ships their source and a compile
command. So adding a job source does not need a CMakeLists edit.
Same story for `bench/*.cpp` — `scripts/bench_all.sh` builds them
directly with `g++` so the bench is reproducible from a terminal
without depending on the project's build system.

### crc32 mismatch on the wire
`crc32` (in `src/protocol.cpp`) is a scalar reference implementation.
If you switch polynomials or move to a hardware-accelerated variant
(crc32c / SSE4.2 / PCLMULQDQ), update **both** sides simultaneously
and bump the protocol version. The current implementation runs at
~1.2 GiB/s on a modern core, which is the bottleneck visible in the
file-transfer benchmark.

### Worker is paranoid about /tmp
`/var/tmp/clustr_worker/` is the install path; `/tmp/clustr/` is the
work dir; `/tmp/clustr_worker.log` is the log. The cleanup story
matters because workers are persistent processes — assume long uptime
and clean up after each job.

---

## How to add things

### A new collective
1. Reserve the next negative tag (`docs/MPI.md` has the table).
2. Implement two methods on `Comm`: the high-level coroutine
   (e.g. `allgather`) and any internal helper.
3. Mirror it on the legacy `MPI<T>` wrapper (the user-facing API the
   TUI demos use) so existing job code keeps the consistent
   `mpi.allgather(...)` shape.
4. Write a job test in `jobs/` and a shell driver in `tests/`.
5. Bench: extend `bench/mpi/bench_collectives.cpp` (one binary, one
   `-DBENCH_OP=N` per op) and `scripts/bench_all.sh`.
6. Run the integration tests against the full transport × recv matrix.

### A new transport
1. Implement `Comm::send_raw<BufferSeq>` and the `(ptr,len)` overload
   in a new `clustr/transport_<name>.hpp`. Match the API of the two
   existing files exactly.
2. Add a `CLUSTR_TRANSPORT=<n>` value in `clustr_mpi.h` and the
   `#elif` chain that includes the new file.
3. Update the TUI submit dialog presets (`src/tui/tui_dialogs.cpp`).
4. Update `scripts/bench_all.sh` so the matrix runs the new transport.
5. Add a smoke test in `tests/`.

### A new recv mode
Same shape as transport. Note the sub-comm capability constraint —
if your mode supports sub-comms, add it to the allow-list in
`clustr_mpi.h`'s `static_assert`.

### A new TUI page
Add to the `Page` enum in `tui.h`, write `draw_*` and
`handle_*_input` functions in `src/tui/tui_draw.cpp` and
`tui_input.cpp`, register them in the page-table dispatch, update the
"Global keys" `1`/`2`/`3`/`4` accelerator. State lives on `TuiState`.

### A new job type that needs companion files
Just put the companion file next to the `.cpp` in `jobs/`. The TUI
auto-detects companions and tarball-bundles them at submit. The
worker auto-extracts. No protocol changes, no scheduler changes.

### A benchmark of a new internal class
Read [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — it has a worked
example using `BenchHook`. Pattern: template the class on `BenchHook`,
default to `DefaultBenchHook`, sprinkle `hook_.begin / end`. Production
builds get a `NullBenchHook` (zero-cost); bench builds pass a
`JsonSinkBenchHook` and pull the sink back via `obj.hook().sink()`.

---

## When to read which doc

| If you're working on… | Read |
|-----------------------|------|
| The scheduler / worker / TUI | `README.md` (top half) |
| The MPI runtime or protocol | `docs/MPI.md` |
| FFT internals or distributed array | `docs/MD-FFT.md`, then `docs/FAST-FFT.md` for algorithm |
| Phase 9 FFT optimisations | `docs/PHASE9_OPTIMIZATIONS.md`, `docs/PHASE9_SUMMARY.md` |
| Adding a Python-bridged job | `docs/PYTHON_BRIDGE.md` |
| The benchmark suite | `docs/BENCHMARKS.md` |
| Historical context (phases) | `docs/archive/ROADMAP.md`, `docs/archive/PHASE8.md` |

`docs/OPTIMIZATION_PLAN.md` is a working scratchpad, not a spec —
treat as informational.

---

## Non-goals (so you don't accidentally try to ship them)

- **External MPI compatibility.** clustr_mpi is not OpenMPI; it does
  not implement MPI-3.x. The OpenMPI baseline in the bench suite
  exists only to give an honest comparison number.
- **Heterogeneous architectures.** Wire format is little-endian raw
  ints. There are guard comments in `clustr_mpi.h` showing where
  `htonl`/`ntohl` would go if anyone ever wired up cross-arch
  clusters.
- **Fault tolerance.** Fail-stop only. Any rank disconnect → the
  scheduler marks the job failed. There is a `FAULT_TOLERANT` build
  flag stub in `clustr_mpi.h` for future checkpoint/restart, but no
  implementation.
- **GPU FFT / GPU-direct RDMA.** PocketFFT is CPU-only. GPU paths are
  out of scope.
- **`mpirun` compatibility.** Job binaries connect via the roster
  file, not via a launcher.

---

## Quick "what is this state?" reference

When you're chasing a bug and need to know what's in flight:

- Job lifecycle: see `WorkerEntry` state in `include/worker_registry.h`
  and the comments around `try_schedule()` in `src/scheduler.cpp`.
- TUI live state: `TuiState` in `include/tui.h` is the single source
  of truth; everything else is rendered from it.
- MPI roster: `/var/tmp/clustr/mpi_roster.conf` on each worker, INI
  format. Same parser as `system.conf`. The scheduler writes it; the
  worker hands it to the spawned binary via the
  `CLUSTR_MPI_ROSTER` env var. (For local tests, the test script
  writes the roster directly.)
- Worker log: `/tmp/clustr_worker.log` on each remote.
- Bench output: `bench/results/*.jsonl` (one record per
  `(bench, config)`, append-only). Plot with
  `scripts/plot_bench.py`.

---

## Getting started checklist for a new agent or contributor

1. `cmake -B build && cmake --build build` — confirm the cluster
   binaries compile clean.
2. `./build/test_protocol` — confirm the unit test passes.
3. `tests/run_fft_3d_local.sh` — confirm the MPI runtime works
   end-to-end on localhost.
4. Skim `include/clustr_mpi.h` top-to-bottom (the design comments are
   the primary documentation for the runtime).
5. Skim `include/protocol.h` and `src/scheduler.cpp` to understand the
   message flow.
6. Read [docs/MPI.md](docs/MPI.md) if you'll be touching the runtime.
7. **Then** start changing things.

When you're done with a non-trivial change, run the integration tests
against **all four** MPI build configurations. The scripts are fast
(seconds, not minutes) — there's no excuse not to.

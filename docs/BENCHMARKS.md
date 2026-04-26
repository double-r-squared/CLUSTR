# Benchmarks: what, why, how to read them

This document explains the benchmark suite under `bench/`, what each metric
actually measures, and how to interpret the numbers honestly. The code is
designed to answer **"where is clustr fast, where is it slow, and why?"** —
not to produce marketing-friendly graphs.

## Running

```bash
# Build + run the full matrix (20–40 min on a laptop, depending on cores):
bash scripts/bench_all.sh

# Skip phases you don't need:
BENCH_SKIP=openmpi,fft bash scripts/bench_all.sh

# Raise per-bench confidence (slower):
BENCH_WARMUP=50 BENCH_ITERATIONS=500 bash scripts/bench_all.sh

# Render plots from the JSONL output:
python3 scripts/plot_bench.py   # requires matplotlib
```

Outputs land in two places:

- `bench/results/*.jsonl` — append-only JSON Lines; each line is one
  `(bench, config)` run with percentile stats and environment metadata.
- `bench/plots/*.png` — matplotlib figures keyed off the JSONL.

Re-running the same config simply appends another record; the plot script
prefers the most recent record per `(bench, config)` tuple, so iteration
is non-destructive.

---

## Measurement philosophy

Every bench follows the same protocol (`bench/include/bench/runner.h`):

1. **Warmup** — `BENCH_WARMUP` iterations (default 10) whose timings are
   discarded. Purpose: prime CPU caches / TLB / page tables, let TCP
   slow-start finish, fire any lazy allocation at least once.
2. **Measurement** — `BENCH_ITERATIONS` iterations (default 100) timed
   individually with `bench::Timer` (`steady_clock`, TSC-backed on x86_64).
3. **Report percentiles, not mean** — `min / p50 / p90 / p99 / max`.
   Mean hides tail latency; in distributed systems the tail is what
   actually matters.
4. **Fences around the hot call** — every sample wraps the op in
   `ClobberMemory()` + `DoNotOptimize()` to prevent the compiler from
   hoisting loads/stores out of the timed region or eliding the result.
5. **One record per config, appended as JSON** — makes diffing easy and
   lets the plot script ignore stale runs without any manual cleanup.

## The observer effect, concretely

Naïve microbenchmarks lie. Specific things this suite does to avoid common
traps:

| Trap | What it looks like | What we do |
|------|--------------------|------------|
| Compiler elides the work | results impossibly fast | `DoNotOptimize(buf.data())` asm fence on every iteration |
| Compiler reorders memory across the timer | results noisy or unstable | `ClobberMemory()` before `start()` and after `stop()` |
| TCP slow-start dominates small-size results | first-iter latency huge | warmup iterations discarded |
| Page faults on first touch | large-buffer throughput poor | warmup touches every page; pre-allocate before timing |
| Mean pulled around by one slow sample | numbers jump run-to-run | report p50/p99; plots show both |
| gzip collapses pseudo-zero data | tar throughput unrealistic | companion files filled with mt19937_64 pseudorandom bytes |
| Kernel schedules ranks onto different cores mid-run | high variance | run at `-O2`; pin affinity manually if reproducing a specific setup |
| "Done when enqueued" accounting | bandwidth inflated | p2p_bandwidth uses a `done` ACK round-trip to close the timed window |

We do **not** manually pin CPU affinity in the default scripts because
the suite has to run on laptops and CI. If you're chasing a specific
regression, add `taskset -c <cores>` (Linux) or use `cpuset` (macOS
requires sudo) around the `run_ranks` call.

---

## Phases

### B1 — Harness
`bench/include/bench/{timer,stats,json_writer,env,runner}.h`

No numbers of its own. Provides the common machinery every other phase
uses. Worth reading if a reported number looks off — the code is short
and the measurement protocol is all in `runner.h`.

### B2 — P2P microbenchmarks
`bench/mpi/bench_p2p_pingpong.cpp`, `bench/mpi/bench_p2p_bandwidth.cpp`

- **Ping-pong**: 2 ranks, timed `send + recv`. Reports one-way p50/p99 by
  dividing RTT by 2. This is the single best microbenchmark for "what
  does a single `send/recv` cost me?"
  Sizes: `8, 64, 512, 4K, 32K, 256K, 2M, 16M` bytes.
- **Bandwidth**: 2 ranks, burst of N messages followed by a 1-byte
  acknowledgement. The ack closes the timed window so we measure
  wire-completion, not enqueue. Reports GiB/s = bytes\*burst / p50_s.

Both benches are run twice: once with the zero-copy transport
(`CLUSTR_TRANSPORT=1`) and once with the pack/unpack transport
(`CLUSTR_TRANSPORT=2`) so the JSON records carry the transport label.

### B3 — Collectives
`bench/mpi/bench_collectives.cpp`

One binary, one bench implementation per collective op, switched at
compile time via `-DBENCH_OP={1..6}`. Each op runs at 2/4/8 ranks.

- **bcast / reduce / allreduce / scatter / gather**: sizes `8, 64, 512,
  4K, 32K, 262K` bytes (scatter/gather round up to a multiple of rank
  count).
- **barrier**: payload-less, reports just the synchronisation cost.
- **Barrier is INSIDE the timed region** to aligns starts across ranks.
  This charges the barrier cost to the collective, but that's the cost
  you actually pay in real pipelines where the next op depends on the
  previous one having completed everywhere.

### B4 — FFT scaling
`bench/fft/bench_fft_3d.cpp`

Exercises `ParallelFFT3D<double, JsonSinkBenchHook>` — the bench builds
with the JSON sink hook so each algorithmic step (`fwd_fft_axis2`,
`fwd_p1_redist`, `fwd_fft_axis1`, `fwd_p0_redist`, `fwd_fft_axis0`, and
their inverses) emits its own JSONL record in addition to the
end-to-end timing.

- **Strong scaling**: `BENCH_FFT_N=48`, same grid size across 2/4/6 ranks.
  Ideal speedup is linear; reality will tail off as redistribute cost
  dominates.
- **Weak scaling**: `BENCH_FFT_N=24` per rank baseline, scaled by
  `cbrt(ranks)`. Ideal is flat; reality will grow due to coordination.

Grid auto-selection: ranks 3→{1,3}, 4→{2,2}, 6→{2,3}.

### B5 — File transfer
`bench/transfer/bench_file_transfer.cpp`

Measures the scheduler and worker CPU paths in isolation (no network):

- `make_file_data_msg` at 1 KiB / 1 MiB / 100 MiB — read-file + crc32
  + wrap-in-Message.
- `handle_file_data` round-trip at the same sizes — verify crc + write
  to disk + (for .tar.gz) `tar xzf`.
- `make_bundle_msg` + `bundle_roundtrip` at 1/5/20/100 companion files
  of 4 KiB each — isolates the cost of the tar shell-out.

Files are filled with pseudo-random bytes so gzip can't cheat.

### B6 — OpenMPI baseline
`bench/openmpi/bench_ompi_pingpong.cpp`, `bench/openmpi/bench_ompi_bcast.cpp`

Same JSON schema, same sizes, same warmup, same per-iteration timer —
so the clustr vs OpenMPI comparison is apples-to-apples.

Skipped gracefully if `mpicxx` / `mpirun` are not installed. Install:
```
brew install open-mpi                             # macOS
apt install openmpi-bin libopenmpi-dev            # Debian/Ubuntu
```

### B7 — Plotting + this doc
`scripts/plot_bench.py`, `docs/BENCHMARKS.md`

---

## How to read the plots

`bench/plots/p2p_pingpong.png` — latency vs message size, log-log.
- The **small-message floor** (bytes ≤ 64) is dominated by syscall +
  loopback cost. Two transports should land at nearly the same floor
  because the critical path is identical at small sizes.
- The **bandwidth regime** (bytes ≥ 32K) slope reveals effective
  throughput. If clustr_zc and clustr_pu diverge here, the pack/unpack
  copy is the dominant cost.
- OpenMPI at p50 is your practical upper bound on loopback. If clustr
  is within 2–3× at small sizes and within 30% at large sizes, it's
  competitive for a pure-userspace TCP transport.

`bench/plots/p2p_bandwidth.png` — GiB/s vs message size.
- Bandwidth ramps until the per-send syscall overhead is amortised
  (typically around 32–256 KiB on loopback).
- Plateau reveals the upper bound. On loopback over TCP without
  `TCP_NODELAY`, expect to be capped well below what shared memory
  would deliver — that is a *real* cost of this design, not a flaw in
  the measurement.

`bench/plots/collectives.png` — p50 latency vs rank count, 4 KiB payload.
- Broadcast and barrier should scale sub-linearly in ranks if a tree is
  used; today they're mostly serialised through rank 0 (centralised
  pattern), so expect roughly linear in `ranks`.
- Reduce + allreduce will look similar to bcast + gather, since
  allreduce is implemented reduce-then-bcast.
- If OpenMPI bcast overlays dramatically lower, the gap is the
  protocol + topology optimisation OpenMPI has that clustr does not.

`bench/plots/fft_scaling.png` — FFT time vs ranks (strong + weak).
- Strong: ideal is `1/ranks`. A flattening curve means redistribute
  cost is growing faster than the local FFT is shrinking.
- Weak: ideal is flat. Growth with ranks indicates coordination cost
  (alltoallw scaling).

`bench/plots/fft_breakdown.png` — stacked per-step time.
- Reveals which of the 5 FFT steps dominates. Typically: local FFTs
  dominate at low rank counts, then the two redistribute steps take
  over as ranks grow. If this inverts, look at the `OPT.md` entries
  about TCP_NODELAY and Subarray allocations.

`bench/plots/file_transfer_throughput.png` — MiB/s vs file size.
- Small-file throughput is dominated by `std::filesystem` / open+read
  syscalls and crc32.
- Large-file throughput approaches pure-I/O bandwidth bounded by
  crc32c speed (we use the naïve scalar crc32, which caps around
  1.2 GiB/s on modern cores — see `OPT.md` for the SSE4.2 opportunity).

`bench/plots/file_transfer_bundles.png` — bundle time vs companion count.
- Nearly-linear growth is expected while the tar invocation is a
  `fork+exec` to the system binary. The y-intercept is the fork cost;
  the slope is the per-file bookkeeping + gzip.

---

## Reproducibility checklist

Before comparing runs across two branches or two machines:

1. Same `BENCH_WARMUP` and `BENCH_ITERATIONS` in both runs.
2. Same compiler version and same optimisation flag (`bench_all.sh`
   hard-codes `-O2`).
3. Machine at similar load. If the machine is not idle, results will
   be noisy in ways percentiles cannot fully mask.
4. Same rank counts for collectives (`2, 4, 8`) and FFT (`2, 4, 6`).
5. Capture both runs' JSONL files — the `env` block in each record
   carries `hostname`, `cpu_model`, `compiler`, `git_sha` so future
   you can remember what was measured.

---

## Using the hook in new benches

If you add a new production-facing class that you want to instrument,
follow the pattern in `include/clustr/parallel_fft_3d.h`:

```cpp
#include "clustr/bench_hook.h"

template <typename T, typename BenchHook = clustr::DefaultBenchHook>
class MyThing {
    BenchHook hook_;
public:
    MyThing(..., BenchHook hook = BenchHook()) : hook_(std::move(hook)) {}

    void do_work() {
        auto tok = hook_.begin("subphase_name");
        // ... the thing you want to time ...
        hook_.end(tok, "subphase_name");
    }
};
```

In production builds `BenchHook = NullBenchHook`; every `begin/end` call
compiles to nothing (empty inline methods, the optimiser elides them
entirely). In bench builds the caller passes `JsonSinkBenchHook` and
gets per-subphase JSONL records for free.

No `#ifdef` at call sites, no runtime branch in the hot path.

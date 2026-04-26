// bench_fft_3d.cpp - parallel 3D FFT end-to-end timing with per-step breakdown.
//
// Uses the compile-time-polymorphic JsonSinkBenchHook so the instrumented
// ParallelFFT3D template picks up per-step wall-time samples without any
// runtime branching. The end-to-end time is measured externally by the
// runner; the per-step data feeds a second JSONL file so plots can show
// e.g. "what fraction of forward() is alltoallw vs. local FFT".
//
// Env:
//   BENCH_FFT_N        cube edge (default 64)  - yields N^3 complex points
//   BENCH_FFT_NX,NY,NZ   non-cubic override (all three must be set)
//   BENCH_FFT_TYPE     "strong" (default) writes shape in config as given
//                      "weak"  multiplies shape by cbrt(ranks) so per-rank
//                              volume is constant
//
// Two JSONL outputs:
//   bench/results/fft_3d.jsonl          - end-to-end forward() timing
//   bench/results/fft_3d_steps.jsonl    - per-step samples (5 series/run)

#include "asio.hpp"
#include "bench/runner.h"
#include "bench/stats.h"
#include "clustr_mpi.h"
#include "clustr/bench_hook.h"
#include "clustr/parallel_fft_3d.h"
#include "dist_array.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using clustr::BenchSink;
using clustr::DistArray;
using clustr::JsonSinkBenchHook;
using clustr::ParallelFFT3D;

namespace {

std::size_t getenv_size(const char* name, std::size_t def) {
    if (const char* v = std::getenv(name)) return std::stoull(v);
    return def;
}

std::string transport_name() {
#if CLUSTR_TRANSPORT == CLUSTR_TRANSPORT_ZERO_COPY
    return "zero_copy";
#else
    return "pack_unpack";
#endif
}

// Round up to nearest multiple of `m`. Pencil FFT requires each axis to
// be divisible by the corresponding grid dim; we don't know the grid
// until construction, but rounding to the LCM of possible small factors
// covers the common cases (2, 3, 4, 6, 8).
std::size_t round_to(std::size_t n, std::size_t m) {
    if (n % m == 0) return n;
    return ((n / m) + 1) * m;
}

}  // namespace

CLUSTR_MPI_MAIN(mpi) {
    const int rank = mpi.rank();
    const int size = mpi.size();
    auto& world = mpi.world();

    // ── Shape selection ─────────────────────────────────────────────────────
    std::size_t NX, NY, NZ;
    const bool custom = std::getenv("BENCH_FFT_NX") &&
                        std::getenv("BENCH_FFT_NY") &&
                        std::getenv("BENCH_FFT_NZ");
    if (custom) {
        NX = getenv_size("BENCH_FFT_NX", 64);
        NY = getenv_size("BENCH_FFT_NY", 64);
        NZ = getenv_size("BENCH_FFT_NZ", 64);
    } else {
        NX = NY = NZ = getenv_size("BENCH_FFT_N", 64);
    }

    const std::string scaling_type =
        std::getenv("BENCH_FFT_TYPE") ? std::getenv("BENCH_FFT_TYPE") : "strong";

    if (scaling_type == "weak") {
        // Scale each axis by cbrt(ranks) so per-rank volume is constant
        // relative to the 1-rank baseline. Round to keep divisibility.
        double factor = std::cbrt(static_cast<double>(size));
        NX = round_to(static_cast<std::size_t>(NX * factor + 0.5), 12);
        NY = round_to(static_cast<std::size_t>(NY * factor + 0.5), 12);
        NZ = round_to(static_cast<std::size_t>(NZ * factor + 0.5), 12);
    }

    // Round for divisibility safety (handles 2, 3, 4, 6 grids).
    NX = round_to(NX, 12);
    NY = round_to(NY, 12);
    NZ = round_to(NZ, 12);

    if (rank == 0) {
        std::cout << "[fft_3d_bench] shape=" << NX << "x" << NY << "x" << NZ
                  << " ranks=" << size
                  << " scaling=" << scaling_type << "\n";
    }

    // ── Set up the FFT object with JSON-sink hook ──────────────────────────
    BenchSink sink;
    JsonSinkBenchHook hook{&sink, rank};

    using shape_t = DistArray<std::complex<double>>::shape_t;
    shape_t global_shape = {NX, NY, NZ};

    ParallelFFT3D<double, JsonSinkBenchHook> fft(
        world, global_shape, /*grid_dims=*/{}, std::move(hook));

    // ── Pre-allocate input/output buffers ONCE outside the timed loop ──────
    DistArray<std::complex<double>> input(
        fft.input_plan_global_shape(), fft.input_local_shape(), 1);
    DistArray<std::complex<double>> freq(
        fft.output_plan_global_shape(), fft.output_local_shape(), 1);

    // Fill with something non-zero so the kernel sees real bytes.
    for (std::size_t i = 0; i < input.size(); ++i)
        input.data()[i] = std::complex<double>(
            static_cast<double>(i & 0xff) / 255.0, 0.0);

    // End-to-end forward() timing.
    const std::string out_path_e2e =
        std::getenv("BENCH_OUT")
            ? std::getenv("BENCH_OUT")
            : std::string("bench/results/fft_3d.jsonl");

    bench::RunOptions opts = bench::default_options("fft_3d_forward");
    opts.config["nx"]        = static_cast<std::int64_t>(NX);
    opts.config["ny"]        = static_cast<std::int64_t>(NY);
    opts.config["nz"]        = static_cast<std::int64_t>(NZ);
    opts.config["ranks"]     = static_cast<std::int64_t>(size);
    opts.config["scaling"]   = scaling_type;
    opts.config["transport"] = transport_name();
    opts.out_path            = out_path_e2e;

    auto op = [&](int iter) -> asio::awaitable<void> {
        // Rewrite input each iteration so the FFT doesn't operate on
        // already-transformed data (forward() destroys its input).
        for (std::size_t i = 0; i < input.size(); ++i) {
            input.data()[i] = std::complex<double>(
                static_cast<double>((i + iter) & 0xff) / 255.0, 0.0);
        }
        co_await fft.forward(input, freq);
        bench::DoNotOptimize(freq.data());
    };

    auto stats = co_await bench::run_bench_async(rank, opts, op);

    if (rank == 0) {
        std::cout << "[fft_3d_forward] p50=" << (stats.p50_ns / 1e6) << "ms"
                  << " p99=" << (stats.p99_ns / 1e6) << "ms"
                  << " min=" << (stats.min_ns / 1e6) << "ms (n="
                  << stats.n << ")\n";
    }

    // ── Emit per-step breakdown from the hook's sink ───────────────────────
    if (rank == 0) {
        const std::string steps_path =
            std::getenv("BENCH_STEPS_OUT")
                ? std::getenv("BENCH_STEPS_OUT")
                : "bench/results/fft_3d_steps.jsonl";

        const auto* sink_ptr = fft.hook().sink();
        const std::vector<clustr::BenchSink::Series> empty;
        for (auto const& series : sink_ptr ? sink_ptr->all() : empty) {
            bench::StatAccumulator acc;
            acc.reserve(series.samples_ns.size());
            for (auto ns : series.samples_ns) acc.record(ns);
            if (acc.size() == 0) continue;

            bench::RunRecord r;
            r.bench         = "fft_3d_step";
            r.config        = opts.config;
            r.config["step"] = series.label;
            r.env           = bench::capture_env();
            r.warmup        = opts.warmup;
            r.iterations    = static_cast<int>(series.samples_ns.size());
            r.stats         = acc.summarize();
            if (opts.record_samples)
                r.samples_ns.assign(series.samples_ns.begin(), series.samples_ns.end());
            bench::write_json_file(steps_path, r);

            std::cout << "  step " << series.label
                      << " p50=" << (r.stats.p50_ns / 1000.0) << "us"
                      << " p99=" << (r.stats.p99_ns / 1000.0) << "us\n";
        }
    }

    co_return 0;
}

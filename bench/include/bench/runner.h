#pragma once

// bench/runner.h - warmup + measurement loop that enforces a common
// protocol across every benchmark in this project.
//
// Every bench follows the same shape:
//
//   1. Run `warmup` iterations of `op`, discard timings.
//      Purpose: prime caches, pager, TLB; let TCP slow-start finish; let
//      any lazy allocation fire at least once.
//
//   2. Run `iterations` iterations of `op`, recording wall-time per
//      iteration via bench::Timer.
//
//   3. Compute percentile stats, attach `env` and `config`, and emit a
//      single JSON record via `write_json_file`.
//
// The op signature is async (asio::awaitable<void>(int)). The int is the
// current iteration index - ops use it to vary tags/seeds so the TCP stream
// doesn't see perfectly identical bytes (which, in pathological cases, can
// hit codec/hash fast paths the compiler doesn't model).
//
// Environment knobs (read from env by run_bench_async):
//   BENCH_WARMUP      default 10
//   BENCH_ITERATIONS  default 100
//   BENCH_OUT         default "bench/results/<bench>.jsonl" (appended)
//   BENCH_SAMPLES     if set, include every sample in the JSON output
//
// Everything here is templated on the op type so the hot loop can inline
// the user function fully. No type-erasure, no std::function.

// Only include asio when the caller actually wants the async runner.
// bench/transfer/* bench files are pure-sync and need to build without
// asio on the include path.
#if __has_include("asio.hpp")
#  include "asio.hpp"
#  define BENCH_HAS_ASIO 1
#endif

#include "bench/env.h"
#include "bench/json_writer.h"
#include "bench/stats.h"
#include "bench/timer.h"

#include <cstdlib>
#include <filesystem>
#include <string>

namespace bench {

struct RunOptions {
    std::string bench_name;
    ConfigMap   config;
    int         warmup     = 10;
    int         iterations = 100;
    bool        record_samples = false;
    std::string out_path;        // default derived from bench_name
};

inline RunOptions default_options(std::string name) {
    RunOptions o;
    o.bench_name = std::move(name);
    if (const char* w = std::getenv("BENCH_WARMUP"))     o.warmup     = std::atoi(w);
    if (const char* n = std::getenv("BENCH_ITERATIONS")) o.iterations = std::atoi(n);
    if (std::getenv("BENCH_SAMPLES"))                    o.record_samples = true;
    if (const char* p = std::getenv("BENCH_OUT")) {
        o.out_path = p;
    } else {
        std::filesystem::create_directories("bench/results");
        o.out_path = "bench/results/" + o.bench_name + ".jsonl";
    }
    if (o.warmup < 0)        o.warmup = 0;
    if (o.iterations <= 0)   o.iterations = 100;
    return o;
}

#ifdef BENCH_HAS_ASIO
// Async runner. `op` must be an invocable returning asio::awaitable<void>
// that accepts a single int (iteration index).
//
// Only rank 0 writes the JSON record. All ranks run the op in lockstep.
// The caller is responsible for ensuring op() is collective-safe.
template <typename Op>
asio::awaitable<Stats>
run_bench_async(int rank, RunOptions opts, Op&& op) {
    // 1. Warmup (throw away)
    for (int i = 0; i < opts.warmup; ++i) {
        co_await op(i);
    }

    // 2. Measurement
    StatAccumulator acc;
    acc.reserve(static_cast<std::size_t>(opts.iterations));

    Timer t;
    for (int i = 0; i < opts.iterations; ++i) {
        ClobberMemory();
        t.start();
        co_await op(i);
        std::int64_t ns = t.stop_ns();
        ClobberMemory();
        acc.record(ns);
    }

    Stats s = acc.summarize();

    // 3. Emit (rank 0 only)
    if (rank == 0) {
        RunRecord r;
        r.bench      = std::move(opts.bench_name);
        r.config     = std::move(opts.config);
        r.env        = capture_env();
        r.warmup     = opts.warmup;
        r.iterations = opts.iterations;
        r.stats      = s;
        if (opts.record_samples) r.samples_ns = acc.raw();
        write_json_file(opts.out_path, r);
    }
    co_return s;
}
#endif // BENCH_HAS_ASIO

// Synchronous variant for benchmarks with no asio runtime (file transfer,
// scheduler in-process). op must be invocable returning void and taking int.
template <typename Op>
Stats run_bench_sync(RunOptions opts, Op&& op) {
    for (int i = 0; i < opts.warmup; ++i) op(i);

    StatAccumulator acc;
    acc.reserve(static_cast<std::size_t>(opts.iterations));

    Timer t;
    for (int i = 0; i < opts.iterations; ++i) {
        ClobberMemory();
        t.start();
        op(i);
        std::int64_t ns = t.stop_ns();
        ClobberMemory();
        acc.record(ns);
    }

    Stats s = acc.summarize();

    RunRecord r;
    r.bench      = std::move(opts.bench_name);
    r.config     = std::move(opts.config);
    r.env        = capture_env();
    r.warmup     = opts.warmup;
    r.iterations = opts.iterations;
    r.stats      = s;
    if (opts.record_samples) r.samples_ns = acc.raw();
    write_json_file(opts.out_path, r);

    return s;
}

}  // namespace bench

// bench_collectives.cpp - unified bench for clustr collectives.
//
// One source file, compile-time selected op via -DBENCH_OP=<name>:
//
//   bcast      - rank 0 ships `bytes` to everyone
//   reduce     - every rank contributes `bytes`, rank 0 owns the sum
//   allreduce  - every rank contributes `bytes`, every rank owns the sum
//   scatter    - rank 0 ships `bytes` total (each rank gets bytes/N)
//   gather     - every rank ships bytes/N, rank 0 owns `bytes`
//   barrier    - ignores bytes
//
// Why one file? Every collective has the same sweep-over-sizes shape and
// the same measurement discipline (warmup, barriers, DoNotOptimize). Five
// near-identical copies is where bugs hide. The op is chosen via
// if-constexpr so the binary for each op is a different artifact (no
// runtime branching in the timed path).
//
// Rank count:
//   Read from the roster (mpi.size()). Any count works; the runner records
//   mpi.size() in the config so results across rank counts stay comparable.
//
// Why a barrier at the top of every iteration?
//   Collectives need every rank to arrive before anyone can start. Without
//   the barrier, the "slowest-starter" cost leaks into the next iteration.
//   The barrier itself is NOT included in the timed region; we measure only
//   the collective under test.

#include "asio.hpp"
#include "bench/runner.h"
#include "clustr_mpi.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef BENCH_OP
#  error "BENCH_OP must be defined: bcast, reduce, allreduce, scatter, gather, barrier"
#endif

#define BENCH_BCAST     1
#define BENCH_REDUCE    2
#define BENCH_ALLREDUCE 3
#define BENCH_SCATTER   4
#define BENCH_GATHER    5
#define BENCH_BARRIER   6

#if BENCH_OP == BENCH_BCAST
constexpr std::string_view kOpName = "bcast";
#elif BENCH_OP == BENCH_REDUCE
constexpr std::string_view kOpName = "reduce";
#elif BENCH_OP == BENCH_ALLREDUCE
constexpr std::string_view kOpName = "allreduce";
#elif BENCH_OP == BENCH_SCATTER
constexpr std::string_view kOpName = "scatter";
#elif BENCH_OP == BENCH_GATHER
constexpr std::string_view kOpName = "gather";
#elif BENCH_OP == BENCH_BARRIER
constexpr std::string_view kOpName = "barrier";
#else
#  error "BENCH_OP must be one of bcast|reduce|allreduce|scatter|gather|barrier"
#endif

namespace {

std::vector<std::size_t> parse_sizes() {
    const char* env = std::getenv("BENCH_SIZES");
    // Collectives: smaller upper bound than P2P since N-way fan-out gets
    // expensive in wall time.
    std::string s = env ? env : "8,64,512,4096,32768,262144,2097152";
    std::vector<std::size_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stoull(tok));
    }
    return out;
}

std::string transport_name() {
#if CLUSTR_TRANSPORT == CLUSTR_TRANSPORT_ZERO_COPY
    return "zero_copy";
#else
    return "pack_unpack";
#endif
}

std::string recv_mode_name() {
#if CLUSTR_RECV == CLUSTR_RECV_CENTRAL
    return "central";
#else
    return "inline";
#endif
}

}  // namespace

CLUSTR_MPI_MAIN(mpi) {
    const int rank = mpi.rank();
    const int size = mpi.size();

    const char* out_env = std::getenv("BENCH_OUT");
    std::string out_path = out_env
        ? out_env
        : (std::string("bench/results/coll_") + std::string(kOpName) + ".jsonl");

#if BENCH_OP == BENCH_BARRIER
    // Barrier ignores size; run a single no-size sweep.
    std::vector<std::size_t> sizes = {0};
#else
    std::vector<std::size_t> sizes = parse_sizes();
#endif

    for (std::size_t bytes : sizes) {
        // Ensure scatter/gather total-bytes are divisible by size (API
        // requirement: data.size() % size() == 0). Round up.
#if BENCH_OP == BENCH_SCATTER || BENCH_OP == BENCH_GATHER
        if (bytes % size != 0) {
            bytes = ((bytes / size) + 1) * size;
        }
#endif

        bench::RunOptions opts = bench::default_options(std::string("coll_") + std::string(kOpName));
        opts.config["bytes"]     = static_cast<std::int64_t>(bytes);
        opts.config["ranks"]     = static_cast<std::int64_t>(size);
        opts.config["transport"] = transport_name();
        opts.config["recv_mode"] = recv_mode_name();
        opts.out_path            = out_path;

        // Pre-allocate the full-size buffer once. The collective-specific
        // chunk size is derived inside op() so no allocations happen in the
        // hot loop.
        std::vector<std::uint8_t> full(bytes, 0x1);
#if BENCH_OP == BENCH_GATHER
        const std::size_t chunk = size > 0 ? (bytes / size) : 0;
#endif

        auto op = [&](int /*iter*/) -> asio::awaitable<void> {
            // Align all ranks before we start timing this iteration. The
            // barrier is outside the Timer scope (the runner starts the
            // clock after this coroutine is entered, but bench::Timer is
            // started per-iteration by run_bench_async). Putting a barrier
            // first inside op() means the timed interval includes the
            // barrier cost - acceptable because it's a real part of
            // every "ready, go" cycle. For a pure-op measurement, remove
            // this line. We keep it because collectives are never used in
            // isolation in practice.
            co_await mpi.barrier();

#if BENCH_OP == BENCH_BCAST
            std::vector<std::uint8_t> data = (rank == 0) ? full : std::vector<std::uint8_t>(bytes);
            co_await mpi.bcast(data, 0);
            bench::DoNotOptimize(data.data());
#elif BENCH_OP == BENCH_REDUCE
            auto r = co_await mpi.reduce(full, clustr::ReduceOp::SUM, 0);
            bench::DoNotOptimize(r.data());
#elif BENCH_OP == BENCH_ALLREDUCE
            auto r = co_await mpi.allreduce(full, clustr::ReduceOp::SUM);
            bench::DoNotOptimize(r.data());
#elif BENCH_OP == BENCH_SCATTER
            auto r = (rank == 0)
                ? co_await mpi.scatter(full, 0)
                : co_await mpi.scatter<std::uint8_t>(0);
            bench::DoNotOptimize(r.data());
#elif BENCH_OP == BENCH_GATHER
            std::vector<std::uint8_t> my_chunk(chunk, 0x1);
            auto r = co_await mpi.gather(my_chunk, 0);
            bench::DoNotOptimize(r.data());
#elif BENCH_OP == BENCH_BARRIER
            // The outer barrier IS the op; nothing else.
            (void)0;
#endif
        };

        auto stats = co_await bench::run_bench_async(rank, opts, op);

        if (rank == 0) {
            std::cout << "[" << kOpName << "] bytes=" << bytes
                      << " ranks=" << size
                      << " p50=" << (stats.p50_ns / 1000.0) << "us"
                      << " p99=" << (stats.p99_ns / 1000.0) << "us"
                      << " (n=" << stats.n << ")\n";
        }

        co_await mpi.barrier();
    }

    co_return 0;
}

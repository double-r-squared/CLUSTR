// bench_p2p_pingpong.cpp - round-trip latency for clustr point-to-point send/recv.
//
// Protocol (2 ranks):
//   rank 0: send payload to rank 1, recv echo from rank 1   (RTT timer)
//   rank 1: recv from rank 0, send identical-size buffer back
//
// Half the RTT is reported as single-direction latency.
//
// Honest-measurement rules this file follows:
//
//   * Buffers are sized ONCE outside the timed loop. Timing a
//     vector<uint8_t>(N) would measure allocator perf, not network.
//
//   * The tag cycles with the iteration index so the receiver cannot
//     short-circuit on an already-mailboxed frame from a prior iteration.
//
//   * We bench::ClobberMemory() around the co_await pair and feed the
//     received buffer to DoNotOptimize so the compiler cannot elide the
//     recv as unused output.
//
//   * Warmup (default 10) covers TCP slow-start; data pages of the
//     send/recv buffers are touched during warmup so the timed iterations
//     see only resident memory.
//
// Sweep:
//   Sizes are taken from the CSV env var BENCH_SIZES (bytes).
//   Default: 8, 64, 512, 4096, 32768, 262144, 2097152, 16777216
//
// Output:
//   One JSONL record per size, at bench/results/p2p_pingpong.jsonl.

#include "asio.hpp"
#include "bench/env.h"
#include "bench/json_writer.h"
#include "bench/runner.h"
#include "bench/stats.h"
#include "bench/timer.h"
#include "clustr_mpi.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::size_t> parse_sizes() {
    const char* env = std::getenv("BENCH_SIZES");
    std::string s = env ? env
                        : "8,64,512,4096,32768,262144,2097152,16777216";
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
    if (size != 2) {
        if (rank == 0)
            std::cerr << "bench_p2p_pingpong: requires exactly 2 ranks (got " << size << ")\n";
        co_return 1;
    }

    const auto sizes = parse_sizes();

    const char* out_env = std::getenv("BENCH_OUT");
    std::string out_path = out_env ? out_env : "bench/results/p2p_pingpong.jsonl";

    for (std::size_t bytes : sizes) {
        // Pre-size a send buffer once. Rank 0 uses it outbound, rank 1 uses it
        // for the reply; both ranks fill with a pattern so the kernel sees
        // real bytes (not zeroed pages) and the compiler sees non-const data.
        std::vector<std::uint8_t> buf(bytes);
        for (std::size_t i = 0; i < bytes; ++i) buf[i] = static_cast<std::uint8_t>(i);

        bench::RunOptions opts = bench::default_options("p2p_pingpong");
        opts.config["bytes"]     = static_cast<std::int64_t>(bytes);
        opts.config["transport"] = transport_name();
        opts.config["recv_mode"] = recv_mode_name();
        opts.out_path            = out_path;

        auto op = [&](int i) -> asio::awaitable<void> {
            // Tag varies per iteration so nothing is confused with a stale
            // mailbox entry from the previous iteration.
            const int tag = i + 1;

            if (rank == 0) {
                bench::DoNotOptimize(buf.data());
                co_await mpi.send(buf, /*dst=*/1, tag);
                auto r = co_await mpi.recv<std::uint8_t>(/*src=*/1, tag);
                bench::DoNotOptimize(r.data());
            } else {
                auto r = co_await mpi.recv<std::uint8_t>(/*src=*/0, tag);
                bench::DoNotOptimize(r.data());
                co_await mpi.send(r, /*dst=*/0, tag);
            }
        };

        auto stats = co_await bench::run_bench_async(rank, opts, op);

        if (rank == 0) {
            const double half_us = stats.p50_ns / 2000.0;
            std::cout << "[pingpong] bytes=" << bytes
                      << " n=" << stats.n
                      << " min=" << (stats.min_ns / 1000.0) << "us"
                      << " p50=" << (stats.p50_ns / 1000.0) << "us"
                      << " p99=" << (stats.p99_ns / 1000.0) << "us"
                      << " one_way_p50=" << half_us << "us\n";
        }

        // Barrier between sizes so the two ranks stay aligned and any
        // in-flight frames drain before we switch tag/size.
        co_await mpi.barrier();
    }

    co_return 0;
}

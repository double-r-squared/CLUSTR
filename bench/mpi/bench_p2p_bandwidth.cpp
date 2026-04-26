// bench_p2p_bandwidth.cpp - streaming send bandwidth (no echo).
//
// Protocol (2 ranks):
//   rank 0: send K messages of `bytes` each, back-to-back. One barrier
//           before the timer starts and one after the last send completes
//           (actually after rank 1 confirms receipt of message K).
//   rank 1: recv K messages, then send a 1-byte "done" token to rank 0.
//
// The "done" token is what closes the timed window on rank 0. Without it
// we'd be measuring enqueue time into the kernel send buffer, not wire-
// completion time - and with TCP buffering that's orders of magnitude
// faster than the actual bandwidth.
//
// Bandwidth is then K * bytes / elapsed.
//
// Honest-measurement rules:
//
//   * K is kept fixed (default 32) so each single "iteration" of the
//     runner measures a full K-message burst. Per-iteration numbers
//     average over the burst, which is what you want for sustained BW
//     because TCP congestion control is steady-state for large enough K.
//
//   * We pre-allocate the send buffer ONCE per size and reuse. Timing a
//     fresh vector per message would measure allocator perf.
//
//   * Each rank is paced by the protocol: rank 0 cannot finish until
//     rank 1 has actually drained the socket. Nagle is off in ASIO's
//     default TCP socket (TCP_NODELAY is not set by default, so it IS
//     on - see OPT.md for the optimization opportunity).
//
// Environment:
//   BENCH_BW_SIZES   default "4096,32768,262144,2097152,16777216"
//   BENCH_BW_BURST   default 32

#include "asio.hpp"
#include "bench/runner.h"
#include "clustr_mpi.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::size_t> parse_sizes() {
    const char* env = std::getenv("BENCH_BW_SIZES");
    std::string s = env ? env : "4096,32768,262144,2097152,16777216";
    std::vector<std::size_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stoull(tok));
    }
    return out;
}

int parse_burst() {
    const char* env = std::getenv("BENCH_BW_BURST");
    return env ? std::max(1, std::atoi(env)) : 32;
}

std::string transport_name() {
#if CLUSTR_TRANSPORT == CLUSTR_TRANSPORT_ZERO_COPY
    return "zero_copy";
#else
    return "pack_unpack";
#endif
}

}  // namespace

CLUSTR_MPI_MAIN(mpi) {
    const int rank = mpi.rank();
    const int size = mpi.size();
    if (size != 2) {
        if (rank == 0)
            std::cerr << "bench_p2p_bandwidth: requires exactly 2 ranks\n";
        co_return 1;
    }

    const auto sizes = parse_sizes();
    const int  burst = parse_burst();

    const char* out_env = std::getenv("BENCH_OUT");
    std::string out_path = out_env ? out_env : "bench/results/p2p_bandwidth.jsonl";

    for (std::size_t bytes : sizes) {
        std::vector<std::uint8_t> buf(bytes);
        for (std::size_t i = 0; i < bytes; ++i) buf[i] = static_cast<std::uint8_t>(i);

        bench::RunOptions opts = bench::default_options("p2p_bandwidth");
        opts.config["bytes"]     = static_cast<std::int64_t>(bytes);
        opts.config["burst"]     = static_cast<std::int64_t>(burst);
        opts.config["transport"] = transport_name();
        opts.out_path            = out_path;

        auto op = [&](int iter) -> asio::awaitable<void> {
            const int base_tag = iter * 10000;
            if (rank == 0) {
                for (int k = 0; k < burst; ++k) {
                    co_await mpi.send(buf, /*dst=*/1, base_tag + k);
                }
                // Wait for the terminal "done" token so we measure wire-
                // completion, not enqueue time.
                auto done = co_await mpi.recv<std::uint8_t>(/*src=*/1, base_tag + burst);
                bench::DoNotOptimize(done.data());
            } else {
                for (int k = 0; k < burst; ++k) {
                    auto r = co_await mpi.recv<std::uint8_t>(/*src=*/0, base_tag + k);
                    bench::DoNotOptimize(r.data());
                }
                std::vector<std::uint8_t> done = {0x01};
                co_await mpi.send(done, /*dst=*/0, base_tag + burst);
            }
        };

        auto stats = co_await bench::run_bench_async(rank, opts, op);

        if (rank == 0) {
            // Bandwidth from the p50 (median) iteration time.
            const double seconds = stats.p50_ns / 1e9;
            const double bytes_total = static_cast<double>(bytes) * burst;
            const double gbps = (bytes_total / seconds) / (1024.0 * 1024.0 * 1024.0);
            std::cout << "[bandwidth] bytes=" << bytes
                      << " burst=" << burst
                      << " p50=" << (stats.p50_ns / 1e6) << "ms"
                      << " bw=" << gbps << " GiB/s"
                      << " (n=" << stats.n << ")\n";
        }

        co_await mpi.barrier();
    }

    co_return 0;
}

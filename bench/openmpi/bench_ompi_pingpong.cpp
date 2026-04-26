// bench_ompi_pingpong.cpp - OpenMPI ping-pong baseline.
//
// Mirrors bench/mpi/bench_p2p_pingpong.cpp as closely as possible so
// the two JSONL outputs are apples-to-apples. Same sizes, same warmup,
// same per-iteration timer protocol, same JSON schema.
//
// The point of this binary is NOT to win or lose; it's to give an
// HONEST reference so we can see where clustr_mpi is competitive and
// where it's paying a cost for being pure-userspace TCP.
//
// Build requires `mpicxx` in PATH. See scripts/bench_openmpi.sh for
// a driver that skips this bench gracefully when MPI is missing.

#include "bench/env.h"
#include "bench/json_writer.h"
#include "bench/stats.h"
#include "bench/timer.h"

#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::size_t> parse_sizes() {
    const char* env = std::getenv("BENCH_SIZES");
    std::string s = env ? env : "8,64,512,4096,32768,262144,2097152,16777216";
    std::vector<std::size_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stoull(tok));
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            std::cerr << "bench_ompi_pingpong requires exactly 2 ranks\n";
        MPI_Finalize();
        return 1;
    }

    const char* out_env = std::getenv("BENCH_OUT");
    std::string out_path = out_env ? out_env : "bench/results/ompi_pingpong.jsonl";

    int warmup     = 10;
    int iterations = 100;
    if (const char* w = std::getenv("BENCH_WARMUP"))     warmup     = std::atoi(w);
    if (const char* n = std::getenv("BENCH_ITERATIONS")) iterations = std::atoi(n);

    for (std::size_t bytes : parse_sizes()) {
        std::vector<std::uint8_t> buf(bytes, 0xAB);

        // Warmup.
        for (int i = 0; i < warmup; ++i) {
            if (rank == 0) {
                MPI_Send(buf.data(), bytes, MPI_BYTE, 1, i + 1, MPI_COMM_WORLD);
                MPI_Recv(buf.data(), bytes, MPI_BYTE, 1, i + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buf.data(), bytes, MPI_BYTE, 0, i + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf.data(), bytes, MPI_BYTE, 0, i + 1, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        bench::StatAccumulator acc;
        acc.reserve(static_cast<std::size_t>(iterations));

        bench::Timer t;
        for (int i = 0; i < iterations; ++i) {
            bench::ClobberMemory();
            t.start();
            if (rank == 0) {
                MPI_Send(buf.data(), bytes, MPI_BYTE, 1, i + 100, MPI_COMM_WORLD);
                MPI_Recv(buf.data(), bytes, MPI_BYTE, 1, i + 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buf.data(), bytes, MPI_BYTE, 0, i + 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf.data(), bytes, MPI_BYTE, 0, i + 100, MPI_COMM_WORLD);
            }
            std::int64_t ns = t.stop_ns();
            bench::ClobberMemory();
            bench::DoNotOptimize(buf.data());
            acc.record(ns);
        }

        auto s = acc.summarize();

        if (rank == 0) {
            bench::RunRecord r;
            r.bench = "ompi_pingpong";
            r.config["bytes"]     = static_cast<std::int64_t>(bytes);
            r.config["transport"] = std::string("openmpi");
            r.env        = bench::capture_env();
            r.warmup     = warmup;
            r.iterations = iterations;
            r.stats      = s;
            if (std::getenv("BENCH_SAMPLES"))
                r.samples_ns = acc.raw();
            bench::write_json_file(out_path, r);

            std::cout << "[ompi_pingpong] bytes=" << bytes
                      << " p50=" << (s.p50_ns / 1000.0) << "us"
                      << " p99=" << (s.p99_ns / 1000.0) << "us\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

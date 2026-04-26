// bench_ompi_bcast.cpp - OpenMPI MPI_Bcast baseline for clustr bcast.
//
// Same measurement protocol as bench/mpi/bench_collectives.cpp (BENCH_OP=bcast):
// barrier-aligned per iteration, per-iteration timer, same JSON schema.

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
    std::string s = env ? env : "8,64,512,4096,32768,262144,2097152";
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

    const char* out_env = std::getenv("BENCH_OUT");
    std::string out_path = out_env ? out_env : "bench/results/ompi_bcast.jsonl";

    int warmup     = 10;
    int iterations = 100;
    if (const char* w = std::getenv("BENCH_WARMUP"))     warmup     = std::atoi(w);
    if (const char* n = std::getenv("BENCH_ITERATIONS")) iterations = std::atoi(n);

    for (std::size_t bytes : parse_sizes()) {
        std::vector<std::uint8_t> buf(bytes, rank == 0 ? 0xAB : 0);

        for (int i = 0; i < warmup; ++i) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(buf.data(), bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
        }

        bench::StatAccumulator acc;
        acc.reserve(static_cast<std::size_t>(iterations));

        bench::Timer t;
        for (int i = 0; i < iterations; ++i) {
            bench::ClobberMemory();
            t.start();
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(buf.data(), bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
            std::int64_t ns = t.stop_ns();
            bench::ClobberMemory();
            bench::DoNotOptimize(buf.data());
            acc.record(ns);
        }

        auto s = acc.summarize();

        if (rank == 0) {
            bench::RunRecord r;
            r.bench = "ompi_bcast";
            r.config["bytes"] = static_cast<std::int64_t>(bytes);
            r.config["ranks"] = static_cast<std::int64_t>(size);
            r.config["transport"] = std::string("openmpi");
            r.env        = bench::capture_env();
            r.warmup     = warmup;
            r.iterations = iterations;
            r.stats      = s;
            if (std::getenv("BENCH_SAMPLES"))
                r.samples_ns = acc.raw();
            bench::write_json_file(out_path, r);

            std::cout << "[ompi_bcast] bytes=" << bytes
                      << " ranks=" << size
                      << " p50=" << (s.p50_ns / 1000.0) << "us\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

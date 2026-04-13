// python_scatter_gather.cpp — scatter/gather with Python doing the compute.
//
// Identical pattern to mpi_scatter_gather.cpp but the per-rank work
// happens inside a Python script instead of inline C++.
//
// Flow (per rank):
//   1. rank 0 builds full dataset, scatter sends chunks to all ranks
//   2. Each rank writes its chunk to a temp file
//   3. Each rank runs: python3 <script> --input <in> --output <out>
//   4. Each rank reads the result file back into memory
//   5. gather collects all results at rank 0
//
// The Python script is a pure function: read input file, compute, write
// output file. It knows nothing about MPI, rosters, or networking.
// All reporting goes through std::cout in the C++ harness, exactly like
// every other job in this folder.
//
// Env:
//   CLUSTR_PYTHON_SCRIPT  path to the .py file (required)
//
// Submit with Ranks = 3.
//
// Expected output (with example_python_job.py doubling values):
//   [rank 0] Scattered chunk: 1 2
//   [rank 1] Scattered chunk: 3 4
//   [rank 2] Scattered chunk: 5 6
//   [rank 0] Python script: example_python_job.py
//   [rank 0] Python exited with code 0
//   [rank 0] Processed chunk: 2 4
//   [rank 1] Python exited with code 0
//   [rank 1] Processed chunk: 6 8
//   [rank 2] Python exited with code 0
//   [rank 2] Processed chunk: 10 12
//   [rank 0] Final result: 2 4 6 8 10 12

#include "clustr_mpi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Write a vector of doubles to a binary file.
bool write_doubles(const std::string& path, const std::vector<double>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(double)));
    return f.good();
}

// Read a binary file back into a vector of doubles.
std::vector<double> read_doubles(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto size = f.tellg();
    f.seekg(0);
    std::vector<double> data(static_cast<std::size_t>(size) / sizeof(double));
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

// Build a temp file path unique to this rank.
std::string tmp_path(int rank, const char* suffix) {
    return "/tmp/clustr_py_rank" + std::to_string(rank) + suffix;
}

}  // namespace

CLUSTR_MPI_MAIN(mpi) {
    int rank = mpi.rank();
    int size = mpi.size();

    // ── Locate Python script ───────────────────────────────────────────────
    // 1. CLUSTR_PYTHON_SCRIPT env var (used by the test harness)
    // 2. Fallback: "example_python_job.py" in the current working dir
    //    (the scheduler deploys companion files alongside the binary)
    std::string script;
    const char* script_env = std::getenv("CLUSTR_PYTHON_SCRIPT");
    if (script_env && script_env[0] != '\0') {
        script = script_env;
    } else {
        script = "example_python_job.py";
        // Verify it exists before proceeding
        std::ifstream probe(script);
        if (!probe.good()) {
            if (rank == 0)
                std::cerr << "[rank 0] ERROR: Python script not found."
                          << " Set CLUSTR_PYTHON_SCRIPT or place "
                          << script << " in the working directory.\n";
            co_return 1;
        }
    }

    if (rank == 0)
        std::cout << "[rank 0] Python script: " << script << std::endl;

    // ── Step 1: scatter ────────────────────────────────────────────────────
    std::vector<double> full_data;
    if (rank == 0) {
        int n = size * 2;
        full_data.resize(n);
        std::iota(full_data.begin(), full_data.end(), 1.0);
    }

    auto chunk = co_await mpi.scatter(full_data, /*root=*/0);

    std::cout << "[rank " << rank << "] Scattered chunk:";
    for (double v : chunk) std::cout << " " << v;
    std::cout << std::endl;

    // ── Step 2: write chunk to temp file ───────────────────────────────────
    std::string in_path  = tmp_path(rank, "_in.bin");
    std::string out_path = tmp_path(rank, "_out.bin");

    if (!write_doubles(in_path, chunk)) {
        std::cerr << "[rank " << rank << "] ERROR: failed to write " << in_path << "\n";
        co_return 1;
    }

    // ── Step 3: run Python ─────────────────────────────────────────────────
    // Build command. Python's stdout/stderr go through popen → we forward
    // to our own stdout so the TUI sees it like any other job output.
    std::ostringstream cmd;
    cmd << "python3 " << script
        << " --input " << in_path
        << " --output " << out_path
        << " --rank " << rank
        << " --size " << size
        << " 2>&1";

    FILE* proc = popen(cmd.str().c_str(), "r");
    if (!proc) {
        std::cerr << "[rank " << rank << "] ERROR: popen failed\n";
        co_return 1;
    }

    // Forward all Python output to our stdout (same stream the TUI reads)
    char buf[256];
    while (fgets(buf, sizeof(buf), proc))
        std::cout << buf;

    int status = pclose(proc);
    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

    std::cout << "[rank " << rank << "] Python exited with code "
              << exit_code << std::endl;

    if (exit_code != 0) {
        std::cerr << "[rank " << rank << "] ERROR: Python failed\n";
        co_return 1;
    }

    // ── Step 4: read result back ───────────────────────────────────────────
    auto result_chunk = read_doubles(out_path);

    if (result_chunk.empty()) {
        std::cerr << "[rank " << rank << "] ERROR: empty result from " << out_path << "\n";
        co_return 1;
    }

    std::cout << "[rank " << rank << "] Processed chunk:";
    for (double v : result_chunk) std::cout << " " << v;
    std::cout << std::endl;

    // Cleanup temp files
    std::remove(in_path.c_str());
    std::remove(out_path.c_str());

    // ── Step 5: gather ─────────────────────────────────────────────────────
    auto result = co_await mpi.gather(result_chunk, /*root=*/0);

    if (rank == 0) {
        std::cout << "[rank 0] Final result:";
        for (double v : result) std::cout << " " << v;
        std::cout << std::endl;
    }

    co_return 0;
}

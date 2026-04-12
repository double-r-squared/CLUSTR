// mpi_scatter_gather.cpp — MPI scatter, parallel work, and gather.
//
// What this demonstrates:
//   - scatter: rank 0 splits a dataset into equal chunks and sends one
//              chunk to each rank (rank 0 keeps its own chunk too)
//   - parallel work: each rank processes its chunk independently
//   - gather: all chunks flow back to rank 0 in rank order
//
// This is the fundamental pattern for data parallelism — the most common
// reason to use MPI. Instead of one machine processing 1000 numbers,
// you split them across N machines and each processes 1000/N numbers
// simultaneously. Total wall time drops by roughly N.
//
// The work here:
//   rank 0 has [1, 2, 3, 4, 5, 6] (with 3 ranks, chunk size = 2)
//   scatter sends:
//     rank 0 → [1, 2]
//     rank 1 → [3, 4]
//     rank 2 → [5, 6]
//   each rank doubles its chunk:
//     rank 0 → [2, 4]
//     rank 1 → [6, 8]
//     rank 2 → [10, 12]
//   gather collects at rank 0: [2, 4, 6, 8, 10, 12]
//
// Submit with Ranks = 3. Data size must be divisible by number of ranks.
//
// Expected output:
//   [rank 0] Scattered chunk: 1 2
//   [rank 1] Scattered chunk: 3 4
//   [rank 2] Scattered chunk: 5 6
//   [rank 0] Processed chunk: 2 4
//   [rank 1] Processed chunk: 6 8
//   [rank 2] Processed chunk: 10 12
//   [rank 0] Final result: 2 4 6 8 10 12
//
// Compile: g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED
//              -I./include -I./asio_include mpi_scatter_gather.cpp -o mpi_scatter_gather -lpthread
// Run:     ./mpi_scatter_gather

#include "clustr_mpi.h"
#include <iostream>
#include <vector>
#include <numeric>

CLUSTR_MPI_MAIN(mpi) {
    int rank = mpi.rank();
    int size = mpi.size();

    // Step 1: rank 0 builds the full dataset.
    // Other ranks pass an empty vector — scatter ignores it on non-root.
    std::vector<double> full_data;
    if (rank == 0) {
        // 6 elements for 3 ranks = 2 per rank. Adjust to match your rank count.
        int n = size * 2;
        full_data.resize(n);
        std::iota(full_data.begin(), full_data.end(), 1.0);  // fill: 1,2,3,4,5,6
    }

    // Step 2: scatter — rank 0 splits full_data into size equal chunks.
    // Each rank receives its own chunk. rank 0 also gets its chunk (no self-send overhead).
    auto chunk = co_await mpi.scatter(full_data, /*root=*/0);

    // Print what we received
    std::cout << "[rank " << rank << "] Scattered chunk:";
    for (double v : chunk) std::cout << " " << v;
    std::cout << std::endl;

    // Step 3: each rank does its own work on its chunk independently.
    // No communication here — this is the parallel speedup.
    for (double& v : chunk) v *= 2.0;

    std::cout << "[rank " << rank << "] Processed chunk:";
    for (double v : chunk) std::cout << " " << v;
    std::cout << std::endl;

    // Step 4: gather — all chunks flow back to rank 0 in rank order.
    // rank 0 gets the full reconstructed dataset.
    // Non-root ranks get an empty vector back.
    auto result = co_await mpi.gather(chunk, /*root=*/0);

    if (rank == 0) {
        std::cout << "[rank 0] Final result:";
        for (double v : result) std::cout << " " << v;
        std::cout << std::endl;
    }

    co_return 0;
}

// mpi_reduce.cpp — MPI reduce and broadcast.
//
// What this demonstrates:
//   - Each rank independently computes something (a partial sum)
//   - reduce() collects all partial results to rank 0 and combines them
//   - bcast() then sends the final answer back out to everyone
//   - Together, reduce + bcast = allreduce (every rank ends up with the result)
//
// Real-world analogy:
//   Imagine 4 workers each counting votes in their district.
//   reduce() is the central tally — all district counts flow to HQ (rank 0).
//   bcast() announces the final total back to every district.
//
// The math here:
//   Each rank contributes the value (rank + 1) * 10.
//   For 3 ranks: 10 + 20 + 30 = 60.
//   rank 0 reduces to get 60, then broadcasts it so everyone knows.
//
// Submit with Ranks = 3.
//
// Expected output:
//   [rank 0] My value: 10
//   [rank 1] My value: 20
//   [rank 2] My value: 30
//   [rank 0] Total across all ranks: 60
//   [rank 0] Broadcast result: 60
//   [rank 1] Received result: 60
//   [rank 2] Received result: 60
//
// Compile: g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED
//              -I./include -I./asio_include mpi_reduce.cpp -o mpi_reduce -lpthread
// Run:     ./mpi_reduce

#include "clustr_mpi.h"
#include <iostream>
#include <vector>

CLUSTR_MPI_MAIN(mpi) {
    int rank = mpi.rank();
    int size = mpi.size();

    // Step 1: each rank computes its own local value independently.
    // No communication yet — this is pure local work.
    double my_value = (rank + 1) * 10.0;
    std::cout << "[rank " << rank << "] My value: " << my_value << std::endl;

    // Wrap it in a vector — reduce/bcast work on vectors of any length.
    // For a single number we use a 1-element vector.
    std::vector<double> local = { my_value };

    // Step 2: reduce — all ranks send their vector to rank 0.
    // rank 0 adds them element-by-element (ReduceOp::SUM).
    // Non-root ranks get back an empty vector.
    auto total = co_await mpi.reduce(local, clustr::ReduceOp::SUM, /*root=*/0);

    if (rank == 0)
        std::cout << "[rank 0] Total across all ranks: " << total[0] << std::endl;

    // Step 3: broadcast — rank 0 sends the result to everyone.
    // After this, every rank has the same 'total' vector.
    // (On non-root ranks, 'total' was empty before this call.)
    co_await mpi.bcast(total, /*root=*/0);

    // Now every rank can use the global result
    std::cout << "[rank " << rank << "] "
              << (rank == 0 ? "Broadcast result: " : "Received result: ")
              << total[0] << std::endl;

    co_return 0;
}

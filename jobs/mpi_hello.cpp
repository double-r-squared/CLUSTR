// mpi_hello.cpp — MPI fundamentals: rank identity and barrier.
//
// What this demonstrates:
//   - Every worker in the group gets a unique rank (0, 1, 2, ...)
//   - mpi.rank() tells you who you are
//   - mpi.size() tells you how many workers are in the group
//   - barrier() is a synchronization point — no rank moves past it
//     until every rank has reached it
//   - rank 0 is conventionally the "root" — it speaks for the group
//
// Think of ranks like runners in a relay race. Each runner knows their
// lane number (rank) and the total number of runners (size). The barrier
// is the starting gun — nobody runs until everyone is at the line.
//
// Submit with Ranks = 2 or more in the Jobs page submit dialog.
//
// Expected output (order of the first prints is not guaranteed):
//   [rank 1] Hello from rank 1 of 3
//   [rank 0] Hello from rank 0 of 3
//   [rank 2] Hello from rank 2 of 3
//   [rank 0] All ranks checked in. Done.
//
// Compile: g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED
//              -I./include -I./asio_include mpi_hello.cpp -o mpi_hello -lpthread
// Run:     ./mpi_hello

#include "clustr_mpi.h"
#include <iostream>

CLUSTR_MPI_MAIN(mpi) {
    int rank = mpi.rank();  // which worker am I?  (0, 1, 2, ...)
    int size = mpi.size();  // how many workers total?

    // Every rank executes this line independently and simultaneously.
    // The output order is non-deterministic — network timing decides it.
    std::cout << "[rank " << rank << "] Hello from rank "
              << rank << " of " << size << std::endl;

    // barrier: every rank blocks here until ALL ranks arrive.
    // Without this, rank 0 might print "Done" before rank 2 says hello.
    co_await mpi.barrier();

    // This only runs on rank 0, and only after the barrier releases.
    if (rank == 0)
        std::cout << "[rank 0] All ranks checked in. Done." << std::endl;

    co_return 0;
}

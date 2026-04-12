// redistribute_test.cpp - Phase 6 exit criterion: RedistributePlan<T>
// correctness on a 2D DistArray.
//
// What this test validates (per ROADMAP.md Phase 6):
//
//   Build a 2D array distributed along axis 0, redistribute to axis 1,
//   verify that element [i][j] on the original equals element [i][j] on
//   the redistributed array - i.e. the data is logically the same, only
//   the distribution changed.
//
//   We fill the input using GLOBAL coordinates:
//     f(i, j) = i * 1000.0 + j
//   Each rank fills its local slab knowing its own my_v_start offset.
//   After execute(), each output rank (distributed along axis 1) checks
//   that local element (i, jl) equals f(i, my_w_start + jl).
//
// Shape choice:
//
//   Global shape is {9, 8}. With 3 ranks:
//     axis 0: 9 / 3 = 3, rem 0  -> all ranks get 3 rows    (uniform)
//     axis 1: 8 / 3 = 2, rem 2  -> ranks 0,1 get 3 cols, rank 2 gets 2
//   With 2 ranks:
//     axis 0: 9 / 2 = 4, rem 1  -> rank 0 gets 5, rank 1 gets 4
//     axis 1: 8 / 2 = 4, rem 0  -> both ranks get 4
//   Either way at least one axis hits the off-by-one path that exposes
//   any start/extent bugs in balanced_block.
//
// Phase 2 (bonus): Round-trip axis1 -> axis0 and verify the result equals
//   the original input. Catches asymmetric bugs where v->w works but the
//   reverse path has a mirrored geometry error.
//
// Compiles under both CLUSTR_TRANSPORT modes and both CLUSTR_RECV modes
// (alltoallw uses the world comm's per-key channel lookup under CENTRAL
// and the inline socket path under INLINE).
//
// Submit with Ranks >= 2. On 1 rank the redistribute is a no-op and the
// test short-circuits with a diagnostic.

#include "clustr_mpi.h"
#include "clustr/redistribute.h"
#include "dist_array.h"

#include <cstddef>
#include <iostream>
#include <vector>

using clustr::DistArray;
using clustr::RedistributePlan;

namespace {

constexpr std::size_t kGlobalRows = 9;
constexpr std::size_t kGlobalCols = 8;

double fill_value(std::size_t global_i, std::size_t global_j) {
    return static_cast<double>(global_i) * 1000.0
         + static_cast<double>(global_j);
}

}  // namespace

CLUSTR_MPI_MAIN(mpi) {
    const int rank = mpi.rank();
    const int size = mpi.size();

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "[redistribute_test] requires Ranks >= 2 (got "
                      << size << ")\n";
        }
        co_return 1;
    }

    auto& world = mpi.world();

    // ── Plan the v=0 -> w=1 redistribute ─────────────────────────────────
    using shape_t = DistArray<double>::shape_t;
    shape_t global_shape = {kGlobalRows, kGlobalCols};

    RedistributePlan<double> fwd(world, global_shape, /*v=*/0, /*w=*/1);

    // Allocate input (distributed on axis 0) and output (distributed on axis 1)
    // using the shapes the plan expects.
    DistArray<double> input (global_shape, fwd.input_local_shape(),  /*axis=*/0);
    DistArray<double> output(global_shape, fwd.output_local_shape(), /*axis=*/1);

    // ── Fill input using GLOBAL coordinates ──────────────────────────────
    // Each rank owns rows [my_v_start, my_v_start + my_v_size) along axis 0
    // and the full 8 columns along axis 1.
    const std::size_t my_v0 = fwd.my_v_start();
    const std::size_t my_vn = fwd.my_v_size();
    for (std::size_t il = 0; il < my_vn; ++il) {
        const std::size_t global_i = my_v0 + il;
        for (std::size_t j = 0; j < kGlobalCols; ++j) {
            input.at(il, j) = fill_value(global_i, j);
        }
    }

    co_await fwd.execute(input, output);

    // ── Verify the output ────────────────────────────────────────────────
    // After forward redistribute, each rank owns the full 9 rows and
    // columns [my_w_start, my_w_start + my_w_size) along axis 1.
    const std::size_t my_w0 = fwd.my_w_start();
    const std::size_t my_wn = fwd.my_w_size();

    std::size_t fwd_mismatches = 0;
    for (std::size_t i = 0; i < kGlobalRows; ++i) {
        for (std::size_t jl = 0; jl < my_wn; ++jl) {
            const std::size_t global_j = my_w0 + jl;
            const double got  = output.at(i, jl);
            const double want = fill_value(i, global_j);
            if (got != want) {
                if (fwd_mismatches < 3) {
                    std::cerr << "[rank " << rank << "] fwd mismatch at ("
                              << i << ", " << jl << ") global (" << i << ", "
                              << global_j << ") got " << got << " want "
                              << want << "\n";
                }
                ++fwd_mismatches;
            }
        }
    }
    if (fwd_mismatches == 0) {
        std::cout << "[rank " << rank << "] forward v=0 -> w=1 OK ("
                  << my_wn << " local cols)\n";
    } else {
        std::cerr << "[rank " << rank << "] forward: " << fwd_mismatches
                  << " mismatches\n";
    }

    co_await world.barrier();

    // ── Round-trip: redistribute back to axis 0 ──────────────────────────
    // Plan the reverse and allocate a fresh sink. Reusing `input` would
    // also work since its shape matches, but a clean buffer proves the
    // reverse path isn't cheating by leftover data.
    RedistributePlan<double> rev(world, global_shape, /*v=*/1, /*w=*/0);
    DistArray<double> round_trip(
        global_shape, rev.output_local_shape(), /*axis=*/0);

    co_await rev.execute(output, round_trip);

    // round_trip should now be bit-exact equal to the original input: same
    // shape, same distribution (axis 0), same my_v_start/size, and every
    // element should be fill_value(global_i, j).
    std::size_t rev_mismatches = 0;
    for (std::size_t il = 0; il < my_vn; ++il) {
        const std::size_t global_i = my_v0 + il;
        for (std::size_t j = 0; j < kGlobalCols; ++j) {
            const double got  = round_trip.at(il, j);
            const double want = fill_value(global_i, j);
            if (got != want) {
                if (rev_mismatches < 3) {
                    std::cerr << "[rank " << rank << "] rev mismatch at ("
                              << il << ", " << j << ") global (" << global_i
                              << ", " << j << ") got " << got << " want "
                              << want << "\n";
                }
                ++rev_mismatches;
            }
        }
    }
    if (rev_mismatches == 0) {
        std::cout << "[rank " << rank << "] reverse w=0 -> v=0 OK ("
                  << my_vn << " local rows)\n";
    } else {
        std::cerr << "[rank " << rank << "] reverse: " << rev_mismatches
                  << " mismatches\n";
    }

    co_await world.barrier();

    const int failures =
        static_cast<int>(fwd_mismatches != 0) +
        static_cast<int>(rev_mismatches != 0);

    if (rank == 0) {
        if (failures == 0)
            std::cout << "[rank 0] redistribute_test: ALL PASS\n";
        else
            std::cout << "[rank 0] redistribute_test: FAIL\n";
    }

    co_return failures;
}

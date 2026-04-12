// fft_3d_test.cpp - Phase 8 exit criterion: parallel 3D pencil FFT.
//
// Test signal: a single complex sinusoid at bin (K0, K1, K2) = (3, 7, 5).
//
//   x[i][j][k] = exp(2*pi*i * (K0*i/N0 + K1*j/N1 + K2*k/N2))
//
// After the forward FFT (unnormalized), the spectrum should have a single
// peak at bin (3, 7, 5) with magnitude N0*N1*N2 = 262144 and all other bins
// at machine-epsilon noise. After inverse, round-trip error must be < 1e-10.
//
// Grid selection:
//   Ranks = 3  → {1, 3} grid (degenerate: P0=1, P1=3; exercises P1 path)
//   Ranks = 4  → {2, 2} grid (balanced)
//   Ranks = 6  → {2, 3} grid (exit criterion)
//   Otherwise  → skip with co_return 0
//
// Compiles under both CLUSTR_TRANSPORT modes and CLUSTR_RECV_CENTRAL.
// Requires CLUSTR_RECV=CLUSTR_RECV_CENTRAL (cart_sub is unavailable otherwise).

#include "clustr_mpi.h"
#include "clustr/parallel_fft_3d.h"
#include "dist_array.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <vector>

using clustr::DistArray;
using clustr::ParallelFFT3D;

namespace {

constexpr std::size_t N0 = 64;
constexpr std::size_t N1 = 64;
constexpr std::size_t N2 = 64;
constexpr std::size_t K0 = 3;
constexpr std::size_t K1 = 7;
constexpr std::size_t K2 = 5;
constexpr double PI2 = 6.283185307179586476925286766559;

std::complex<double> signal_value(std::size_t i, std::size_t j, std::size_t k) {
    double phase = PI2 * (static_cast<double>(K0 * i) / static_cast<double>(N0)
                        + static_cast<double>(K1 * j) / static_cast<double>(N1)
                        + static_cast<double>(K2 * k) / static_cast<double>(N2));
    return {std::cos(phase), std::sin(phase)};
}

// Select grid dimensions based on rank count.
// Returns {P0, P1} or throws if no valid grid exists.
std::vector<int> select_grid(int world_size) {
    if (world_size == 3)
        return {1, 3};   // degenerate, exercises P1 path fully
    if (world_size == 4)
        return {2, 2};   // balanced 2D grid
    if (world_size == 6)
        return {2, 3};   // exit criterion grid
    throw std::runtime_error(
        "fft_3d_test: no valid grid for Ranks=" + std::to_string(world_size)
        + "; supported: 3 (1×3), 4 (2×2), 6 (2×3)");
}

}  // namespace

CLUSTR_MPI_MAIN(mpi) {
    const int rank = mpi.rank();
    const int size = mpi.size();

    auto& world = mpi.world();

    // ── Grid selection ──────────────────────────────────────────────────────
    std::vector<int> grid_dims;
    try {
        grid_dims = select_grid(size);
    } catch (const std::runtime_error& e) {
        if (rank == 0)
            std::cerr << "[fft_3d_test] " << e.what() << "\n";
        co_return 0;  // skip silently
    }

    if (rank == 0)
        std::cout << "[fft_3d_test] Using grid " << grid_dims[0] << " × "
                  << grid_dims[1] << " (" << size << " ranks)\n";

    // ── Construct FFT object ────────────────────────────────────────────────
    using shape_t = DistArray<std::complex<double>>::shape_t;
    shape_t global_shape = {N0, N1, N2};

    ParallelFFT3D<double> fft(world, global_shape, grid_dims);

    if (rank == 0) {
        std::cout << "[fft_3d_test] Grid: " << grid_dims[0] << " x " << grid_dims[1] << "\n";
        std::cout << "[fft_3d_test] my_n0: [" << fft.my_n0_start() << ", " << (fft.my_n0_start() + fft.my_n0_size()) << ")\n";
        std::cout << "[fft_3d_test] my_n1 (P1): [" << fft.my_n1_start() << ", " << (fft.my_n1_start() + fft.my_n1_size()) << ")\n";
        std::cout << "[fft_3d_test] my_n2 (P1): [" << fft.my_n2_start() << ", " << (fft.my_n2_start() + fft.my_n2_size()) << ")\n";
        std::cout << "[fft_3d_test] input_local_shape: {" << fft.input_local_shape()[0] << ", " << fft.input_local_shape()[1] << ", " << fft.input_local_shape()[2] << "}\n";
        std::cout << "[fft_3d_test] output_local_shape: {" << fft.output_local_shape()[0] << ", " << fft.output_local_shape()[1] << ", " << fft.output_local_shape()[2] << "}\n";
    }

    // ── Allocate buffers with plan global shapes ────────────────────────────
    // For pencil decomposition, arrays are distributed on TWO axes.
    // Since DistArray only supports tracking one, we use the axis that
    // the first redistribution operates on:
    // - input: distributed_axis=1 (P1 will redistribute this axis)
    // - output: distributed_axis=1 (P0 will redistribute this axis)
    DistArray<std::complex<double>> input(
        fft.input_plan_global_shape(), fft.input_local_shape(),
        1);  // P1 operates on this (axis 1 is split by P1 in input state)
    DistArray<std::complex<double>> freq(
        fft.output_plan_global_shape(), fft.output_local_shape(),
        1);  // P0 operates on this (axis 1 is split by P0 in output state)

    const std::size_t n0_start = fft.my_n0_start();
    const std::size_t n0_size  = fft.my_n0_size();
    const std::size_t n1_start = fft.my_n1_start();  // P1's split of axis 1
    const std::size_t n1_size  = fft.my_n1_size();

    // ── Fill input with test signal ─────────────────────────────────────────
    // Input state A: {N0/P0, N1/P1, N2} - axis 2 is fully replicated (not distributed)
    for (std::size_t il = 0; il < n0_size; ++il) {
        const std::size_t gi = n0_start + il;
        for (std::size_t jl = 0; jl < n1_size; ++jl) {
            const std::size_t gj = n1_start + jl;
            for (std::size_t kl = 0; kl < N2; ++kl) {  // All of axis 2
                input.at(il, jl, kl) = signal_value(gi, gj, kl);
            }
        }
    }

    std::vector<std::complex<double>> input_copy(
        input.data(), input.data() + input.size());

    if (rank == 0)
        std::cout << "[rank 0] input allocated and filled\n";

    // ── Forward transform ───────────────────────────────────────────────────
    co_await fft.forward(input, freq);

    // ── Check spectrum ──────────────────────────────────────────────────────
    const double expected_peak = static_cast<double>(N0 * N1 * N2);
    const double peak_tol  = 1e-6;
    const double noise_tol = 1e-6;

    std::size_t peak_errors = 0;
    std::size_t noise_errors = 0;
    double max_noise = 0.0;

    const std::size_t n1_p0_start = fft.my_n1_p0_start();  // P0's split of axis 1
    const std::size_t n1_p0_size  = fft.my_n1_p0_size();
    const std::size_t n2_p1_start = fft.my_n2_start();      // P1's split of axis 2
    const std::size_t n2_p1_size  = fft.my_n2_size();

    for (std::size_t il = 0; il < N0; ++il) {
        for (std::size_t jl = 0; jl < n1_p0_size; ++jl) {
            const std::size_t gj = n1_p0_start + jl;
            for (std::size_t kl = 0; kl < n2_p1_size; ++kl) {
                const std::size_t gk = n2_p1_start + kl;
                const double mag = std::abs(freq.at(il, jl, kl));
                if (il == K0 && gj == K1 && gk == K2) {
                    if (std::abs(mag - expected_peak) > peak_tol) {
                        std::cerr << "[rank " << rank << "] peak (" << K0 << ","
                                  << K1 << "," << K2 << ") mag=" << mag
                                  << " expected=" << expected_peak << "\n";
                        ++peak_errors;
                    }
                } else {
                    if (mag > noise_tol) ++noise_errors;
                    if (mag > max_noise) max_noise = mag;
                }
            }
        }
    }

    bool owns_peak = (K1 >= n1_p0_start && K1 < n1_p0_start + n1_p0_size
                   && K2 >= n2_p1_start && K2 < n2_p1_start + n2_p1_size);
    if (owns_peak) {
        std::cout << "[rank " << rank << "] peak mag="
                  << std::abs(freq.at(K0, K1 - n1_p0_start, K2 - n2_p1_start))
                  << " expected=" << expected_peak
                  << " max_noise=" << max_noise << "\n";
    } else {
        std::cout << "[rank " << rank << "] axis1 [" << n1_p0_start << ","
                  << n1_p0_start + n1_p0_size << ") axis2 [" << n2_p1_start << ","
                  << n2_p1_start + n2_p1_size << ") max_noise=" << max_noise << "\n";
    }

    co_await world.barrier();

    // ── Inverse transform (round-trip) ──────────────────────────────────────
    DistArray<std::complex<double>> result(
        global_shape, fft.input_local_shape(), 1);

    co_await fft.inverse(freq, result);

    // Result has same shape as input: {N0/P0, N1/P1, N2}
    const std::size_t result_n2 = N2;  // axis 2 is fully replicated in result
    double max_err = 0.0;
    for (std::size_t il = 0; il < n0_size; ++il) {
        for (std::size_t jl = 0; jl < n1_size; ++jl) {
            for (std::size_t kl = 0; kl < result_n2; ++kl) {
                const std::size_t lidx = il * n1_size * result_n2 + jl * result_n2 + kl;
                double err = std::abs(result.at(il, jl, kl) - input_copy[lidx]);
                if (err > max_err) max_err = err;
            }
        }
    }

    std::cout << "[rank " << rank << "] round-trip max_err=" << max_err << "\n";

    co_await world.barrier();

    int failures = static_cast<int>(peak_errors != 0)
                 + static_cast<int>(noise_errors != 0)
                 + static_cast<int>(max_err >= 1e-10);

    if (rank == 0) {
        if (failures == 0)
            std::cout << "[rank 0] fft_3d_test: ALL PASS\n";
        else
            std::cout << "[rank 0] fft_3d_test: FAIL (peak_errors=" << peak_errors
                      << " noise_errors=" << noise_errors
                      << " max_err=" << max_err << ")\n";
    }

    co_return failures;
}

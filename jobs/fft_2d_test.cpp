// fft_2d_test.cpp - Phase 7 exit criterion: parallel 2D slab FFT.
//
// Test signal: a single complex sinusoid at bin (K0, K1) = (3, 7).
//
//   x[i][j] = exp(2*pi*i * (K0*i/N0 + K1*j/N1))
//
// After the forward FFT (unnormalized), the spectrum should have a single
// peak at bin (3, 7) with magnitude N0*N1 = 65536 and all other bins at
// machine-epsilon noise. After inverse, round-trip error must be < 1e-10.
//
// Submit with Ranks >= 2. On 3 ranks, axis 0 decomposes 256 -> {86,85,85}
// and axis 1 also -> {86,85,85}, exercising the balanced-block remainder.
//
// Compiles under both CLUSTR_TRANSPORT modes and both CLUSTR_RECV modes.

#include "clustr_mpi.h"
#include "clustr/parallel_fft_2d.h"
#include "dist_array.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <vector>

using clustr::DistArray;
using clustr::ParallelFFT2D;

namespace {

constexpr std::size_t N0 = 256;
constexpr std::size_t N1 = 256;
constexpr std::size_t K0 = 3;
constexpr std::size_t K1 = 7;
constexpr double PI2 = 6.283185307179586476925286766559;

std::complex<double> signal_value(std::size_t i, std::size_t j) {
    double phase = PI2 * (static_cast<double>(K0 * i) / static_cast<double>(N0)
                        + static_cast<double>(K1 * j) / static_cast<double>(N1));
    return {std::cos(phase), std::sin(phase)};
}

}  // namespace

CLUSTR_MPI_MAIN(mpi) {
    const int rank = mpi.rank();
    const int size = mpi.size();

    if (size < 2) {
        if (rank == 0)
            std::cerr << "[fft_2d_test] requires Ranks >= 2 (got "
                      << size << ")\n";
        co_return 1;
    }

    auto& world = mpi.world();

    using shape_t = DistArray<std::complex<double>>::shape_t;
    shape_t global_shape = {N0, N1};

    ParallelFFT2D<double> fft(world, global_shape);

    // ── Allocate ─────────────────────────────────────────────────────────
    DistArray<std::complex<double>> input(
        global_shape, fft.input_local_shape(), 0);
    DistArray<std::complex<double>> freq(
        global_shape, fft.output_local_shape(), 1);

    const auto& fplan = fft.fwd_plan();
    const std::size_t v0 = fplan.my_v_start();
    const std::size_t vn = fplan.my_v_size();
    const std::size_t w0 = fplan.my_w_start();
    const std::size_t wn = fplan.my_w_size();

    // ── Fill input ───────────────────────────────────────────────────────
    for (std::size_t il = 0; il < vn; ++il)
        for (std::size_t j = 0; j < N1; ++j)
            input.at(il, j) = signal_value(v0 + il, j);

    std::vector<std::complex<double>> input_copy(
        input.data(), input.data() + input.size());

    // ── Forward ──────────────────────────────────────────────────────────
    co_await fft.forward(input, freq);

    // ── Check spectrum ───────────────────────────────────────────────────
    const double expected_peak = static_cast<double>(N0 * N1);
    const double peak_tol  = 1e-6;
    const double noise_tol = 1e-6;

    std::size_t peak_errors = 0;
    std::size_t noise_errors = 0;
    double max_noise = 0.0;

    for (std::size_t i = 0; i < N0; ++i) {
        for (std::size_t jl = 0; jl < wn; ++jl) {
            const std::size_t gj = w0 + jl;
            const double mag = std::abs(freq.at(i, jl));
            if (i == K0 && gj == K1) {
                if (std::abs(mag - expected_peak) > peak_tol) {
                    std::cerr << "[rank " << rank << "] peak (" << K0 << ","
                              << K1 << ") mag=" << mag << " expected="
                              << expected_peak << "\n";
                    ++peak_errors;
                }
            } else {
                if (mag > noise_tol) ++noise_errors;
                if (mag > max_noise) max_noise = mag;
            }
        }
    }

    bool owns_peak = (K1 >= w0 && K1 < w0 + wn);
    if (owns_peak) {
        std::cout << "[rank " << rank << "] peak mag="
                  << std::abs(freq.at(K0, K1 - w0))
                  << " expected=" << expected_peak
                  << " max_noise=" << max_noise << "\n";
    } else {
        std::cout << "[rank " << rank << "] cols [" << w0 << "," << w0 + wn
                  << ") max_noise=" << max_noise << "\n";
    }

    co_await world.barrier();

    // ── Inverse (round-trip) ─────────────────────────────────────────────
    DistArray<std::complex<double>> result(
        global_shape, fft.input_local_shape(), 0);

    co_await fft.inverse(freq, result);

    double max_err = 0.0;
    for (std::size_t il = 0; il < vn; ++il)
        for (std::size_t j = 0; j < N1; ++j) {
            double err = std::abs(result.at(il, j) - input_copy[il * N1 + j]);
            if (err > max_err) max_err = err;
        }

    std::cout << "[rank " << rank << "] round-trip max_err=" << max_err << "\n";

    co_await world.barrier();

    int failures = static_cast<int>(peak_errors != 0)
                 + static_cast<int>(noise_errors != 0)
                 + static_cast<int>(max_err >= 1e-10);

    if (rank == 0) {
        if (failures == 0)
            std::cout << "[rank 0] fft_2d_test: ALL PASS\n";
        else
            std::cout << "[rank 0] fft_2d_test: FAIL (" << failures
                      << " checks failed)\n";
    }

    co_return failures;
}

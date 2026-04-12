// fft_3d_test_auto.cpp - Phase 9: test auto grid selection
//
// Same as fft_3d_test.cpp but passes empty grid_dims to trigger auto_grid_selection()
// Built with -DCLUSTR_BENCHMARK to show performance breakdown per step.

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

}  // namespace

CLUSTR_MPI_MAIN(mpi) {
    const int rank = mpi.rank();
    const int size = mpi.size();
    auto& world = mpi.world();

    if (rank == 0)
        std::cout << "[fft_3d_test_auto] Testing auto grid selection with " << size << " ranks\n";

    // ── Construct FFT object with EMPTY grid_dims to trigger auto-selection ──
    using shape_t = DistArray<std::complex<double>>::shape_t;
    shape_t global_shape = {N0, N1, N2};

    std::vector<int> empty_grid_dims;  // Empty → auto-select
    ParallelFFT3D<double> fft(world, global_shape, empty_grid_dims);

    if (rank == 0) {
        std::cout << "[fft_3d_test_auto] FFT configured, proceeding with test\n";
    }

    // ── Allocate buffers ────────────────────────────────────────────────────
    DistArray<std::complex<double>> input(
        fft.input_plan_global_shape(), fft.input_local_shape(), 1);
    DistArray<std::complex<double>> freq(
        fft.output_plan_global_shape(), fft.output_local_shape(), 1);

    const std::size_t n0_start = fft.my_n0_start();
    const std::size_t n0_size  = fft.my_n0_size();
    const std::size_t n1_start = fft.my_n1_start();
    const std::size_t n1_size  = fft.my_n1_size();

    // ── Fill input with test signal ─────────────────────────────────────────
    for (std::size_t il = 0; il < n0_size; ++il) {
        const std::size_t gi = n0_start + il;
        for (std::size_t jl = 0; jl < n1_size; ++jl) {
            const std::size_t gj = n1_start + jl;
            for (std::size_t kl = 0; kl < N2; ++kl) {
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

    const std::size_t n1_p0_start = fft.my_n1_p0_start();
    const std::size_t n1_p0_size  = fft.my_n1_p0_size();
    const std::size_t n2_p1_start = fft.my_n2_start();
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

    const std::size_t result_n2 = N2;
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
            std::cout << "[rank 0] fft_3d_test_auto: ALL PASS\n";
        else
            std::cout << "[rank 0] fft_3d_test_auto: FAIL (peak_errors=" << peak_errors
                      << " noise_errors=" << noise_errors
                      << " max_err=" << max_err << ")\n";
    }

    co_return failures;
}

// fft_serial_test.cpp - Phase 0 sanity test: serial multidimensional FFT.
//
// What this does:
//   - Builds a small 3D complex array with deterministic content
//   - Runs an in-place PocketFFT forward transform
//   - Runs an inverse transform
//   - Verifies element-wise round-trip error is below 1e-10
//
// No MPI, no redistribution, no sub-communicators. This exists to prove the
// serial building block (PocketFFT) works and that DistArray hands PocketFFT
// the right shape + byte strides.
//
// Submit with Ranks = 1.
//
// Compile: g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED
//              -I./include -I./asio_include fft_serial_test.cpp
//              -o fft_serial_test -lpthread
// Run:     ./fft_serial_test

#include "clustr_mpi.h"
#include "dist_array.h"
#include "pocketfft_hdronly.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <vector>

namespace pf = pocketfft;

using cplx = std::complex<double>;

CLUSTR_MPI_MAIN(mpi) {
    if (mpi.size() != 1) {
        if (mpi.rank() == 0) {
            std::cerr << "fft_serial_test expects Ranks=1, got " << mpi.size()
                      << std::endl;
        }
        co_return 1;
    }

    // Small but non-trivial: 8 x 16 x 32 complex doubles.
    const std::vector<std::size_t> shape = { 8, 16, 32 };

    clustr::DistArray<cplx> x  = clustr::DistArray<cplx>::serial(shape);
    clustr::DistArray<cplx> x0 = clustr::DistArray<cplx>::serial(shape);

    // Fill with a deterministic pattern. Arithmetic is done in signed ints
    // to avoid unsigned underflow from size_t loop counters.
    std::size_t idx = 0;
    for (std::size_t i = 0; i < shape[0]; ++i) {
        for (std::size_t j = 0; j < shape[1]; ++j) {
            for (std::size_t k = 0; k < shape[2]; ++k) {
                int ii = static_cast<int>(i);
                int jj = static_cast<int>(j);
                int kk = static_cast<int>(k);
                double re = static_cast<double>(ii * 7 + jj * 3 + kk);
                double im = static_cast<double>(ii - 2 * jj + kk / 4);
                x.at(i, j, k)  = cplx{ re, im };
                x0.at(i, j, k) = cplx{ re, im };
                ++idx;
            }
        }
    }

    std::cout << "[rank 0] shape = " << shape[0] << "x" << shape[1] << "x"
              << shape[2] << " (" << idx << " elements)" << std::endl;

    // All axes transformed, in place, same stride for in and out.
    const pf::shape_t  pf_shape  = { shape[0], shape[1], shape[2] };
    const pf::stride_t pf_stride = x.strides_bytes();
    const pf::shape_t  axes      = { 0, 1, 2 };

    // Forward: normalize by 1.0, inverse by 1/N to make round-trip identity.
    const double norm = 1.0 / static_cast<double>(x.size());

    pf::c2c<double>(pf_shape, pf_stride, pf_stride, axes,
                    /*forward=*/true, x.data(), x.data(), /*fct=*/1.0);

    pf::c2c<double>(pf_shape, pf_stride, pf_stride, axes,
                    /*forward=*/false, x.data(), x.data(), /*fct=*/norm);

    // Verify element-wise round-trip error.
    double max_err = 0.0;
    for (std::size_t i = 0; i < shape[0]; ++i) {
        for (std::size_t j = 0; j < shape[1]; ++j) {
            for (std::size_t k = 0; k < shape[2]; ++k) {
                cplx diff = x.at(i, j, k) - x0.at(i, j, k);
                double e  = std::abs(diff);
                if (e > max_err) max_err = e;
            }
        }
    }

    std::cout << "[rank 0] max round-trip error = " << max_err << std::endl;

    constexpr double tol = 1e-10;
    if (max_err > tol) {
        std::cout << "[rank 0] FAIL: error exceeds tolerance " << tol
                  << std::endl;
        co_return 1;
    }

    std::cout << "[rank 0] PASS: serial 3D FFT round-trip within "
              << tol << std::endl;
    co_return 0;
}

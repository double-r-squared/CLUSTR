#pragma once

// parallel_fft_2d.h - Phase 7: parallel 2D slab FFT.
//
// Distributed 2D complex-to-complex FFT using slab decomposition over a
// 1D process group. The array is distributed along one axis at a time;
// each direction requires exactly one redistribute (alltoallw) and two
// local 1D FFTs (PocketFFT).
//
// Forward path (input distributed on axis 0):
//   1. Local FFT along axis 1 (contiguous, no communication)
//   2. Redistribute axis 0 -> axis 1
//   3. Local FFT along axis 0 (now contiguous after redistribute)
//   Output is distributed on axis 1.
//
// Inverse path (input distributed on axis 1):
//   Mirror of forward. Output is distributed on axis 0.
//
// A forward+inverse round-trip restores the original distribution and
// data (within floating-point tolerance).
//
// Normalization: forward fct=1, inverse fct=1/N (split per-axis).
// This is the standard physicist convention matching PocketFFT's default.
//
// Both forward() and inverse() destroy the input buffer (the first local
// FFT operates in-place on the input before the redistribute). Caller
// must copy if the original data is needed.

#include "asio.hpp"
#include "pocketfft_hdronly.h"
#include "clustr/redistribute.h"
#include "dist_array.h"

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace clustr {

template <typename T>
class ParallelFFT2D {
    static_assert(std::is_floating_point_v<T>,
        "ParallelFFT2D<T>: T must be a floating-point type (double, float)");

public:
    using complex_t = std::complex<T>;
    using shape_t   = typename DistArray<complex_t>::shape_t;

    ParallelFFT2D(Comm& comm, shape_t global_shape)
        : comm_(comm),
          global_shape_(validate_2d(std::move(global_shape))),
          fwd_plan_(comm, global_shape_, 0, 1),
          inv_plan_(comm, global_shape_, 1, 0),
          inv_norm_axis0_(T(1) / static_cast<T>(global_shape_[0])),
          inv_norm_axis1_(T(1) / static_cast<T>(global_shape_[1]))
    {}

    const shape_t& global_shape()       const noexcept { return global_shape_; }
    const shape_t& input_local_shape()  const noexcept { return fwd_plan_.input_local_shape(); }
    const shape_t& output_local_shape() const noexcept { return fwd_plan_.output_local_shape(); }

    const RedistributePlan<complex_t>& fwd_plan() const noexcept { return fwd_plan_; }
    const RedistributePlan<complex_t>& inv_plan() const noexcept { return inv_plan_; }

    asio::awaitable<void> forward(DistArray<complex_t>& input,
                                  DistArray<complex_t>& output) {
        validate_forward(input, output);

        pocketfft::c2c<T>(
            input.local_shape(), input.strides_bytes(), input.strides_bytes(),
            {1}, true, input.data(), input.data(), T(1));

        co_await fwd_plan_.execute(input, output);

        pocketfft::c2c<T>(
            output.local_shape(), output.strides_bytes(), output.strides_bytes(),
            {0}, true, output.data(), output.data(), T(1));
    }

    asio::awaitable<void> inverse(DistArray<complex_t>& input,
                                  DistArray<complex_t>& output) {
        validate_inverse(input, output);

        pocketfft::c2c<T>(
            input.local_shape(), input.strides_bytes(), input.strides_bytes(),
            {0}, false, input.data(), input.data(), inv_norm_axis0_);

        co_await inv_plan_.execute(input, output);

        pocketfft::c2c<T>(
            output.local_shape(), output.strides_bytes(), output.strides_bytes(),
            {1}, false, output.data(), output.data(), inv_norm_axis1_);
    }

private:
    Comm&       comm_;
    shape_t     global_shape_;
    RedistributePlan<complex_t> fwd_plan_;
    RedistributePlan<complex_t> inv_plan_;
    T           inv_norm_axis0_;
    T           inv_norm_axis1_;

    static shape_t validate_2d(shape_t s) {
        if (s.size() != 2)
            throw std::invalid_argument(
                "ParallelFFT2D: global_shape must be 2D");
        return s;
    }

    void validate_forward(const DistArray<complex_t>& input,
                          const DistArray<complex_t>& output) const {
        if (input.local_shape() != fwd_plan_.input_local_shape())
            throw std::invalid_argument(
                "ParallelFFT2D::forward: input local_shape mismatch");
        if (output.local_shape() != fwd_plan_.output_local_shape())
            throw std::invalid_argument(
                "ParallelFFT2D::forward: output local_shape mismatch");
        if (input.distributed_axis() != 0)
            throw std::invalid_argument(
                "ParallelFFT2D::forward: input must be distributed on axis 0");
        if (output.distributed_axis() != 1)
            throw std::invalid_argument(
                "ParallelFFT2D::forward: output must be distributed on axis 1");
    }

    void validate_inverse(const DistArray<complex_t>& input,
                          const DistArray<complex_t>& output) const {
        if (input.local_shape() != inv_plan_.input_local_shape())
            throw std::invalid_argument(
                "ParallelFFT2D::inverse: input local_shape mismatch");
        if (output.local_shape() != inv_plan_.output_local_shape())
            throw std::invalid_argument(
                "ParallelFFT2D::inverse: output local_shape mismatch");
        if (input.distributed_axis() != 1)
            throw std::invalid_argument(
                "ParallelFFT2D::inverse: input must be distributed on axis 1");
        if (output.distributed_axis() != 0)
            throw std::invalid_argument(
                "ParallelFFT2D::inverse: output must be distributed on axis 0");
    }
};

}  // namespace clustr

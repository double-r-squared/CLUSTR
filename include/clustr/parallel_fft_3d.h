#pragma once

// parallel_fft_3d.h - Phase 8: parallel 3D pencil FFT.
//
// Distributed 3D complex-to-complex FFT using pencil decomposition over a
// 2D process grid (P0 rows × P1 columns). The array is distributed along
// two axes simultaneously; each forward transform requires exactly two
// redistributions (one per sub-comm) and three local 1D FFTs (PocketFFT).
//
// Forward path (input pencil along axis 2: [N0/P0, N1/P1, N2]):
//   1. Local FFT along axis 2 (contiguous, no communication)
//   2. Redistribute via P1 comm: axis 1 ↔ 2 (alltoallw)
//   3. Local FFT along axis 1 (now contiguous after redistribute)
//   4. Redistribute via P0 comm: axis 0 ↔ 1 (alltoallw)
//   5. Local FFT along axis 0 (now contiguous after redistribute)
//   Output is pencil along axis 0: [N0, N1/P0, N2/P1].
//
// Inverse path (input pencil along axis 0):
//   Mirror of forward (reverse order, inverse FFTs, swapped axes).
//   Output is pencil along axis 2 (same distribution as input).
//
// A forward+inverse round-trip restores the original distribution and
// data (within floating-point tolerance).
//
// Normalization: forward fct=1, inverse split 1/N across three axes.
// Physicist convention matching PocketFFT's default.
//
// Both forward() and inverse() destroy the input buffer (the first local
// FFT operates in-place). Caller must copy if the original data is needed.
//
// Pencil state representation: Each intermediate buffer is a DistArray with
// a "sub-global" shape that absorbs the OTHER sub-comm's partition. This
// avoids extending DistArray to support multi-axis distribution. Each
// RedistributePlan operates on a 1D sub-comm and sees the contracted global.

#include "asio.hpp"
#include "pocketfft_hdronly.h"
#include "clustr/redistribute.h"
#include "dist_array.h"

#include <complex>
#include <cstddef>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace clustr {

template <typename T>
class ParallelFFT3D {
    static_assert(std::is_floating_point_v<T>,
        "ParallelFFT3D<T>: T must be a floating-point type (double, float)");

public:
    using complex_t = std::complex<T>;
    using shape_t   = typename DistArray<complex_t>::shape_t;

    // Constructor: set up 2D Cartesian topology and all redistribution plans.
    // Validates 3D shape and that product(grid_dims) == world.size().
    ParallelFFT3D(Comm& world, shape_t global_shape, std::vector<int> grid_dims)
        : world_(world),
          global_shape_(validate_3d(std::move(global_shape))),
          grid_dims_(validate_grid_dims(std::move(grid_dims), world.size()))
    {
        // ── Establish Cartesian topology ────────────────────────────────────
        world_.cart_create(grid_dims_);

        // ── Initialize sub-communicators (must be after cart_create) ─────────
        p0_comm_.emplace(world_.cart_sub(0));  // col comms, size=P0
        p1_comm_.emplace(world_.cart_sub(1));  // row comms, size=P1

        const int P0 = grid_dims_[0];
        const int P1 = grid_dims_[1];
        const int my_p0 = p0_comm_->rank();
        const int my_p1 = p1_comm_->rank();

        // ── Compute local sizes via balanced block decomposition ────────────
        const auto [n0_start, n0_size] = redistribute_detail::balanced_block(
            global_shape_[0], static_cast<std::size_t>(P0), static_cast<std::size_t>(my_p0));
        const auto [n1_by_p1_start, n1_by_p1_size] = redistribute_detail::balanced_block(
            global_shape_[1], static_cast<std::size_t>(P1), static_cast<std::size_t>(my_p1));
        const auto [n2_start, n2_size] = redistribute_detail::balanced_block(
            global_shape_[2], static_cast<std::size_t>(P1), static_cast<std::size_t>(my_p1));
        const auto [n1_by_p0_start, n1_by_p0_size] = redistribute_detail::balanced_block(
            global_shape_[1], static_cast<std::size_t>(P0), static_cast<std::size_t>(my_p0));

        my_n0_start_    = n0_start;
        my_n0_size_     = n0_size;
        my_n1_by_p1_start_ = n1_by_p1_start;
        my_n1_by_p1_size_  = n1_by_p1_size;
        my_n1_by_p0_start_ = n1_by_p0_start;
        my_n1_by_p0_size_  = n1_by_p0_size;
        my_n2_start_    = n2_start;
        my_n2_size_     = n2_size;

        // ── Compute expected local shapes for validation ─────────────────────
        input_local_shape_  = {n0_size, n1_by_p1_size, global_shape_[2]};
        output_local_shape_ = {global_shape_[0], n1_by_p0_size, n2_size};

        // ── Intermediate buffer (scratch) local shape ───────────────────────
        // After P1 redistribute: [N0/P0, N1, N2/P1]
        scratch_b_local_shape_ = {n0_size, global_shape_[1], n2_size};

        // ── Sub-global shapes for each redistribution plan ──────────────────
        // P1 plan: sees [N0/P0, N1, N2] (P0's axis-0 partition is fixed)
        shape_t p1_plan_global = {n0_size, global_shape_[1], global_shape_[2]};
        // P0 plan: sees [N0, N1, N2/P1] (P1's axis-2 partition is fixed)
        shape_t p0_plan_global = {global_shape_[0], global_shape_[1], n2_size};

        // ── Construct the four redistribution plans ─────────────────────────
        //
        // Forward:
        //   P1 step 2: v=1 (un-distribute axis 1), w=2 (distribute axis 2)
        //   P0 step 4: v=0 (un-distribute axis 0), w=1 (distribute axis 1)
        // Inverse:
        //   P0 step 2: v=1 (un-distribute axis 1), w=0 (distribute axis 0)
        //   P1 step 4: v=2 (un-distribute axis 2), w=1 (distribute axis 1)
        p1_fwd_plan_.emplace(*p1_comm_, p1_plan_global, /*v=*/1, /*w=*/2);
        p0_fwd_plan_.emplace(*p0_comm_, p0_plan_global, /*v=*/0, /*w=*/1);
        p1_inv_plan_.emplace(*p1_comm_, p1_plan_global, /*v=*/2, /*w=*/1);
        p0_inv_plan_.emplace(*p0_comm_, p0_plan_global, /*v=*/1, /*w=*/0);

        // ── Allocate internal scratch buffers ──────────────────────────────────
        // Forward: State B is distributed on axes 0 (by P0) and 2 (by P1).
        // RedistributePlan will check distributed_axis on both input and output.
        // To make both operations happy, we must share data but can't use separate DistArray
        // objects (each allocates its own storage). Instead, allocate two buffers and
        // copy data between them. This is acceptable since alltoallw dominates the cost.
        scratch_b_p1_view_.emplace(p1_plan_global, scratch_b_local_shape_, /*distributed_axis=*/2);
        scratch_b_p0_view_.emplace(p1_plan_global, scratch_b_local_shape_, /*distributed_axis=*/0);

        // Inverse: State B' (after P0 inverse redistribution) similarly needs two buffers.
        scratch_b_p0_inv_view_.emplace(p0_plan_global, scratch_b_local_shape_, /*distributed_axis=*/0);
        scratch_b_p1_inv_view_.emplace(p1_plan_global, scratch_b_local_shape_, /*distributed_axis=*/2);

        // ── Normalization factors for inverse FFTs ──────────────────────────
        inv_norm_axis0_ = T(1) / static_cast<T>(global_shape_[0]);
        inv_norm_axis1_ = T(1) / static_cast<T>(global_shape_[1]);
        inv_norm_axis2_ = T(1) / static_cast<T>(global_shape_[2]);
    }

    // ── Accessors for caller to allocate input/output buffers ──────────────

    const shape_t& global_shape()       const noexcept { return global_shape_; }
    const shape_t& input_local_shape()  const noexcept { return input_local_shape_; }
    const shape_t& output_local_shape() const noexcept { return output_local_shape_; }

    // Global index ranges for filling test data or verifying results
    std::size_t my_n0_start()    const noexcept { return my_n0_start_; }
    std::size_t my_n0_size()     const noexcept { return my_n0_size_; }
    std::size_t my_n1_start()    const noexcept { return my_n1_by_p1_start_; }  // P1's split of axis 1
    std::size_t my_n1_size()     const noexcept { return my_n1_by_p1_size_; }
    std::size_t my_n1_p0_start() const noexcept { return my_n1_by_p0_start_; }  // P0's split of axis 1
    std::size_t my_n1_p0_size()  const noexcept { return my_n1_by_p0_size_; }
    std::size_t my_n2_start()    const noexcept { return my_n2_start_; }
    std::size_t my_n2_size()     const noexcept { return my_n2_size_; }

    // ── Forward and inverse transforms ───────────────────────────────────

    // Accessors for "plan global shapes" (sub-global shapes from each sub-comm's perspective)
    // Caller should create DistArrays with these global_shapes to pass to forward/inverse
    shape_t input_plan_global_shape() const noexcept {
        return {my_n0_size_, global_shape_[1], global_shape_[2]};
    }
    shape_t output_plan_global_shape() const noexcept {
        return {global_shape_[0], global_shape_[1], my_n2_size_};
    }

    asio::awaitable<void> forward(DistArray<complex_t>& input,
                                  DistArray<complex_t>& output) {
        validate_forward(input, output);

        // Step 1: Local FFT along axis 2 (in-place on input)
        pocketfft::c2c<T>(
            input.local_shape(), input.strides_bytes(), input.strides_bytes(),
            {2}, true, input.data(), input.data(), T(1));

        // Step 2: Redistribute axis 1→2 via P1 comm (input → scratch_b_p1_view_)
        co_await p1_fwd_plan_->execute(input, *scratch_b_p1_view_);

        // Sync: copy data from scratch_b_p1_view_ to scratch_b_p0_view_ for next operation
        // Both have same local_shape and strides, just different distributed_axis annotation
        std::copy(scratch_b_p1_view_->data(),
                  scratch_b_p1_view_->data() + scratch_b_p1_view_->size(),
                  scratch_b_p0_view_->data());

        // Step 3: Local FFT along axis 1 (in-place on scratch_b_p0_view_)
        pocketfft::c2c<T>(
            scratch_b_p0_view_->local_shape(), scratch_b_p0_view_->strides_bytes(), scratch_b_p0_view_->strides_bytes(),
            {1}, true, scratch_b_p0_view_->data(), scratch_b_p0_view_->data(), T(1));

        // Step 4: Redistribute axis 0→1 via P0 comm (scratch_b_p0_view_ → output)
        co_await p0_fwd_plan_->execute(*scratch_b_p0_view_, output);

        // Step 5: Local FFT along axis 0 (in-place on output)
        pocketfft::c2c<T>(
            output.local_shape(), output.strides_bytes(), output.strides_bytes(),
            {0}, true, output.data(), output.data(), T(1));
    }

    asio::awaitable<void> inverse(DistArray<complex_t>& input,
                                  DistArray<complex_t>& output) {
        validate_inverse(input, output);

        // Step 1: Inverse FFT along axis 0 with normalization (in-place on input)
        pocketfft::c2c<T>(
            input.local_shape(), input.strides_bytes(), input.strides_bytes(),
            {0}, false, input.data(), input.data(), inv_norm_axis0_);

        // Step 2: Redistribute axis 1→0 via P0 comm (input → scratch_b_p0_inv_view_)
        co_await p0_inv_plan_->execute(input, *scratch_b_p0_inv_view_);

        // Step 3: Inverse FFT along axis 1 with normalization (in-place on scratch_b_p0_inv_view_)
        pocketfft::c2c<T>(
            scratch_b_p0_inv_view_->local_shape(), scratch_b_p0_inv_view_->strides_bytes(), scratch_b_p0_inv_view_->strides_bytes(),
            {1}, false, scratch_b_p0_inv_view_->data(), scratch_b_p0_inv_view_->data(), inv_norm_axis1_);

        // Sync: copy data from scratch_b_p0_inv_view_ to scratch_b_p1_inv_view_ for next operation
        std::copy(scratch_b_p0_inv_view_->data(),
                  scratch_b_p0_inv_view_->data() + scratch_b_p0_inv_view_->size(),
                  scratch_b_p1_inv_view_->data());

        // Step 4: Redistribute axis 2→1 via P1 comm (scratch_b_p1_inv_view_ → output)
        co_await p1_inv_plan_->execute(*scratch_b_p1_inv_view_, output);

        // Step 5: Inverse FFT along axis 2 with normalization (in-place on output)
        pocketfft::c2c<T>(
            output.local_shape(), output.strides_bytes(), output.strides_bytes(),
            {2}, false, output.data(), output.data(), inv_norm_axis2_);
    }

private:
    Comm&                           world_;
    std::optional<Comm>             p0_comm_;  // col sub-comm, size=P0
    std::optional<Comm>             p1_comm_;  // row sub-comm, size=P1
    shape_t                         global_shape_;
    std::vector<int>                grid_dims_;

    // Local sizes computed via balanced_block
    std::size_t my_n0_start_, my_n0_size_;
    std::size_t my_n1_by_p1_start_, my_n1_by_p1_size_;
    std::size_t my_n1_by_p0_start_, my_n1_by_p0_size_;
    std::size_t my_n2_start_, my_n2_size_;

    // Expected shapes for validation
    shape_t input_local_shape_;
    shape_t output_local_shape_;
    shape_t scratch_b_local_shape_;

    // Redistribution plans (4 total: fwd P1, fwd P0, inv P0, inv P1)
    std::optional<RedistributePlan<complex_t>> p1_fwd_plan_;
    std::optional<RedistributePlan<complex_t>> p0_fwd_plan_;
    std::optional<RedistributePlan<complex_t>> p1_inv_plan_;
    std::optional<RedistributePlan<complex_t>> p0_inv_plan_;

    // Internal scratch buffer (state B: after P1 redistribute)
    // Since state B is distributed on both axes 0 (by P0) and 2 (by P1),
    // we maintain two DistArray views of the same data:
    // - scratch_b_p1_view_: distributed_axis=2 (for P1 output, P0 input view)
    // - scratch_b_p0_view_: distributed_axis=0 (for P0 input)
    // Similarly for inverse state B':
    // - scratch_b_p0_inv_view_: distributed_axis=1 (for P0 output, P1 input view)
    // - scratch_b_p1_inv_view_: distributed_axis=2 (for P1 input)
    std::optional<DistArray<complex_t>> scratch_b_p1_view_;  // P1 output view (dist_axis=2)
    std::optional<DistArray<complex_t>> scratch_b_p0_view_;  // P0 input view (dist_axis=0)
    std::optional<DistArray<complex_t>> scratch_b_p0_inv_view_;  // inv P0 output (dist_axis=1)
    std::optional<DistArray<complex_t>> scratch_b_p1_inv_view_;  // inv P1 input (dist_axis=2)

    // Normalization factors for inverse FFTs
    T inv_norm_axis0_;
    T inv_norm_axis1_;
    T inv_norm_axis2_;

    // ── Validation helpers ──────────────────────────────────────────────────

    static shape_t validate_3d(shape_t s) {
        if (s.size() != 3)
            throw std::invalid_argument(
                "ParallelFFT3D: global_shape must be 3D");
        if (s[0] == 0 || s[1] == 0 || s[2] == 0)
            throw std::invalid_argument(
                "ParallelFFT3D: all dimensions must be > 0");
        return s;
    }

    static std::vector<int> validate_grid_dims(std::vector<int> dims, int world_size) {
        if (dims.size() != 2)
            throw std::invalid_argument(
                "ParallelFFT3D: grid_dims must be 2D [P0, P1]");
        if (dims[0] <= 0 || dims[1] <= 0)
            throw std::invalid_argument(
                "ParallelFFT3D: all grid dimensions must be > 0");
        const int product = dims[0] * dims[1];
        if (product != world_size)
            throw std::invalid_argument(
                "ParallelFFT3D: product(grid_dims) (" + std::to_string(product)
                + ") must equal world.size() (" + std::to_string(world_size) + ")");
        return dims;
    }

    void validate_forward(const DistArray<complex_t>& input,
                          const DistArray<complex_t>& output) const {
        if (input.local_shape() != input_local_shape_)
            throw std::invalid_argument(
                "ParallelFFT3D::forward: input local_shape mismatch");
        if (output.local_shape() != output_local_shape_)
            throw std::invalid_argument(
                "ParallelFFT3D::forward: output local_shape mismatch");
    }

    void validate_inverse(const DistArray<complex_t>& input,
                          const DistArray<complex_t>& output) const {
        if (input.local_shape() != output_local_shape_)
            throw std::invalid_argument(
                "ParallelFFT3D::inverse: input local_shape mismatch "
                "(should match forward's output_local_shape)");
        if (output.local_shape() != input_local_shape_)
            throw std::invalid_argument(
                "ParallelFFT3D::inverse: output local_shape mismatch "
                "(should match forward's input_local_shape)");
    }
};

}  // namespace clustr

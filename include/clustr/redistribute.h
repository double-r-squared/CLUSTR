#pragma once

// redistribute.h - Phase 6 core FFT primitive.
//
// Swap the distributed axis of a DistArray<T> from `v` to `w` within a 1D
// process group (the world communicator or any 1D sub-comm from cart_sub).
// This is the Section 3 routine from docs/FAST-FFT.md and the building block
// the 2D slab / 3D pencil FFTs (Phase 7/8) call once per axis transition.
//
// Why it's a plan object:
//   The balanced block decomposition and per-peer (start, extent) geometry
//   depend only on (global_shape, comm.size(), v, w) - not on the data.
//   An FFT forward-then-inverse hits the same geometry many times. The plan
//   computes it once in its constructor and caches N start/extent vector
//   pairs for reuse by every execute() call.
//
//   Subarray<T> itself is non-copyable and tied to a data pointer, so it
//   cannot literally live in the plan. execute() builds N stack-local
//   Subarrays (one heap allocation each via unique_ptr because Subarray
//   deletes its move/copy operations), wires them up with the cached shape
//   vectors, and hands them to Comm::alltoallw. All the expensive math is
//   amortized; execute() is O(N) pointer plumbing around one alltoallw.
//
// Scope: 1D process group only. Multi-dim grids are Phase 7/8's job -
// there the caller projects the world Comm onto a 1D axis via cart_sub and
// hands the resulting sub-comm here.
//
// Caller responsibilities:
//   - Allocate input and output DistArrays with the shapes reported by
//     plan.input_local_shape() / plan.output_local_shape(). The plan
//     rechecks at execute() time and throws on mismatch.
//   - Fill the input before calling execute(); the output is overwritten.
//   - Keep both DistArrays alive for the duration of the awaitable.
//
// Empty-slab edge case: when global_shape[axis] < comm.size(), some ranks
// get extent 0 on the partitioned axis. Subarray<T> reports total_bytes()==0,
// alltoallw short-circuits the send/recv op for that peer, nothing special
// needed here.

#include "asio.hpp"
#include "clustr_mpi.h"
#include "clustr/alltoallw.h"
#include "clustr/subarray.h"
#include "dist_array.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clustr {

namespace redistribute_detail {

// Balanced block decomposition from docs/FAST-FFT.md section 3.
// Given an axis of length `N` partitioned across `P` ranks, returns the
// (start, size) pair that rank `r` owns.
//
// Layout (row-major along ranks, fatter blocks first):
//   first (N mod P) ranks each own (floor(N/P) + 1) entries
//   remaining ranks each own floor(N/P) entries
//
// Invariants:
//   sum of sizes across all ranks == N
//   start[r+1] == start[r] + size[r]
//   size[r] >= 0; size[r] == 0 iff N < P and r >= N
inline std::pair<std::size_t, std::size_t>
balanced_block(std::size_t n, std::size_t p, std::size_t r) {
    if (p == 0)
        throw std::invalid_argument("balanced_block: p must be > 0");
    const std::size_t base = n / p;
    const std::size_t rem  = n % p;
    if (r < rem) {
        const std::size_t size  = base + 1;
        const std::size_t start = r * size;
        return {start, size};
    } else {
        const std::size_t size  = base;
        const std::size_t start = rem * (base + 1) + (r - rem) * base;
        return {start, size};
    }
}

}  // namespace redistribute_detail

// ─────────────────────────────────────────────────────────────────────────────
// RedistributePlan<T>
//
// Constructor does all the shape math. execute() is the hot path used by
// every FFT forward/inverse pair.
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
class RedistributePlan {
public:
    using shape_t = typename DistArray<T>::shape_t;

    RedistributePlan(Comm& comm,
                     shape_t global_shape,
                     int v,
                     int w)
        : comm_(comm),
          n_(static_cast<std::size_t>(comm.size())),
          my_rank_(static_cast<std::size_t>(comm.rank())),
          v_(v),
          w_(w),
          global_shape_(std::move(global_shape))
    {
        const int ndim = static_cast<int>(global_shape_.size());
        if (ndim == 0)
            throw std::invalid_argument("RedistributePlan: empty global_shape");
        if (v_ < 0 || v_ >= ndim)
            throw std::out_of_range(
                "RedistributePlan: v (" + std::to_string(v_)
                + ") out of range [0, " + std::to_string(ndim) + ")");
        if (w_ < 0 || w_ >= ndim)
            throw std::out_of_range(
                "RedistributePlan: w (" + std::to_string(w_)
                + ") out of range [0, " + std::to_string(ndim) + ")");
        if (v_ == w_)
            throw std::invalid_argument(
                "RedistributePlan: v == w is not a redistribute");
        if (n_ == 0)
            throw std::invalid_argument("RedistributePlan: empty comm");

        // ── My own local slabs along each partitioned axis ──────────────
        auto my_v = redistribute_detail::balanced_block(
            global_shape_[v_], n_, my_rank_);
        auto my_w = redistribute_detail::balanced_block(
            global_shape_[w_], n_, my_rank_);
        my_v_start_ = my_v.first;
        my_v_size_  = my_v.second;
        my_w_start_ = my_w.first;
        my_w_size_  = my_w.second;

        // ── Expected DistArray local shapes for validation at execute() ─
        //
        // Input (distributed on v): full everywhere except axis v.
        // Output (distributed on w): full everywhere except axis w.
        input_local_shape_  = global_shape_;
        output_local_shape_ = global_shape_;
        input_local_shape_ [v_] = my_v_size_;
        output_local_shape_[w_] = my_w_size_;

        // ── Precompute per-peer start/extent vectors ────────────────────
        //
        // Send to peer p, in MY INPUT local coordinates:
        //   axis v: full local  (start=0, extent=my_v_size_)
        //   axis w: peer p's w-slab in global coords (which equal local
        //           coords on the input because input is full along w)
        //   others: full global
        //
        // Recv from peer p, in MY OUTPUT local coordinates:
        //   axis v: peer p's v-slab in global coords (output is full along v)
        //   axis w: full local  (start=0, extent=my_w_size_)
        //   others: full global
        send_starts_.resize(n_, shape_t(static_cast<std::size_t>(ndim), 0));
        send_extents_.resize(n_, global_shape_);
        recv_starts_.resize(n_, shape_t(static_cast<std::size_t>(ndim), 0));
        recv_extents_.resize(n_, global_shape_);

        for (std::size_t p = 0; p < n_; ++p) {
            auto peer_v = redistribute_detail::balanced_block(
                global_shape_[v_], n_, p);
            auto peer_w = redistribute_detail::balanced_block(
                global_shape_[w_], n_, p);

            // Send[p]: my input carries my v-slab (local) × peer p's w-slab.
            send_starts_[p][v_]  = 0;
            send_extents_[p][v_] = my_v_size_;
            send_starts_[p][w_]  = peer_w.first;
            send_extents_[p][w_] = peer_w.second;

            // Recv[p]: my output stores peer p's v-slab × my w-slab (local).
            recv_starts_[p][v_]  = peer_v.first;
            recv_extents_[p][v_] = peer_v.second;
            recv_starts_[p][w_]  = 0;
            recv_extents_[p][w_] = my_w_size_;
        }
    }

    // ── Plan accessors ──────────────────────────────────────────────────
    const shape_t& global_shape()       const noexcept { return global_shape_; }
    const shape_t& input_local_shape()  const noexcept { return input_local_shape_; }
    const shape_t& output_local_shape() const noexcept { return output_local_shape_; }
    int            v()                  const noexcept { return v_; }
    int            w()                  const noexcept { return w_; }

    // My own v-slab (same math as balanced_block(global[v], n, my_rank)).
    // Exposed so callers can fill the input buffer with the right global
    // coordinates without repeating the decomposition.
    std::size_t my_v_start() const noexcept { return my_v_start_; }
    std::size_t my_v_size()  const noexcept { return my_v_size_; }
    std::size_t my_w_start() const noexcept { return my_w_start_; }
    std::size_t my_w_size()  const noexcept { return my_w_size_; }

    // ── execute ─────────────────────────────────────────────────────────
    //
    // Build N send Subarrays over `input` and N recv Subarrays over `output`
    // using the cached geometry, then fan them out via alltoallw.
    //
    // Validates that input/output have the shapes this plan was built for.
    // Soft-checks the distributed_axis annotations too (non-fatal if wrong:
    // DistArray still stores the data correctly, but the tag is wrong - so
    // we throw to catch caller bugs early).
    asio::awaitable<void> execute(DistArray<T>& input, DistArray<T>& output) {
        if (input.local_shape() != input_local_shape_)
            throw std::invalid_argument(
                "RedistributePlan::execute: input local_shape mismatch");
        if (output.local_shape() != output_local_shape_)
            throw std::invalid_argument(
                "RedistributePlan::execute: output local_shape mismatch");
        if (input.distributed_axis() != v_)
            throw std::invalid_argument(
                "RedistributePlan::execute: input.distributed_axis != v");
        if (output.distributed_axis() != w_)
            throw std::invalid_argument(
                "RedistributePlan::execute: output.distributed_axis != w");

        // Subarray deletes its move/copy ops, so we own them via unique_ptr
        // and pass raw pointers to alltoallw. One heap alloc per peer; cheap
        // compared with the actual network round-trip.
        std::vector<std::unique_ptr<Subarray<T>>> send_owned;
        std::vector<std::unique_ptr<Subarray<T>>> recv_owned;
        send_owned.reserve(n_);
        recv_owned.reserve(n_);

        std::vector<Subarray<T>*> send_ptrs(n_, nullptr);
        std::vector<Subarray<T>*> recv_ptrs(n_, nullptr);

        for (std::size_t p = 0; p < n_; ++p) {
            send_owned.push_back(std::make_unique<Subarray<T>>(
                input, send_starts_[p], send_extents_[p]));
            recv_owned.push_back(std::make_unique<Subarray<T>>(
                output, recv_starts_[p], recv_extents_[p]));
            send_ptrs[p] = send_owned.back().get();
            recv_ptrs[p] = recv_owned.back().get();
        }

        co_await alltoallw<T>(
            comm_,
            std::span<Subarray<T>* const>(send_ptrs.data(), send_ptrs.size()),
            std::span<Subarray<T>* const>(recv_ptrs.data(), recv_ptrs.size()));
    }

private:
    Comm&       comm_;
    std::size_t n_;
    std::size_t my_rank_;
    int         v_;
    int         w_;

    shape_t     global_shape_;
    shape_t     input_local_shape_;
    shape_t     output_local_shape_;

    std::size_t my_v_start_ = 0;
    std::size_t my_v_size_  = 0;
    std::size_t my_w_start_ = 0;
    std::size_t my_w_size_  = 0;

    // Per-peer cached geometry. send_starts_[p] / send_extents_[p] apply
    // to the input buffer; recv_starts_[p] / recv_extents_[p] to the output.
    std::vector<shape_t> send_starts_;
    std::vector<shape_t> send_extents_;
    std::vector<shape_t> recv_starts_;
    std::vector<shape_t> recv_extents_;
};

}  // namespace clustr

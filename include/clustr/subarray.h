#pragma once

// subarray.h - non-owning box descriptor over a DistArray<T>.
//
// Phase 3 of the parallel FFT roadmap. A Subarray<T> bridges distributed-array
// geometry and the zero-copy transport: given a box (start + extent) inside a
// row-major DistArray, it produces an asio buffer sequence whose entries are
// coalesced runs of contiguous bytes.
//
// Coalesce rule (recursive contiguous-prefix collapse):
//   Walk axes from innermost outward. Fold axis k into the fragment if and
//   only if every axis strictly inside k is full extent
//   (extent[j] == local_shape[j] for all j > k). Equivalently: starting from
//   the inner axis, keep absorbing outward as long as the previously-absorbed
//   axis was full extent. The inner axis is always part of the fragment, even
//   if partial.
//
//   This mirrors what MPI_Type_create_subarray does internally and collapses
//   a 64^3 full face to a single fragment, which is the Phase 3 exit
//   criterion in ROADMAP.md.
//
// Lifetime: Subarray is a non-owning view. The caller MUST ensure the parent
// DistArray outlives the Subarray and is not resized between construction and
// last use. Same contract as std::span.
//
// Move/copy disabled: the cache holds raw pointers into the parent's storage.
// Build the descriptor in place where it is used (e.g. inside an FFT plan
// struct that has the same lifetime as the data).

#include "asio.hpp"
#include "dist_array.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clustr {

template <typename T>
class Subarray {
public:
    using shape_t  = typename DistArray<T>::shape_t;
    using stride_t = typename DistArray<T>::stride_t;

    Subarray(DistArray<T>& arr, shape_t start, shape_t extent)
        : data_(arr.data()),
          parent_strides_(arr.strides_bytes()),
          local_shape_(arr.local_shape()),
          start_(std::move(start)),
          extent_(std::move(extent)),
          ndim_(local_shape_.size()) {
        if (start_.size() != ndim_ || extent_.size() != ndim_) {
            throw std::invalid_argument(
                "Subarray: start/extent rank mismatch with parent DistArray");
        }
        for (std::size_t i = 0; i < ndim_; ++i) {
            if (start_[i] + extent_[i] > local_shape_[i]) {
                throw std::out_of_range(
                    "Subarray: box exceeds parent local_shape on axis "
                    + std::to_string(i));
            }
        }
        total_bytes_ = sizeof(T);
        for (auto e : extent_) total_bytes_ *= e;
    }

    Subarray(const Subarray&)            = delete;
    Subarray(Subarray&&)                 = delete;
    Subarray& operator=(const Subarray&) = delete;
    Subarray& operator=(Subarray&&)      = delete;

    // ── geometry queries (cheap, no cache) ───────────────────────────────
    std::size_t    ndim()        const noexcept { return ndim_; }
    const shape_t& start()       const noexcept { return start_; }
    const shape_t& extent()      const noexcept { return extent_; }
    std::size_t    total_bytes() const noexcept { return total_bytes_; }

    // The deepest axis at which the fragment starts. Fragment spans
    // [coalesced_axis()..ndim-1]; outer loop axes are [0..coalesced_axis()-1].
    // For a fully-collapsible box this returns 0 (one fragment, whole box).
    // For a box whose inner axis is partial this returns ndim-1 (one fragment
    // per outer multi-index, each spanning the inner partial row).
    std::size_t coalesced_axis() const noexcept {
        if (ndim_ == 0) return 0;
        std::size_t c = ndim_ - 1;
        while (c > 0 && extent_[c] == local_shape_[c]) --c;
        return c;
    }

    std::size_t fragment_count() const noexcept {
        if (total_bytes_ == 0) return 0;
        const std::size_t c = coalesced_axis();
        std::size_t n = 1;
        for (std::size_t k = 0; k < c; ++k) n *= extent_[k];
        return n;
    }

    // ── materializing accessors (lazy, cached) ───────────────────────────
    //
    // First call walks the box and builds the Fragment table + the requested
    // asio buffer vector. Subsequent calls return the cached vector. The
    // Fragment table is shared between const and mutable variants - only the
    // pointer cast differs.

    const std::vector<asio::const_buffer>& as_const_buffers() {
        if (const_cache_.empty() && fragment_count() != 0) {
            ensure_fragment_table();
            const_cache_.reserve(fragments_.size());
            const auto* base = reinterpret_cast<const std::byte*>(data_);
            for (const auto& f : fragments_) {
                const_cache_.emplace_back(
                    asio::buffer(static_cast<const void*>(base + f.byte_offset),
                                 f.byte_size));
            }
        }
        return const_cache_;
    }

    const std::vector<asio::mutable_buffer>& as_mutable_buffers() {
        if (mutable_cache_.empty() && fragment_count() != 0) {
            ensure_fragment_table();
            mutable_cache_.reserve(fragments_.size());
            auto* base = reinterpret_cast<std::byte*>(data_);
            for (const auto& f : fragments_) {
                mutable_cache_.emplace_back(
                    asio::buffer(static_cast<void*>(base + f.byte_offset),
                                 f.byte_size));
            }
        }
        return mutable_cache_;
    }

private:
    struct Fragment {
        std::ptrdiff_t byte_offset;
        std::size_t    byte_size;
    };

    T*           data_;
    stride_t     parent_strides_;
    shape_t      local_shape_;
    shape_t      start_;
    shape_t      extent_;
    std::size_t  ndim_;
    std::size_t  total_bytes_;

    std::vector<Fragment>             fragments_;
    std::vector<asio::const_buffer>   const_cache_;
    std::vector<asio::mutable_buffer> mutable_cache_;

    void ensure_fragment_table() {
        if (!fragments_.empty()) return;

        const std::size_t c = coalesced_axis();

        std::size_t frag_bytes = sizeof(T);
        for (std::size_t k = c; k < ndim_; ++k) frag_bytes *= extent_[k];

        std::size_t n_frags = 1;
        for (std::size_t k = 0; k < c; ++k) n_frags *= extent_[k];
        if (n_frags == 0 || frag_bytes == 0) return;

        std::ptrdiff_t origin = 0;
        for (std::size_t i = 0; i < ndim_; ++i) {
            origin += static_cast<std::ptrdiff_t>(start_[i]) * parent_strides_[i];
        }

        fragments_.reserve(n_frags);

        // Mixed-radix row-major iteration over outer axes [0..c-1].
        std::vector<std::size_t> idx(c, 0);
        for (std::size_t f = 0; f < n_frags; ++f) {
            std::ptrdiff_t off = origin;
            for (std::size_t k = 0; k < c; ++k) {
                off += static_cast<std::ptrdiff_t>(idx[k]) * parent_strides_[k];
            }
            fragments_.push_back({off, frag_bytes});

            for (std::size_t k = c; k-- > 0;) {
                if (++idx[k] < extent_[k]) break;
                idx[k] = 0;
            }
        }
    }
};

}  // namespace clustr

#pragma once

// DistArray<T> - row-major multidimensional array with an optional
// "distributed axis" annotation, used by the parallel FFT.
//
// Phase 0 (serial): distributed_axis == kNoDistributedAxis and
// local_shape == global_shape. The array behaves as a plain row-major buffer.
//
// Later phases populate distributed_axis with a real axis index and a
// local_shape that reflects this rank's slab.
//
// Strides are stored in BYTES so they can be handed directly to PocketFFT's
// c2c / r2c / c2r entry points without a conversion step.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace clustr {

inline constexpr int kNoDistributedAxis = -1;

template <typename T>
class DistArray {
public:
    using shape_t  = std::vector<std::size_t>;
    using stride_t = std::vector<std::ptrdiff_t>;  // bytes, matches PocketFFT

    // Serial constructor: not distributed, local_shape == global_shape.
    static DistArray serial(shape_t shape) {
        return DistArray(shape, shape, kNoDistributedAxis);
    }

    // Distributed constructor: caller supplies local_shape (the slab this
    // rank owns) along with the global shape and which axis is split.
    DistArray(shape_t global_shape,
              shape_t local_shape,
              int     distributed_axis)
        : global_shape_(std::move(global_shape)),
          local_shape_(std::move(local_shape)),
          distributed_axis_(distributed_axis) {
        if (global_shape_.empty()) {
            throw std::invalid_argument("DistArray: empty shape");
        }
        if (local_shape_.size() != global_shape_.size()) {
            throw std::invalid_argument(
                "DistArray: local_shape rank mismatch");
        }
        strides_bytes_ = compute_row_major_strides(local_shape_);
        storage_.assign(product(local_shape_), T{});
    }

    // ── accessors ─────────────────────────────────────────────────────────
    const shape_t&  global_shape() const noexcept { return global_shape_; }
    const shape_t&  local_shape()  const noexcept { return local_shape_;  }
    int             distributed_axis() const noexcept { return distributed_axis_; }
    std::size_t     ndim() const noexcept { return local_shape_.size(); }
    std::size_t     size() const noexcept { return storage_.size(); }

    // Row-major strides in BYTES (PocketFFT convention).
    const stride_t& strides_bytes() const noexcept { return strides_bytes_; }

    T*       data()       noexcept { return storage_.data(); }
    const T* data() const noexcept { return storage_.data(); }

    // Flat row-major indexing for serial use and tests.
    template <typename... I>
    T& at(I... idx) {
        static_assert(sizeof...(I) > 0, "DistArray::at needs at least one index");
        std::size_t indices[] = { static_cast<std::size_t>(idx)... };
        return storage_[linear_index(indices, sizeof...(I))];
    }

    template <typename... I>
    const T& at(I... idx) const {
        static_assert(sizeof...(I) > 0, "DistArray::at needs at least one index");
        std::size_t indices[] = { static_cast<std::size_t>(idx)... };
        return storage_[linear_index(indices, sizeof...(I))];
    }

private:
    shape_t              global_shape_;
    shape_t              local_shape_;
    int                  distributed_axis_{kNoDistributedAxis};
    stride_t             strides_bytes_;
    std::vector<T>       storage_;

    static std::size_t product(const shape_t& s) {
        std::size_t p = 1;
        for (auto v : s) p *= v;
        return p;
    }

    // Row-major (C-order) strides in BYTES.
    // For shape (A, B, C): strides = (B*C*sizeof(T), C*sizeof(T), sizeof(T)).
    static stride_t compute_row_major_strides(const shape_t& shape) {
        stride_t strides(shape.size());
        std::ptrdiff_t running = static_cast<std::ptrdiff_t>(sizeof(T));
        for (std::size_t i = shape.size(); i-- > 0;) {
            strides[i] = running;
            running *= static_cast<std::ptrdiff_t>(shape[i]);
        }
        return strides;
    }

    std::size_t linear_index(const std::size_t* idx, std::size_t n) const {
        if (n != local_shape_.size()) {
            throw std::out_of_range("DistArray::at wrong number of indices");
        }
        std::size_t flat   = 0;
        std::size_t stride = 1;
        for (std::size_t i = n; i-- > 0;) {
            if (idx[i] >= local_shape_[i]) {
                throw std::out_of_range("DistArray::at index out of range");
            }
            flat   += idx[i] * stride;
            stride *= local_shape_[i];
        }
        return flat;
    }
};

}  // namespace clustr

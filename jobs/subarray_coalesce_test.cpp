// subarray_coalesce_test.cpp - Phase 3 unit test: Subarray coalesce geometry.
//
// What this does:
//   - Constructs Subarray<T> instances over a row-major DistArray
//   - Asserts fragment_count, total_bytes, and the actual fragment offsets
//     produced by the recursive contiguous-prefix coalesce rule
//   - Covers the Phase 3 exit criterion: a 64^3 full face collapses to 1
//     fragment, full size = 64^3 * sizeof(T) bytes
//
// No MPI, no networking, no PocketFFT. Pure geometry.
//
// Submit with Ranks = 1, or run locally:
//   g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED \
//       -I./include -I./asio_include subarray_coalesce_test.cpp \
//       -o subarray_coalesce_test -lpthread
//   ./subarray_coalesce_test

#include "clustr/subarray.h"
#include "dist_array.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using clustr::DistArray;
using clustr::Subarray;

namespace {

int g_failures = 0;

#define EXPECT_EQ(a, b)                                                       \
    do {                                                                      \
        auto _av = (a);                                                       \
        auto _bv = (b);                                                       \
        if (_av != _bv) {                                                     \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__               \
                      << " " #a " (" << _av << ") != " #b " (" << _bv << ")\n"; \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

void test_full_box_3d_collapses_to_one() {
    auto arr = DistArray<double>::serial({64, 64, 64});
    Subarray<double> sub(arr, {0, 0, 0}, {64, 64, 64});

    EXPECT_EQ(sub.coalesced_axis(), std::size_t{0});
    EXPECT_EQ(sub.fragment_count(), std::size_t{1});
    EXPECT_EQ(sub.total_bytes(), std::size_t{64} * 64 * 64 * sizeof(double));

    const auto& bufs = sub.as_const_buffers();
    EXPECT_EQ(bufs.size(), std::size_t{1});
    EXPECT_EQ(bufs[0].size(), std::size_t{64} * 64 * 64 * sizeof(double));
    EXPECT_EQ(bufs[0].data(), static_cast<const void*>(arr.data()));
}

void test_partial_outer_axis_still_one_fragment() {
    auto arr = DistArray<double>::serial({64, 64, 64});
    Subarray<double> sub(arr, {0, 0, 0}, {32, 64, 64});

    EXPECT_EQ(sub.coalesced_axis(), std::size_t{0});
    EXPECT_EQ(sub.fragment_count(), std::size_t{1});
    EXPECT_EQ(sub.total_bytes(), std::size_t{32} * 64 * 64 * sizeof(double));

    const auto& bufs = sub.as_const_buffers();
    EXPECT_EQ(bufs.size(), std::size_t{1});
    EXPECT_EQ(bufs[0].size(), std::size_t{32} * 64 * 64 * sizeof(double));
}

void test_partial_middle_axis_emits_outer_count_fragments() {
    auto arr = DistArray<double>::serial({64, 64, 64});
    Subarray<double> sub(arr, {0, 0, 0}, {64, 32, 64});

    EXPECT_EQ(sub.coalesced_axis(), std::size_t{1});
    EXPECT_EQ(sub.fragment_count(), std::size_t{64});

    const auto& bufs = sub.as_const_buffers();
    EXPECT_EQ(bufs.size(), std::size_t{64});
    for (const auto& b : bufs) {
        EXPECT_EQ(b.size(), std::size_t{32} * 64 * sizeof(double));
    }
}

void test_partial_inner_axis_emits_n_squared_fragments() {
    auto arr = DistArray<double>::serial({64, 64, 64});
    Subarray<double> sub(arr, {0, 0, 0}, {64, 64, 32});

    EXPECT_EQ(sub.coalesced_axis(), std::size_t{2});
    EXPECT_EQ(sub.fragment_count(), std::size_t{64} * 64);

    const auto& bufs = sub.as_const_buffers();
    EXPECT_EQ(bufs.size(), std::size_t{64} * 64);
    for (const auto& b : bufs) {
        EXPECT_EQ(b.size(), std::size_t{32} * sizeof(double));
    }
}

void test_offset_box_origin_is_correct() {
    auto arr = DistArray<double>::serial({4, 4, 4});
    Subarray<double> sub(arr, {1, 2, 0}, {2, 2, 4});

    EXPECT_EQ(sub.coalesced_axis(), std::size_t{1});
    EXPECT_EQ(sub.fragment_count(), std::size_t{2});

    const auto& bufs = sub.as_const_buffers();
    EXPECT_EQ(bufs.size(), std::size_t{2});

    // Each fragment is extent[1] * extent[2] = 2 * 4 = 8 doubles.
    for (const auto& b : bufs) {
        EXPECT_EQ(b.size(), std::size_t{8} * sizeof(double));
    }

    // Origin offset in bytes: start = (1, 2, 0).
    // parent_strides = (16*8, 4*8, 8) = (128, 32, 8).
    // origin = 1*128 + 2*32 + 0*8 = 192 bytes.
    const auto* base = reinterpret_cast<const std::byte*>(arr.data());
    EXPECT_EQ(bufs[0].data(), static_cast<const void*>(base + 192));

    // Second fragment increments outer axis 0 by 1: +128 bytes.
    EXPECT_EQ(bufs[1].data(), static_cast<const void*>(base + 192 + 128));
}

void test_1d_array_always_one_fragment() {
    auto arr = DistArray<double>::serial({100});

    Subarray<double> full(arr, {0}, {100});
    EXPECT_EQ(full.fragment_count(), std::size_t{1});
    EXPECT_EQ(full.as_const_buffers()[0].size(), std::size_t{100} * sizeof(double));

    Subarray<double> partial(arr, {10}, {30});
    EXPECT_EQ(partial.fragment_count(), std::size_t{1});
    EXPECT_EQ(partial.as_const_buffers()[0].size(), std::size_t{30} * sizeof(double));
}

void test_mutable_buffers_return_writable_pointers() {
    auto arr = DistArray<std::int32_t>::serial({4, 4});
    for (std::size_t i = 0; i < arr.size(); ++i) arr.data()[i] = 0;

    Subarray<std::int32_t> sub(arr, {0, 0}, {4, 4});
    const auto& mut = sub.as_mutable_buffers();
    EXPECT_EQ(mut.size(), std::size_t{1});

    auto* p = static_cast<std::int32_t*>(mut[0].data());
    for (std::size_t i = 0; i < 16; ++i) p[i] = static_cast<std::int32_t>(i + 1);

    for (std::size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(arr.data()[i], static_cast<std::int32_t>(i + 1));
    }
}

void test_empty_box_yields_zero_fragments() {
    auto arr = DistArray<double>::serial({4, 4, 4});
    Subarray<double> sub(arr, {0, 0, 0}, {2, 0, 4});

    EXPECT_EQ(sub.total_bytes(), std::size_t{0});
    EXPECT_EQ(sub.fragment_count(), std::size_t{0});
    EXPECT_EQ(sub.as_const_buffers().size(), std::size_t{0});
}

void test_out_of_range_throws() {
    auto arr = DistArray<double>::serial({4, 4, 4});
    bool threw = false;
    try {
        Subarray<double> sub(arr, {0, 0, 0}, {5, 4, 4});
    } catch (const std::out_of_range&) {
        threw = true;
    }
    if (!threw) {
        std::cerr << "FAIL out-of-range box did not throw\n";
        ++g_failures;
    }
}

}  // namespace

int main() {
    test_full_box_3d_collapses_to_one();
    test_partial_outer_axis_still_one_fragment();
    test_partial_middle_axis_emits_outer_count_fragments();
    test_partial_inner_axis_emits_n_squared_fragments();
    test_offset_box_origin_is_correct();
    test_1d_array_always_one_fragment();
    test_mutable_buffers_return_writable_pointers();
    test_empty_box_yields_zero_fragments();
    test_out_of_range_throws();

    if (g_failures == 0) {
        std::cout << "subarray_coalesce_test: ALL PASS\n";
        return 0;
    }
    std::cout << "subarray_coalesce_test: " << g_failures << " FAILURE(S)\n";
    return 1;
}

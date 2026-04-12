// subarray_send_test.cpp - Phase 3 round-trip: Subarray over the wire.
//
// What this demonstrates:
//   - Building a Subarray<T> over a DistArray on rank 0 and sending its
//     coalesced asio::const_buffer sequence directly to rank 1 via
//     Comm::send_raw, with no intermediate packing.
//   - Both transports (zero-copy and pack/unpack) deliver bit-identical bytes
//     so this same source compiles and passes under MPI-zc and MPI-pu.
//
// Two scenarios are exercised:
//   1. Full 8x8x8 box. Coalesces to 1 fragment. The most common FFT face case.
//   2. Inner-half partial box {4,8,8} with start {2,0,0}. Still coalesces to
//      1 fragment (axes 1 and 2 are full extent, so all 4 slabs are
//      contiguous in memory).
//
// Receiver path: rank 1 calls recv_raw to pull bytes into a vector<uint8_t>
// and then memcmps against the expected packed reference it generates locally
// from the same deterministic fill formula. This verifies the bytes that
// traveled the wire match what the Subarray geometry says it sent.
//
// Submit with Ranks >= 2. Extra ranks idle.
//
// Compile: g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED
//              -DCLUSTR_TRANSPORT=CLUSTR_TRANSPORT_ZERO_COPY
//              -I./include -I./asio_include subarray_send_test.cpp
//              -o subarray_send_test -lpthread
//   (or CLUSTR_TRANSPORT_PACK_UNPACK for the reference baseline)

#include "clustr_mpi.h"
#include "clustr/subarray.h"
#include "dist_array.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using clustr::DistArray;
using clustr::Subarray;
using clustr::PeerHeader;
using clustr::PeerMsgType;

namespace {

// Deterministic fill - both ranks compute the same value for the same indices
// so the receiver can reconstruct the expected packed bytes locally.
double fill_value(std::size_t i, std::size_t j, std::size_t k) {
    return static_cast<double>(i) * 1000.0
         + static_cast<double>(j) * 10.0
         + static_cast<double>(k) * 0.01;
}

void fill_array(DistArray<double>& arr) {
    const auto& s = arr.local_shape();
    for (std::size_t i = 0; i < s[0]; ++i)
        for (std::size_t j = 0; j < s[1]; ++j)
            for (std::size_t k = 0; k < s[2]; ++k)
                arr.at(i, j, k) = fill_value(i, j, k);
}

// Walk the box in row-major order and pack the doubles into a flat byte
// buffer. This is the "what the receiver should see" reference - bit-equal
// to what a coalesced Subarray would put on the wire.
std::vector<std::uint8_t> pack_box_reference(
        const DistArray<double>& arr,
        const std::vector<std::size_t>& start,
        const std::vector<std::size_t>& extent) {
    std::vector<std::uint8_t> out;
    out.reserve(extent[0] * extent[1] * extent[2] * sizeof(double));
    for (std::size_t i = 0; i < extent[0]; ++i) {
        for (std::size_t j = 0; j < extent[1]; ++j) {
            for (std::size_t k = 0; k < extent[2]; ++k) {
                double v = arr.at(start[0] + i, start[1] + j, start[2] + k);
                const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
                out.insert(out.end(), p, p + sizeof(double));
            }
        }
    }
    return out;
}

}  // namespace

CLUSTR_MPI_MAIN(mpi) {
    const int rank = mpi.rank();
    const int size = mpi.size();

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "subarray_send_test: requires Ranks >= 2 (got "
                      << size << ")\n";
        }
        co_return 1;
    }

    auto& world = mpi.world();

    // Both ranks build the same DistArray with the same deterministic fill.
    DistArray<double> arr = DistArray<double>::serial({8, 8, 8});
    fill_array(arr);

    int failures = 0;

    // ── Scenario 1: full box, single coalesced fragment ──────────────────
    {
        std::vector<std::size_t> start  = {0, 0, 0};
        std::vector<std::size_t> extent = {8, 8, 8};

        if (rank == 0) {
            Subarray<double> sub(arr, start, extent);

            if (sub.fragment_count() != 1) {
                std::cerr << "[rank 0] full-box fragment_count="
                          << sub.fragment_count() << " (expected 1)\n";
                ++failures;
            }

            PeerHeader hdr{};
            hdr.magic[0]    = 0xC1;
            hdr.magic[1]    = 0x52;
            hdr.type        = PeerMsgType::SEND;
            hdr.src_rank    = static_cast<std::uint32_t>(rank);
            hdr.dst_rank    = 1;
            hdr.tag         = 1001;
            hdr.payload_len = static_cast<std::uint32_t>(sub.total_bytes());
            hdr.comm_id     = clustr::kWorldCommId;

            co_await world.send_raw(1, hdr, sub.as_const_buffers());
            std::cout << "[rank 0] full-box sent " << sub.total_bytes()
                      << " bytes in " << sub.fragment_count() << " fragment(s)\n";
        } else if (rank == 1) {
            auto raw = co_await world.recv_raw(0, 1001, PeerMsgType::SEND);
            auto expected = pack_box_reference(arr, start, extent);

            if (raw.size() != expected.size()) {
                std::cerr << "[rank 1] full-box size " << raw.size()
                          << " != expected " << expected.size() << "\n";
                ++failures;
            } else if (std::memcmp(raw.data(), expected.data(), raw.size()) != 0) {
                std::cerr << "[rank 1] full-box bytes mismatch\n";
                ++failures;
            } else {
                std::cout << "[rank 1] full-box OK (" << raw.size()
                          << " bytes match)\n";
            }
        }
    }

    co_await world.barrier(91);

    // ── Scenario 2: partial outer slab, still 1 coalesced fragment ───────
    {
        std::vector<std::size_t> start  = {2, 0, 0};
        std::vector<std::size_t> extent = {4, 8, 8};

        if (rank == 0) {
            Subarray<double> sub(arr, start, extent);

            if (sub.fragment_count() != 1) {
                std::cerr << "[rank 0] partial-slab fragment_count="
                          << sub.fragment_count() << " (expected 1)\n";
                ++failures;
            }

            PeerHeader hdr{};
            hdr.magic[0]    = 0xC1;
            hdr.magic[1]    = 0x52;
            hdr.type        = PeerMsgType::SEND;
            hdr.src_rank    = static_cast<std::uint32_t>(rank);
            hdr.dst_rank    = 1;
            hdr.tag         = 1002;
            hdr.payload_len = static_cast<std::uint32_t>(sub.total_bytes());
            hdr.comm_id     = clustr::kWorldCommId;

            co_await world.send_raw(1, hdr, sub.as_const_buffers());
            std::cout << "[rank 0] partial-slab sent " << sub.total_bytes()
                      << " bytes in " << sub.fragment_count() << " fragment(s)\n";
        } else if (rank == 1) {
            auto raw = co_await world.recv_raw(0, 1002, PeerMsgType::SEND);
            auto expected = pack_box_reference(arr, start, extent);

            if (raw.size() != expected.size()) {
                std::cerr << "[rank 1] partial-slab size " << raw.size()
                          << " != expected " << expected.size() << "\n";
                ++failures;
            } else if (std::memcmp(raw.data(), expected.data(), raw.size()) != 0) {
                std::cerr << "[rank 1] partial-slab bytes mismatch\n";
                ++failures;
            } else {
                std::cout << "[rank 1] partial-slab OK (" << raw.size()
                          << " bytes match)\n";
            }
        }
    }

    co_await world.barrier(92);

    // ── Scenario 3: partial inner axis, fragment_count = outer^2 ─────────
    {
        std::vector<std::size_t> start  = {0, 0, 0};
        std::vector<std::size_t> extent = {8, 8, 4};

        if (rank == 0) {
            Subarray<double> sub(arr, start, extent);

            if (sub.fragment_count() != 64) {
                std::cerr << "[rank 0] inner-partial fragment_count="
                          << sub.fragment_count() << " (expected 64)\n";
                ++failures;
            }

            PeerHeader hdr{};
            hdr.magic[0]    = 0xC1;
            hdr.magic[1]    = 0x52;
            hdr.type        = PeerMsgType::SEND;
            hdr.src_rank    = static_cast<std::uint32_t>(rank);
            hdr.dst_rank    = 1;
            hdr.tag         = 1003;
            hdr.payload_len = static_cast<std::uint32_t>(sub.total_bytes());
            hdr.comm_id     = clustr::kWorldCommId;

            co_await world.send_raw(1, hdr, sub.as_const_buffers());
            std::cout << "[rank 0] inner-partial sent " << sub.total_bytes()
                      << " bytes in " << sub.fragment_count() << " fragment(s)\n";
        } else if (rank == 1) {
            auto raw = co_await world.recv_raw(0, 1003, PeerMsgType::SEND);
            auto expected = pack_box_reference(arr, start, extent);

            if (raw.size() != expected.size()) {
                std::cerr << "[rank 1] inner-partial size " << raw.size()
                          << " != expected " << expected.size() << "\n";
                ++failures;
            } else if (std::memcmp(raw.data(), expected.data(), raw.size()) != 0) {
                std::cerr << "[rank 1] inner-partial bytes mismatch\n";
                ++failures;
            } else {
                std::cout << "[rank 1] inner-partial OK (" << raw.size()
                          << " bytes match)\n";
            }
        }
    }

    co_await world.barrier(93);

    if (rank == 1) {
        if (failures == 0) {
            std::cout << "[rank 1] subarray_send_test: ALL PASS\n";
        } else {
            std::cout << "[rank 1] subarray_send_test: " << failures << " FAILURE(S)\n";
        }
    }

    co_return failures;
}

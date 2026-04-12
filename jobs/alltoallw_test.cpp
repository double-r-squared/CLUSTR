// alltoallw_test.cpp - Phase 4 round-trip: Comm::alltoallw correctness.
//
// Each rank owns an 8x8x8 input array (its data depends on the rank,
// deterministic) and a freshly allocated recv buffer for every peer. The
// test plans send/recv Subarray descriptors so that the exchange covers:
//
//   1. Self-send (loopback memcpy path)        - kind 0
//   2. Single-fragment box (full coalesce)     - kind 1
//   3. Multi-fragment box (partial inner axis) - kind 2
//   4. Zero-extent slot (no actual exchange)   - kind 3
//
// The peer chosen for each kind is determined by delta = (peer - rank) mod N.
// Which kinds are exercised depends on rank count:
//
//   Ranks = 2 -> kinds {0, 1}        (self + full)
//   Ranks = 3 -> kinds {0, 1, 2}     (self + full + partial-inner)
//   Ranks = 4 -> kinds {0, 1, 2, 3}  (all four)
//   Ranks > 4 -> cycles back through the four kinds
//
// The empty-extent fragment path is also covered by
// subarray_coalesce_test.cpp, so Ranks = 3 still gives full alltoallw
// coverage if you only have three machines.
//
// Submit with Ranks >= 2. Compiles under both MPI-zc and MPI-pu.

#include "clustr_mpi.h"
#include "clustr/alltoallw.h"
#include "clustr/subarray.h"
#include "dist_array.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

using clustr::DistArray;
using clustr::Subarray;
using clustr::alltoallw;

namespace {

// Deterministic per-rank fill - the producer rank id is folded in so the
// receiver can recompute exactly what should have arrived from each peer.
double fill_value(int producer_rank, std::size_t i, std::size_t j, std::size_t k) {
    return static_cast<double>(producer_rank) * 1e6
         + static_cast<double>(i) * 1e4
         + static_cast<double>(j) * 1e2
         + static_cast<double>(k);
}

void fill_array(DistArray<double>& arr, int producer_rank) {
    const auto& s = arr.local_shape();
    for (std::size_t i = 0; i < s[0]; ++i)
        for (std::size_t j = 0; j < s[1]; ++j)
            for (std::size_t k = 0; k < s[2]; ++k)
                arr.at(i, j, k) = fill_value(producer_rank, i, j, k);
}

}  // namespace

CLUSTR_MPI_MAIN(mpi) {
    const int rank = mpi.rank();
    const int size = mpi.size();

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "alltoallw_test: requires Ranks >= 2 (got "
                      << size << ")\n";
        }
        co_return 1;
    }

    auto& world = mpi.world();

    // Each rank's "input" array - filled with this rank's signature data.
    DistArray<double> in  = DistArray<double>::serial({8, 8, 8});
    DistArray<double> out = DistArray<double>::serial({8, 8, 8});
    fill_array(in, rank);

    // Plan layout: each rank sends a different shape to each peer to exercise
    // all four scenarios in one collective.
    //
    //   self (p == rank): full 8x8x8 box (loopback memcpy, single fragment)
    //   peer (rank+1)%4 : 8x8x8 full box (single fragment over the wire)
    //   peer (rank+2)%4 : 8x8x4 partial-inner box (64 fragments)
    //   peer (rank+3)%4 : zero-extent (no exchange)
    //
    // The recv plan is the symmetric mirror: rank r receives from peer p the
    // box that peer p planned to send to rank r.

    // delta in [0, n-1], folded into 4 kinds:
    //   0 = self full, 1 = full, 2 = partial-inner, 3 = empty
    // Self-send is always kind 0; other deltas cycle through 1..3.
    auto box_kind = [](int from, int to, int n) {
        const int delta = ((to - from) % n + n) % n;
        if (delta == 0) return 0;
        return 1 + ((delta - 1) % 3);
    };

    // ── Build send descriptors (one per dst) ─────────────────────────────
    std::vector<std::unique_ptr<Subarray<double>>> sends;
    sends.reserve(size);
    for (int dst = 0; dst < size; ++dst) {
        const int kind = box_kind(rank, dst, size);
        switch (kind) {
            case 0: case 1:  // full 8x8x8
                sends.emplace_back(std::make_unique<Subarray<double>>(
                    in, std::vector<std::size_t>{0, 0, 0},
                        std::vector<std::size_t>{8, 8, 8}));
                break;
            case 2:  // partial-inner: 8x8x4 starting at k=0
                sends.emplace_back(std::make_unique<Subarray<double>>(
                    in, std::vector<std::size_t>{0, 0, 0},
                        std::vector<std::size_t>{8, 8, 4}));
                break;
            case 3:  // empty
                sends.emplace_back(std::make_unique<Subarray<double>>(
                    in, std::vector<std::size_t>{0, 0, 0},
                        std::vector<std::size_t>{0, 0, 0}));
                break;
        }
    }

    // ── Build recv descriptors (one per src) ─────────────────────────────
    // Each receives into a distinct region of `out` so we can verify all
    // four arrivals independently after the collective:
    //   from peer p with delta d -> recv into out's "slot d"
    //
    // Slot layout in out (8x8x8 logical, sliced along axis 0):
    //   slot 0 (self full)         -> out[0..2, :, :] (16 doubles per row * 16 rows = 128 doubles, fits 2x8x8 = 128)
    //                                   wait - 8x8x8 box has 512 doubles, doesn't fit a 2x8x8 slot.
    //
    // Simpler: give each arrival its own freshly allocated DistArray<double>
    // sized to match its descriptor exactly. No slot juggling, no cross-talk.
    // Trades a tiny bit of geometry purity for verification clarity.

    std::vector<std::unique_ptr<DistArray<double>>> recv_storage;
    std::vector<std::unique_ptr<Subarray<double>>>  recvs;
    recv_storage.reserve(size);
    recvs.reserve(size);

    for (int src = 0; src < size; ++src) {
        const int kind = box_kind(src, rank, size);  // what src is sending us
        switch (kind) {
            case 0: case 1:  // full 8x8x8
                recv_storage.emplace_back(std::make_unique<DistArray<double>>(
                    DistArray<double>::serial({8, 8, 8})));
                recvs.emplace_back(std::make_unique<Subarray<double>>(
                    *recv_storage.back(),
                    std::vector<std::size_t>{0, 0, 0},
                    std::vector<std::size_t>{8, 8, 8}));
                break;
            case 2:  // 8x8x4
                recv_storage.emplace_back(std::make_unique<DistArray<double>>(
                    DistArray<double>::serial({8, 8, 4})));
                recvs.emplace_back(std::make_unique<Subarray<double>>(
                    *recv_storage.back(),
                    std::vector<std::size_t>{0, 0, 0},
                    std::vector<std::size_t>{8, 8, 4}));
                break;
            case 3:  // empty
                recv_storage.emplace_back(std::make_unique<DistArray<double>>(
                    DistArray<double>::serial({1, 1, 1})));
                recvs.emplace_back(std::make_unique<Subarray<double>>(
                    *recv_storage.back(),
                    std::vector<std::size_t>{0, 0, 0},
                    std::vector<std::size_t>{0, 0, 0}));
                break;
        }
    }

    // Spans of raw pointers - alltoallw signature.
    std::vector<Subarray<double>*> send_ptrs, recv_ptrs;
    send_ptrs.reserve(size);
    recv_ptrs.reserve(size);
    for (auto& p : sends) send_ptrs.push_back(p.get());
    for (auto& p : recvs) recv_ptrs.push_back(p.get());

    co_await alltoallw<double>(world,
        std::span<Subarray<double>* const>(send_ptrs.data(), send_ptrs.size()),
        std::span<Subarray<double>* const>(recv_ptrs.data(), recv_ptrs.size()));

    // ── Verify ───────────────────────────────────────────────────────────
    int failures = 0;

    for (int src = 0; src < size; ++src) {
        const int kind = box_kind(src, rank, size);
        const auto& store = *recv_storage[src];

        std::vector<std::size_t> extent;
        if (kind == 0 || kind == 1) extent = {8, 8, 8};
        else if (kind == 2)         extent = {8, 8, 4};
        else                        extent = {0, 0, 0};

        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < extent[0]; ++i) {
            for (std::size_t j = 0; j < extent[1]; ++j) {
                for (std::size_t k = 0; k < extent[2]; ++k) {
                    const double got = store.at(i, j, k);
                    const double want = fill_value(src, i, j, k);
                    if (got != want) {
                        if (mismatches < 3) {
                            std::cerr << "[rank " << rank << "] from " << src
                                      << " kind " << kind << " (" << i << "," << j << "," << k
                                      << ") got " << got << " want " << want << "\n";
                        }
                        ++mismatches;
                    }
                }
            }
        }
        if (mismatches != 0) {
            std::cerr << "[rank " << rank << "] from " << src << ": "
                      << mismatches << " mismatches\n";
            ++failures;
        } else {
            std::cout << "[rank " << rank << "] from " << src
                      << " kind " << kind << " OK\n";
        }
    }

    co_await world.barrier(81);

    if (rank == 0) {
        if (failures == 0)
            std::cout << "[rank 0] alltoallw_test: ALL PASS\n";
        else
            std::cout << "[rank 0] alltoallw_test: " << failures << " local FAILURE(S)\n";
    }

    co_return failures;
}

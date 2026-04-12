#pragma once

// alltoallw.h - generalized all-to-all over per-peer Subarray descriptors.
//
// Phase 4 of the parallel FFT roadmap. This is the collective the FFT
// redistribute step (Phase 6) calls. Each peer pair (src, dst) gets its own
// send/recv Subarray descriptor; the function fans out one send and one
// recv per peer concurrently and waits for the whole exchange to complete.
//
// Concurrency model:
//   - Every non-self peer's send and recv is spawned with co_spawn + deferred
//     and collected into a vector<deferred_op>.
//   - asio::experimental::ranged_parallel_group(ops) waits on all of them
//     in a single co_await.
//   - Self-loopback (rank == self) is handled inline via a two-cursor
//     fragment-stream memcpy. There is no self-socket in the connect mesh,
//     so this path is mandatory, not just an optimization.
//
// Tag reservation: kAlltoallwTag = -6 (next free after -1 barrier, -2 bcast,
// -3 reduce, -4 scatter, -5 gather). Documented in docs/MPI.md.
//
// Slot semantics: every send_types[p] / recv_types[p] must be a valid
// Subarray pointer (no nullptr). A "no exchange with peer p" slot is just a
// Subarray with extent 0 in some axis - total_bytes() == 0 short-circuits
// the loop without posting an op. This keeps the inner loop branch-free of
// null checks.
//
// Lifetime: caller owns the Subarray objects (typically via a long-lived
// RedistributePlan). Their underlying DistArrays must outlive the awaitable.
//
// API contract: the send Subarray's total_bytes() must equal the matching
// recv Subarray's total_bytes() on the peer. alltoallw does not have a way
// to negotiate sizes - both ends were planned from the same global shape.

#include "asio.hpp"
#include "asio/experimental/parallel_group.hpp"

#include "clustr_mpi.h"
#include "clustr/subarray.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace clustr {

inline constexpr std::int32_t kAlltoallwTag = -6;

namespace alltoallw_detail {

// Walk a Subarray's mutable buffer cache and memcpy `src` (a flat byte
// stream of exactly dst.total_bytes() bytes) into the fragments in order.
// Used by recv: recv_raw returns one contiguous vector<uint8_t> per message,
// and we have to scatter it back into the destination subarray's geometry.
template <typename T>
void scatter_bytes_into(const std::uint8_t* src, std::size_t src_len,
                        Subarray<T>& dst) {
    if (src_len != dst.total_bytes()) {
        throw std::runtime_error(
            "alltoallw: recv size " + std::to_string(src_len)
            + " does not match dst Subarray total_bytes "
            + std::to_string(dst.total_bytes()));
    }
    const auto& bufs = dst.as_mutable_buffers();
    std::size_t off = 0;
    for (const auto& b : bufs) {
        std::memcpy(b.data(), src + off, b.size());
        off += b.size();
    }
}

// Loopback path: copy a send Subarray directly into a recv Subarray with no
// network round-trip. Send and recv may have different fragment counts (e.g.
// fully-coalesced send vs. partial-inner recv), so we walk both as byte
// streams with two cursors and copy chunks of min(remaining_src, remaining_dst).
template <typename T>
void copy_subarray_to_subarray(Subarray<T>& src, Subarray<T>& dst) {
    if (src.total_bytes() != dst.total_bytes()) {
        throw std::runtime_error(
            "alltoallw: self-loopback size mismatch (send "
            + std::to_string(src.total_bytes()) + ", recv "
            + std::to_string(dst.total_bytes()) + ")");
    }
    if (src.total_bytes() == 0) return;

    const auto& src_bufs = src.as_const_buffers();
    const auto& dst_bufs = dst.as_mutable_buffers();

    std::size_t si = 0, soff = 0;
    std::size_t di = 0, doff = 0;

    while (si < src_bufs.size() && di < dst_bufs.size()) {
        const std::size_t src_remaining = src_bufs[si].size() - soff;
        const std::size_t dst_remaining = dst_bufs[di].size() - doff;
        const std::size_t chunk = std::min(src_remaining, dst_remaining);

        std::memcpy(
            static_cast<std::uint8_t*>(dst_bufs[di].data()) + doff,
            static_cast<const std::uint8_t*>(src_bufs[si].data()) + soff,
            chunk);

        soff += chunk;
        doff += chunk;
        if (soff == src_bufs[si].size()) { ++si; soff = 0; }
        if (doff == dst_bufs[di].size()) { ++di; doff = 0; }
    }
}

}  // namespace alltoallw_detail

// ── alltoallw ────────────────────────────────────────────────────────────────
//
// Each rank exchanges one Subarray with every peer:
//   - send_types[p] is the box this rank ships to peer p
//   - recv_types[p] is the box this rank receives from peer p
//
// span sizes must equal comm.size(). Self-slot (p == comm.rank()) is copied
// in place via memcpy. Zero-byte slots are skipped.

template <typename T>
asio::awaitable<void> alltoallw(Comm& comm,
                                std::span<Subarray<T>* const> send_types,
                                std::span<Subarray<T>* const> recv_types) {
    const int n    = comm.size();
    const int self = comm.rank();

    if (static_cast<int>(send_types.size()) != n
        || static_cast<int>(recv_types.size()) != n) {
        throw std::invalid_argument(
            "alltoallw: send_types/recv_types span size must equal comm.size()");
    }

    // ── Self-loopback first (no socket, no awaits) ───────────────────────
    if (send_types[self]->total_bytes() != 0 || recv_types[self]->total_bytes() != 0) {
        alltoallw_detail::copy_subarray_to_subarray(*send_types[self],
                                                     *recv_types[self]);
    }

    auto exec = co_await asio::this_coro::executor;

    // Each operation is a coroutine launched via co_spawn(..., asio::deferred).
    // The deferred wrapper produces a value-typed handle suitable for
    // ranged_parallel_group; the underlying coroutine doesn't start running
    // until the group's async_wait is co_awaited.
    //
    // Capture is by value where possible (peer index, raw pointer to the
    // Subarray) and by reference for `comm`. The lambdas live inside the
    // coroutine frame for the duration of the awaitable, which is the entire
    // alltoallw call - same lifetime as the parent stack frame.

    auto make_send_op = [&comm, self](int dst, Subarray<T>* sub)
            -> asio::awaitable<void> {
        PeerHeader hdr{};
        hdr.magic[0]    = 0xC1;
        hdr.magic[1]    = 0x52;
        hdr.type        = PeerMsgType::SEND;
        hdr.src_rank    = static_cast<std::uint32_t>(self);
        hdr.dst_rank    = static_cast<std::uint32_t>(dst);
        hdr.tag         = kAlltoallwTag;
        hdr.payload_len = static_cast<std::uint32_t>(sub->total_bytes());
        hdr.comm_id     = comm.comm_id();
        co_await comm.send_raw(dst, hdr, sub->as_const_buffers());
    };

    auto make_recv_op = [&comm](int src, Subarray<T>* sub)
            -> asio::awaitable<void> {
        auto bytes = co_await comm.recv_raw(src, kAlltoallwTag, PeerMsgType::SEND);
        alltoallw_detail::scatter_bytes_into<T>(bytes.data(), bytes.size(), *sub);
    };

    using deferred_op_t = decltype(
        asio::co_spawn(exec, make_send_op(0, send_types[0]), asio::deferred));

    std::vector<deferred_op_t> ops;
    ops.reserve(2 * static_cast<std::size_t>(n));

    for (int p = 0; p < n; ++p) {
        if (p == self) continue;

        if (send_types[p]->total_bytes() != 0) {
            ops.push_back(asio::co_spawn(exec,
                make_send_op(p, send_types[p]), asio::deferred));
        }
        if (recv_types[p]->total_bytes() != 0) {
            ops.push_back(asio::co_spawn(exec,
                make_recv_op(p, recv_types[p]), asio::deferred));
        }
    }

    if (ops.empty()) co_return;

    // ranged_parallel_group with wait_for_all() returns
    //   (completion_order, vector<exception_ptr>)
    // when each underlying operation is a coroutine spawned via deferred.
    // We rethrow the first non-null exception (if any).
    auto [order, exceptions] = co_await
        asio::experimental::make_parallel_group(std::move(ops))
            .async_wait(asio::experimental::wait_for_all(),
                         asio::use_awaitable);

    for (auto& eptr : exceptions) {
        if (eptr) std::rethrow_exception(eptr);
    }
}

}  // namespace clustr

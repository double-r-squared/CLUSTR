#pragma once

// cart.hpp — out-of-class definitions for Comm::cart_sub and the sub-comm
// private constructor. Included from the bottom of clustr_mpi.h after Comm is
// fully declared. Only compiled under CLUSTR_RECV=CLUSTR_RECV_CENTRAL.
//
// What lives here:
//
//   1. detail::FnvHash — FNV-1a 64-bit, returning the upper-folded 32 bits.
//      Used to allocate sub-comm comm_ids deterministically across the cohort.
//      Every member of a cohort feeds the same byte sequence (parent comm_id,
//      axis index, then the coords of every NON-projected axis), so they all
//      arrive at the same comm_id without any communication.
//
//   2. Comm::Comm(parent, ...) — sub-comm private constructor. Clones the
//      parent's TCP sockets at world-rank indices into the sub-comm's
//      sockets_ vector at sub-comm-rank indices, so the existing send_raw
//      (which indexes sockets_ by logical destination rank) just works on
//      sub-comms without any sub-comm-aware code path.
//
//   3. Comm::cart_sub(int axis) — projects a Cartesian Comm onto a single
//      axis. See clustr_mpi.h for the design rationale and the public-API
//      contract.
//
// API contract with clustr_mpi.h:
//   - Comm::cart_sub is *declared* on Comm under the CENTRAL guard there.
//     This file *defines* it. Any signature drift is a build break.
//   - The sub-comm ctor is similarly declared in the private section under
//     the CENTRAL guard and defined here.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace clustr {

namespace detail {

// FNV-1a 64-bit, folded down to 32 bits at finalize time. Deterministic
// across ranks given the same input bytes — no endian conversion needed
// because every rank is the same architecture (cluster is homogeneous;
// see PeerHeader endian note in clustr_mpi.h).
struct FnvHash {
    static constexpr std::uint64_t kOffset = 0xcbf29ce484222325ULL;
    static constexpr std::uint64_t kPrime  = 0x00000100000001b3ULL;

    std::uint64_t state = kOffset;

    void feed_byte(std::uint8_t b) noexcept {
        state ^= b;
        state *= kPrime;
    }

    void feed_u32(std::uint32_t v) noexcept {
        feed_byte(static_cast<std::uint8_t>( v        & 0xFF));
        feed_byte(static_cast<std::uint8_t>((v >> 8 ) & 0xFF));
        feed_byte(static_cast<std::uint8_t>((v >> 16) & 0xFF));
        feed_byte(static_cast<std::uint8_t>((v >> 24) & 0xFF));
    }

    void feed_i32(int v) noexcept {
        feed_u32(static_cast<std::uint32_t>(v));
    }

    // Fold the 64-bit state down to 32 bits by xor'ing the halves. Keeps the
    // collision rate close to a true 32-bit hash without truncating away
    // entropy from either half.
    std::uint32_t finalize_u32() noexcept {
        return static_cast<std::uint32_t>(state) ^
               static_cast<std::uint32_t>(state >> 32);
    }
};

}  // namespace detail

// ── Sub-comm private constructor ─────────────────────────────────────────────

inline Comm::Comm(Comm& parent_comm, std::uint32_t sub_comm_id,
                  int sub_logical_rank, int sub_size,
                  std::vector<int> sub_dims, std::vector<int> sub_coords,
                  std::vector<int> world_ranks_in)
    : io_(parent_comm.io_),
      roster_(),                            // unused on sub-comms
      comm_id_(sub_comm_id),
      transport_iov_cap_(parent_comm.transport_iov_cap_),
      my_logical_rank_(sub_logical_rank),
      my_size_(sub_size),
      parent_(&parent_comm),
      child_count_(0),
      dims_(std::move(sub_dims)),
      coords_(std::move(sub_coords)),
      world_ranks_(std::move(world_ranks_in))
{
    if (static_cast<int>(world_ranks_.size()) != sub_size)
        throw std::logic_error(
            "clustr::Comm sub-ctor: world_ranks_.size() ("
            + std::to_string(world_ranks_.size())
            + ") != sub_size (" + std::to_string(sub_size) + ")");

    if (sub_logical_rank < 0 || sub_logical_rank >= sub_size)
        throw std::logic_error(
            "clustr::Comm sub-ctor: sub_logical_rank "
            + std::to_string(sub_logical_rank) + " out of range [0, "
            + std::to_string(sub_size) + ")");

    // Clone parent sockets at world-rank indices into our sockets_ vector at
    // sub-comm-rank indices. The self slot stays empty (we never send to
    // ourselves over a socket — collectives short-circuit self-sends).
    sockets_.resize(sub_size);
    for (int sub_r = 0; sub_r < sub_size; ++sub_r) {
        if (sub_r == sub_logical_rank) continue;

        const int wr = world_ranks_[sub_r];
        if (wr < 0 || wr >= static_cast<int>(parent_comm.sockets_.size()))
            throw std::logic_error(
                "clustr::Comm sub-ctor: cohort world rank "
                + std::to_string(wr) + " out of parent socket range [0, "
                + std::to_string(parent_comm.sockets_.size()) + ")");

        auto sp = parent_comm.sockets_[wr];
        if (!sp)
            throw std::logic_error(
                "clustr::Comm sub-ctor: parent sockets_["
                + std::to_string(wr) + "] is null (peer never connected?)");

        // shared_ptr clone — bumps refcount, both Comms own the socket and
        // either may close it during shutdown. The dispatch loop also holds
        // a copy via central_dispatch_loop_ capture, so the socket survives
        // even if the sub-comm is destroyed mid-flight.
        sockets_[sub_r] = sp;
    }

    // Bump *last*: any throw above leaves child_count_ untouched so the
    // parent's destructor assertion remains accurate.
    ++parent_comm.child_count_;
}

// ── Comm::cart_sub ───────────────────────────────────────────────────────────

inline Comm Comm::cart_sub(int axis) {
    if (dims_.empty())
        throw std::runtime_error(
            "clustr::Comm::cart_sub: this Comm has no Cartesian topology — "
            "call cart_create first");

    if (parent_)
        throw std::runtime_error(
            "clustr::Comm::cart_sub: nested cart_sub off a sub-comm is not "
            "yet supported (Phase 5 only projects off the world Comm)");

    if (axis < 0 || axis >= static_cast<int>(dims_.size()))
        throw std::out_of_range(
            "clustr::Comm::cart_sub: axis " + std::to_string(axis)
            + " out of range [0, " + std::to_string(dims_.size()) + ")");

    const int sub_size = dims_[axis];
    const int sub_my   = coords_[axis];

    // Stride for the projected axis under row-major rank layout (last axis
    // varies fastest, matching cart_create's coord assignment).
    long stride = 1;
    for (std::size_t a = static_cast<std::size_t>(axis) + 1;
         a < dims_.size(); ++a)
        stride *= dims_[a];

    // base = my world rank with coords_[axis] zeroed out. Walking from base
    // by +stride enumerates the cohort in sub-comm-rank order.
    const long base =
        static_cast<long>(my_logical_rank_) - static_cast<long>(sub_my) * stride;

    std::vector<int> world_ranks_table(sub_size);
    for (int i = 0; i < sub_size; ++i)
        world_ranks_table[i] = static_cast<int>(base + i * stride);

    // Sanity: every cohort entry must be a real world rank, and the table
    // must hash back to my own world rank at my own sub-comm-rank position.
    for (int wr : world_ranks_table) {
        if (wr < 0 || wr >= my_size_)
            throw std::logic_error(
                "clustr::Comm::cart_sub: computed cohort world rank "
                + std::to_string(wr) + " out of range [0, "
                + std::to_string(my_size_) + ") — cart math bug");
    }
    if (world_ranks_table[sub_my] != my_logical_rank_)
        throw std::logic_error(
            "clustr::Comm::cart_sub: cohort table[my_sub_rank] != my world "
            "rank — cart math bug");

    // Deterministic comm_id: every cohort member feeds the same bytes
    // (parent comm_id, axis, then every coord EXCEPT the projected one),
    // so they all converge on the same hash without any communication.
    detail::FnvHash h;
    h.feed_u32(comm_id_);
    h.feed_i32(axis);
    for (std::size_t a = 0; a < dims_.size(); ++a) {
        if (static_cast<int>(a) == axis) continue;
        h.feed_i32(coords_[a]);
    }
    std::uint32_t sub_id = h.finalize_u32();

    // Never collide with the world id. The probability is 1/2^32, but the
    // consequence (frames silently routed to the wrong central channel) is
    // bad enough to justify the cheap guard.
    if (sub_id == kWorldCommId) sub_id = 1;

#ifndef NDEBUG
    if (!issued_sub_ids_.insert(sub_id).second) {
        // Local collision: this Comm already issued the same id earlier.
        // Either a hash bug or a duplicate cart_sub for the same cohort.
        throw std::logic_error(
            "clustr::Comm::cart_sub: local comm_id collision (id="
            + std::to_string(sub_id) + ") — duplicate cart_sub call?");
    }
#endif

    std::vector<int> sub_dims_v   { sub_size };
    std::vector<int> sub_coords_v { sub_my   };

    return Comm(*this, sub_id, sub_my, sub_size,
                std::move(sub_dims_v), std::move(sub_coords_v),
                std::move(world_ranks_table));
}

}  // namespace clustr

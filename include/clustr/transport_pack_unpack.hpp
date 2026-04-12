#pragma once

// transport_pack_unpack.hpp — reference pack/unpack transport.
//
// Allocates a single contiguous scratch buffer per message, memcpy's the
// header followed by every fragment of the payload buffer sequence into it,
// and issues a single async_write. Receive mirrors the pattern: one
// async_read into a preallocated vector<uint8_t>, then the caller unpacks.
//
// This is the *reference* implementation. It is intentionally the simplest
// thing that can be correct, and serves as both:
//
//   - a sanity check against the zero-copy transport during development, and
//   - the baseline half of the Phase 9.5 benchmark (ROADMAP.md "Dual transport
//     strategy").
//
// API contract with transport_zero_copy.hpp:
//   - Both files define Comm::send_raw<BufferSeq>(dst, hdr, payload_bufs)
//   - Both files define Comm::recv_raw(src, tag, type)
//   - Both files define the (ptr, len) send_raw convenience overload
//   - Any divergence is a bug. Build and test the full matrix against both.

#include "asio.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace clustr {

// ── Comm::send_raw (buffer-sequence form) ─────────────────────────────────────
//
// Walks the ConstBufferSequence, sums its total size, allocates one scratch
// vector of (header + total), memcpy's everything in, and issues one async_write.
// Deliberately uses a single write call — this is the "obvious" baseline the
// zero-copy path must beat.

template <typename BufferSeq>
asio::awaitable<void> Comm::send_raw(int dst, const PeerHeader& hdr,
                                      const BufferSeq& payload_bufs) {
    // Sum fragment sizes to size the scratch buffer.
    std::size_t total = 0;
    for (auto it = asio::buffer_sequence_begin(payload_bufs);
         it != asio::buffer_sequence_end(payload_bufs); ++it) {
        total += asio::buffer_size(*it);
    }

    if (total != hdr.payload_len) {
        throw std::runtime_error(
            "clustr_mpi (pack_unpack): payload sequence size "
            + std::to_string(total) + " does not match PeerHeader::payload_len "
            + std::to_string(hdr.payload_len));
    }

    std::vector<std::uint8_t> scratch(sizeof(PeerHeader) + total);
    std::memcpy(scratch.data(), &hdr, sizeof(PeerHeader));

    // memcpy every fragment into place.
    std::size_t off = sizeof(PeerHeader);
    for (auto it = asio::buffer_sequence_begin(payload_bufs);
         it != asio::buffer_sequence_end(payload_bufs); ++it) {
        const std::size_t n = asio::buffer_size(*it);
        if (n) std::memcpy(scratch.data() + off, (*it).data(), n);
        off += n;
    }

    co_await asio::async_write(*sockets_[dst],
        asio::buffer(scratch), asio::use_awaitable);
}

// ── Comm::send_raw (ptr/len convenience form) ────────────────────────────────
//
// Wraps a bare (ptr, len) in a one-element const_buffer sequence and forwards
// to the buffer-sequence form. Same signature is provided by the zero-copy
// header, so barrier() and other internal call sites stay transport-agnostic.

inline asio::awaitable<void> Comm::send_raw(int dst, const PeerHeader& hdr,
                                             const void* payload, std::uint32_t len) {
    std::array<asio::const_buffer, 1> seq{{ asio::buffer(payload, len) }};
    co_await send_raw(dst, hdr, seq);
}

// ── Comm::recv_raw ───────────────────────────────────────────────────────────
//
// recv_raw is no longer defined in this file. As of Phase 5 it is selected
// at compile time via CLUSTR_RECV in clustr_mpi.h:
//
//   CLUSTR_RECV_INLINE  -> include/clustr/recv_inline.hpp  (Phase 0..4 path)
//   CLUSTR_RECV_CENTRAL -> include/clustr/recv_central.hpp (sub-comm capable)
//
// Both modes work with this transport. Build the full transport x recv
// matrix to catch any drift.

}  // namespace clustr

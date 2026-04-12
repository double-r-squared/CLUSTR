#pragma once

// transport_zero_copy.hpp — scatter-gather transport (default).
//
// Header and payload are handed to the kernel as an iovec without any
// intermediate packing. This is the implementation described in ROADMAP.md
// Phase 2/3/4 and is what the parallel FFT relies on for bandwidth.
//
// Layered IOV_MAX strategy:
//
//   Layer 1 — Coalesce: the producer of the buffer sequence (Subarray in
//             Phase 3) merges adjacent fragments upstream. The transport
//             trusts the caller to have done this.
//
//   Layer 2 — Chunked writev/readv: write_seq_chunked and read_seq_chunked
//             loop over the input in batches of (transport_iov_cap_ - 1)
//             entries, leaving one slot for the header on sends. TCP
//             preserves order on a connection, so the receiver transparently
//             reassembles the bytes whether they arrived in 1 syscall or K.
//
//   Layer 3 — Staging fallback: currently unreachable in practice; would fire
//             if a single logical message produced more than IOV_MAX
//             fragments *after* coalescing. Not implemented yet (Phase 3
//             adds coalesce, Phase 4 can re-evaluate if benchmarks show a
//             pathological input).
//
// API contract with transport_pack_unpack.hpp:
//   - Both files define Comm::send_raw<BufferSeq>(dst, hdr, payload_bufs)
//   - Both files define Comm::recv_raw(src, tag, type)
//   - Both files define the (ptr, len) send_raw convenience overload
//   - Any divergence is a bug. Build and test the full matrix against both.

#include "asio.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clustr {
namespace transport_detail {

// ── write_seq_chunked ────────────────────────────────────────────────────────
//
// Loops over a ConstBufferSequence in batches of at most `cap` entries,
// issuing one async_write per batch. Caller must provide `cap <= iov_max`.
//
// We deliberately materialize each batch as a std::vector<const_buffer>
// instead of using fancy sequence adapters — the vector is tiny (at most
// `cap` elements, typically 1-8), lives only for the duration of one
// async_write, and keeps the control flow dead simple.

template <typename BufferSeq>
asio::awaitable<void> write_seq_chunked(asio::ip::tcp::socket& s,
                                         const BufferSeq& seq,
                                         std::size_t cap) {
    std::vector<asio::const_buffer> batch;
    batch.reserve(cap);

    auto it   = asio::buffer_sequence_begin(seq);
    auto end  = asio::buffer_sequence_end(seq);

    while (it != end) {
        batch.clear();
        while (it != end && batch.size() < cap) {
            // Skip empty fragments — writev would accept them but they waste
            // an iovec slot.
            if (asio::buffer_size(*it) != 0) {
                batch.push_back(*it);
            }
            ++it;
        }
        if (!batch.empty()) {
            co_await asio::async_write(s, batch, asio::use_awaitable);
        }
    }
}

}  // namespace transport_detail

// ── Comm::send_raw (buffer-sequence form) ─────────────────────────────────────
//
// Emits the 24-byte header followed by the caller's payload sequence in a
// single logical message. When the combined iovec fits in one writev (the
// common case), this is one syscall. When it doesn't, the chunking loop
// splits the sequence; TCP preserves order so reassembly is free.

template <typename BufferSeq>
asio::awaitable<void> Comm::send_raw(int dst, const PeerHeader& hdr,
                                      const BufferSeq& payload_bufs) {
    // Sanity: the payload sequence must match the header's declared length.
    // The receiver uses payload_len to size its read, so any mismatch here
    // corrupts the stream on the wire.
    std::size_t total = 0;
    for (auto it = asio::buffer_sequence_begin(payload_bufs);
         it != asio::buffer_sequence_end(payload_bufs); ++it) {
        total += asio::buffer_size(*it);
    }
    if (total != hdr.payload_len) {
        throw std::runtime_error(
            "clustr_mpi (zero_copy): payload sequence size "
            + std::to_string(total) + " does not match PeerHeader::payload_len "
            + std::to_string(hdr.payload_len));
    }

    // Count non-empty payload fragments so we can decide whether the whole
    // (header + payload) fits in one writev.
    std::size_t frag_count = 0;
    for (auto it = asio::buffer_sequence_begin(payload_bufs);
         it != asio::buffer_sequence_end(payload_bufs); ++it) {
        if (asio::buffer_size(*it) != 0) ++frag_count;
    }

    // Fast path: header + all fragments fit in a single writev.
    if (frag_count + 1 <= transport_iov_cap_) {
        std::vector<asio::const_buffer> combined;
        combined.reserve(frag_count + 1);
        combined.emplace_back(asio::buffer(&hdr, sizeof(PeerHeader)));
        for (auto it = asio::buffer_sequence_begin(payload_bufs);
             it != asio::buffer_sequence_end(payload_bufs); ++it) {
            if (asio::buffer_size(*it) != 0) combined.push_back(*it);
        }
        co_await asio::async_write(*sockets_[dst], combined, asio::use_awaitable);
        co_return;
    }

    // Slow path: more fragments than IOV_MAX allows. Send the header alone in
    // one call, then stream the payload in chunks of (cap) entries. TCP
    // delivers everything in order on the connection so the receiver sees a
    // normal contiguous stream.
    co_await asio::async_write(*sockets_[dst],
        asio::buffer(&hdr, sizeof(PeerHeader)), asio::use_awaitable);

    co_await transport_detail::write_seq_chunked(*sockets_[dst],
                                                  payload_bufs,
                                                  transport_iov_cap_);
}

// ── Comm::send_raw (ptr/len convenience form) ────────────────────────────────

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

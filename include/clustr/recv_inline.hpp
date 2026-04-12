#pragma once

// recv_inline.hpp - inline (per-call) receive path. World-comm only.
//
// This is the original Phase 0/1/2 recv_raw implementation, lifted out of
// the two transport headers so it can be selected at compile time alongside
// the future central-dispatch path. Behavior is byte-identical to what
// transport_zero_copy.hpp and transport_pack_unpack.hpp used to define
// inline:
//
//   1. Check the per-(src, tag) mailbox first. If a buffered frame is
//      already there, return it.
//   2. Otherwise, loop on async_read against sockets_[src]:
//        a. Read the 24-byte PeerHeader.
//        b. Validate magic bytes and comm_id (world == 0).
//        c. Allocate vector<uint8_t>(payload_len) and read into it.
//        d. If (type, src_rank, tag) match, return.
//        e. Otherwise buffer the frame in the mailbox keyed by its actual
//           (src_rank, tag) and keep reading.
//
// This path does NOT support sub-communicators. Calling cart_sub() under
// CLUSTR_RECV_INLINE is a compile-time error (static_assert in clustr_mpi.h).
// Use CLUSTR_RECV_CENTRAL when sub-comms are required (Phase 5+).
//
// API contract with recv_central.hpp:
//   - Both files define Comm::recv_raw(int src, int tag, PeerMsgType type)
//     out-of-class, returning asio::awaitable<std::vector<std::uint8_t>>
//   - Any divergence is a bug. Build the full matrix against both modes.

#include "asio.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace clustr {

inline asio::awaitable<std::vector<std::uint8_t>>
Comm::recv_raw(int src, int tag, PeerMsgType type) {
    MailboxKey key{src, tag};

    auto it = mailbox_.find(key);
    if (it != mailbox_.end() && !it->second.empty()) {
        auto msg = std::move(it->second.front().data);
        it->second.pop_front();
        co_return msg;
    }

    while (true) {
        PeerHeader hdr{};
        co_await asio::async_read(*sockets_[src],
            asio::buffer(&hdr, sizeof(PeerHeader)), asio::use_awaitable);

        if (hdr.magic[0] != 0xC1 || hdr.magic[1] != 0x52)
            throw std::runtime_error("clustr_mpi (recv_inline): bad peer magic bytes");

        if (hdr.comm_id != comm_id_)
            throw std::runtime_error(
                "clustr_mpi (recv_inline): frame with unknown comm_id="
                + std::to_string(hdr.comm_id)
                + " (expected " + std::to_string(comm_id_) + ")");

        std::vector<std::uint8_t> payload(hdr.payload_len);
        if (hdr.payload_len)
            co_await asio::async_read(*sockets_[src],
                asio::buffer(payload), asio::use_awaitable);

        if (hdr.type == type && static_cast<int>(hdr.src_rank) == src && hdr.tag == tag) {
            co_return payload;
        }

        MailboxKey mkey{static_cast<int>(hdr.src_rank), hdr.tag};
        mailbox_[mkey].push_back({std::move(payload)});
    }
}

}  // namespace clustr

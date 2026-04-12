#pragma once

// recv_central.hpp - central dispatch loop receive path. Required for
// sub-communicators (Phase 5+) because all sub-comms share the underlying
// TCP connection with the world communicator and must be demultiplexed by
// PeerHeader::comm_id.
//
// How it works:
//
//   1. Comm::connect() finishes the socket mesh, then spawns one detached
//      coroutine per remote peer (start_central_dispatch_loops_).
//   2. Each dispatch coroutine reads a PeerHeader, validates magic, allocates
//      a payload buffer sized by hdr.payload_len, reads the payload, then
//      pushes it into the per-(comm_id, src_rank, tag, type) channel.
//   3. recv_raw (this file) computes the same key from its (src, tag, type)
//      arguments plus this Comm's comm_id_, looks up (or lazily creates)
//      the matching channel, and async_receives one payload from it.
//
// Channels are unbounded: we never want the dispatch loop to stall waiting
// for a slow consumer, because that would block readv on the shared socket
// and starve every other (comm_id, tag, type) tuple sharing the connection.
//
// Single-threaded io_context: get_or_create_channel_ has no locking. If
// CLUSTR ever moves to a multi-threaded io_context, the channel map needs
// either a strand or a mutex.
//
// API contract with recv_inline.hpp:
//   - Both files define Comm::recv_raw(int src, int tag, PeerMsgType type)
//     out-of-class, returning asio::awaitable<std::vector<std::uint8_t>>
//   - Any divergence is a bug. Build the full matrix against both modes.

#include "asio.hpp"
#include "asio/experimental/channel.hpp"
#include <cstdint>
#include <utility>
#include <vector>

namespace clustr {

inline asio::awaitable<std::vector<std::uint8_t>>
Comm::recv_raw(int src, int tag, PeerMsgType type) {
    // Dispatch loops live on the world Comm — there is exactly one per remote
    // socket and they read the shared TCP stream into per-(comm_id, src, tag,
    // type) channels stored in world's CentralState. Sub-comms must therefore
    // look up channels in the world Comm, not in their own (default-empty)
    // central_ map. Walk the parent chain to the root in case nested cart_sub
    // is added later.
    Comm* root = this;
    while (root->parent_) root = root->parent_;

    // Demux by (this comm's id, sender rank in this comm's logical space, tag,
    // type). The dispatch loop pushes with the same key because send_raw on
    // the sender side stamped hdr.comm_id and hdr.src_rank from the same
    // logical view, so the (comm_id, src) tuple matches across send/recv
    // without any sub-comm-aware translation.
    CentralKey key{comm_id_, src, tag, type};
    auto& ch = root->get_or_create_channel_(key);

    // Channels are unbounded — async_receive only blocks when the channel is
    // empty (no frame has arrived for this key yet). On shutdown the channel
    // is close()'d, which completes this await with operation_aborted.
    auto payload = co_await ch.async_receive(asio::use_awaitable);
    co_return payload;
}

}  // namespace clustr

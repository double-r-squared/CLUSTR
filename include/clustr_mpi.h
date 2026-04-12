#pragma once

// ============================================================================
// clustr_mpi.h — Header-only C++20 coroutine MPI API for CLUSTR job files.
//
// Design decisions (see doc/mpi_design.md for full rationale):
//
//   Transport:    ASIO async TCP. One persistent connection per rank pair.
//                 No third-party MPI dependency. Ships with the worker binary
//                 since ASIO is already bundled via CMake FetchContent.
//
//   Wire format:  Fixed-size PeerHeader (16 bytes) + variable payload.
//                 All multi-byte integers are little-endian (cluster is
//                 assumed homogeneous x86_64 or ARM64). Add htonl/ntohl
//                 guards here if heterogeneous architectures are added.
//
//   Roster:       Read from /var/tmp/clustr/mpi_roster.conf (INI format).
//                 Written by the worker daemon before exec. No JSON, no
//                 protobuf — same parser as system_conf.h.
//
//   Barrier:      Centralized (rank 0 as coordinator). O(N) messages.
//                 A dissemination/butterfly barrier (O(log N) rounds) is
//                 stubbed below for future implementation when N > ~16.
//
//   Fault model:  Fail-stop. If any rank disconnects, remaining ranks will
//                 see a broken pipe on the next send/recv and throw. The
//                 scheduler detects missing RANK_DONE/TASK_RESULT and marks
//                 the job failed.
//                 FAULT_TOLERANT build flag (stub): checkpoint/restart support
//                 would go here — see fault_tolerance_stub() below.
//
//   Entry point:  Use CLUSTR_MPI_MAIN(mpi) macro instead of int main().
//                 This sets up the ASIO io_context, loads the roster, and
//                 establishes all peer connections before calling user code.
//
// Usage:
//
//   #include "clustr_mpi.h"
//
//   CLUSTR_MPI_MAIN(mpi) {
//       int r = mpi.rank();
//       int n = mpi.size();
//
//       if (r == 0) {
//           std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
//           co_await mpi.scatter(data);           // send chunk[i] to rank i
//       } else {
//           auto chunk = co_await mpi.scatter<double>(); // receive my chunk
//           for (auto& v : chunk) v *= 2.0;
//           co_await mpi.gather(chunk);            // send back to rank 0
//       }
//       if (r == 0) {
//           auto results = co_await mpi.gather<double>(); // collect all
//       }
//       co_return 0;
//   }
//
// Compile:
//   g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED \
//       -I./include -I./asio_include \
//       myjob.cpp -o myjob -lpthread
//
// ============================================================================

#include "asio.hpp"
#include "clustr/transport_common.hpp"
#include <array>
#include <cassert>
#include <coroutine>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <mutex>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>

// ============================================================================
// Receive-path selector (constants only — the include is at the bottom).
//
// We need CLUSTR_RECV visible *before* the Comm class declaration so the
// CENTRAL-mode state members (channel map, dispatch loops) can be guarded
// inline. The actual recv_*.hpp include still happens at the bottom of this
// file, after Comm is fully declared, because those headers define
// Comm::recv_raw out-of-class.
// ============================================================================

#define CLUSTR_RECV_INLINE  1
#define CLUSTR_RECV_CENTRAL 2

#ifndef CLUSTR_RECV
#  define CLUSTR_RECV CLUSTR_RECV_INLINE
#endif

#if CLUSTR_RECV == CLUSTR_RECV_CENTRAL
#  include "asio/experimental/channel.hpp"
#endif

namespace clustr {

// ============================================================================
// Peer wire protocol
// ============================================================================
//
// Every peer-to-peer message uses this fixed header followed by payload_len
// bytes of raw data. The magic bytes catch misaligned reads early.
//
// Layout (24 bytes total):
//   [0-1]  magic      0xC1 0x52  ("ClustR peer")
//   [2]    type       PeerMsgType
//   [3]    reduce_op  ReduceOp (only meaningful for REDUCE/ALLREDUCE)
//   [4-7]  src_rank   uint32_t LE
//   [8-11] dst_rank   uint32_t LE  (0xFFFFFFFF = broadcast)
//   [12-15] tag       int32_t LE
//   [16-19] payload_len  uint32_t LE
//   [20-23] comm_id   uint32_t LE  (0 = world; sub-communicators get nonzero
//                                   ids assigned by Comm::cart_sub in Phase 5)

enum class PeerMsgType : uint8_t {
    SEND    = 0x01,  // Point-to-point data
    BARRIER = 0x02,  // Synchronization token (no payload)
    BCAST   = 0x03,  // Root broadcasts data to all ranks
    REDUCE  = 0x04,  // All ranks send data to root for reduction
    SCATTER = 0x05,  // Root distributes one chunk per rank
    GATHER  = 0x06,  // All ranks send chunk to root
    // ALLREDUCE = 0x07  // Future: ring-allreduce (reduce + bcast in O(1) rounds)
};

enum class ReduceOp : uint8_t {
    SUM  = 0x00,
    MIN  = 0x01,
    MAX  = 0x02,
    PROD = 0x03,
};

#pragma pack(push, 1)
struct PeerHeader {
    uint8_t     magic[2]    = {0xC1, 0x52};
    PeerMsgType type;
    ReduceOp    reduce_op   = ReduceOp::SUM;
    uint32_t    src_rank;
    uint32_t    dst_rank;
    int32_t     tag;
    uint32_t    payload_len;
    uint32_t    comm_id     = 0;    // 0 = world; nonzero = sub-communicator (Phase 5)
};
static_assert(sizeof(PeerHeader) == 24, "PeerHeader must be 24 bytes");
#pragma pack(pop)

// World communicator id. Sub-communicators created in Phase 5 get nonzero ids.
inline constexpr uint32_t kWorldCommId = 0;

// ============================================================================
// Roster — parsed from /var/tmp/clustr/mpi_roster.conf
// ============================================================================
//
// Format (INI, same style as system.conf):
//   rank=0
//   size=3
//   peer_port=10000
//   peer.0=100.92.163.46:10000
//   peer.1=100.100.147.7:10001
//   peer.2=100.92.163.50:10002

struct PeerInfo {
    int         rank;
    std::string ip;
    uint16_t    port;
};

struct Roster {
    int                   my_rank;
    int                   size;
    uint16_t              my_peer_port;
    std::vector<PeerInfo> peers;   // all peers including self (skip self on connect)

    static Roster load(const std::string& path = "/var/tmp/clustr/mpi_roster.conf") {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("clustr_mpi: cannot open roster: " + path);

        Roster r{};
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);

            if      (key == "rank")      r.my_rank      = std::stoi(val);
            else if (key == "size")      r.size         = std::stoi(val);
            else if (key == "peer_port") r.my_peer_port = static_cast<uint16_t>(std::stoi(val));
            else if (key.rfind("peer.", 0) == 0) {
                // peer.N=ip:port
                int rank = std::stoi(key.substr(5));
                auto colon = val.rfind(':');
                PeerInfo pi;
                pi.rank = rank;
                pi.ip   = val.substr(0, colon);
                pi.port = static_cast<uint16_t>(std::stoi(val.substr(colon + 1)));
                if ((int)r.peers.size() <= rank) r.peers.resize(rank + 1);
                r.peers[rank] = pi;
            }
        }
        if (r.size == 0) throw std::runtime_error("clustr_mpi: roster has size=0");
        return r;
    }
};

// ============================================================================
// Mailbox — holds messages that arrived before recv() was called
// ============================================================================

struct MailboxKey {
    int src_rank;
    int tag;
    bool operator==(const MailboxKey& o) const {
        return src_rank == o.src_rank && tag == o.tag;
    }
};

struct MailboxKeyHash {
    size_t operator()(const MailboxKey& k) const {
        return std::hash<int>()(k.src_rank) ^ (std::hash<int>()(k.tag) << 16);
    }
};

#if CLUSTR_RECV == CLUSTR_RECV_CENTRAL

// CentralKey demultiplexes the central dispatch loop's incoming frames.
// World comm and every sub-communicator on the same socket pair share the
// underlying TCP stream, so the (comm_id, src, tag, type) tuple is what
// uniquely identifies "who is waiting for this frame".
struct CentralKey {
    uint32_t    comm_id;
    int         src_rank;
    int         tag;
    PeerMsgType type;
    bool operator==(const CentralKey& o) const noexcept {
        return comm_id == o.comm_id && src_rank == o.src_rank
            && tag == o.tag && type == o.type;
    }
};

struct CentralKeyHash {
    std::size_t operator()(const CentralKey& k) const noexcept {
        // boost::hash_combine pattern.
        std::size_t h = std::hash<uint32_t>{}(k.comm_id);
        h ^= std::hash<int>{}(k.src_rank) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.tag)      + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.type))
              + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

#endif // CLUSTR_RECV == CLUSTR_RECV_CENTRAL

// ============================================================================
// Comm — a communicator (world or sub-group). The unit of MPI operations.
//
// Phase 1: only the world communicator exists. All messages carry comm_id=0.
// Phase 5: Comm::cart_sub will produce row/column sub-communicators that share
// the parent's sockets but tag messages with a unique comm_id.
//
// The existing `clustr::MPI` name is kept as a thin facade (below) that owns
// a world Comm and forwards rank()/size()/connect() + collectives, so existing
// user code (CLUSTR_MPI_MAIN, mpi.rank(), etc.) keeps working unchanged.
// ============================================================================

class Comm {
public:
    // World-communicator constructor: loads the roster and prepares for connect().
    explicit Comm(asio::io_context& io, const std::string& roster_path =
                  "/var/tmp/clustr/mpi_roster.conf")
        : io_(io),
          roster_(Roster::load(roster_path)),
          comm_id_(kWorldCommId),
          transport_iov_cap_(query_iov_max()),
          my_logical_rank_(roster_.my_rank),
          my_size_(roster_.size) {}

    // Non-copyable (owns a TCP acceptor) and non-move-assignable (io_ is a
    // reference). Move-construct is custom — it transfers parent ownership
    // so the source object's destructor doesn't double-decrement the parent's
    // child counter. cart_sub returns by value and relies on this.
    Comm(const Comm&)            = delete;
    Comm& operator=(const Comm&) = delete;
    Comm& operator=(Comm&&)      = delete;

    Comm(Comm&& o) noexcept
        : io_(o.io_),
          roster_(std::move(o.roster_)),
          comm_id_(o.comm_id_),
          transport_iov_cap_(o.transport_iov_cap_),
          acceptor_(std::move(o.acceptor_)),
          sockets_(std::move(o.sockets_)),
          mailbox_(std::move(o.mailbox_)),
#if CLUSTR_RECV == CLUSTR_RECV_CENTRAL
          central_(std::move(o.central_)),
#endif
          my_logical_rank_(o.my_logical_rank_),
          my_size_(o.my_size_),
          parent_(o.parent_),
          child_count_(o.child_count_),
          dims_(std::move(o.dims_)),
          coords_(std::move(o.coords_)),
          world_ranks_(std::move(o.world_ranks_))
#ifndef NDEBUG
          , issued_sub_ids_(std::move(o.issued_sub_ids_))
#endif
    {
        // Source no longer owns any registration with the parent.
        o.parent_      = nullptr;
        o.child_count_ = 0;
    }

    // Destructor: a sub-comm decrements its parent's child counter; a parent
    // asserts that no sub-comms still reference it (debug only — production
    // sub-comm-after-parent calls are UB on the parent's central_ map).
    ~Comm() {
        if (parent_) --parent_->child_count_;
        assert(child_count_ == 0
               && "clustr::Comm destroyed while sub-comms still reference it");
    }

    int      rank()    const { return my_logical_rank_; }
    int      size()    const { return my_size_;         }
    uint32_t comm_id() const { return comm_id_;         }

    // ── Setup ─────────────────────────────────────────────────────────────────
    //
    // Call once before any send/recv. Binds the peer acceptor, connects to all
    // higher-ranked peers, and accepts connections from all lower-ranked peers.
    // Returns when the full mesh is established.

    asio::awaitable<void> connect() {
        using tcp = asio::ip::tcp;
        using namespace std::chrono_literals;

        // Bind acceptor on the pre-assigned port written by the worker daemon.
        // ASIO sets SO_REUSEADDR automatically, so re-binding the port the daemon
        // released a moment ago works reliably.
        acceptor_ = std::make_unique<tcp::acceptor>(
            io_, tcp::endpoint(tcp::v4(), roster_.my_peer_port));

        int my_rank = roster_.my_rank;
        int n       = roster_.size;
        sockets_.resize(n);

        // Accept connections from all lower-ranked peers.
        // Lower-ranked peers connect to us first, so this can't deadlock:
        //   rank 0 has no accepts → goes straight to the connect loop.
        //   rank 1 waits for rank 0, which has already moved on to connecting.
        //   rank 2 waits for ranks 0 and 1, which both connect before looping.
        for (int i = 0; i < my_rank; i++) {
            auto sock = std::make_shared<tcp::socket>(io_);
            co_await acceptor_->async_accept(*sock, asio::use_awaitable);
            // 4-byte handshake from the connecting side tells us its rank
            uint32_t peer_rank = 0;
            co_await asio::async_read(*sock,
                asio::buffer(&peer_rank, sizeof(peer_rank)), asio::use_awaitable);
            sockets_[peer_rank] = std::move(sock);
        }

        // Connect to all higher-ranked peers.
        //
        // All binaries start at nearly the same time after EXEC_CMD, so the
        // remote acceptor may not be bound yet when we first try. Retry with
        // exponential backoff (100ms -> 200ms -> ... capped at 2s, 20 attempts
        // total = up to ~20s window).  "Connection refused" is the only expected
        // transient error; anything else is fatal immediately.
        static constexpr int MAX_ATTEMPTS = 20;

        for (int r = my_rank + 1; r < n; r++) {
            tcp::resolver resolver(io_);
            auto endpoints = co_await resolver.async_resolve(
                roster_.peers[r].ip,
                std::to_string(roster_.peers[r].port),
                asio::use_awaitable);

            auto delay = 100ms;
            for (int attempt = 0; ; attempt++) {
                auto sock = std::make_shared<tcp::socket>(io_);
                asio::error_code ec;
                co_await asio::async_connect(*sock, endpoints,
                    asio::redirect_error(asio::use_awaitable, ec));

                if (!ec) {
                    // Send handshake: our rank number
                    uint32_t my_rank_u = static_cast<uint32_t>(my_rank);
                    co_await asio::async_write(*sock,
                        asio::buffer(&my_rank_u, sizeof(my_rank_u)),
                        asio::use_awaitable);
                    sockets_[r] = std::move(sock);
                    break;
                }

                if (attempt >= MAX_ATTEMPTS)
                    throw std::system_error(ec,
                        "clustr_mpi: rank " + std::to_string(my_rank)
                        + " could not reach rank " + std::to_string(r)
                        + " at " + roster_.peers[r].ip
                        + ":" + std::to_string(roster_.peers[r].port)
                        + " after " + std::to_string(MAX_ATTEMPTS) + " attempts");

                asio::steady_timer t(io_, delay);
                co_await t.async_wait(asio::use_awaitable);
                // Exponential backoff, capped at 2s
                delay = std::min(delay * 2, std::chrono::milliseconds(2000));
            }
        }

#if CLUSTR_RECV == CLUSTR_RECV_CENTRAL
        // Spawn one detached dispatch coroutine per remote socket. Each
        // drains its socket eagerly and routes frames into per-key channels
        // that recv_raw waits on. See start_central_dispatch_loops_ below.
        start_central_dispatch_loops_();
#endif
    }

    // ── Point-to-point ────────────────────────────────────────────────────────

    // Send a vector of trivially-copyable T to dst_rank with tag.
    template<typename T>
    asio::awaitable<void> send(const std::vector<T>& data, int dst, int tag = 0) {
        static_assert(std::is_trivially_copyable_v<T>);
        PeerHeader hdr{};
        hdr.type        = PeerMsgType::SEND;
        hdr.src_rank    = static_cast<uint32_t>(my_logical_rank_);
        hdr.dst_rank    = static_cast<uint32_t>(dst);
        hdr.tag         = tag;
        hdr.payload_len = static_cast<uint32_t>(data.size() * sizeof(T));
        hdr.comm_id     = comm_id_;

        // One-element buffer sequence wrapping the vector's storage. The
        // transport layer (zero-copy or pack/unpack) decides what to do with it.
        std::array<asio::const_buffer, 1> seq{{
            asio::buffer(data.data(), hdr.payload_len)
        }};
        co_await send_raw(dst, hdr, seq);
    }

    // Receive a vector of T from src_rank with tag.
    // Blocks until the message arrives; buffered in the mailbox if it came early.
    template<typename T>
    asio::awaitable<std::vector<T>> recv(int src, int tag = 0) {
        static_assert(std::is_trivially_copyable_v<T>);
        auto raw = co_await recv_raw(src, tag, PeerMsgType::SEND);
        std::vector<T> out(raw.size() / sizeof(T));
        std::memcpy(out.data(), raw.data(), raw.size());
        co_return out;
    }

    // ── Barrier ───────────────────────────────────────────────────────────────
    //
    // Centralized: all ranks send a BARRIER token to rank 0.
    // Rank 0 waits for N-1 tokens, then broadcasts release to all.
    // O(2N) messages total.
    //
    // STUB: dissemination_barrier() — O(2 log N) rounds, no central bottleneck.
    // Preferred for N > 16. Algorithm:
    //   for d in 0..ceil(log2(N)):
    //     send token to (my_rank + 2^d) % N
    //     recv token from (my_rank - 2^d + N) % N
    // Uncomment and replace centralized_barrier() call when needed.

    asio::awaitable<void> barrier(int tag = -1) {
        co_await centralized_barrier(tag);
    }

    // ── Broadcast ─────────────────────────────────────────────────────────────
    //
    // Root sends its data to all other ranks. O(N) messages.
    // data must be non-empty on root; on non-root, contents are replaced.

    template<typename T>
    asio::awaitable<void> bcast(std::vector<T>& data, int root = 0, int tag = -2) {
        static_assert(std::is_trivially_copyable_v<T>);
        int my = my_logical_rank_;
        int n  = my_size_;

        if (my == root) {
            for (int r = 0; r < n; r++) {
                if (r == root) continue;
                co_await send(data, r, tag);
            }
        } else {
            data = co_await recv<T>(root, tag);
        }
    }

    // ── Reduce ────────────────────────────────────────────────────────────────
    //
    // All ranks send their data to root. Root applies op element-wise and
    // returns the result. Non-root ranks receive an empty vector.
    // Requires equal-length data on all ranks.

    template<typename T>
    asio::awaitable<std::vector<T>> reduce(const std::vector<T>& data,
                                            ReduceOp op = ReduceOp::SUM,
                                            int root = 0, int tag = -3) {
        static_assert(std::is_trivially_copyable_v<T>);
        int my = my_logical_rank_;
        int n  = my_size_;

        if (my != root) {
            co_await send(data, root, tag);
            co_return std::vector<T>{};
        }

        // Root: collect from all non-root ranks and reduce
        std::vector<T> result = data;
        for (int r = 0; r < n; r++) {
            if (r == root) continue;
            auto incoming = co_await recv<T>(r, tag);
            apply_reduce(result, incoming, op);
        }
        co_return result;
    }

    // ── Scatter ───────────────────────────────────────────────────────────────
    //
    // Root splits `data` into `size` equal chunks and sends chunk[r] to rank r.
    // Root's own chunk is returned directly (no self-send).
    // data.size() must be divisible by size().

    template<typename T>
    asio::awaitable<std::vector<T>> scatter(const std::vector<T>& data,
                                             int root = 0, int tag = -4) {
        static_assert(std::is_trivially_copyable_v<T>);
        int my = my_logical_rank_;
        int n  = my_size_;

        if (my == root) {
            if (data.size() % n != 0)
                throw std::invalid_argument("clustr_mpi::scatter: data.size() not divisible by size()");
            size_t chunk = data.size() / n;
            for (int r = 0; r < n; r++) {
                if (r == root) continue;
                std::vector<T> slice(data.begin() + r * chunk,
                                     data.begin() + (r + 1) * chunk);
                co_await send(slice, r, tag);
            }
            co_return std::vector<T>(data.begin() + my * chunk,
                                     data.begin() + (my + 1) * chunk);
        } else {
            co_return co_await recv<T>(root, tag);
        }
    }

    // ── Gather ────────────────────────────────────────────────────────────────
    //
    // All ranks send their chunk to root. Root returns the concatenated result
    // in rank order. Non-root ranks receive an empty vector.

    template<typename T>
    asio::awaitable<std::vector<T>> gather(const std::vector<T>& chunk,
                                            int root = 0, int tag = -5) {
        static_assert(std::is_trivially_copyable_v<T>);
        int my = my_logical_rank_;
        int n  = my_size_;

        if (my != root) {
            co_await send(chunk, root, tag);
            co_return std::vector<T>{};
        }

        // Root: collect all chunks in rank order
        std::vector<std::vector<T>> parts(n);
        parts[root] = chunk;
        for (int r = 0; r < n; r++) {
            if (r == root) continue;
            parts[r] = co_await recv<T>(r, tag);
        }

        std::vector<T> result;
        for (auto& p : parts) result.insert(result.end(), p.begin(), p.end());
        co_return result;
    }

    // ── Allreduce (stub) ──────────────────────────────────────────────────────
    //
    // Ring-allreduce: O(2(N-1)) messages, bandwidth-optimal.
    // Used in distributed ML (same algorithm as NCCL/Horovod).
    // Implementation: N phases of (reduce-scatter → allgather) around the ring.
    //
    // TODO: implement when ML/large-tensor workloads are targeted.

    template<typename T>
    asio::awaitable<std::vector<T>> allreduce(const std::vector<T>& data,
                                               ReduceOp op = ReduceOp::SUM) {
        // Fallback: reduce to root 0 then broadcast. Correct, not bandwidth-optimal.
        auto result = co_await reduce(data, op, 0);
        co_await bcast(result, 0);
        co_return result;
    }

    // ── Cartesian topology ────────────────────────────────────────────────────
    //
    // In-place: marks this Comm as carrying a Cartesian topology with the
    // given dim vector. product(dims) MUST equal size(). Coordinates are
    // assigned in row-major order (last axis varies fastest), matching
    // MPI_Cart_create's default. Calling cart_create on a sub-comm is
    // disallowed — only the world Comm can be promoted.
    //
    // Idempotent for the same dim vector; throws if called twice with a
    // different one (catches accidental re-decomposition).
    void cart_create(const std::vector<int>& dims) {
        if (parent_) throw std::runtime_error(
            "clustr::Comm::cart_create: only the world Comm can be promoted to a Cartesian topology");

        long product = 1;
        for (int d : dims) {
            if (d <= 0) throw std::invalid_argument(
                "clustr::Comm::cart_create: every dim must be > 0");
            product *= d;
        }
        if (product != static_cast<long>(my_size_))
            throw std::invalid_argument(
                "clustr::Comm::cart_create: product(dims) (" + std::to_string(product)
                + ") must equal size() (" + std::to_string(my_size_) + ")");

        if (!dims_.empty()) {
            if (dims_ == dims) return;
            throw std::runtime_error(
                "clustr::Comm::cart_create: already promoted with a different dim vector");
        }

        dims_ = dims;
        coords_.assign(dims.size(), 0);
        long stride = 1;
        for (std::size_t a = dims.size(); a-- > 0; ) {
            coords_[a] = static_cast<int>((my_logical_rank_ / stride) % dims[a]);
            stride *= dims[a];
        }
    }

    // Read-only access to the Cartesian metadata. Empty until cart_create.
    const std::vector<int>& dims()   const { return dims_;   }
    const std::vector<int>& coords() const { return coords_; }

    // ── Sub-communicator ──────────────────────────────────────────────────────
    //
    // Project this Cartesian Comm onto a single axis: returns a sub-Comm whose
    // members are the ranks that share all of this Comm's coordinates EXCEPT
    // the one along `axis`. The sub-Comm's logical rank equals coords_[axis],
    // its size equals dims_[axis], and its comm_id is a deterministic FNV-1a
    // hash of (parent comm_id, axis, fixed coords) — every member of the
    // cohort hashes to the same value because only the projected axis varies.
    //
    // Sub-comms share the parent's TCP sockets, so the parent must outlive
    // every sub-comm it produces. Debug builds assert this in ~Comm via
    // child_count_.
    //
    // Requires CLUSTR_RECV=CLUSTR_RECV_CENTRAL: recv_raw demultiplexes by
    // comm_id, which only the central dispatch loop can do off the shared
    // socket. Calling this under CLUSTR_RECV_INLINE is a compile-time error.
#if CLUSTR_RECV == CLUSTR_RECV_CENTRAL
    Comm cart_sub(int axis);   // defined in include/clustr/cart.hpp
#else
    template <bool Always = false>
    Comm cart_sub(int /*axis*/) {
        static_assert(Always,
            "Comm::cart_sub requires -DCLUSTR_RECV=CLUSTR_RECV_CENTRAL "
            "(only the central dispatch loop can demultiplex sub-comm frames "
            "off the shared TCP socket).");
        std::abort();
    }
#endif

private:
#if CLUSTR_RECV == CLUSTR_RECV_CENTRAL
    // Sub-communicator constructor used only by Comm::cart_sub. Defined in
    // include/clustr/cart.hpp so the body can refer to fully-declared Comm.
    //
    //   parent_comm     — Comm we're projecting (must outlive *this)
    //   sub_comm_id     — FNV-1a hash already computed by cart_sub
    //   sub_logical_rank— this rank's position along the projected axis
    //   sub_size        — number of ranks along the projected axis
    //   sub_dims        — single-element { sub_size }
    //   sub_coords      — single-element { sub_logical_rank }
    //   world_ranks_in  — sub-comm-rank → world-rank table; must have
    //                     exactly sub_size entries and contain my_world_rank
    //                     at index sub_logical_rank.
    //
    // Bumps parent_comm.child_count_ at the end of the body so a partially-
    // constructed sub-comm (e.g., on a thrown range check) never bumps it.
    Comm(Comm& parent_comm, std::uint32_t sub_comm_id,
         int sub_logical_rank, int sub_size,
         std::vector<int> sub_dims, std::vector<int> sub_coords,
         std::vector<int> world_ranks_in);
#endif


    asio::io_context& io_;
    Roster            roster_;
    uint32_t          comm_id_;
    std::size_t       transport_iov_cap_;   // runtime sysconf(_SC_IOV_MAX)
    std::unique_ptr<asio::ip::tcp::acceptor>    acceptor_;
    std::vector<std::shared_ptr<asio::ip::tcp::socket>> sockets_;

    // Mailbox: messages that arrived before recv() was called.
    // Key: (src_rank, tag) — same as MPI_ANY_TAG not yet supported.
    // Used by recv_inline.hpp only; under CENTRAL each per-key channel
    // does its own buffering and this map is unused.
    struct Envelope { std::vector<uint8_t> data; };
    std::unordered_map<MailboxKey, std::deque<Envelope>, MailboxKeyHash> mailbox_;

#if CLUSTR_RECV == CLUSTR_RECV_CENTRAL
    // ── CentralState ─────────────────────────────────────────────────────────
    //
    // One unbounded channel per (comm_id, src_rank, tag, type) tuple. Lazily
    // created by either the dispatch loop (sender side) or recv_raw (receiver
    // side); whichever wins the race calls try_emplace and the other side
    // finds the existing entry.
    //
    // The channel value type is unique_ptr<channel> rather than channel by
    // value because basic_channel is non-movable once constructed.
    //
    // Single-threaded io_context: no mutex required around `channels`. If
    // CLUSTR ever moves to a multi-threaded io_context, wrap channel lookup
    // in a strand or add a mutex here.
    using CentralChannel = asio::experimental::channel<
        void(asio::error_code, std::vector<std::uint8_t>)>;

    struct CentralState {
        std::unordered_map<CentralKey,
                           std::unique_ptr<CentralChannel>,
                           CentralKeyHash> channels;
        bool dispatch_running = false;
        bool shutting_down    = false;
    };
    CentralState central_;
#endif

    // ── Comm identity (logical view) ─────────────────────────────────────────
    //
    // For the world communicator these mirror roster_.{my_rank,size}. For a
    // sub-communicator they hold the position-along-axis and length-along-axis
    // returned by cart_sub. rank()/size() always read these — never roster_.
    int my_logical_rank_ = -1;
    int my_size_         = -1;

    // ── Sub-comm linkage (Phase 5) ───────────────────────────────────────────
    //
    // parent_ is non-null exactly when this Comm was produced by cart_sub.
    // Sub-comms share the parent's TCP sockets (cloned into our sockets_ at
    // sub-comm rank indices) and route recv_raw through the parent's central
    // dispatch state — there is exactly one dispatch loop per socket and it
    // lives on the world Comm.
    //
    // child_count_ is bumped by sub-comm constructors and decremented by
    // their destructors. The Comm destructor asserts child_count_ == 0 to
    // catch parent-destroyed-while-sub-comm-alive bugs in debug builds.
    Comm* parent_      = nullptr;
    int   child_count_ = 0;

    // ── Cartesian topology (set by cart_create / cart_sub) ───────────────────
    //
    // dims_ / coords_ are populated on the world Comm by cart_create and
    // copied (with axis-position adjusted) into sub-comms by cart_sub.
    // world_ranks_ is sub-comm only: index is sub-comm logical rank, value
    // is the corresponding world rank — used at sub-comm construction time
    // to clone the parent's sockets at the right indices.
    std::vector<int> dims_;
    std::vector<int> coords_;
    std::vector<int> world_ranks_;

#ifndef NDEBUG
    // Debug-only set of comm_ids issued by cart_sub on this Comm. Catches
    // accidental hash collisions per-rank (cross-rank collisions are
    // statistically negligible at the grids we target). Defined in cart.hpp.
    std::unordered_set<uint32_t> issued_sub_ids_;
#endif

public:
    // ── Transport primitives (definitions in include/clustr/transport_*.hpp) ─
    //
    // These are the low-level entry points the typed send<T>/recv<T> wrappers
    // are built on. They are also the public surface that Phase 3+ code uses
    // directly when sending Subarray buffer sequences (which are not expressible
    // as a single contiguous vector<T>) and the future recv-into-Subarray API.
    //
    // Two implementations exist, selected at compile time via CLUSTR_TRANSPORT:
    //
    //   ZERO_COPY   - scatter-gather writev/readv, sysconf(_SC_IOV_MAX) chunking.
    //                 No intermediate packing; header + payload fragments go
    //                 directly to the kernel via iovec.
    //
    //   PACK_UNPACK - traditional path: allocate one contiguous scratch buffer,
    //                 memcpy header + all payload fragments into it, single
    //                 async_write. Reference implementation for benchmarking.
    //
    // The public send<T> / recv<T> / collectives API is identical on both.
    // See ROADMAP.md "Dual transport strategy" and docs/MPI.md.

    // Buffer-sequence form (the primary primitive). Each transport defines this.
    template <typename BufferSeq>
    asio::awaitable<void> send_raw(int dst, const PeerHeader& hdr,
                                    const BufferSeq& payload_bufs);

    // Convenience (ptr, len) overload. Defined in transport_common inline below
    // and delegates to the buffer-sequence form, so both transports get it free.
    asio::awaitable<void> send_raw(int dst, const PeerHeader& hdr,
                                    const void* payload, uint32_t len);

    // Receive a message matching (src, tag, type). Checks the mailbox first
    // and otherwise reads from the socket, buffering mismatched frames. Each
    // transport defines this in its split header.
    asio::awaitable<std::vector<uint8_t>> recv_raw(int src, int tag, PeerMsgType type);

    // ── Shutdown ────────────────────────────────────────────────────────────
    //
    // Stops any background dispatch coroutines (CENTRAL mode only) and closes
    // the acceptor + all peer sockets so io_context::run() can return. Safe to
    // call multiple times. Under INLINE mode this is a no-op aside from the
    // socket close.
    //
    // CLUSTR_MPI_MAIN invokes this automatically after the user coroutine
    // returns. Tests that drive Comm directly should call it before letting
    // io_context go out of scope.
    void shutdown() {
#if CLUSTR_RECV == CLUSTR_RECV_CENTRAL
        if (central_.shutting_down) return;
        central_.shutting_down = true;
        // Close every per-key channel so any pending recv_raw wakes with
        // operation_aborted instead of hanging forever.
        for (auto& [k, ch] : central_.channels) {
            if (ch) ch->close();
        }
#endif
        if (acceptor_) {
            asio::error_code ec;
            acceptor_->close(ec);
        }
        for (auto& sp : sockets_) {
            if (sp) {
                asio::error_code ec;
                sp->close(ec);
            }
        }
    }

private:
    // ── Internal: centralized barrier ────────────────────────────────────────

    asio::awaitable<void> centralized_barrier(int tag) {
        int my = my_logical_rank_;
        int n  = my_size_;
        PeerHeader hdr{};
        hdr.type     = PeerMsgType::BARRIER;
        hdr.src_rank = static_cast<uint32_t>(my);
        hdr.tag      = tag;

        hdr.comm_id = comm_id_;

        if (my != 0) {
            // Send arrival token to rank 0
            hdr.dst_rank    = 0;
            hdr.payload_len = 0;
            co_await send_raw(0, hdr, nullptr, 0);
            // Wait for release token from rank 0
            co_await recv_raw(0, tag, PeerMsgType::BARRIER);
        } else {
            // Collect arrival from all non-root ranks
            for (int r = 1; r < n; r++)
                co_await recv_raw(r, tag, PeerMsgType::BARRIER);
            // Release all
            for (int r = 1; r < n; r++) {
                hdr.dst_rank    = r;
                hdr.payload_len = 0;
                co_await send_raw(r, hdr, nullptr, 0);
            }
        }
    }

    // ── Internal: dissemination barrier (stub) ────────────────────────────────
    // Uncomment to replace centralized_barrier() for N > 16.
    //
    // asio::awaitable<void> dissemination_barrier(int tag) {
    //     int my = roster_.my_rank;
    //     int n  = roster_.size;
    //     for (int d = 1; d < n; d <<= 1) {
    //         int dst = (my + d) % n;
    //         int src = (my - d + n) % n;
    //         PeerHeader hdr{};
    //         hdr.type = PeerMsgType::BARRIER;
    //         hdr.src_rank = my; hdr.dst_rank = dst; hdr.tag = tag;
    //         co_await send_raw(dst, hdr, nullptr, 0);
    //         co_await recv_raw(src, tag, PeerMsgType::BARRIER);
    //     }
    // }

    // ── Internal: fault tolerance stub ───────────────────────────────────────
    //
    // #ifdef CLUSTR_FAULT_TOLERANT
    // Checkpoint/restart support. On rank failure, surviving ranks call
    // checkpoint() to write their state, then block until the scheduler
    // respawns the failed rank and sends a PEER_ROSTER update.
    //
    // asio::awaitable<void> checkpoint(const void* state, size_t len) {
    //     // Write to /var/tmp/clustr/checkpoint_rank<N>.bin
    //     // Notify scheduler via a RANK_CHECKPOINT message (not yet in protocol)
    // }
    //
    // asio::awaitable<void> restore(void* state, size_t len) {
    //     // Read from checkpoint file
    // }
    // #endif

    // ── Internal: element-wise reduction ─────────────────────────────────────

    template<typename T>
    static void apply_reduce(std::vector<T>& acc, const std::vector<T>& incoming,
                              ReduceOp op) {
        if (acc.size() != incoming.size())
            throw std::runtime_error("clustr_mpi::reduce: mismatched sizes");
        for (size_t i = 0; i < acc.size(); i++) {
            switch (op) {
            case ReduceOp::SUM:  acc[i] += incoming[i]; break;
            case ReduceOp::MIN:  acc[i]  = std::min(acc[i], incoming[i]); break;
            case ReduceOp::MAX:  acc[i]  = std::max(acc[i], incoming[i]); break;
            case ReduceOp::PROD: acc[i] *= incoming[i]; break;
            }
        }
    }

#if CLUSTR_RECV == CLUSTR_RECV_CENTRAL
public:
    // Lazily get-or-create the per-key channel. Called from both the dispatch
    // loop (sender side) and recv_raw (receiver side); whichever runs first
    // creates the channel and the other side picks it up. Single-threaded
    // io_context — no locking. Public so recv_central.hpp can call it from
    // its out-of-class Comm::recv_raw definition.
    CentralChannel& get_or_create_channel_(const CentralKey& key) {
        auto it = central_.channels.find(key);
        if (it == central_.channels.end()) {
            // Unbounded buffer: dispatch loop must drain socket reads as fast
            // as possible without blocking on receiver readiness.
            auto ch = std::make_unique<CentralChannel>(
                io_, std::numeric_limits<std::size_t>::max());
            it = central_.channels.emplace(key, std::move(ch)).first;
        }
        return *it->second;
    }

private:
    // ── Internal: central dispatch loop (one per remote socket) ──────────────
    //
    // Reads frames eagerly from sockets_[peer], demuxes by
    // (comm_id, src_rank, tag, type), and pushes the payload into the matching
    // per-key channel. recv_raw simply async_receives from its key's channel.
    //
    // The shared_ptr<socket> is captured by value so the coroutine keeps the
    // socket alive even if Comm is destroyed mid-flight (matters for shutdown
    // sequencing where ~Comm runs before io_context is fully drained).
    asio::awaitable<void>
    central_dispatch_loop_(std::shared_ptr<asio::ip::tcp::socket> sock, int peer) {
        try {
            for (;;) {
                PeerHeader hdr{};
                co_await asio::async_read(*sock,
                    asio::buffer(&hdr, sizeof(PeerHeader)), asio::use_awaitable);

                if (hdr.magic[0] != 0xC1 || hdr.magic[1] != 0x52)
                    throw std::runtime_error(
                        "clustr_mpi (recv_central): bad peer magic from rank "
                        + std::to_string(peer));

                std::vector<std::uint8_t> payload(hdr.payload_len);
                if (hdr.payload_len) {
                    co_await asio::async_read(*sock,
                        asio::buffer(payload), asio::use_awaitable);
                }

                CentralKey key{hdr.comm_id,
                               static_cast<int>(hdr.src_rank),
                               hdr.tag,
                               hdr.type};
                auto& ch = get_or_create_channel_(key);
                co_await ch.async_send(asio::error_code{},
                                        std::move(payload),
                                        asio::use_awaitable);
            }
        } catch (...) {
            // Socket closed, peer hung up, or shutdown was called. Wake any
            // pending receivers on this peer's channels so they don't hang.
            for (auto& [k, ch] : central_.channels) {
                if (k.src_rank == peer && ch) ch->close();
            }
        }
    }

    // Spawn one detached dispatch coroutine per remote peer. Called from
    // connect() once the full socket mesh is established.
    void start_central_dispatch_loops_() {
        if (central_.dispatch_running) return;
        int my = roster_.my_rank;
        int n  = roster_.size;
        for (int r = 0; r < n; ++r) {
            if (r == my) continue;
            if (!sockets_[r]) continue;
            asio::co_spawn(io_,
                central_dispatch_loop_(sockets_[r], r),
                asio::detached);
        }
        central_.dispatch_running = true;
    }
#endif // CLUSTR_RECV == CLUSTR_RECV_CENTRAL
};

// ============================================================================
// MPI — thin facade that owns the world Comm and forwards the classic API.
//
// Kept so existing user code (mpi.rank(), mpi.send(...), mpi.barrier()) keeps
// working unchanged after the Phase 1 Comm refactor.  New code that needs to
// reach into sub-communicators (Phase 5 onward) should grab `mpi.world()` and
// call `cart_sub()` / `cart_create()` directly on the returned Comm&.
// ============================================================================

class MPI {
public:
    explicit MPI(asio::io_context& io, const std::string& roster_path =
                 "/var/tmp/clustr/mpi_roster.conf")
        : world_(io, roster_path) {}

    // Access the world communicator. Use this when writing code that needs
    // sub-communicators (cart_sub, cart_create) or passes Comm& to helpers.
    Comm&       world()       { return world_; }
    const Comm& world() const { return world_; }

    // ── Forwarded read-only accessors ────────────────────────────────────────
    int rank() const { return world_.rank(); }
    int size() const { return world_.size(); }

    // ── Forwarded setup ──────────────────────────────────────────────────────
    asio::awaitable<void> connect()  { return world_.connect(); }
    void                  shutdown() { world_.shutdown();       }

    // ── Forwarded point-to-point ─────────────────────────────────────────────
    template<typename T>
    asio::awaitable<void> send(const std::vector<T>& data, int dst, int tag = 0) {
        return world_.send(data, dst, tag);
    }

    template<typename T>
    asio::awaitable<std::vector<T>> recv(int src, int tag = 0) {
        return world_.template recv<T>(src, tag);
    }

    // ── Forwarded collectives ────────────────────────────────────────────────
    asio::awaitable<void> barrier(int tag = -1) { return world_.barrier(tag); }

    template<typename T>
    asio::awaitable<void> bcast(std::vector<T>& data, int root = 0, int tag = -2) {
        return world_.bcast(data, root, tag);
    }

    template<typename T>
    asio::awaitable<std::vector<T>> reduce(const std::vector<T>& data,
                                            ReduceOp op = ReduceOp::SUM,
                                            int root = 0, int tag = -3) {
        return world_.reduce(data, op, root, tag);
    }

    template<typename T>
    asio::awaitable<std::vector<T>> scatter(const std::vector<T>& data,
                                             int root = 0, int tag = -4) {
        return world_.scatter(data, root, tag);
    }

    template<typename T>
    asio::awaitable<std::vector<T>> gather(const std::vector<T>& chunk,
                                            int root = 0, int tag = -5) {
        return world_.gather(chunk, root, tag);
    }

    template<typename T>
    asio::awaitable<std::vector<T>> allreduce(const std::vector<T>& data,
                                               ReduceOp op = ReduceOp::SUM) {
        return world_.allreduce(data, op);
    }

private:
    Comm world_;
};

} // namespace clustr

// ============================================================================
// Transport selector
//
// Both implementations must define the same Comm::send_raw<BufferSeq> and
// Comm::recv_raw out-of-class, plus the (ptr, len) convenience overload.
// The public Comm API above is unchanged by the choice — any API drift
// between the two files is a bug and should be caught by building the full
// test matrix against both transports.
// ============================================================================

#define CLUSTR_TRANSPORT_ZERO_COPY   1
#define CLUSTR_TRANSPORT_PACK_UNPACK 2

#ifndef CLUSTR_TRANSPORT
#  define CLUSTR_TRANSPORT CLUSTR_TRANSPORT_ZERO_COPY
#endif

#if CLUSTR_TRANSPORT == CLUSTR_TRANSPORT_ZERO_COPY
#  include "clustr/transport_zero_copy.hpp"
#elif CLUSTR_TRANSPORT == CLUSTR_TRANSPORT_PACK_UNPACK
#  include "clustr/transport_pack_unpack.hpp"
#else
#  error "CLUSTR_TRANSPORT must be CLUSTR_TRANSPORT_ZERO_COPY or CLUSTR_TRANSPORT_PACK_UNPACK"
#endif

// ============================================================================
// Receive-path selector — implementation include.
//
// The CLUSTR_RECV constants and default selection live near the top of this
// file because they guard state members on Comm itself. The actual include
// of recv_*.hpp happens here, after Comm is fully declared, so the headers
// can define Comm::recv_raw out-of-class:
//
//   INLINE  - per-call async_read on sockets_[src]. World-comm only.
//             Original Phase 0..4 path. Cannot demultiplex sub-comms.
//
//   CENTRAL - one dispatch coroutine per socket reads frames eagerly and
//             routes them by (comm_id, src, tag, type) into per-key channels.
//             Required by Phase 5 sub-communicators.
//
// Selecting INLINE and then calling Comm::cart_sub() is a hard compile-time
// error via static_assert in the cart_sub body itself.
// ============================================================================

#if CLUSTR_RECV == CLUSTR_RECV_INLINE
#  include "clustr/recv_inline.hpp"
#elif CLUSTR_RECV == CLUSTR_RECV_CENTRAL
#  include "clustr/recv_central.hpp"
#  include "clustr/cart.hpp"
#else
#  error "CLUSTR_RECV must be CLUSTR_RECV_INLINE or CLUSTR_RECV_CENTRAL"
#endif

// ============================================================================
// CLUSTR_MPI_MAIN — entry point macro for MPI jobs
//
// Replaces int main(). Sets up the ASIO io_context, loads the roster,
// establishes peer connections, then runs the user coroutine.
//
// Example:
//   CLUSTR_MPI_MAIN(mpi) {
//       co_await mpi.barrier();
//       co_return 0;
//   }
// ============================================================================

// CLUSTR_MPI_ROSTER env var, when set, overrides the default roster path.
// Production cluster jobs do not set it (the worker daemon writes the roster
// to the default path); local test harnesses use it to point each spawned
// process at its own per-rank roster file. Purely additive — leaving it
// unset preserves the original cluster behavior exactly.
#define CLUSTR_MPI_MAIN(mpi_var)                                                \
    asio::awaitable<int> _clustr_user_main(clustr::MPI& mpi_var);              \
    int main() {                                                                 \
        const char* _clustr_roster_env = std::getenv("CLUSTR_MPI_ROSTER");      \
        asio::io_context io;                                                     \
        clustr::MPI mpi(io,                                                      \
            _clustr_roster_env ? _clustr_roster_env                              \
                              : "/var/tmp/clustr/mpi_roster.conf");             \
        int ret = 0;                                                             \
        asio::co_spawn(io,                                                       \
            [&]() -> asio::awaitable<void> {                                    \
                co_await mpi.connect();                                          \
                ret = co_await _clustr_user_main(mpi);                          \
                mpi.shutdown();                                                  \
            },                                                                   \
            [](std::exception_ptr e) {                                           \
                if (e) std::rethrow_exception(e);                               \
            });                                                                  \
        io.run();                                                                \
        return ret;                                                              \
    }                                                                            \
    asio::awaitable<int> _clustr_user_main(clustr::MPI& mpi_var)

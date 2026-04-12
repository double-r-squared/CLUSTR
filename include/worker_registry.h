#pragma once

#include "protocol.h"
#include "tcp_server.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace clustr {

// ============================================================================
// WorkerEntry — one connected worker node
// ============================================================================

enum class WorkerState : uint8_t {
    CONNECTED,   // TCP open, handshake in progress
    FILE_READY,  // Source file received and checksum verified
    COMPILED,    // Binary compiled successfully on worker
    RUNNING,     // Process spawned, job executing
    IDLE,        // Job finished, ready for next task
    FAILED,      // Setup or execution error — do not dispatch
};

struct WorkerEntry {
    std::string  worker_id;       // Primary key — from HELLO handshake
    std::string  ip;              // Remote IP (informational / reconnect)
    uint16_t     port;            // Remote port
    float        capacity;        // Normalized 0.0–1.0 derived from hardware
    bool         available;       // Not currently executing a task
    WorkerState  state = WorkerState::CONNECTED;
    std::string  last_error;      // Set when state == FAILED
    HardwareCapability hardware;  // Raw capability data
    Connection::Ptr conn;         // Live socket — send messages through here
    uint16_t     peer_port = 0;  // OS-assigned MPI peer listener port (0 = not yet reported)
};

using WorkerPtr = std::shared_ptr<WorkerEntry>;

// Derive a normalized capacity score from hardware benchmarks.
// Weights: 70% compute (GFLOPS), 30% memory bandwidth (GB/s).
// Clamped to [0.0, 1.0] against reference maximums (80 GFLOPS, 50 GB/s).
inline float derive_capacity(const HardwareCapability& hw) {
    constexpr float MAX_GFLOPS  = 80.0f;
    constexpr float MAX_BW_GBPS = 50.0f;

    float compute = hw.compute_score        / MAX_GFLOPS;
    float memory  = hw.memory_bandwidth_gbps / MAX_BW_GBPS;

    float score = 0.7f * compute + 0.3f * memory;

    if (score < 0.0f) return 0.0f;
    if (score > 1.0f) return 1.0f;
    return score;
}

// ============================================================================
// WorkerRegistry — owns WorkerEntry objects, thread-safe
// ============================================================================
//
// All WorkerPtr values in the work queue point into this registry.
// Entries remain alive as long as either the registry or a queue holds them.

class WorkerRegistry {
public:
    // Register a new worker after HELLO + CAPABILITY_REPORT handshake.
    void add(WorkerPtr worker) {
        std::lock_guard<std::mutex> lock(mutex_);
        workers_[worker->worker_id] = std::move(worker);
    }

    // Remove worker on GOODBYE or heartbeat timeout.
    void remove(const std::string& worker_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        workers_.erase(worker_id);
    }

    // Lookup by worker_id.
    WorkerPtr get(const std::string& worker_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = workers_.find(worker_id);
        return (it != workers_.end()) ? it->second : nullptr;
    }

    // Return all workers sorted descending by capacity.
    // Used to build/rebuild the work queue.
    std::vector<WorkerPtr> sorted_by_capacity() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<WorkerPtr> result;
        result.reserve(workers_.size());
        for (const auto& [id, ptr] : workers_)
            result.push_back(ptr);

        std::sort(result.begin(), result.end(),
            [](const WorkerPtr& a, const WorkerPtr& b) {
                return a->capacity > b->capacity;
            });
        return result;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return workers_.size();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, WorkerPtr> workers_;
};

// ============================================================================
// WorkQueue — ordered list of WorkerPtr for task dispatch
//
// Holds shared_ptr so entries stay alive even if removed from the registry
// (e.g. worker disconnects mid-queue). Callers can check worker->available
// and worker->conn before dispatching.
// ============================================================================

class WorkQueue {
public:
    // Rebuild the queue from the registry, sorted by capacity descending.
    void rebuild(const WorkerRegistry& registry) {
        queue_ = registry.sorted_by_capacity();
    }

    // Return the next available worker, or nullptr if none ready.
    WorkerPtr next_available() {
        for (auto& worker : queue_) {
            if (worker && worker->available)
                return worker;
        }
        return nullptr;
    }

    // Return all workers weighted by capacity for proportional dispatch.
    // Caller picks based on share = worker->capacity / total_capacity.
    const std::vector<WorkerPtr>& all() const { return queue_; }

    bool empty() const { return queue_.empty(); }
    size_t size() const { return queue_.size(); }

private:
    std::vector<WorkerPtr> queue_;
};

}  // namespace clustr

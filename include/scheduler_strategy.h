#pragma once

#include "worker_registry.h"
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <memory>
#include <functional>
#include <chrono>
#include <atomic>

namespace clustr {

// ============================================================================
// Job — a unit of work waiting to be dispatched to a worker
// ============================================================================

struct Job {
    std::string job_id;
    std::string source_file;
    std::string compile_cmd;
    std::string run_cmd;

    // Extra files to transfer alongside source_file (e.g. Python scripts).
    // Each entry is a full local path; only the basename is sent to the worker.
    std::vector<std::string> companion_files;

    // If non-empty, only dispatch to this specific worker (manual pin).
    std::string pinned_worker_id;

    // MPI: number of worker ranks this job requires (1 = single-worker, legacy).
    uint32_t num_ranks = 1;

    // MPI: tracks which ranks have reported PEER_READY. Scheduler sends
    // EXEC_CMD to all ranks only once this reaches num_ranks.
    uint32_t ranks_ready = 0;

    // MPI: worker_ids of all allocated ranks, in rank order.
    std::vector<std::string> rank_workers;

    std::chrono::steady_clock::time_point submitted_at =
        std::chrono::steady_clock::now();
};

using JobPtr = std::shared_ptr<Job>;

// ============================================================================
// DispatchDecision — what the strategy returns
// ============================================================================

struct DispatchDecision {
    std::string worker_id;
    std::string job_id;
};

// ============================================================================
// SchedulingStrategy — pluggable interface
//
// Implement this to define your own dispatch logic.
// Called whenever a worker becomes idle or a new job is submitted.
// Returns a list of (worker_id, job_id) pairs to dispatch immediately.
// Workers and jobs NOT in the returned list remain pending.
// ============================================================================

class SchedulingStrategy {
public:
    virtual ~SchedulingStrategy() = default;

    virtual const char* name() const = 0;

    // idle_workers: workers currently in IDLE state with an open connection.
    // pending_jobs: jobs in the queue not yet dispatched.
    // Returns pairs to dispatch; may return fewer than available.
    virtual std::vector<DispatchDecision>
    schedule(const std::vector<WorkerPtr>& idle_workers,
             const std::vector<JobPtr>&    pending_jobs) = 0;
};

// ============================================================================
// Built-in strategies
// ============================================================================

// Dispatch each pending job to the first available idle worker (FIFO).
class FirstAvailableStrategy : public SchedulingStrategy {
public:
    const char* name() const override { return "FirstAvailable"; }

    std::vector<DispatchDecision>
    schedule(const std::vector<WorkerPtr>& idle_workers,
             const std::vector<JobPtr>&    pending_jobs) override
    {
        std::vector<DispatchDecision> decisions;
        std::vector<bool> worker_used(idle_workers.size(), false);

        for (const auto& job : pending_jobs) {
            for (size_t i = 0; i < idle_workers.size(); i++) {
                if (worker_used[i]) continue;
                const auto& w = idle_workers[i];
                // Respect manual pin if set
                if (!job->pinned_worker_id.empty() &&
                    job->pinned_worker_id != w->worker_id) continue;

                decisions.push_back({w->worker_id, job->job_id});
                worker_used[i] = true;
                break;
            }
        }
        return decisions;
    }
};

// Dispatch jobs to the highest-capacity idle worker first.
class CapacityWeightedStrategy : public SchedulingStrategy {
public:
    const char* name() const override { return "CapacityWeighted"; }

    std::vector<DispatchDecision>
    schedule(const std::vector<WorkerPtr>& idle_workers,
             const std::vector<JobPtr>&    pending_jobs) override
    {
        // Workers arrive pre-sorted by capacity descending (from sorted_by_capacity).
        // Just match FIFO jobs to capacity-ranked workers.
        std::vector<DispatchDecision> decisions;
        std::vector<bool> worker_used(idle_workers.size(), false);

        for (const auto& job : pending_jobs) {
            // Find highest-capacity worker that satisfies pin constraint
            for (size_t i = 0; i < idle_workers.size(); i++) {
                if (worker_used[i]) continue;
                const auto& w = idle_workers[i];
                if (!job->pinned_worker_id.empty() &&
                    job->pinned_worker_id != w->worker_id) continue;

                decisions.push_back({w->worker_id, job->job_id});
                worker_used[i] = true;
                break;
            }
        }
        return decisions;
    }
};

// Round-robin across idle workers.
class RoundRobinStrategy : public SchedulingStrategy {
public:
    const char* name() const override { return "RoundRobin"; }

    std::vector<DispatchDecision>
    schedule(const std::vector<WorkerPtr>& idle_workers,
             const std::vector<JobPtr>&    pending_jobs) override
    {
        if (idle_workers.empty() || pending_jobs.empty()) return {};

        std::vector<DispatchDecision> decisions;
        size_t n = idle_workers.size();

        for (size_t j = 0; j < pending_jobs.size(); j++) {
            const auto& job = pending_jobs[j];
            // Round-robin index advances per job
            size_t idx = (rr_counter_++) % n;

            // Respect pin
            if (!job->pinned_worker_id.empty()) {
                bool found = false;
                for (size_t i = 0; i < n; i++) {
                    if (idle_workers[i]->worker_id == job->pinned_worker_id) {
                        decisions.push_back({idle_workers[i]->worker_id, job->job_id});
                        found = true;
                        break;
                    }
                }
                if (!found) continue;  // pinned worker not idle yet
            } else {
                decisions.push_back({idle_workers[idx]->worker_id, job->job_id});
            }

            if (decisions.size() >= n) break;  // don't over-assign
        }
        return decisions;
    }

private:
    std::atomic<uint64_t> rr_counter_{0};
};

// Manual-only: never auto-dispatches. All assignments come from TUI.
class ManualStrategy : public SchedulingStrategy {
public:
    const char* name() const override { return "Manual"; }

    std::vector<DispatchDecision>
    schedule(const std::vector<WorkerPtr>&,
             const std::vector<JobPtr>&) override
    {
        return {};  // always empty — user drives dispatch explicitly
    }
};

// ============================================================================
// JobQueue — thread-safe pending job store
// ============================================================================

class JobQueue {
public:
    void push(JobPtr job) {
        std::lock_guard<std::mutex> lk(mutex_);
        jobs_.push_back(std::move(job));
    }

    // Remove and return a job by id.  Returns nullptr if not found.
    JobPtr take(const std::string& job_id) {
        std::lock_guard<std::mutex> lk(mutex_);
        for (auto it = jobs_.begin(); it != jobs_.end(); ++it) {
            if ((*it)->job_id == job_id) {
                auto j = *it;
                jobs_.erase(it);
                return j;
            }
        }
        return nullptr;
    }

    void remove(const std::string& job_id) { take(job_id); }

    // Find without removing — used by on_peer_ready to accumulate PEER_READY counts.
    JobPtr peek(const std::string& job_id) const {
        std::lock_guard<std::mutex> lk(mutex_);
        for (const auto& j : jobs_)
            if (j->job_id == job_id) return j;
        return nullptr;
    }

    // Snapshot for scheduling / display
    std::vector<JobPtr> snapshot() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return std::vector<JobPtr>(jobs_.begin(), jobs_.end());
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return jobs_.size();
    }

    bool empty() const { return size() == 0; }

    static std::string make_id() {
        static std::atomic<uint32_t> ctr{1};
        return "job_" + std::to_string(ctr.fetch_add(1));
    }

private:
    mutable std::mutex     mutex_;
    std::deque<JobPtr>     jobs_;
};

}  // namespace clustr

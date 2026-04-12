#pragma once

#include "tcp_server.h"
#include "worker_registry.h"
#include "file_transfer.h"
#include "scheduler_strategy.h"
#include "system_conf.h"
#include "tui.h"
#include <string>
#include <memory>
#include <unordered_map>
#include <atomic>

namespace clustr {

// ============================================================================
// Scheduler
//
// Owns the TCP server, worker registry, job queue, and scheduling strategy.
// All methods (except the constructor and deploy_worker) run on the io_context
// thread.  The TUI posts commands via asio::post() for thread safety.
// deploy_worker() spawns its own std::thread internally.
// ============================================================================

class Scheduler {
public:
    Scheduler(asio::io_context& io, uint16_t port, TuiState& ui,
              SystemConf system_conf);

    void start();

    // ── Commands posted from the TUI thread ───────────────────────────────────

    void submit_job(const std::string& pinned_worker_id);
    void force_dispatch(const std::string& job_id, const std::string& worker_id);
    void kill_job(const std::string& worker_id);
    void request_status(const std::string& worker_id);
    void cancel_job(const std::string& job_id);
    void set_strategy(const std::string& name);

    // Spawns deploy.sh in a background thread; streams output to the TUI log.
    // Safe to call from any thread.  Multiple concurrent deploys are supported.
    void deploy_worker(const std::string& name);

    // Re-parses system.conf and updates known_workers in the TUI state.
    // Must be called on the io_context thread (via asio::post).
    void reload_config(const std::string& conf_path);

private:
    // ── Core subsystems ───────────────────────────────────────────────────────
    TcpServer      server_;
    WorkerRegistry registry_;
    WorkQueue      work_queue_;
    TuiState&      ui_;
    JobQueue       job_queue_;
    SystemConf     system_conf_;

    std::unique_ptr<SchedulingStrategy>          strategy_;
    std::unordered_map<Connection*, std::string> conn_to_id_;
    std::unordered_map<std::string, uint32_t>    running_pids_;
    std::unordered_map<std::string, std::string> pending_compile_;
    std::unordered_map<std::string, std::string> pending_run_;
    std::unordered_map<std::string, std::string> pending_job_id_;

    std::atomic<uint32_t> next_msg_id_{100};

    // ── Internal helpers ──────────────────────────────────────────────────────
    uint32_t    new_msg_id() { return next_msg_id_.fetch_add(1); }
    void        log(const std::string& msg,
                    LogCategory        cat         = LogCategory::Worker,
                    const std::string& worker_id   = {},
                    const std::string& job_id      = {},
                    bool               is_summary  = false,
                    bool               collapsible = false);
    void        try_schedule();
    void        dispatch_to(WorkerPtr worker, JobPtr job);
    // MPI: allocate num_ranks idle workers and send PEER_ROSTER to each.
    // Returns true if enough workers were available.
    bool        dispatch_group(JobPtr job);
    void        on_peer_ready(Connection::Ptr conn, const Message& msg);
    void        mark_failed(WorkerPtr worker, const std::string& reason);
    void        update_ui_worker(const WorkerPtr& w,
                                 const std::string& display_name = "");
    void        sync_jobs_to_ui();
    std::string worker_id_for(const Connection::Ptr& conn) const;
    uint32_t    worker_pid(const std::string& worker_id) const;
    static std::string state_str(WorkerState s);

    // ── Connection lifecycle ──────────────────────────────────────────────────
    void handle_new_connection(Connection::Ptr conn);
    void handle_disconnect(Connection::Ptr conn);
    void handle_message(Connection::Ptr conn, const Message& msg);

    // ── Message handlers ──────────────────────────────────────────────────────
    void on_hello(Connection::Ptr conn, const Message& msg);
    void on_capability(Connection::Ptr conn, const Message& msg);
    void on_file_ack(Connection::Ptr conn, const Message& msg);
    void on_exec_result(Connection::Ptr conn, const Message& msg);
    void on_process_spawned(Connection::Ptr conn, const Message& msg);
    void on_task_result(Connection::Ptr conn, const Message& msg);
    void on_status_resp(Connection::Ptr conn, const Message& msg);

    // ── Senders ───────────────────────────────────────────────────────────────
    void send_compile(Connection::Ptr conn, const std::string& worker_id,
                      const std::string& compile_cmd);
    void send_run(Connection::Ptr conn, const std::string& worker_id,
                  const std::string& run_cmd);
};

}  // namespace clustr

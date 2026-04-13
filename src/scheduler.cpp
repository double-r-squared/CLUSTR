#include "scheduler.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace clustr {

// ============================================================================
// Construction
// ============================================================================

Scheduler::Scheduler(asio::io_context& io, uint16_t port, TuiState& ui,
                     SystemConf system_conf)
    : server_(io, port), ui_(ui),
      system_conf_(std::move(system_conf))
{
    strategy_ = std::make_unique<FirstAvailableStrategy>();
}

void Scheduler::start() {
    server_.on_connection = [this](Connection::Ptr conn) {
        handle_new_connection(conn);
    };
    server_.start();
}

// ============================================================================
// Public commands (called via asio::post from the TUI thread)
// ============================================================================

void Scheduler::submit_job(const std::string& pinned_worker_id) {
    std::string src, compile, run;
    std::vector<std::string> companions;
    uint32_t num_ranks = 1;
    {
        std::lock_guard<std::mutex> lk(ui_.mutex);
        src        = ui_.source_file;
        compile    = ui_.compile_cmd;
        run        = ui_.run_cmd;
        companions = ui_.companion_files;
        num_ranks  = ui_.num_ranks;
    }
    if (src.empty() || compile.empty() || run.empty()) {
        log("[ERROR] Job template incomplete.", LogCategory::Job);
        return;
    }

    auto job              = std::make_shared<Job>();
    job->job_id           = JobQueue::make_id();
    job->source_file      = src;
    job->compile_cmd      = compile;
    job->run_cmd          = run;
    job->companion_files  = std::move(companions);
    job->pinned_worker_id = pinned_worker_id;
    job->num_ranks        = num_ranks;

    job_queue_.push(job);

    // Record which job_id is the most recent for this source file
    {
        std::lock_guard<std::mutex> lk(ui_.mutex);
        ui_.recent_job_for_file[src] = job->job_id;
    }

    log("[QUEUED] " + job->job_id + " src=" + src +
        (pinned_worker_id.empty() ? "" : " pinned=" + pinned_worker_id),
        LogCategory::Job, "", job->job_id, /*is_summary=*/true);

    sync_jobs_to_ui();
    try_schedule();
}

void Scheduler::force_dispatch(const std::string& job_id,
                                const std::string& worker_id) {
    auto job = job_queue_.take(job_id);
    if (!job) {
        log("[ERROR] force_dispatch: job " + job_id + " not in queue",
            LogCategory::Job, worker_id, job_id);
        return;
    }

    // Multi-rank jobs: the forced worker_id cannot be honored because the
    // scheduler allocates the full rank set itself (see dispatch_group).
    // Pinning a single worker as rank 0 would also break the roster math,
    // which assumes workers are picked in capacity order. Ignore the forced
    // worker, route through dispatch_group, and log why.
    if (job->num_ranks > 1) {
        log("[FORCE] " + job_id + " (num_ranks=" + std::to_string(job->num_ranks)
            + ") ignoring worker pin " + worker_id
            + " - MPI jobs auto-allocate ranks",
            LogCategory::Job, worker_id, job_id, /*is_summary=*/true);
        // dispatch_group re-queues on its own if allocation fails.
        dispatch_group(job);
        sync_jobs_to_ui();
        return;
    }

    auto worker = registry_.get(worker_id);
    if (!worker || !worker->conn) {
        log("[ERROR] force_dispatch: worker " + worker_id + " not found",
            LogCategory::Worker, worker_id, job_id);
        job_queue_.push(job);
        return;
    }
    log("[FORCE] " + job_id + " -> " + worker_id,
        LogCategory::Job, worker_id, job_id, /*is_summary=*/true);
    dispatch_to(worker, job);
    sync_jobs_to_ui();
}

void Scheduler::kill_job(const std::string& worker_id) {
    auto worker = registry_.get(worker_id);
    if (!worker || !worker->conn) return;

    uint32_t pid = worker_pid(worker_id);
    if (!pid) {
        log("[WARN] kill: no PID for " + worker_id, LogCategory::Worker, worker_id);
        return;
    }

    ProcessKillPayload kp{};
    kp.pid    = pid;
    kp.signal = 15;

    Message msg;
    msg.type             = MessageType::PROCESS_KILL;
    msg.message_id       = new_msg_id();
    msg.protocol_version = PROTOCOL_VERSION;
    msg.payload.resize(sizeof(ProcessKillPayload));
    std::memcpy(msg.payload.data(), &kp, sizeof(ProcessKillPayload));
    worker->conn->send_message(msg);
    log("[KILL] SIGTERM pid=" + std::to_string(pid) + " on " + worker_id,
        LogCategory::Worker, worker_id);
}

void Scheduler::request_status(const std::string& worker_id) {
    auto worker = registry_.get(worker_id);
    if (!worker || !worker->conn) return;
    uint32_t pid = worker_pid(worker_id);
    if (!pid) return;

    ProcessStatusReqPayload req{};
    req.pid = pid;
    Message msg;
    msg.type             = MessageType::PROCESS_STATUS_REQ;
    msg.message_id       = new_msg_id();
    msg.protocol_version = PROTOCOL_VERSION;
    msg.payload.resize(sizeof(ProcessStatusReqPayload));
    std::memcpy(msg.payload.data(), &req, sizeof(ProcessStatusReqPayload));
    worker->conn->send_message(msg);
}

void Scheduler::cancel_job(const std::string& job_id) {
    job_queue_.remove(job_id);
    log("[CANCEL] " + job_id, LogCategory::Job, "", job_id);
    sync_jobs_to_ui();
}

void Scheduler::set_strategy(const std::string& name) {
    if      (name == "FirstAvailable")   strategy_ = std::make_unique<FirstAvailableStrategy>();
    else if (name == "CapacityWeighted") strategy_ = std::make_unique<CapacityWeightedStrategy>();
    else if (name == "RoundRobin")       strategy_ = std::make_unique<RoundRobinStrategy>();
    else if (name == "Manual")           strategy_ = std::make_unique<ManualStrategy>();
    else { log("[ERROR] Unknown strategy: " + name, LogCategory::Worker); return; }

    log("[STRATEGY] Switched to " + name, LogCategory::Worker);
    std::lock_guard<std::mutex> lk(ui_.mutex);
    ui_.strategy_name = name;
}

void Scheduler::reload_config(const std::string& conf_path) {
    SystemConf new_conf = SystemConf::load(conf_path);
    system_conf_ = new_conf;
    {
        std::lock_guard<std::mutex> lk(ui_.mutex);
        ui_.known_workers = new_conf.workers;
        ui_.add_log("Config reloaded - " + std::to_string(new_conf.workers.size()) +
                    " worker(s) known", LogCategory::Worker);
    }
}

void Scheduler::deploy_worker(const std::string& name) {
    const KnownWorker* kw = system_conf_.find_by_name(name);
    if (!kw) {
        log("[DEPLOY] Unknown worker: " + name, LogCategory::Deploy, name);
        return;
    }
    if (kw->deploy_host.empty() || kw->deploy_user.empty()) {
        log("[DEPLOY:" + name + "] Missing deploy_host or deploy_user",
            LogCategory::Deploy, name);
        return;
    }
    if (kw->scheduler_ip.empty()) {
        log("[DEPLOY:" + name + "] Missing scheduler_ip",
            LogCategory::Deploy, name);
        return;
    }

    {
        std::lock_guard<std::mutex> lk(ui_.mutex);
        if (ui_.deploying_workers.count(name)) {
            ui_.add_log("[DEPLOY:" + name + "] Already deploying - wait for it to finish",
                        LogCategory::Deploy, name);
            return;
        }
        ui_.deploying_workers.insert(name);
    }

    std::string tmp_conf = "/tmp/clustr_deploy_" + name + ".conf";
    {
        std::ofstream out(tmp_conf);
        out << "scheduler_ip="        << kw->scheduler_ip        << "\n";
        out << "scheduler_port="      << kw->scheduler_port      << "\n";
        out << "deploy_user="         << kw->deploy_user         << "\n";
        out << "deploy_host="         << kw->deploy_host         << "\n";
        out << "ssh_key_path="        << kw->ssh_key_path        << "\n";
        out << "remote_install_path=" << kw->remote_install_path << "\n";
        out << "work_dir="            << kw->work_dir            << "\n";
    }

    log("[DEPLOY:" + name + "] Starting -> " +
        kw->deploy_user + "@" + kw->deploy_host,
        LogCategory::Deploy, name);

    std::string wname  = name;
    std::string tmp    = tmp_conf;
    TuiState&   ui_ref = ui_;

    std::thread([wname, tmp, &ui_ref]() {
        auto ui_log = [&](const std::string& msg, bool is_summary = false) {
            std::lock_guard<std::mutex> lk(ui_ref.mutex);
            ui_ref.add_log(msg, LogCategory::Deploy, wname, {}, is_summary);
        };

        std::string cmd = "./scripts/deploy.sh " + tmp + " 2>&1";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            ui_log("[DEPLOY:" + wname + "] ERROR: could not start deploy.sh");
            std::lock_guard<std::mutex> lk(ui_ref.mutex);
            ui_ref.deploying_workers.erase(wname);
            std::remove(tmp.c_str());
            return;
        }

        char buf[512];
        while (fgets(buf, sizeof(buf), pipe)) {
            std::string line = buf;
            while (!line.empty() &&
                   (line.back() == '\n' || line.back() == '\r'))
                line.pop_back();
            if (!line.empty())
                ui_log("[DEPLOY:" + wname + "] " + line);
        }

        int rc = pclose(pipe);
        std::remove(tmp.c_str());

        std::lock_guard<std::mutex> lk(ui_ref.mutex);
        ui_ref.deploying_workers.erase(wname);
        if (rc == 0)
            ui_ref.add_log("[DEPLOY:" + wname + "] Complete!",
                           LogCategory::Deploy, wname, {}, /*is_summary=*/true);
        else
            ui_ref.add_log("[DEPLOY:" + wname + "] FAILED (exit=" +
                           std::to_string(rc) + ")",
                           LogCategory::Deploy, wname, {}, /*is_summary=*/true);
    }).detach();
}

// ============================================================================
// Internal helpers
// ============================================================================

void Scheduler::log(const std::string& msg,
                    LogCategory        cat,
                    const std::string& worker_id,
                    const std::string& job_id,
                    bool               is_summary,
                    bool               collapsible) {
    std::lock_guard<std::mutex> lk(ui_.mutex);
    ui_.add_log(msg, cat, worker_id, job_id, is_summary, collapsible);
}

void Scheduler::try_schedule() {
    auto idle_workers = registry_.sorted_by_capacity();
    idle_workers.erase(
        std::remove_if(idle_workers.begin(), idle_workers.end(),
            [](const WorkerPtr& w) {
                return !w->available || w->state != WorkerState::IDLE;
            }),
        idle_workers.end());

    auto pending = job_queue_.snapshot();
    if (idle_workers.empty() || pending.empty()) return;

    for (const auto& job : pending) {
        if (job->num_ranks > 1) {
            // MPI job: need num_ranks idle workers simultaneously
            if ((int)idle_workers.size() >= (int)job->num_ranks) {
                job_queue_.take(job->job_id);
                dispatch_group(job);
                // Rebuild idle list after allocation
                idle_workers.erase(idle_workers.begin(),
                                   idle_workers.begin() + job->num_ranks);
            }
            // else: not enough workers yet — leave in queue
        }
    }

    // Single-rank jobs: use the strategy as before
    auto decisions = strategy_->schedule(idle_workers, pending);
    for (const auto& d : decisions) {
        auto job = job_queue_.take(d.job_id);
        if (!job) continue;
        if (job->num_ranks > 1) { job_queue_.push(job); continue; } // already handled
        auto worker = registry_.get(d.worker_id);
        if (!worker || !worker->conn) { job_queue_.push(job); continue; }
        dispatch_to(worker, job);
    }
    sync_jobs_to_ui();
}

bool Scheduler::dispatch_group(JobPtr job) {
    // Collect num_ranks idle workers (best capacity first — registry already sorted)
    auto idle = registry_.sorted_by_capacity();
    idle.erase(std::remove_if(idle.begin(), idle.end(),
        [](const WorkerPtr& w) { return !w->available || w->state != WorkerState::IDLE; }),
        idle.end());

    if ((int)idle.size() < (int)job->num_ranks) {
        job_queue_.push(job);
        return false;
    }

    // Assign ranks
    job->rank_workers.clear();
    for (uint32_t r = 0; r < job->num_ranks; r++) {
        job->rank_workers.push_back(idle[r]->worker_id);
        idle[r]->available = false;
        idle[r]->state     = WorkerState::CONNECTED;
        pending_job_id_[idle[r]->worker_id] = job->job_id;
        update_ui_worker(idle[r]);
    }

    log("[MPI] Allocated " + std::to_string(job->num_ranks) + " ranks for " + job->job_id,
        LogCategory::Job, {}, job->job_id, /*is_summary=*/true);

    // Send PEER_ROSTER to each rank.
    // Each worker will report back with PEER_READY (carrying its OS peer port).
    // We send a partial roster now — peer_port fields will be 0 until PEER_READY
    // arrives. The scheduler updates the roster and re-sends a complete one once
    // all ranks have reported their ports (handled in on_peer_ready).
    for (uint32_t my_rank = 0; my_rank < job->num_ranks; my_rank++) {
        auto worker = registry_.get(job->rank_workers[my_rank]);
        if (!worker) continue;

        PeerRosterPayload roster{};
        std::strncpy(roster.job_id, job->job_id.c_str(), sizeof(roster.job_id) - 1);
        roster.my_rank   = my_rank;
        roster.num_ranks = job->num_ranks;

        for (uint32_t r = 0; r < job->num_ranks; r++) {
            roster.peers[r].rank = r;
            auto peer = registry_.get(job->rank_workers[r]);
            if (peer) {
                std::strncpy(roster.peers[r].ip, peer->ip.c_str(),
                             sizeof(roster.peers[r].ip) - 1);
                roster.peers[r].peer_port = peer->peer_port; // 0 until PEER_READY
            }
        }

        Message msg;
        msg.protocol_version = PROTOCOL_VERSION;
        msg.type             = MessageType::PEER_ROSTER;
        msg.message_id       = new_msg_id();
        msg.payload.resize(sizeof(PeerRosterPayload));
        std::memcpy(msg.payload.data(), &roster, sizeof(PeerRosterPayload));

        try { worker->conn->send_message(msg); }
        catch (const std::exception& e) { mark_failed(worker, e.what()); }
    }

    // Store job so on_peer_ready can find it
    job_queue_.push(job); // temporarily re-queue; taken again in on_peer_ready
    return true;
}

void Scheduler::on_peer_ready(Connection::Ptr conn, const Message& msg) {
    if (msg.payload.size() < sizeof(PeerReadyPayload)) return;
    PeerReadyPayload pr{};
    std::memcpy(&pr, msg.payload.data(), sizeof(PeerReadyPayload));

    std::string worker_id = worker_id_for(conn);
    std::string job_id(pr.job_id);

    // Store the OS-assigned peer port on the worker entry
    auto worker = registry_.get(worker_id);
    if (worker) worker->peer_port = pr.peer_port;

    log("[PEER_READY] rank=" + std::to_string(pr.my_rank) +
        " port=" + std::to_string(pr.peer_port) + " job=" + job_id,
        LogCategory::Network, worker_id, job_id, false, /*collapsible=*/true);

    // Find the job (it was temporarily re-queued in dispatch_group)
    auto job = job_queue_.peek(job_id);
    if (!job) return;

    job->ranks_ready++;
    if (job->ranks_ready < job->num_ranks) return; // still waiting on others

    // All ranks ready — send complete rosters then EXEC_CMD to everyone
    job_queue_.take(job_id);

    log("[MPI] All " + std::to_string(job->num_ranks) + " ranks ready for " + job_id +
        " - sending EXEC_CMD", LogCategory::Job, {}, job_id,
        /*is_summary=*/true, /*collapsible=*/true);

    // Build and re-send complete PEER_ROSTER with all peer ports now known
    for (uint32_t my_rank = 0; my_rank < job->num_ranks; my_rank++) {
        auto w = registry_.get(job->rank_workers[my_rank]);
        if (!w) continue;

        PeerRosterPayload roster{};
        std::strncpy(roster.job_id, job_id.c_str(), sizeof(roster.job_id) - 1);
        roster.my_rank   = my_rank;
        roster.num_ranks = job->num_ranks;

        for (uint32_t r = 0; r < job->num_ranks; r++) {
            roster.peers[r].rank = r;
            auto peer = registry_.get(job->rank_workers[r]);
            if (peer) {
                std::strncpy(roster.peers[r].ip, peer->ip.c_str(),
                             sizeof(roster.peers[r].ip) - 1);
                roster.peers[r].peer_port = peer->peer_port;
            }
        }

        Message rmsg;
        rmsg.protocol_version = PROTOCOL_VERSION;
        rmsg.type             = MessageType::PEER_ROSTER;
        rmsg.message_id       = new_msg_id();
        rmsg.payload.resize(sizeof(PeerRosterPayload));
        std::memcpy(rmsg.payload.data(), &roster, sizeof(PeerRosterPayload));
        try { w->conn->send_message(rmsg); } catch (...) {}

        // Now send compile + run commands (same as single-rank dispatch_to)
        pending_compile_[w->worker_id] = job->compile_cmd;
        pending_run_[w->worker_id]     = job->run_cmd;

        try {
            Message fmsg = job->companion_files.empty()
                ? make_file_data_msg(job->source_file, new_msg_id())
                : make_bundle_msg(job->source_file, job->companion_files, new_msg_id());
            w->conn->send_message(fmsg);
        } catch (const std::exception& e) { mark_failed(w, e.what()); }
    }
}

void Scheduler::dispatch_to(WorkerPtr worker, JobPtr job) {
    worker->state     = WorkerState::CONNECTED;
    worker->available = false;
    work_queue_.rebuild(registry_);
    update_ui_worker(worker);

    pending_compile_[worker->worker_id] = job->compile_cmd;
    pending_run_[worker->worker_id]     = job->run_cmd;
    pending_job_id_[worker->worker_id]  = job->job_id;

    log("[DISPATCH] " + job->job_id + " -> " + worker->worker_id,
        LogCategory::Job, worker->worker_id, job->job_id, /*is_summary=*/true);

    try {
        Message fmsg = job->companion_files.empty()
            ? make_file_data_msg(job->source_file, new_msg_id())
            : make_bundle_msg(job->source_file, job->companion_files, new_msg_id());
        worker->conn->send_message(fmsg);
        log("[FILE_DATA] " + job->source_file + " -> " + worker->worker_id,
            LogCategory::Job, worker->worker_id, job->job_id);
    } catch (const std::exception& e) {
        mark_failed(worker, std::string("send_file: ") + e.what());
    }
}

void Scheduler::mark_failed(WorkerPtr worker, const std::string& reason) {
    std::string job_id = pending_job_id_.count(worker->worker_id)
                         ? pending_job_id_[worker->worker_id] : "";
    worker->state      = WorkerState::FAILED;
    worker->available  = false;
    worker->last_error = reason;
    work_queue_.rebuild(registry_);
    update_ui_worker(worker);
    log("[FAILED] " + worker->worker_id + ": " + reason,
        LogCategory::Worker, worker->worker_id, job_id);
}

void Scheduler::update_ui_worker(const WorkerPtr& w,
                                  const std::string& display_name) {
    WorkerDisplay wd;
    wd.worker_id      = w->worker_id;
    wd.display_name   = display_name;
    wd.ip             = w->ip;
    wd.capacity       = w->capacity;
    wd.state_str      = state_str(w->state);
    wd.pid            = worker_pid(w->worker_id);
    wd.hw_capability  = w->hardware;
    wd.has_capability = true;

    std::lock_guard<std::mutex> lk(ui_.mutex);
    for (const auto& existing : ui_.workers) {
        if (existing.worker_id == w->worker_id) {
            wd.run_start    = existing.run_start;
            wd.last_output  = existing.last_output;
            wd.mem_bytes    = existing.mem_bytes;
            wd.cpu_percent  = existing.cpu_percent;
            if (wd.display_name.empty())
                wd.display_name = existing.display_name;
            break;
        }
    }
    ui_.upsert_worker(wd);
}

void Scheduler::sync_jobs_to_ui() {
    std::lock_guard<std::mutex> lk(ui_.mutex);
    ui_.pending_jobs = job_queue_.snapshot();
}

std::string Scheduler::worker_id_for(const Connection::Ptr& conn) const {
    auto it = conn_to_id_.find(conn.get());
    return (it != conn_to_id_.end()) ? it->second : "";
}

uint32_t Scheduler::worker_pid(const std::string& worker_id) const {
    auto it = running_pids_.find(worker_id);
    return (it != running_pids_.end()) ? it->second : 0;
}

std::string Scheduler::state_str(WorkerState s) {
    switch (s) {
    case WorkerState::CONNECTED:  return "CONNECTED";
    case WorkerState::FILE_READY: return "FILE_READY";
    case WorkerState::COMPILED:   return "COMPILED";
    case WorkerState::RUNNING:    return "RUNNING";
    case WorkerState::IDLE:       return "IDLE";
    case WorkerState::FAILED:     return "FAILED";
    }
    return "UNKNOWN";
}

// ============================================================================
// Connection lifecycle
// ============================================================================

void Scheduler::handle_new_connection(Connection::Ptr conn) {
    std::string ip = conn->socket().remote_endpoint().address().to_string();
    log("[CONNECT] " + ip, LogCategory::Network);
    conn->on_message = [this, conn](Connection::Ptr, const Message& msg) {
        handle_message(conn, msg);
    };
    conn->on_disconnect = [this](Connection::Ptr c) {
        handle_disconnect(c);
    };
}

void Scheduler::handle_disconnect(Connection::Ptr conn) {
    auto it = conn_to_id_.find(conn.get());
    if (it != conn_to_id_.end()) {
        std::string wid = it->second;
        log("[DISCONNECT] " + wid, LogCategory::Network, wid);
        registry_.remove(wid);
        conn_to_id_.erase(it);
        running_pids_.erase(wid);
        pending_compile_.erase(wid);
        pending_run_.erase(wid);
        pending_job_id_.erase(wid);
        work_queue_.rebuild(registry_);
        std::lock_guard<std::mutex> lk(ui_.mutex);
        ui_.remove_worker(wid);
    } else {
        log("[DISCONNECT] unknown worker", LogCategory::Network);
    }
}

// ============================================================================
// Message dispatch
// ============================================================================

void Scheduler::handle_message(Connection::Ptr conn, const Message& msg) {
    switch (msg.type) {
    case MessageType::HELLO:               on_hello(conn, msg);           break;
    case MessageType::CAPABILITY_REPORT:   on_capability(conn, msg);      break;
    case MessageType::FILE_ACK:            on_file_ack(conn, msg);        break;
    case MessageType::EXEC_RESULT:         on_exec_result(conn, msg);     break;
    case MessageType::PROCESS_SPAWNED:     on_process_spawned(conn, msg); break;
    case MessageType::TASK_RESULT:         on_task_result(conn, msg);     break;
    case MessageType::PROCESS_STATUS_RESP: on_status_resp(conn, msg);     break;
    case MessageType::PEER_READY:          on_peer_ready(conn, msg);      break;
    case MessageType::RANK_DONE: {
        // Non-root rank finished - log it, surface its output, mark worker idle
        if (msg.payload.size() >= sizeof(RankDonePayload)) {
            RankDonePayload rd{};
            std::memcpy(&rd, msg.payload.data(), sizeof(RankDonePayload));
            std::string wid    = worker_id_for(conn);
            std::string job_id = std::string(rd.job_id);

            log("[RANK_DONE] rank=" + std::to_string(rd.rank) +
                " exit=" + std::to_string(rd.exit_code) + " job=" + job_id,
                LogCategory::Job, wid, job_id);

            // Log any output from this rank line-by-line (same as TASK_RESULT)
            if (rd.output_size > 0 &&
                msg.payload.size() >= sizeof(RankDonePayload) + rd.output_size) {
                std::string output(
                    reinterpret_cast<const char*>(
                        msg.payload.data() + sizeof(RankDonePayload)),
                    rd.output_size);
                std::istringstream ss(output);
                std::string line;
                while (std::getline(ss, line))
                    log("  > " + line, LogCategory::Job, wid, job_id);
            }

            auto worker = registry_.get(wid);
            if (worker) {
                worker->state     = WorkerState::IDLE;
                worker->available = true;
                pending_job_id_.erase(wid);
                work_queue_.rebuild(registry_);
                update_ui_worker(worker);
            }
            // Drain the queue: without this, a non-root rank finishing after
            // rank 0's TASK_RESULT (or an MPI job where rank 0 finishes first)
            // leaves the pending queue stuck until the next external event.
            try_schedule();
        }
        break;
    }
    case MessageType::HEARTBEAT: {
        std::string wid = worker_id_for(conn);
        log("[HEARTBEAT] " + wid, LogCategory::Heartbeat, wid);
        break;
    }
    case MessageType::GOODBYE:
        log("[GOODBYE] " + worker_id_for(conn), LogCategory::Network,
            worker_id_for(conn));
        break;
    default:
        break;
    }
}

// ============================================================================
// Message handlers
// ============================================================================

void Scheduler::on_hello(Connection::Ptr conn, const Message& msg) {
    if (msg.payload.size() < sizeof(HelloPayload)) return;
    HelloPayload hello{};
    std::memcpy(&hello, msg.payload.data(), sizeof(HelloPayload));
    std::string worker_id(hello.worker_id);
    conn_to_id_[conn.get()] = worker_id;
    log("[HELLO] " + worker_id + " v" + hello.worker_version,
        LogCategory::Network, worker_id);

    HelloAckPayload ack{};
    std::strncpy(ack.scheduler_id, "scheduler_main", sizeof(ack.scheduler_id) - 1);
    ack.task_timeout_seconds = 300;

    Message ack_msg;
    ack_msg.type             = MessageType::HELLO_ACK;
    ack_msg.message_id       = new_msg_id();
    ack_msg.protocol_version = PROTOCOL_VERSION;
    ack_msg.payload.resize(sizeof(HelloAckPayload));
    std::memcpy(ack_msg.payload.data(), &ack, sizeof(HelloAckPayload));
    conn->send_message(ack_msg);
}

void Scheduler::on_capability(Connection::Ptr conn, const Message& msg) {
    std::string worker_id = worker_id_for(conn);
    if (worker_id.empty() || msg.payload.size() < sizeof(CapabilityPayload)) return;

    CapabilityPayload cap{};
    std::memcpy(&cap, msg.payload.data(), sizeof(CapabilityPayload));

    std::string remote_ip = conn->socket().remote_endpoint().address().to_string();

    auto entry        = std::make_shared<WorkerEntry>();
    entry->worker_id  = worker_id;
    entry->ip         = remote_ip;
    entry->port       = conn->socket().remote_endpoint().port();
    entry->hardware   = cap.capability;
    entry->capacity   = derive_capacity(cap.capability);
    entry->state      = WorkerState::IDLE;
    entry->available  = true;
    entry->conn       = conn;

    std::string display_name;
    if (const KnownWorker* kw = system_conf_.find_by_ip(remote_ip))
        display_name = kw->name;

    registry_.add(entry);
    work_queue_.rebuild(registry_);
    update_ui_worker(entry, display_name);

    std::ostringstream s;
    s << "[READY] " << worker_id
      << "  " << cap.capability.cpu_model
      << "  " << cap.capability.cpu_cores << " cores"
      << "  cap=" << std::fixed << std::setprecision(3) << entry->capacity;
    log(s.str(), LogCategory::Worker, worker_id, {}, /*is_summary=*/true);

    try_schedule();
}

void Scheduler::on_file_ack(Connection::Ptr conn, const Message& msg) {
    std::string worker_id = worker_id_for(conn);
    auto worker = registry_.get(worker_id);
    if (!worker || msg.payload.size() < sizeof(FileAckPayload)) return;

    FileAckPayload ack{};
    std::memcpy(&ack, msg.payload.data(), sizeof(FileAckPayload));

    std::string job_id = pending_job_id_.count(worker_id)
                         ? pending_job_id_[worker_id] : "";

    if (!ack.success) {
        mark_failed(worker, "FILE_ACK checksum mismatch: " + std::string(ack.filename));
        return;
    }

    log("[FILE_ACK] " + std::string(ack.filename) + " ok on " + worker_id,
        LogCategory::Job, worker_id, job_id, false, /*collapsible=*/true);

    worker->state = WorkerState::FILE_READY;
    update_ui_worker(worker);

    auto it = pending_compile_.find(worker_id);
    if (it == pending_compile_.end()) {
        mark_failed(worker, "FILE_ACK: no compile command queued");
        return;
    }
    send_compile(worker->conn, worker_id, it->second);
}

void Scheduler::on_exec_result(Connection::Ptr conn, const Message& msg) {
    std::string worker_id = worker_id_for(conn);
    auto worker = registry_.get(worker_id);
    if (!worker || msg.payload.size() < sizeof(ExecResultPayload)) return;

    ExecResultPayload result{};
    std::memcpy(&result, msg.payload.data(), sizeof(ExecResultPayload));

    std::string job_id = pending_job_id_.count(worker_id)
                         ? pending_job_id_[worker_id] : "";

    std::string stderr_str;
    size_t off = sizeof(ExecResultPayload) + result.stdout_size;
    if (off + result.stderr_size <= msg.payload.size())
        stderr_str.assign(
            reinterpret_cast<const char*>(msg.payload.data() + off),
            result.stderr_size);

    if (result.exit_code != 0) {
        mark_failed(worker, "Compile failed (exit=" +
                    std::to_string(result.exit_code) + "): " + stderr_str);
        return;
    }

    log("[COMPILED] " + worker_id, LogCategory::Job, worker_id, job_id,
        false, /*collapsible=*/true);
    worker->state = WorkerState::COMPILED;
    update_ui_worker(worker);

    auto it = pending_run_.find(worker_id);
    if (it == pending_run_.end()) {
        mark_failed(worker, "EXEC_RESULT: no run command queued");
        return;
    }
    send_run(worker->conn, worker_id, it->second);
}

void Scheduler::on_process_spawned(Connection::Ptr conn, const Message& msg) {
    std::string worker_id = worker_id_for(conn);
    auto worker = registry_.get(worker_id);
    if (!worker || msg.payload.size() < sizeof(ProcessSpawnedPayload)) return;

    ProcessSpawnedPayload spawned{};
    std::memcpy(&spawned, msg.payload.data(), sizeof(ProcessSpawnedPayload));

    std::string job_id = pending_job_id_.count(worker_id)
                         ? pending_job_id_[worker_id] : "";

    worker->state     = WorkerState::RUNNING;
    worker->available = false;
    running_pids_[worker_id] = spawned.pid;
    work_queue_.rebuild(registry_);

    auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lk(ui_.mutex);
        for (auto& w : ui_.workers) {
            if (w.worker_id == worker_id) {
                w.state_str = "RUNNING";
                w.pid       = spawned.pid;
                w.run_start = now;
                w.task_id   = spawned.task_id;
                break;
            }
        }
        ui_.add_log("[RUNNING] " + worker_id + " pid=" + std::to_string(spawned.pid),
                    LogCategory::Job, worker_id, job_id, false, /*collapsible=*/true);
    }
}

void Scheduler::on_task_result(Connection::Ptr conn, const Message& msg) {
    std::string worker_id = worker_id_for(conn);
    auto worker = registry_.get(worker_id);
    if (!worker || msg.payload.size() < sizeof(TaskResultPayload)) return;

    TaskResultPayload result{};
    std::memcpy(&result, msg.payload.data(), sizeof(TaskResultPayload));

    std::string job_id = pending_job_id_.count(worker_id)
                         ? pending_job_id_[worker_id] : "";

    std::string output;
    size_t off = sizeof(TaskResultPayload);
    if (result.result_data_size > 0 && off + result.result_data_size <= msg.payload.size())
        output.assign(
            reinterpret_cast<const char*>(msg.payload.data() + off),
            result.result_data_size);

    worker->state     = WorkerState::IDLE;
    worker->available = true;
    running_pids_.erase(worker_id);
    pending_compile_.erase(worker_id);
    pending_run_.erase(worker_id);
    pending_job_id_.erase(worker_id);
    work_queue_.rebuild(registry_);

    {
        std::lock_guard<std::mutex> lk(ui_.mutex);
        for (auto& w : ui_.workers) {
            if (w.worker_id == worker_id) {
                w.state_str   = "IDLE";
                w.pid         = 0;
                w.run_start   = {};
                w.last_output = output;
                break;
            }
        }
        // Sweep intermediate noise (COMPILED, RUN, RUNNING, etc.) for this job
        if (!job_id.empty()) ui_.collapse_job(job_id);

        std::string status = result.success ? "SUCCESS" : "FAILED";
        ui_.add_log("[TASK_RESULT] " + worker_id + " " + status,
                    LogCategory::Job, worker_id, job_id, /*is_summary=*/true);
        if (!output.empty()) {
            std::istringstream iss(output);
            std::string line;
            int n = 0;
            while (std::getline(iss, line) && n++ < 6)
                ui_.add_log("  > " + line, LogCategory::Job, worker_id, job_id);
        }
    }

    try_schedule();
}

void Scheduler::on_status_resp(Connection::Ptr conn, const Message& msg) {
    std::string worker_id = worker_id_for(conn);
    if (msg.payload.size() < sizeof(ProcessStatusRespPayload)) return;

    ProcessStatusRespPayload st{};
    std::memcpy(&st, msg.payload.data(), sizeof(ProcessStatusRespPayload));

    {
        std::lock_guard<std::mutex> lk(ui_.mutex);
        for (auto& w : ui_.workers) {
            if (w.worker_id == worker_id) {
                w.cpu_percent = st.cpu_percent;
                w.mem_bytes   = st.mem_bytes;
                break;
            }
        }
        std::ostringstream s;
        s << "[STATUS] " << worker_id
          << " mem=" << (st.mem_bytes / (1024 * 1024)) << "MB"
          << " cpu=" << st.cpu_percent << "%";
        ui_.add_log(s.str(), LogCategory::Worker, worker_id);
    }
}

// ============================================================================
// Senders
// ============================================================================

void Scheduler::send_compile(Connection::Ptr conn, const std::string& worker_id,
                              const std::string& compile_cmd) {
    ExecCmdPayload cmd{};
    std::strncpy(cmd.task_id, worker_id.c_str(), sizeof(cmd.task_id) - 1);
    std::strncpy(cmd.command, compile_cmd.c_str(), sizeof(cmd.command) - 1);
    cmd.timeout_seconds = 120;
    cmd.detach          = 0;

    Message msg;
    msg.type             = MessageType::EXEC_CMD;
    msg.message_id       = new_msg_id();
    msg.protocol_version = PROTOCOL_VERSION;
    msg.payload.resize(sizeof(ExecCmdPayload));
    std::memcpy(msg.payload.data(), &cmd, sizeof(ExecCmdPayload));
    conn->send_message(msg);

    std::string job_id = pending_job_id_.count(worker_id)
                         ? pending_job_id_[worker_id] : "";
    log("[COMPILE] -> " + worker_id + ": " + compile_cmd,
        LogCategory::Job, worker_id, job_id);
}

void Scheduler::send_run(Connection::Ptr conn, const std::string& worker_id,
                         const std::string& run_cmd) {
    ExecCmdPayload cmd{};
    std::strncpy(cmd.task_id, worker_id.c_str(), sizeof(cmd.task_id) - 1);
    std::strncpy(cmd.command, run_cmd.c_str(), sizeof(cmd.command) - 1);
    cmd.timeout_seconds = 0;
    cmd.detach          = 1;

    Message msg;
    msg.type             = MessageType::EXEC_CMD;
    msg.message_id       = new_msg_id();
    msg.protocol_version = PROTOCOL_VERSION;
    msg.payload.resize(sizeof(ExecCmdPayload));
    std::memcpy(msg.payload.data(), &cmd, sizeof(ExecCmdPayload));
    conn->send_message(msg);

    std::string job_id = pending_job_id_.count(worker_id)
                         ? pending_job_id_[worker_id] : "";
    log("[RUN] -> " + worker_id + ": " + run_cmd,
        LogCategory::Job, worker_id, job_id, false, /*collapsible=*/true);
}

}  // namespace clustr

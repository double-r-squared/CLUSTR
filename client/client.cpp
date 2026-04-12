#include "tcp_server.h"
#include "capability_detector.h"
#include "protocol.h"
#include "config.h"
#include "file_transfer.h"
#include "remote_exec.h"
#include "process_monitor.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace clustr;

// ============================================================================
// Worker
// ============================================================================
//
// Lifecycle:
//   Connect → HELLO → HELLO_ACK → CAPABILITY_REPORT
//   FILE_DATA → FILE_ACK
//   EXEC_CMD(detach=0) → exec_sync → EXEC_RESULT       (compile)
//   EXEC_CMD(detach=1) → spawn_async → PROCESS_SPAWNED (run)
//     [process runs... exits...]
//   TASK_RESULT sent when spawned process exits
//   PROCESS_STATUS_REQ → query_process → PROCESS_STATUS_RESP  (on demand)
//   PROCESS_KILL → kill_process                               (on demand)
//   GOODBYE → shutdown

class Worker {
public:
    Worker(const std::string& scheduler_host,
           uint16_t           scheduler_port,
           const std::string& work_dir,
           const std::string& worker_id)
        : io_context_(),
          client_(io_context_),
          scheduler_host_(scheduler_host),
          scheduler_port_(scheduler_port),
          work_dir_(work_dir),
          worker_id_(worker_id) {}

    void run() {
        std::cout << "Detecting hardware capabilities..." << std::endl;
        capabilities_ = CapabilityDetector::detect_all();

        std::cout << "Connecting to " << scheduler_host_
                  << ":" << scheduler_port_ << "..." << std::endl;

        bool connected = false;

        client_.on_message = [this](const Message& msg) {
            handle_message(msg);
        };

        client_.connect(scheduler_host_, scheduler_port_,
            [&connected](bool ok) { connected = ok; });

        // Poll until connected or timeout
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!connected) {
            io_context_.poll();
            if (std::chrono::steady_clock::now() > deadline) {
                std::cerr << "Connection timed out." << std::endl;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        std::cout << "Connected. Sending HELLO..." << std::endl;
        send_hello();

        // Main IO loop — runs until GOODBYE or connection drop
        static constexpr int HEARTBEAT_INTERVAL_S = 5;
        while (running_) {
            io_context_.poll();

            // Proactive heartbeat when idle
            if (idle_) {
                auto now = std::chrono::steady_clock::now();
                if (last_heartbeat_ == std::chrono::steady_clock::time_point{} ||
                    std::chrono::duration_cast<std::chrono::seconds>(
                        now - last_heartbeat_).count() >= HEARTBEAT_INTERVAL_S) {
                    send_heartbeat();
                    last_heartbeat_ = now;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

private:
    asio::io_context io_context_;
    TcpClient        client_;
    std::string      scheduler_host_;
    uint16_t         scheduler_port_;
    std::string      work_dir_;
    std::string      worker_id_;
    HardwareCapability capabilities_{};
    bool             running_       = true;
    bool             idle_          = false;  // true once capability report sent
    uint32_t         next_msg_id_   = 0;
    std::chrono::steady_clock::time_point last_heartbeat_{};

    // MPI group state — reset after each job
    std::string current_job_id_;
    uint32_t    my_mpi_rank_    = 0;
    uint32_t    mpi_size_       = 1;
    uint16_t    peer_port_      = 0;   // 0 = first PEER_ROSTER not yet handled
    bool        roster_written_ = false;

    uint32_t new_msg_id() { return next_msg_id_++; }

    // ------------------------------------------------------------------
    // Message dispatch
    // ------------------------------------------------------------------

    void handle_message(const Message& msg) {
        switch (msg.type) {
            case MessageType::HELLO_ACK:
                on_hello_ack();
                break;

            case MessageType::PEER_ROSTER:
                on_peer_roster(msg);
                break;

            case MessageType::FILE_DATA:
                on_file_data(msg);
                break;

            case MessageType::EXEC_CMD:
                on_exec_cmd(msg);
                break;

            case MessageType::PROCESS_KILL:
                on_process_kill(msg);
                break;

            case MessageType::PROCESS_STATUS_REQ:
                on_status_req(msg);
                break;

            case MessageType::HEARTBEAT:
                // Echo heartbeat back
                send_heartbeat();
                break;

            case MessageType::GOODBYE:
                std::cout << "[GOODBYE] Scheduler shutting down." << std::endl;
                running_ = false;
                break;

            default:
                std::cout << "[UNKNOWN] type=0x"
                          << std::hex << static_cast<int>(msg.type)
                          << std::dec << std::endl;
        }
    }

    // ------------------------------------------------------------------
    // HELLO_ACK → send CAPABILITY_REPORT
    // ------------------------------------------------------------------

    void on_hello_ack() {
        std::cout << "[HELLO_ACK] Sending capability report..." << std::endl;
        send_capability_report();
        idle_ = true;  // worker is now idle, start heartbeating
    }

    // ------------------------------------------------------------------
    // FILE_DATA → write to work_dir, send FILE_ACK
    // ------------------------------------------------------------------

    void on_file_data(const Message& msg) {
        idle_ = false;  // work is incoming — pause heartbeats
        std::cout << "[FILE_DATA] Receiving source file..." << std::endl;
        try {
            Message ack = handle_file_data(msg, work_dir_, new_msg_id());

            // Log outcome before sending
            FileAckPayload ack_payload{};
            std::memcpy(&ack_payload, ack.payload.data(), sizeof(FileAckPayload));
            if (ack_payload.success) {
                std::cout << "[FILE_ACK] " << ack_payload.filename
                          << " written to " << work_dir_ << " (checksum ok)" << std::endl;
            } else {
                std::cerr << "[FILE_ACK] " << ack_payload.filename
                          << " checksum MISMATCH — rejecting" << std::endl;
            }

            client_.send_message(ack);
        } catch (const std::exception& e) {
            std::cerr << "[FILE_DATA] Error: " << e.what() << std::endl;
            send_error(e.what());
        }
    }

    // ------------------------------------------------------------------
    // EXEC_CMD → dispatch based on detach flag
    //   detach=0: exec_sync  → EXEC_RESULT
    //   detach=1: spawn_async → PROCESS_SPAWNED, then TASK_RESULT on exit
    // ------------------------------------------------------------------

    void on_exec_cmd(const Message& msg) {
        if (msg.payload.size() < sizeof(ExecCmdPayload)) return;

        ExecCmdPayload cmd{};
        std::memcpy(&cmd, msg.payload.data(), sizeof(ExecCmdPayload));

        // If scheduler left working_dir empty, use our configured work_dir
        if (cmd.working_dir[0] == '\0')
            std::strncpy(cmd.working_dir, work_dir_.c_str(),
                         sizeof(cmd.working_dir) - 1);

        std::string task_id(cmd.task_id);
        std::string command(cmd.command);

        if (cmd.detach == 0) {
            // ---- Synchronous: compile ----
            std::cout << "[EXEC_CMD] compile: " << command << std::endl;
            try {
                Message result = exec_sync(cmd, new_msg_id());

                // Log exit code and any stderr
                ExecResultPayload res{};
                std::memcpy(&res, result.payload.data(), sizeof(ExecResultPayload));
                std::cout << "[EXEC_RESULT] exit=" << res.exit_code << std::endl;
                if (res.stderr_size > 0) {
                    std::string err_str(
                        reinterpret_cast<const char*>(
                            result.payload.data() + sizeof(ExecResultPayload) + res.stdout_size),
                        res.stderr_size);
                    std::cerr << "[STDERR]\n" << err_str << std::endl;
                }

                client_.send_message(result);
            } catch (const std::exception& e) {
                std::cerr << "[EXEC_CMD] exec_sync failed: " << e.what() << std::endl;
                send_error(e.what());
            }

        } else {
            // ---- Async: run the binary ----
            std::cout << "[EXEC_CMD] run: " << command << std::endl;

            // Write a 1-rank fallback roster if the scheduler did not send
            // PEER_ROSTER (e.g. single-rank MPI job submitted with Ranks=1).
            // MPI jobs always need the roster file at startup.
            if (!roster_written_)
                write_default_roster();

            // Capture MPI rank so the exit callback can decide TASK_RESULT vs RANK_DONE
            uint32_t    exec_rank   = my_mpi_rank_;
            std::string exec_job_id = current_job_id_;

            try {
                // on_exit is called from a background thread — post back to
                // io_context so send_task_result touches the socket safely.
                Message spawned = spawn_async(cmd, new_msg_id(),
                    [this, task_id, exec_rank, exec_job_id]
                    (int32_t exit_code, std::string output) {
                        asio::post(io_context_,
                            [this, task_id, exec_rank, exec_job_id,
                             exit_code, out = std::move(output)]() mutable {
                            // Root rank reports TASK_RESULT; non-root sends RANK_DONE
                            if (exec_rank == 0 || exec_job_id.empty())
                                send_task_result(task_id, exit_code, std::move(out));
                            else
                                send_rank_done(exec_job_id, exec_rank, exit_code, std::move(out));
                            // Reset MPI state for next job
                            roster_written_ = false;
                            peer_port_      = 0;
                            my_mpi_rank_    = 0;
                            mpi_size_       = 1;
                            current_job_id_.clear();
                        });
                    });

                ProcessSpawnedPayload sp{};
                std::memcpy(&sp, spawned.payload.data(), sizeof(ProcessSpawnedPayload));
                std::cout << "[PROCESS_SPAWNED] pid=" << sp.pid << std::endl;

                client_.send_message(spawned);
            } catch (const std::exception& e) {
                std::cerr << "[EXEC_CMD] spawn_async failed: " << e.what() << std::endl;
                send_error(e.what());
            }
        }
    }

    // ------------------------------------------------------------------
    // PROCESS_KILL → send POSIX signal
    // ------------------------------------------------------------------

    void on_process_kill(const Message& msg) {
        if (msg.payload.size() < sizeof(ProcessKillPayload)) return;

        ProcessKillPayload kill_req{};
        std::memcpy(&kill_req, msg.payload.data(), sizeof(ProcessKillPayload));

        bool ok = kill_process(kill_req.pid, kill_req.signal);
        std::cout << "[PROCESS_KILL] pid=" << kill_req.pid
                  << " signal=" << static_cast<int>(kill_req.signal)
                  << (ok ? " ok" : " failed") << std::endl;
    }

    // ------------------------------------------------------------------
    // PROCESS_STATUS_REQ → query and respond
    // ------------------------------------------------------------------

    void on_status_req(const Message& msg) {
        if (msg.payload.size() < sizeof(ProcessStatusReqPayload)) return;

        ProcessStatusReqPayload req{};
        std::memcpy(&req, msg.payload.data(), sizeof(ProcessStatusReqPayload));

        ProcessStatusRespPayload status = query_process(req.pid);
        Message resp = make_status_resp_msg(status, new_msg_id());
        client_.send_message(resp);
    }

    // ------------------------------------------------------------------
    // PEER_ROSTER — MPI group setup
    // ------------------------------------------------------------------

    void on_peer_roster(const Message& msg) {
        if (msg.payload.size() < sizeof(PeerRosterPayload)) return;

        PeerRosterPayload roster{};
        std::memcpy(&roster, msg.payload.data(), sizeof(PeerRosterPayload));

        std::string job_id(roster.job_id);

        if (peer_port_ == 0) {
            // First PEER_ROSTER for this job: pick a free port and report back
            my_mpi_rank_    = roster.my_rank;
            mpi_size_       = roster.num_ranks;
            current_job_id_ = job_id;
            peer_port_      = pick_free_port();

            std::cout << "[PEER_ROSTER] job=" << job_id
                      << " rank=" << roster.my_rank << "/" << roster.num_ranks
                      << " -> listening on peer_port=" << peer_port_ << std::endl;

            send_peer_ready(job_id, roster.my_rank, peer_port_);
        }

        // Detect whether this is the complete roster (all peer_ports filled)
        bool complete = true;
        for (uint32_t i = 0; i < roster.num_ranks; i++) {
            if (roster.peers[i].peer_port == 0) { complete = false; break; }
        }

        if (complete) {
            write_roster_file(roster);
            roster_written_ = true;
            std::cout << "[PEER_ROSTER] complete roster written for job=" << job_id << std::endl;
        }
    }

    // Bind port 0, read the OS-assigned port, release the socket.
    // The MPI binary will re-bind with SO_REUSEADDR on the same port.
    static uint16_t pick_free_port() {
        int sock = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return 0;
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port        = 0;
        if (::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            ::close(sock); return 0;
        }
        socklen_t len = sizeof(addr);
        ::getsockname(sock, reinterpret_cast<sockaddr*>(&addr), &len);
        uint16_t port = ntohs(addr.sin_port);
        ::close(sock);
        return port;
    }

    void write_roster_file(const PeerRosterPayload& r) {
        ::mkdir("/var/tmp/clustr", 0755);
        std::ofstream f("/var/tmp/clustr/mpi_roster.conf",
                        std::ios::out | std::ios::trunc);
        f << "rank="      << r.my_rank    << "\n"
          << "size="      << r.num_ranks  << "\n"
          << "peer_port=" << peer_port_   << "\n";
        for (uint32_t i = 0; i < r.num_ranks; i++) {
            f << "peer." << r.peers[i].rank << "="
              << r.peers[i].ip << ":" << r.peers[i].peer_port << "\n";
        }
    }

    // Fallback for single-rank or non-MPI jobs: write a rank=0/size=1 roster
    // so that clustr_mpi.h can open the file without throwing.
    void write_default_roster() {
        ::mkdir("/var/tmp/clustr", 0755);
        std::ofstream f("/var/tmp/clustr/mpi_roster.conf",
                        std::ios::out | std::ios::trunc);
        uint16_t port = pick_free_port();
        f << "rank=0\nsize=1\npeer_port=" << port << "\npeer.0=127.0.0.1:" << port << "\n";
        std::cout << "[MPI] Wrote default 1-rank roster (peer_port=" << port << ")" << std::endl;
        roster_written_ = true;
    }

    // ------------------------------------------------------------------
    // Outgoing messages
    // ------------------------------------------------------------------

    void send_hello() {
        HelloPayload payload{};
        std::strncpy(payload.worker_id,      worker_id_.c_str(), sizeof(payload.worker_id) - 1);
        std::strncpy(payload.worker_version, "0.1.0",            sizeof(payload.worker_version) - 1);
        payload.protocol_version = PROTOCOL_VERSION;

        Message msg;
        msg.type             = MessageType::HELLO;
        msg.message_id       = new_msg_id();
        msg.protocol_version = PROTOCOL_VERSION;
        msg.payload.resize(sizeof(HelloPayload));
        std::memcpy(msg.payload.data(), &payload, sizeof(HelloPayload));
        client_.send_message(msg);
    }

    void send_capability_report() {
        CapabilityPayload payload{};
        payload.capability = capabilities_;

        Message msg;
        msg.type             = MessageType::CAPABILITY_REPORT;
        msg.message_id       = new_msg_id();
        msg.protocol_version = PROTOCOL_VERSION;
        msg.payload.resize(sizeof(CapabilityPayload));
        std::memcpy(msg.payload.data(), &payload, sizeof(CapabilityPayload));
        client_.send_message(msg);
    }

    void send_task_result(const std::string& task_id, int32_t exit_code,
                          std::string output = {}) {
        bool ok = (exit_code == 0);
        std::cout << "[TASK_RESULT] task=" << task_id
                  << " exit=" << exit_code
                  << (ok ? " SUCCESS" : " FAILED")
                  << " output=" << output.size() << " bytes" << std::endl;

        // Echo job output to local log too
        if (!output.empty()) {
            std::cout << "--- job output ---\n" << output;
            if (output.back() != '\n') std::cout << '\n';
            std::cout << "------------------" << std::endl;
        }

        TaskResultPayload payload{};
        std::strncpy(payload.task_id, task_id.c_str(), sizeof(payload.task_id) - 1);
        payload.success          = ok ? 1 : 0;
        payload.result_data_size = static_cast<uint32_t>(output.size());
        if (!ok) {
            std::string err = "Process exited with code " + std::to_string(exit_code);
            std::strncpy(payload.error_message, err.c_str(),
                         sizeof(payload.error_message) - 1);
        }

        Message msg;
        msg.type             = MessageType::TASK_RESULT;
        msg.message_id       = new_msg_id();
        msg.protocol_version = PROTOCOL_VERSION;
        msg.payload.resize(sizeof(TaskResultPayload) + output.size());
        std::memcpy(msg.payload.data(), &payload, sizeof(TaskResultPayload));
        if (!output.empty())
            std::memcpy(msg.payload.data() + sizeof(TaskResultPayload),
                        output.data(), output.size());
        client_.send_message(msg);
        idle_ = true;  // task result sent — worker is idle again
    }

    void send_heartbeat() {
        Message msg;
        msg.type             = MessageType::HEARTBEAT;
        msg.message_id       = new_msg_id();
        msg.protocol_version = PROTOCOL_VERSION;
        client_.send_message(msg);
    }

    void send_peer_ready(const std::string& job_id, uint32_t rank, uint16_t port) {
        PeerReadyPayload payload{};
        std::strncpy(payload.job_id, job_id.c_str(), sizeof(payload.job_id) - 1);
        payload.my_rank   = rank;
        payload.peer_port = port;

        Message msg;
        msg.type             = MessageType::PEER_READY;
        msg.message_id       = new_msg_id();
        msg.protocol_version = PROTOCOL_VERSION;
        msg.payload.resize(sizeof(PeerReadyPayload));
        std::memcpy(msg.payload.data(), &payload, sizeof(PeerReadyPayload));
        client_.send_message(msg);
    }

    // Non-root MPI ranks send RANK_DONE instead of TASK_RESULT on exit
    void send_rank_done(const std::string& job_id, uint32_t rank,
                        int32_t exit_code, std::string output = {}) {
        std::cout << "[RANK_DONE] job=" << job_id
                  << " rank=" << rank << " exit=" << exit_code << std::endl;
        if (!output.empty()) {
            std::cout << "--- rank " << rank << " output ---\n" << output;
            if (output.back() != '\n') std::cout << '\n';
            std::cout << "-------------------" << std::endl;
        }

        RankDonePayload payload{};
        std::strncpy(payload.job_id, job_id.c_str(), sizeof(payload.job_id) - 1);
        payload.rank        = rank;
        payload.exit_code   = exit_code;
        payload.output_size = static_cast<uint32_t>(output.size());

        Message msg;
        msg.type             = MessageType::RANK_DONE;
        msg.message_id       = new_msg_id();
        msg.protocol_version = PROTOCOL_VERSION;
        msg.payload.resize(sizeof(RankDonePayload) + output.size());
        std::memcpy(msg.payload.data(), &payload, sizeof(RankDonePayload));
        if (!output.empty())
            std::memcpy(msg.payload.data() + sizeof(RankDonePayload),
                        output.data(), output.size());
        client_.send_message(msg);
        idle_ = true;
    }

    void send_error(const std::string& reason) {
        Message msg;
        msg.type             = MessageType::ERROR_MSG;
        msg.message_id       = new_msg_id();
        msg.protocol_version = PROTOCOL_VERSION;
        msg.payload.assign(reason.begin(), reason.end());
        client_.send_message(msg);
    }
};

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "CLUSTR Worker v0.1.0 (Bonesy)" << std::endl << std::endl;

    // Load config — fall back to CLI args or defaults
    std::string scheduler_host = "localhost";
    uint16_t    scheduler_port = DEFAULT_PORT;
    std::string work_dir       = "/tmp/clustr";
    std::string worker_id      = generate_worker_id();

    // Search for config in priority order:
    //   1. CLI argument (explicit path)
    //   2. /etc/clustr/worker.conf  (system install, written by deploy.sh)
    //   3. worker.conf              (local dev/fallback)
    static const char* CONFIG_SEARCH[] = {
        nullptr,                     // placeholder for argv[1]
        "/etc/clustr/worker.conf",
        "worker.conf",
    };
    if (argc > 1) CONFIG_SEARCH[0] = argv[1];

    bool config_loaded = false;
    for (const char* path : CONFIG_SEARCH) {
        if (!path) continue;
        try {
            Config cfg = Config::from_file(path);
            scheduler_host = cfg.scheduler_ip();
            scheduler_port = cfg.scheduler_port();
            work_dir       = cfg.work_dir();
            worker_id      = cfg.worker_id();
            std::cout << "Config loaded from " << path << std::endl;
            config_loaded = true;
            break;
        } catch (...) {}
    }
    if (!config_loaded) {
        std::cout << "No config found in any search path — using defaults\n"
                  << "  (checked: /etc/clustr/worker.conf, worker.conf)\n"
                  << "Using defaults: localhost:" << scheduler_port << std::endl;
    }

    std::cout << "  scheduler: " << scheduler_host << ":" << scheduler_port << std::endl;
    std::cout << "  work_dir:  " << work_dir       << std::endl;
    std::cout << "  worker_id: " << worker_id      << std::endl << std::endl;

    Worker worker(scheduler_host, scheduler_port, work_dir, worker_id);
    worker.run();

    return 0;
}

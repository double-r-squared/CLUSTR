#pragma once

#include "scheduler_strategy.h"
#include "system_conf.h"
#include "protocol.h"
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <mutex>
#include <chrono>
#include <functional>
#include <cstdint>

namespace clustr {

// ============================================================================
// LogCategory — tag applied to every log entry for per-page filtering
// ============================================================================

enum class LogCategory : uint8_t {
    Job       = 0,  // bit 0
    Worker    = 1,  // bit 1
    Network   = 2,  // bit 2
    Heartbeat = 3,  // bit 3
    Deploy    = 4,  // bit 4
};

inline uint8_t cat_bit(LogCategory c) {
    return static_cast<uint8_t>(1u << static_cast<unsigned>(c));
}

// ============================================================================
// WorkerDisplay — snapshot rendered by the TUI each frame
// ============================================================================

struct WorkerDisplay {
    std::string  worker_id;
    std::string  display_name;   // From system.conf section header; empty if unknown
    std::string  ip;
    float        capacity    = 0.0f;
    std::string  state_str;
    uint32_t     pid         = 0;
    std::string  task_id;
    std::chrono::steady_clock::time_point run_start{};
    std::string  last_output;
    uint64_t     mem_bytes   = 0;
    float        cpu_percent = 0.0f;

    HardwareCapability hw_capability{};
    bool               has_capability = false;
};

// ============================================================================
// LogEntry — one timestamped log line, tagged for filtering
// ============================================================================

struct LogEntry {
    std::string  timestamp;
    std::string  text;
    LogCategory  category   = LogCategory::Worker;
    std::string  worker_id;   // empty = not worker-specific
    std::string  job_id;      // empty = not job-specific
    bool         is_summary  = false;  // show in Dashboard summary strip
    bool         collapsible = false;  // removed from log when the job completes
};

// ============================================================================
// TuiState — shared between io_context thread (writer) and TUI thread (reader)
// ============================================================================

struct TuiState {
    std::mutex                 mutex;
    std::vector<WorkerDisplay> workers;
    std::deque<LogEntry>       log;

    // Job template written by submit dialog, read by scheduler
    std::string source_file;
    std::string compile_cmd;
    std::string run_cmd;
    std::vector<std::string> companion_files;   // extra files to send (e.g. .py)
    uint32_t    num_ranks   = 1;
    uint16_t    port = 9999;

    // Pending queue snapshot (scheduler owns the real JobQueue)
    std::vector<JobPtr> pending_jobs;

    // Active strategy name for display
    std::string strategy_name = "FirstAvailable";

    // Known workers from system.conf — set once at startup, immutable after
    std::vector<KnownWorker> known_workers;

    // Names of known workers currently running a deploy subprocess
    std::set<std::string> deploying_workers;

    // Maps source_file path → most recently submitted job_id for that file.
    // Used by the Jobs page to filter the log panel to a specific job's logs.
    std::map<std::string, std::string> recent_job_for_file;

    static constexpr size_t MAX_LOG = 2000;

    // All must be called with mutex held.
    void add_log(std::string      msg,
                 LogCategory      cat          = LogCategory::Worker,
                 std::string      worker_id    = {},
                 std::string      job_id       = {},
                 bool             is_summary   = false,
                 bool             collapsible  = false);
    // Remove all collapsible entries for a completed job.
    void collapse_job(const std::string& job_id);
    void upsert_worker(const WorkerDisplay& w);
    void remove_worker(const std::string& id);
};

// ============================================================================
// TuiCommands — callbacks executed on the io_context thread via asio::post
// ============================================================================

struct TuiCommands {
    std::function<void(const std::string& pinned_worker_id)> submit_job;
    std::function<void(const std::string& job_id,
                       const std::string& worker_id)>       force_dispatch;
    std::function<void(const std::string& worker_id)>       kill_job;
    std::function<void(const std::string& worker_id)>       request_status;
    std::function<void(const std::string& job_id)>          cancel_job;
    std::function<void(const std::string& strategy_name)>   set_strategy;
    std::function<void(const std::string& name)>            deploy_worker;
    std::function<void()>                                    reload_config;
    std::function<void()>                                    quit;
};

// ============================================================================
// Page — the four TUI screens
// ============================================================================

enum class Page { Dashboard = 1, Jobs = 2, Workers = 3, Logs = 4 };

// ============================================================================
// Tui — ncurses terminal UI, runs on the main thread
// ============================================================================

class Tui {
public:
    Tui(TuiState& state, TuiCommands cmds, const std::string& conf_path);
    ~Tui();

    void run();   // blocks until quit

private:
    TuiState&   state_;
    TuiCommands cmds_;
    std::string conf_path_;

    // ── Page and selection state ───────────────────────────────────────────────
    Page page_    = Page::Dashboard;
    int  sel_w_   = 0;   // selected worker row (Dashboard + Workers page)
    int  sel_file_= 0;   // selected file in Jobs page browser

    bool dash_cap_expanded_ = false;   // Dashboard: capability panel open

    int  log_scroll_w_ = 0;   // Workers page: lines scrolled up from bottom
    int  log_scroll_j_ = 0;   // Jobs page: lines scrolled up from bottom
    int  log_scroll_l_ = 0;   // Logs page: lines scrolled up from bottom
    uint8_t log_filter_ = 0xFF;  // Logs page: bitmask of visible categories

    std::vector<std::string> job_files_;  // .cpp files found in jobs/

    bool running_ = true;

    // ── Rendering ─────────────────────────────────────────────────────────────
    void redraw();

    void draw_page_dashboard(int rows, int cols,
                             const std::vector<WorkerDisplay>& wkrs,
                             const std::vector<KnownWorker>&   known,
                             const std::set<std::string>&      deploying,
                             const std::deque<LogEntry>&       log);

    void draw_page_jobs(int rows, int cols,
                        const std::deque<LogEntry>&                 log,
                        const std::map<std::string, std::string>&   recent);

    void draw_page_workers(int rows, int cols,
                           const std::vector<WorkerDisplay>& wkrs,
                           const std::vector<KnownWorker>&   known,
                           const std::set<std::string>&      deploying,
                           const std::deque<LogEntry>&       log);

    void draw_page_logs(int rows, int cols, const std::deque<LogEntry>& log);

    // Shared primitives used by multiple page renderers
    int  draw_page_header(int cols, Page page,
                          const std::vector<WorkerDisplay>& wkrs,
                          size_t queue_size);
    int  draw_worker_table(int row, int cols,
                           const std::vector<WorkerDisplay>& wkrs,
                           const std::vector<WorkerDisplay>& offline,
                           bool show_actions);
    int  draw_capability_panel(int row, int cols, const WorkerDisplay& w);
    int  draw_log_panel(int row, int end_row, int cols,
                        const std::deque<LogEntry>& log,
                        int                         scroll,
                        const std::string&          filter_worker_id = {},
                        const std::string&          filter_job_id    = {},
                        uint8_t                     cat_mask         = 0xFF);
    void draw_statusbar(int rows, int cols);
    void draw_modal(int dh, int dw, int& dy, int& dx, const char* title);

    // ── Input ─────────────────────────────────────────────────────────────────
    bool handle_key(int ch);
    void handle_key_dashboard(int ch, const std::vector<WorkerDisplay>& wkrs,
                              int total_workers);
    void handle_key_jobs(int ch);
    void handle_key_workers(int ch, const std::vector<WorkerDisplay>& wkrs,
                            const std::vector<WorkerDisplay>& offline);
    void handle_key_logs(int ch);

    // ── Dialogs ───────────────────────────────────────────────────────────────
    void dialog_submit_job_file(const std::string& filepath);
    void dialog_force_assign();
    void dialog_set_strategy();
    void dialog_edit_config();
    void dialog_add_worker();
    void dialog_deploy_confirm(const std::string& name, const std::string& ip);

    std::string input_line(int y, int x, int fw, const std::string& initial);

    // ── Helpers ───────────────────────────────────────────────────────────────
    void scan_job_files();
    std::vector<WorkerDisplay> build_offline(
        const std::vector<WorkerDisplay>& wkrs,
        const std::vector<KnownWorker>&   known,
        const std::set<std::string>&      deploying) const;
};

}  // namespace clustr

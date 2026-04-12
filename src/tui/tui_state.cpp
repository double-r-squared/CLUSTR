#include "tui.h"
#include "tui_impl.h"
#include <algorithm>
#include <filesystem>

namespace clustr {

void TuiState::add_log(std::string msg,
                        LogCategory cat,
                        std::string worker_id,
                        std::string job_id,
                        bool        is_summary,
                        bool        collapsible) {
    if (log.size() >= MAX_LOG) log.pop_front();
    LogEntry e;
    e.timestamp   = now_hms();
    e.text        = std::move(msg);
    e.category    = cat;
    e.worker_id   = std::move(worker_id);
    e.job_id      = std::move(job_id);
    e.is_summary  = is_summary;
    e.collapsible = collapsible;
    log.push_back(std::move(e));
}

void TuiState::collapse_job(const std::string& job_id) {
    // Called (under mutex) when a job completes. Removes all in-progress noise
    // (COMPILED, RUN, RUNNING, FILE_ACK, PEER_READY, MPI setup) for this job,
    // leaving only result entries (TASK_RESULT, RANK_DONE, output lines).
    log.erase(
        std::remove_if(log.begin(), log.end(),
            [&](const LogEntry& e) {
                return e.collapsible && e.job_id == job_id;
            }),
        log.end());
}

void TuiState::upsert_worker(const WorkerDisplay& w) {
    for (auto& e : workers) {
        if (e.worker_id == w.worker_id) { e = w; return; }
    }
    workers.push_back(w);
}

void TuiState::remove_worker(const std::string& id) {
    workers.erase(
        std::remove_if(workers.begin(), workers.end(),
            [&](const WorkerDisplay& e) { return e.worker_id == id; }),
        workers.end());
}

// ── Tui construction / destruction ────────────────────────────────────────────

Tui::Tui(TuiState& state, TuiCommands cmds, const std::string& conf_path)
    : state_(state), cmds_(std::move(cmds)), conf_path_(conf_path)
{
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    nodelay(stdscr, TRUE);
    curs_set(0);

    if (has_colors()) {
        start_color();
        use_default_colors();
        init_pair(CP_HEADER,  COLOR_WHITE,   COLOR_BLUE);
        init_pair(CP_IDLE,    COLOR_GREEN,   -1);
        init_pair(CP_RUNNING, COLOR_YELLOW,  -1);
        init_pair(CP_FAILED,  COLOR_RED,     -1);
        init_pair(CP_SEL,     COLOR_BLACK,   COLOR_WHITE);
        init_pair(CP_LOG_TS,  COLOR_WHITE,   -1);
        init_pair(CP_SETUP,   COLOR_MAGENTA, -1);
        init_pair(CP_DIM,     COLOR_WHITE,   -1);

        init_pair(CP_LOG_HANDSHAKE, COLOR_YELLOW,  -1);
        init_pair(CP_LOG_TRANSFER,  COLOR_CYAN,    -1);
        init_pair(CP_LOG_EXEC,      COLOR_MAGENTA, -1);
        init_pair(CP_LOG_SPAWN,     COLOR_YELLOW,  -1);
        init_pair(CP_LOG_SUCCESS,   COLOR_GREEN,   -1);
        init_pair(CP_LOG_ERROR,     COLOR_RED,     -1);
        init_pair(CP_LOG_OUTPUT,    COLOR_CYAN,    -1);
    }

    scan_job_files();
}

Tui::~Tui() {
    endwin();
}

// ── Helpers ───────────────────────────────────────────────────────────────────

std::vector<WorkerDisplay> Tui::build_offline(
    const std::vector<WorkerDisplay>& wkrs,
    const std::vector<KnownWorker>&   known,
    const std::set<std::string>&      deploying) const
{
    std::vector<WorkerDisplay> offline;
    for (const auto& kw : known) {
        bool connected = false;
        for (const auto& w : wkrs)
            if (w.ip == kw.deploy_host) { connected = true; break; }
        if (!connected) {
            WorkerDisplay d;
            d.worker_id    = kw.name;
            d.display_name = kw.name;
            d.ip           = kw.deploy_host;
            d.state_str    = deploying.count(kw.name) ? "DEPLOYING" : "OFFLINE";
            offline.push_back(d);
        }
    }
    return offline;
}

void Tui::scan_job_files() {
    job_files_.clear();

    namespace fs = std::filesystem;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator("jobs", ec)) {
        if (ec) break;
        const auto& p = entry.path();
        if (p.extension() == ".cpp")
            job_files_.push_back(p.filename().string());
    }
    std::sort(job_files_.begin(), job_files_.end());

    if (sel_file_ >= (int)job_files_.size())
        sel_file_ = std::max(0, (int)job_files_.size() - 1);
}

}  // namespace clustr

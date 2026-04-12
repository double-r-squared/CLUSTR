#include "tui.h"
#include "tui_impl.h"
#include <sstream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cstdio>

namespace clustr {

// ============================================================================
// Main event loop
// ============================================================================

void Tui::run() {
    while (running_) {
        redraw();
        int ch = getch();
        if (ch != ERR && !handle_key(ch))
            running_ = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// ============================================================================
// redraw — snapshot state and dispatch to the active page renderer
// ============================================================================

void Tui::redraw() {
    int rows, cols;
    getmaxyx(stdscr, rows, cols);
    erase();

    std::vector<WorkerDisplay>              wkrs;
    std::deque<LogEntry>                    log;
    std::vector<JobPtr>                     jobs;
    std::vector<KnownWorker>                known;
    std::set<std::string>                   deploying;
    std::map<std::string, std::string>      recent;
    std::string                             strategy;
    {
        std::lock_guard<std::mutex> lk(state_.mutex);
        wkrs      = state_.workers;
        log       = state_.log;
        jobs      = state_.pending_jobs;
        known     = state_.known_workers;
        deploying = state_.deploying_workers;
        recent    = state_.recent_job_for_file;
        strategy  = state_.strategy_name;
    }

    switch (page_) {
    case Page::Dashboard:
        draw_page_dashboard(rows, cols, wkrs, known, deploying, log);
        break;
    case Page::Jobs:
        draw_page_jobs(rows, cols, log, recent);
        break;
    case Page::Workers:
        draw_page_workers(rows, cols, wkrs, known, deploying, log);
        break;
    case Page::Logs:
        draw_page_logs(rows, cols, log);
        break;
    }

    draw_statusbar(rows, cols);
    refresh();
}

// ============================================================================
// draw_page_header — shared top bar with page indicator and summary counts
// ============================================================================

int Tui::draw_page_header(int cols, Page page,
                           const std::vector<WorkerDisplay>& wkrs,
                           size_t queue_size) {
    int idle = 0, running = 0, failed = 0;

    for (const auto& w : wkrs) {
        if (w.state_str == "IDLE")    idle++;
        if (w.state_str == "RUNNING") running++;
        if (w.state_str == "FAILED")  failed++;
    }

    const char* page_name = "DASHBOARD";
    if      (page == Page::Jobs)     page_name = "JOBS";
    else if (page == Page::Workers)  page_name = "WORKERS";
    else if (page == Page::Logs)     page_name = "LOGS";

    std::string strategy_suffix;
    if (page == Page::Jobs) {
        std::lock_guard<std::mutex> lk(state_.mutex);
        strategy_suffix = "  Strategy: " + state_.strategy_name;
    }

    attron(COLOR_PAIR(CP_HEADER) | A_BOLD);

    std::string left = " " + std::string(page_name) + strategy_suffix;

    std::ostringstream rs;
    rs << "W:" << wkrs.size()
       << " I:" << idle
       << " R:" << running;
    if (failed) rs << " F:" << failed;
    rs << " Q:" << queue_size << " ";
    std::string right = rs.str();

    std::string hdr = left;
    int pad = cols - (int)left.size() - (int)right.size();
    if (pad > 0) hdr.append(pad, ' ');
    hdr += right;
    if ((int)hdr.size() > cols) hdr.resize(cols);
    mvprintw(0, 0, "%s", hdr.c_str());
    attroff(COLOR_PAIR(CP_HEADER) | A_BOLD);
    return 1;
}

// ============================================================================
// draw_worker_table — renders connected + offline rows, returns next row
// ============================================================================

int Tui::draw_worker_table(int row, int cols,
                            const std::vector<WorkerDisplay>& wkrs,
                            const std::vector<WorkerDisplay>& offline,
                            bool show_actions) {

    // Column header
    attron(A_DIM);
    mvprintw(row, 0, "  %-22s %-16s %-12s %s",
             "Name / Worker ID", "IP", "State", "Capacity");
    attroff(A_DIM);
    row++;

    int total = (int)wkrs.size() + (int)offline.size();
    if (total == 0) {
        attron(A_DIM);
        mvprintw(row,     2, "No workers configured. To add a machine, edit system.conf:");
        mvprintw(row + 2, 4, "[worker-name]");
        mvprintw(row + 3, 4, "deploy_host=<ip-address>       # must be an IP, not a hostname");
        mvprintw(row + 4, 4, "deploy_user=<ssh-user>");
        mvprintw(row + 6, 2, "Then run:  ./scripts/setup_worker.sh   (once per machine)");
        mvprintw(row + 7, 2, "Then press [e] to reload config, select the worker, and press [d] to deploy.");
        attroff(A_DIM);
        return row + 9;
    }

    auto render_row = [&](int idx, const WorkerDisplay& w, bool is_offline) {
        bool selected = (idx == sel_w_);

        std::string label = w.display_name.empty() ? w.worker_id : w.display_name;
        if (!w.display_name.empty() && !is_offline &&
            w.worker_id != w.display_name && (int)label.size() < 18)
            label += " (" + w.worker_id.substr(0, std::min((size_t)8, w.worker_id.size())) + ")";

        char cap_buf[8];
        if (is_offline) snprintf(cap_buf, sizeof(cap_buf), "  -   ");
        else            snprintf(cap_buf, sizeof(cap_buf), "%.3f", w.capacity);

        if (selected) attron(COLOR_PAIR(CP_SEL) | A_BOLD);

        mvprintw(row, 0, "  %-22s %-16s ",
                 pad(label, 22).c_str(),
                 pad(w.ip, 16).c_str());

        int scol = state_color(w.state_str);
        if (!selected) attron(COLOR_PAIR(scol) | A_BOLD);
        mvprintw(row, 2 + 22 + 16 + 1, "%-12s", pad(w.state_str, 12).c_str());
        if (!selected) attroff(COLOR_PAIR(scol) | A_BOLD);

        mvprintw(row, 2 + 22 + 16 + 1 + 12 + 1, "%s", cap_buf);

        // Elapsed time for running workers
        if (!is_offline && w.state_str == "RUNNING") {
            std::string ela = elapsed_str(w.run_start);
            mvprintw(row, 2 + 22 + 16 + 1 + 12 + 1 + 8, "  %s", ela.c_str());
        }

        if (selected) attroff(COLOR_PAIR(CP_SEL) | A_BOLD);
        row++;
    };

    for (int i = 0; i < (int)wkrs.size(); i++)
        render_row(i, wkrs[i], false);
    for (int i = 0; i < (int)offline.size(); i++)
        render_row((int)wkrs.size() + i, offline[i], true);

    return row;
}

// ============================================================================
// draw_capability_panel — shown on Dashboard when a worker is selected+expanded
// ============================================================================

int Tui::draw_capability_panel(int row, int cols, const WorkerDisplay& w) {
    for (int c = 0; c < cols; c++) mvaddch(row, c, '-');
    row++;

    if (!w.has_capability) {
        attron(A_DIM);
        mvprintw(row, 2, "(capability data not yet received)");
        attroff(A_DIM);
        return row + 1;
    }

    const auto& hw = w.hw_capability;

    // CPU line
    char avx_buf[32];
    snprintf(avx_buf, sizeof(avx_buf), "AVX2:%s  AVX512:%s  NEON:%s",
             hw.has_avx2   ? "yes" : "no",
             hw.has_avx512 ? "yes" : "no",
             hw.has_neon   ? "yes" : "no");
    attron(A_BOLD);
    mvprintw(row, 2, "CPU");
    attroff(A_BOLD);
    mvprintw(row, 6, "%-40s  %u cores / %u threads  %s",
             trunc(hw.cpu_model, 40).c_str(),
             hw.cpu_cores, hw.cpu_threads, avx_buf);
    row++;

    // Memory line
    double total_gb = hw.total_ram_bytes / 1073741824.0;
    double avail_gb = hw.available_ram_bytes / 1073741824.0;
    attron(A_BOLD);
    mvprintw(row, 2, "RAM");
    attroff(A_BOLD);
    mvprintw(row, 6, "%.1f GB total   %.1f GB available   BW: %.2f GB/s",
             total_gb, avail_gb, hw.memory_bandwidth_gbps);
    row++;

    // Compute / storage line
    double disk_gb = hw.storage_bytes / 1073741824.0;
    attron(A_BOLD);
    mvprintw(row, 2, "COMPUTE");
    attroff(A_BOLD);
    mvprintw(row, 10, "Score: %.3f GFLOPS   Storage: %.0f GB   Capacity: %.3f",
             hw.compute_score, disk_gb, w.capacity);
    row++;

    return row;
}

// ============================================================================
// draw_log_panel — scrollable filtered log section
// ============================================================================

int Tui::draw_log_panel(int row, int end_row, int cols,
                         const std::deque<LogEntry>& log,
                         int                         scroll,
                         const std::string&          filter_worker_id,
                         const std::string&          filter_job_id,
                         uint8_t                     cat_mask) {
    // Build filtered view
    std::vector<const LogEntry*> visible;
    for (const auto& e : log) {
        if (!(cat_mask & cat_bit(e.category)))          continue;
        if (!filter_worker_id.empty() && e.worker_id != filter_worker_id) continue;
        if (!filter_job_id.empty()    && e.job_id    != filter_job_id)    continue;
        visible.push_back(&e);
    }

    int avail = end_row - row;
    if (avail <= 0) return row;

    if (visible.empty()) {
        attron(A_DIM);
        if (!filter_job_id.empty())
            mvprintw(row, 2, "No logs for this job yet.");
        else if (!filter_worker_id.empty())
            mvprintw(row, 2, "No logs for this worker yet.");
        else
            mvprintw(row, 2, "(log is empty)");
        attroff(A_DIM);
        return row + 1;
    }

    // scroll=0 means show tail; scroll>0 means scrolled that many lines up
    int total = (int)visible.size();
    int start = std::max(0, total - avail - scroll);
    int end   = std::min(total, start + avail);

    for (int i = start; i < end && row < end_row; i++) {
        const auto& e = *visible[i];

        attron(COLOR_PAIR(CP_LOG_TS) | A_DIM);
        mvprintw(row, 0, "%s ", e.timestamp.c_str());
        attroff(COLOR_PAIR(CP_LOG_TS) | A_DIM);

        auto [cp, attr] = log_color(e.text);
        attron(COLOR_PAIR(cp) | attr);
        mvprintw(row, 9, "%s", trunc(e.text, cols - 10).c_str());
        attroff(COLOR_PAIR(cp) | attr);
        row++;
    }

    // Scroll indicator
    if (scroll > 0 && row < end_row) {
        attron(A_DIM);
        mvprintw(row - 1, cols - 16, " [+%d lines] ", scroll);
        attroff(A_DIM);
    }

    return row;
}

// ============================================================================
// Page 1 — Dashboard
// ============================================================================

void Tui::draw_page_dashboard(int rows, int cols,
                               const std::vector<WorkerDisplay>& wkrs,
                               const std::vector<KnownWorker>&   known,
                               const std::set<std::string>&      deploying,
                               const std::deque<LogEntry>&       log) {
    std::vector<JobPtr> jobs;
    { std::lock_guard<std::mutex> lk(state_.mutex); jobs = state_.pending_jobs; }

    int row = draw_page_header(cols, Page::Dashboard, wkrs, jobs.size());

    auto offline = build_offline(wkrs, known, deploying);
    int total_workers = (int)wkrs.size() + (int)offline.size();
    if (total_workers > 0)
        sel_w_ = std::max(0, std::min(sel_w_, total_workers - 1));

    row = draw_worker_table(row, cols, wkrs, offline, /*show_actions=*/false);

    // Cluster stats separator
    {
        uint32_t total_cores = 0;
        uint64_t total_mem   = 0;
        float    cap_sum     = 0.0f;
        int      cap_n       = 0;
        for (const auto& w : wkrs) {
            cap_sum += w.capacity;
            cap_n++;
            if (w.has_capability) {
                total_cores += w.hw_capability.cpu_threads;
                total_mem   += w.hw_capability.total_ram_bytes;
            }
        }
        float avg_cap  = cap_n > 0 ? cap_sum / cap_n : 0.0f;
        float mem_gb   = total_mem / (1024.0f * 1024.0f * 1024.0f);

        attron(A_DIM);
        for (int c = 0; c < cols; c++) mvaddch(row, c, '-');
        row++;
        attroff(A_DIM);

        char stats[128];
        if (cap_n > 0)
            snprintf(stats, sizeof(stats),
                     " cores: %u   avg capacity: %.3f   memory: %.1f GB ",
                     total_cores, avg_cap, mem_gb);
        else
            snprintf(stats, sizeof(stats), " no connected workers ");

        attron(A_DIM | A_BOLD);
        mvprintw(row, 0, "%s", stats);
        attroff(A_DIM | A_BOLD);
        row++;
    }

    // Capability panel (if expanded and a connected worker is selected)
    if (dash_cap_expanded_ && sel_w_ < (int)wkrs.size()) {
        row = draw_capability_panel(row, cols, wkrs[sel_w_]);
    }

    // Summary log strip — always last 6 rows before statusbar
    constexpr int SUMMARY_ROWS = 6;
    int summary_start = rows - 1 - SUMMARY_ROWS;  // -1 for statusbar
    if (summary_start > row) {
        // Draw separator
        for (int c = 0; c < cols; c++) mvaddch(summary_start, c, '-');
        attron(A_BOLD);
        mvprintw(summary_start, 0, " SUMMARY ");
        attroff(A_BOLD);

        // Collect is_summary entries
        std::vector<const LogEntry*> summary;
        for (const auto& e : log)
            if (e.is_summary) summary.push_back(&e);

        int start = std::max(0, (int)summary.size() - SUMMARY_ROWS + 1);
        int srow  = summary_start + 1;
        for (int i = start; i < (int)summary.size() && srow < rows - 1; i++) {
            const auto& e = *summary[i];
            attron(COLOR_PAIR(CP_LOG_TS) | A_DIM);
            mvprintw(srow, 0, "%s ", e.timestamp.c_str());
            attroff(COLOR_PAIR(CP_LOG_TS) | A_DIM);
            auto [cp, attr] = log_color(e.text);
            attron(COLOR_PAIR(cp) | attr);
            mvprintw(srow, 9, "%s", trunc(e.text, cols - 10).c_str());
            attroff(COLOR_PAIR(cp) | attr);
            srow++;
        }
    }
}

// ============================================================================
// Page 2 — Jobs (file browser + per-job log)
// ============================================================================

void Tui::draw_page_jobs(int rows, int cols,
                          const std::deque<LogEntry>&              log,
                          const std::map<std::string, std::string>& recent) {
    std::vector<JobPtr> jobs;
    std::string strategy;
    { std::lock_guard<std::mutex> lk(state_.mutex);
      jobs     = state_.pending_jobs;
      strategy = state_.strategy_name; }

    std::vector<WorkerDisplay> wkrs;
    { std::lock_guard<std::mutex> lk(state_.mutex); wkrs = state_.workers; }

    int row = draw_page_header(cols, Page::Jobs, wkrs, jobs.size());

    // Column header
    attron(A_DIM);
    mvprintw(row, 0, "  %-30s  %-20s  %s",
             "File", "Last Job ID", "Status");
    attroff(A_DIM);
    row++;


    // File browser
    int browser_start = row;
    if (job_files_.empty()) {
        attron(A_DIM);
        mvprintw(row, 2, "(no .cpp files found in jobs/ - press [r] to refresh)");
        attroff(A_DIM);
        row++;
    } else {
        if (sel_file_ >= (int)job_files_.size())
            sel_file_ = (int)job_files_.size() - 1;

        for (int i = 0; i < (int)job_files_.size(); i++) {
            bool selected = (i == sel_file_);

            // Look up the most recent job_id for this file
            std::string filepath = "jobs/" + job_files_[i];
            std::string job_id, job_status;
            auto it = recent.find(filepath);
            if (it != recent.end()) {
                job_id = it->second;
                // Check if this job_id is still in the pending queue
                bool pending = false;
                for (const auto& j : jobs)
                    if (j->job_id == job_id) { pending = true; break; }
                job_status = pending ? "queued/running" : "completed";
            } else {
                job_id     = "(not submitted)";
                job_status = "-";
            }

            if (selected) attron(COLOR_PAIR(CP_SEL) | A_BOLD);
            mvprintw(row, 0, "  %-30s  %-20s  %s",
                     pad(job_files_[i], 30).c_str(),
                     pad(trunc(job_id, 20), 20).c_str(),
                     job_status.c_str());
            if (selected) attroff(COLOR_PAIR(CP_SEL) | A_BOLD);
            row++;
        }
    }
    (void)browser_start;

    // Separator + log panel for selected job
    if (row < rows - 3) {
        for (int c = 0; c < cols; c++) mvaddch(row, c, '-');
        attron(A_BOLD);
        mvprintw(row, 0, " LOG ");
        attroff(A_BOLD);
        row++;

        // Get the job_id for the selected file (empty = never submitted)
        std::string filter_job_id;
        if (!job_files_.empty() && sel_file_ < (int)job_files_.size()) {
            std::string filepath = "jobs/" + job_files_[sel_file_];
            auto it = recent.find(filepath);
            if (it != recent.end())
                filter_job_id = it->second;
        }

        if (filter_job_id.empty()) {
            // File has never been submitted — show placeholder instead of
            // leaking every other job's logs into the panel.
            attron(A_DIM);
            mvprintw(row, 2, "No logs for this file yet. Press [Enter] to submit.");
            attroff(A_DIM);
        } else {
            draw_log_panel(row, rows - 1, cols, log, log_scroll_j_,
                           /*worker_id=*/{}, filter_job_id,
                           cat_bit(LogCategory::Job));
        }
    }
}

// ============================================================================
// Page 3 — Workers (full list + per-worker log)
// ============================================================================

void Tui::draw_page_workers(int rows, int cols,
                              const std::vector<WorkerDisplay>& wkrs,
                              const std::vector<KnownWorker>&   known,
                              const std::set<std::string>&      deploying,
                              const std::deque<LogEntry>&       log) {
    std::vector<JobPtr> jobs;
    { std::lock_guard<std::mutex> lk(state_.mutex); jobs = state_.pending_jobs; }

    int row = draw_page_header(cols, Page::Workers, wkrs, jobs.size());

    auto offline  = build_offline(wkrs, known, deploying);
    int  total_w  = (int)wkrs.size() + (int)offline.size();
    if (total_w > 0)
        sel_w_ = std::max(0, std::min(sel_w_, total_w - 1));

    row = draw_worker_table(row, cols, wkrs, offline, /*show_actions=*/true);

    // Per-worker log panel
    if (row < rows - 3) {
        for (int c = 0; c < cols; c++) mvaddch(row, c, '-');
        attron(A_BOLD);
        mvprintw(row, 0, " LOG ");
        attroff(A_BOLD);
        row++;

        // Determine which worker is selected
        std::string filter_worker_id;
        if (sel_w_ < (int)wkrs.size()) {
            filter_worker_id = wkrs[sel_w_].worker_id;
        } else if (sel_w_ < total_w) {
            // Offline worker — use display_name as worker_id key since it
            // hasn't connected yet; deploy logs use the name as worker_id tag
            filter_worker_id = offline[sel_w_ - (int)wkrs.size()].display_name;
        }

        draw_log_panel(row, rows - 1, cols, log, log_scroll_w_,
                       filter_worker_id, /*job_id=*/{});
    }
}

// ============================================================================
// Page 4 — Logs (full log with category filters)
// ============================================================================

void Tui::draw_page_logs(int rows, int cols, const std::deque<LogEntry>& log) {
    std::vector<WorkerDisplay> wkrs;
    std::vector<JobPtr> jobs;
    { std::lock_guard<std::mutex> lk(state_.mutex);
      wkrs = state_.workers;
      jobs = state_.pending_jobs; }

    int row = draw_page_header(cols, Page::Logs, wkrs, jobs.size());

    // Filter toggle bar
    auto toggle_label = [&](LogCategory cat, const char* label) {
        bool on = (log_filter_ & cat_bit(cat)) != 0;
        if (on) attron(A_BOLD);
        else    attron(A_DIM);
        printw("%s", label);
        if (on) attroff(A_BOLD);
        else    attroff(A_DIM);
        printw(" ");
    };

    draw_log_panel(row, rows - 1, cols, log, log_scroll_l_,
                   /*worker_id=*/{}, /*job_id=*/{}, log_filter_);
}

// ============================================================================
// draw_statusbar
// ============================================================================

void Tui::draw_statusbar(int rows, int cols) {
    attron(COLOR_PAIR(CP_HEADER));
    std::string bar;
    switch (page_) {
    case Page::Dashboard:
        bar = " [up/dn]Select  [Enter]Capability  [d]Deploy  [D]Deploy All  [1-4]Page  [Tab]Cycle  [e]Conf  [q]Quit";
        break;
    case Page::Jobs:
        bar = " [up/dn]Select  [Enter]Submit  [f]Force  [S]Strategy  [r]Refresh  [1-4]Page  [q]Quit";
        break;
    case Page::Workers:
        bar = " [up/dn]Select  [d]Deploy  [D]Deploy All  [k]Kill  [s]Status  [1-4]Page  [Tab]Cycle  [e]Conf  [q]Quit";
        break;
    case Page::Logs:
        bar = " [up/dn]Scroll  [J/W/N/H/D]Filter  [g]Bottom  [1-4]Page  [q]Quit";
        break;
    }
    bar.resize(cols, ' ');
    mvprintw(rows - 1, 0, "%s", bar.c_str());
    attroff(COLOR_PAIR(CP_HEADER));
}

// ============================================================================
// draw_modal
// ============================================================================

void Tui::draw_modal(int dh, int dw, int& dy, int& dx, const char* title) {
    int rows, cols;
    getmaxyx(stdscr, rows, cols);
    dy = std::max(0, (rows - dh) / 2);
    dx = std::max(0, (cols - dw) / 2);

    for (int r = dy; r < dy + dh && r < rows; r++) {
        for (int c = dx; c < dx + dw && c < cols; c++) {
            if (r == dy || r == dy + dh - 1) mvaddch(r, c, '-');
            else if (c == dx || c == dx + dw - 1) mvaddch(r, c, '|');
            else mvaddch(r, c, ' ');
        }
    }
    mvaddch(dy, dx, '+');
    mvaddch(dy, dx + dw - 1, '+');
    if (dy + dh - 1 < rows) {
        mvaddch(dy + dh - 1, dx, '+');
        mvaddch(dy + dh - 1, dx + dw - 1, '+');
    }
    if (title) {
        attron(A_BOLD);
        mvprintw(dy, dx + 2, " %s ", title);
        attroff(A_BOLD);
    }
}

}  // namespace clustr

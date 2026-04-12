#include "tui.h"
#include "tui_impl.h"

namespace clustr {

// ============================================================================
// input_line — single-line text field with cursor, called from dialogs
// ============================================================================

std::string Tui::input_line(int y, int x, int fw, const std::string& initial) {
    nodelay(stdscr, FALSE);
    noecho();
    curs_set(1);

    std::string buf    = initial;
    int         cur    = (int)buf.size();
    int         scroll = std::max(0, cur - fw + 1);

    auto clamp_scroll = [&]() {
        if (cur < scroll)          scroll = cur;
        if (cur >= scroll + fw)    scroll = cur - fw + 1;
        if (scroll < 0)            scroll = 0;
    };

    auto draw = [&]() {
        clamp_scroll();
        attron(COLOR_PAIR(CP_SEL));
        // Show the visible window of the buffer, padded to fw
        std::string disp = buf.size() > (size_t)scroll
                           ? buf.substr(scroll) : std::string{};
        if ((int)disp.size() > fw) disp.resize(fw);
        else                       disp.resize(fw, ' ');
        mvprintw(y, x, "%s", disp.c_str());
        // Show scroll indicator in the last cell when content is hidden
        if (scroll > 0)
            mvaddch(y, x, '<');
        if ((int)buf.size() > scroll + fw)
            mvaddch(y, x + fw - 1, '>');
        move(y, x + (cur - scroll));
        attroff(COLOR_PAIR(CP_SEL));
        refresh();
    };

    while (true) {
        draw();
        int ch = getch();
        if (ch == '\n' || ch == '\r' || ch == KEY_ENTER) break;
        if (ch == 27) { buf = initial; break; }
        if (ch == KEY_BACKSPACE || ch == 127 || ch == '\b') {
            if (cur > 0) { buf.erase(cur - 1, 1); cur--; }
        } else if (ch == KEY_DC) {
            if (cur < (int)buf.size()) buf.erase(cur, 1);
        } else if (ch == KEY_LEFT  && cur > 0)              cur--;
        else if (ch == KEY_RIGHT && cur < (int)buf.size()) cur++;
        else if (ch == KEY_HOME) cur = 0;
        else if (ch == KEY_END)  cur = (int)buf.size();
        else if (ch >= 32 && ch < 256) {
            buf.insert(cur, 1, (char)ch);
            cur++;
        }
    }

    curs_set(0);
    nodelay(stdscr, TRUE);
    return buf;
}

// ============================================================================
// handle_key — global keys first, then dispatch to per-page handler
// ============================================================================

bool Tui::handle_key(int ch) {
    // ── Global: quit ──────────────────────────────────────────────────────────
    if (ch == 'q' || ch == 'Q') {
        cmds_.quit();
        return false;
    }

    // ── Global: direct page jump ──────────────────────────────────────────────
    if (ch == '1') { page_ = Page::Dashboard; return true; }
    if (ch == '2') { page_ = Page::Jobs;      return true; }
    if (ch == '3') { page_ = Page::Workers;   return true; }
    if (ch == '4') { page_ = Page::Logs;      return true; }

    // ── Global: Tab cycles pages ──────────────────────────────────────────────
    if (ch == '\t') {
        switch (page_) {
        case Page::Dashboard: page_ = Page::Jobs;       break;
        case Page::Jobs:      page_ = Page::Workers;    break;
        case Page::Workers:   page_ = Page::Logs;       break;
        case Page::Logs:      page_ = Page::Dashboard;  break;
        }
        return true;
    }

    // ── Global: edit config ───────────────────────────────────────────────────
    if (ch == 'e' || ch == 'E') {
        dialog_edit_config();
        redraw();
        return true;
    }

    // ── Global: deploy all known workers ─────────────────────────────────────
    if (ch == 'D') {
        if (cmds_.deploy_worker) {
            std::vector<KnownWorker> known;
            { std::lock_guard<std::mutex> lk(state_.mutex); known = state_.known_workers; }
            for (const auto& kw : known)
                cmds_.deploy_worker(kw.name);
        }
        return true;
    }

    // ── Snapshot state needed by per-page handlers ────────────────────────────
    std::vector<WorkerDisplay> wkrs;
    std::vector<KnownWorker>   known;
    std::set<std::string>      deploying;
    {
        std::lock_guard<std::mutex> lk(state_.mutex);
        wkrs      = state_.workers;
        known     = state_.known_workers;
        deploying = state_.deploying_workers;
    }
    auto offline      = build_offline(wkrs, known, deploying);
    int  total_workers = (int)wkrs.size() + (int)offline.size();

    switch (page_) {
    case Page::Dashboard:
        handle_key_dashboard(ch, wkrs, total_workers);
        break;
    case Page::Jobs:
        handle_key_jobs(ch);
        break;
    case Page::Workers:
        handle_key_workers(ch, wkrs, offline);
        break;
    case Page::Logs:
        handle_key_logs(ch);
        break;
    }
    return true;
}

// ============================================================================
// handle_key_dashboard
// ============================================================================

void Tui::handle_key_dashboard(int ch,
                                const std::vector<WorkerDisplay>& wkrs,
                                int total_workers) {
    switch (ch) {
    case KEY_UP:
        if (sel_w_ > 0) { sel_w_--; dash_cap_expanded_ = false; }
        break;
    case KEY_DOWN:
        if (sel_w_ < total_workers - 1) { sel_w_++; dash_cap_expanded_ = false; }
        break;

    case '\n': case '\r': case KEY_ENTER:
        // Toggle capability panel; only for connected workers
        if (sel_w_ < (int)wkrs.size())
            dash_cap_expanded_ = !dash_cap_expanded_;
        break;

    case 'd': case 'D':
        if (cmds_.deploy_worker) {
            if (sel_w_ < (int)wkrs.size()) {
                const auto& w = wkrs[sel_w_];
                std::string name = w.display_name.empty() ? w.worker_id : w.display_name;
                dialog_deploy_confirm(name, w.ip);
            } else {
                // Build offline list to get the right entry
                std::vector<KnownWorker>   known;
                std::set<std::string>      deploying;
                { std::lock_guard<std::mutex> lk(state_.mutex);
                  known     = state_.known_workers;
                  deploying = state_.deploying_workers; }
                auto offline = build_offline(wkrs, known, deploying);
                int  idx     = sel_w_ - (int)wkrs.size();
                if (idx < (int)offline.size())
                    dialog_deploy_confirm(offline[idx].display_name, offline[idx].ip);
            }
        }
        break;

    default:
        break;
    }
}

// ============================================================================
// handle_key_jobs
// ============================================================================

void Tui::handle_key_jobs(int ch) {
    switch (ch) {
    case KEY_UP:
        if (ch == KEY_UP) {
            // Scroll log if we have files; otherwise move file selection
            if (log_scroll_j_ > 0) {
                log_scroll_j_--;
            } else if (sel_file_ > 0) {
                sel_file_--;
                log_scroll_j_ = 0;
            }
        }
        break;
    case KEY_DOWN:
        if (sel_file_ < (int)job_files_.size() - 1) {
            sel_file_++;
            log_scroll_j_ = 0;
        }
        break;

    // Scroll the log panel with PgUp/PgDn
    case KEY_PPAGE:
        log_scroll_j_ += 10;
        break;
    case KEY_NPAGE:
        log_scroll_j_ = std::max(0, log_scroll_j_ - 10);
        break;

    case '\n': case '\r': case KEY_ENTER:
        if (!job_files_.empty() && sel_file_ < (int)job_files_.size())
            dialog_submit_job_file("jobs/" + job_files_[sel_file_]);
        break;

    case 'f': case 'F':
        dialog_force_assign();
        break;

    case 'S':
        dialog_set_strategy();
        break;

    case 'r': case 'R':
        scan_job_files();
        break;

    case 'g': case 'G':
        log_scroll_j_ = 0;
        break;

    default:
        break;
    }
}

// ============================================================================
// handle_key_workers
// ============================================================================

void Tui::handle_key_workers(int ch,
                               const std::vector<WorkerDisplay>& wkrs,
                               const std::vector<WorkerDisplay>& offline) {
    int total_workers = (int)wkrs.size() + (int)offline.size();

    switch (ch) {
    case KEY_UP:
        if (sel_w_ > 0) { sel_w_--; log_scroll_w_ = 0; }
        break;
    case KEY_DOWN:
        if (sel_w_ < total_workers - 1) { sel_w_++; log_scroll_w_ = 0; }
        break;

    // Scroll worker log with PgUp/PgDn
    case KEY_PPAGE:
        log_scroll_w_ += 10;
        break;
    case KEY_NPAGE:
        log_scroll_w_ = std::max(0, log_scroll_w_ - 10);
        break;

    case 'g': case 'G':
        log_scroll_w_ = 0;
        break;

    case 'd': case 'D':
        if (cmds_.deploy_worker) {
            if (sel_w_ < (int)wkrs.size()) {
                const auto& w = wkrs[sel_w_];
                std::string name = w.display_name.empty() ? w.worker_id : w.display_name;
                dialog_deploy_confirm(name, w.ip);
            } else {
                int idx = sel_w_ - (int)wkrs.size();
                if (idx < (int)offline.size())
                    dialog_deploy_confirm(offline[idx].display_name, offline[idx].ip);
            }
        }
        break;

    case 'k': case 'K':
        if (sel_w_ < (int)wkrs.size())
            cmds_.kill_job(wkrs[sel_w_].worker_id);
        break;

    case 's':
        if (sel_w_ < (int)wkrs.size())
            cmds_.request_status(wkrs[sel_w_].worker_id);
        break;

    case 'S':
        dialog_set_strategy();
        break;

    default:
        break;
    }
}

// ============================================================================
// handle_key_logs
// ============================================================================

void Tui::handle_key_logs(int ch) {
    switch (ch) {
    case KEY_UP:
        log_scroll_l_++;
        break;
    case KEY_DOWN:
        log_scroll_l_ = std::max(0, log_scroll_l_ - 1);
        break;
    case KEY_PPAGE:
        log_scroll_l_ += 20;
        break;
    case KEY_NPAGE:
        log_scroll_l_ = std::max(0, log_scroll_l_ - 20);
        break;
    case 'g': case 'G':
        log_scroll_l_ = 0;
        break;

    // Category filter toggles
    case 'j': case 'J':
        log_filter_ ^= cat_bit(LogCategory::Job);
        break;
    case 'w': case 'W':
        log_filter_ ^= cat_bit(LogCategory::Worker);
        break;
    case 'n': case 'N':
        log_filter_ ^= cat_bit(LogCategory::Network);
        break;
    case 'h': case 'H':
        log_filter_ ^= cat_bit(LogCategory::Heartbeat);
        break;
    case 'd': case 'D':
        log_filter_ ^= cat_bit(LogCategory::Deploy);
        break;

    default:
        break;
    }
}

}  // namespace clustr

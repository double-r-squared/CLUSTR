#include "tui.h"
#include "tui_impl.h"
#include <fstream>
#include <filesystem>
#include <unordered_map>

namespace clustr {

// ============================================================================
// dialog_submit_job_file — file browser submit: auto-gen compile+run, pick worker
// ============================================================================

void Tui::dialog_submit_job_file(const std::string& filepath) {
    // Derive binary name from file stem (e.g. "jobs/matrix_multiply.cpp" → "matrix_multiply")
    namespace fs = std::filesystem;
    std::string stem = fs::path(filepath).stem().string();

    // Compile presets - default (index 0) is MPI with the zero-copy transport
    // and the inline (per-call) recv path.
    //
    // Suffix legend:
    //   -i  CLUSTR_RECV_INLINE   Phase 0..4 path. World-comm only.
    //   -c  CLUSTR_RECV_CENTRAL  Per-socket dispatch loop into per-key channels.
    //                            Required by Phase 5 sub-communicators.
    //
    // The full 4-way matrix exists so each transport can be benchmarked against
    // each recv path side by side (ROADMAP "Dual transport strategy" + Phase 5).
    struct Preset { const char* label; std::string cmd; };
    std::vector<Preset> presets = {
        { "MPI-zc-i",
          "g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED"
          " -DCLUSTR_TRANSPORT=CLUSTR_TRANSPORT_ZERO_COPY"
          " -DCLUSTR_RECV=CLUSTR_RECV_INLINE"
          " -I/var/tmp/clustr/include -I/var/tmp/clustr/asio_include"
          " " + stem + ".cpp -o " + stem + " -lpthread" },
        { "MPI-zc-c",
          "g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED"
          " -DCLUSTR_TRANSPORT=CLUSTR_TRANSPORT_ZERO_COPY"
          " -DCLUSTR_RECV=CLUSTR_RECV_CENTRAL"
          " -I/var/tmp/clustr/include -I/var/tmp/clustr/asio_include"
          " " + stem + ".cpp -o " + stem + " -lpthread" },
        { "MPI-pu-i",
          "g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED"
          " -DCLUSTR_TRANSPORT=CLUSTR_TRANSPORT_PACK_UNPACK"
          " -DCLUSTR_RECV=CLUSTR_RECV_INLINE"
          " -I/var/tmp/clustr/include -I/var/tmp/clustr/asio_include"
          " " + stem + ".cpp -o " + stem + " -lpthread" },
        { "MPI-pu-c",
          "g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED"
          " -DCLUSTR_TRANSPORT=CLUSTR_TRANSPORT_PACK_UNPACK"
          " -DCLUSTR_RECV=CLUSTR_RECV_CENTRAL"
          " -I/var/tmp/clustr/include -I/var/tmp/clustr/asio_include"
          " " + stem + ".cpp -o " + stem + " -lpthread" },
        { "Simple", "g++ -O2 -o " + stem + " " + stem + ".cpp" },
        { "Opt",    "g++ -std=c++20 -O3 -march=native -o " + stem
                    + " " + stem + ".cpp" },
    };
    int preset_idx = 0;

    std::string default_run = "./" + stem;

    // Auto-detect companion files (e.g. .py scripts) in the same directory.
    // Any non-C++ source file is offered as a companion to be transferred
    // alongside the .cpp to the worker.
    std::vector<std::string> companions;
    {
        fs::path src_dir = fs::path(filepath).parent_path();
        if (fs::is_directory(src_dir)) {
            for (auto& entry : fs::directory_iterator(src_dir)) {
                if (!entry.is_regular_file()) continue;
                std::string ext = entry.path().extension().string();
                if (ext == ".cpp" || ext == ".h" || ext == ".hpp") continue;
                companions.push_back(entry.path().string());
            }
            std::sort(companions.begin(), companions.end());
        }
    }

    // Snapshot connected workers for the worker picker
    std::vector<WorkerDisplay> wkrs;
    {
        std::lock_guard<std::mutex> lk(state_.mutex);
        wkrs = state_.workers;
    }

    // Use most of the terminal width so long compile commands fit
    int dw = std::max(80, std::min(COLS - 4, 140));
    int dh = 13 + std::min((int)wkrs.size(), 6);
    int dy, dx;
    draw_modal(dh, dw, dy, dx, "Submit Job");

    // fw is the editable field width (from column 11 to the right margin)
    int fw = dw - 13;

    // ── Compile preset selector ───────────────────────────────────────────────
    // Up/Down cycles presets; Enter drops into free-text edit; Esc aborts.
    // Hint is printed centered on the bottom border, not inside the form.
    static const char* PRESET_HINT = " up/dn: cycle preset   Enter: edit   Esc: cancel ";
    auto draw_hint = [&]() {
        int hlen = (int)strlen(PRESET_HINT);
        int hx   = dx + (dw - hlen) / 2;
        mvprintw(dy + dh - 1, hx, "%s", PRESET_HINT);
    };

    auto draw_compile_row = [&]() {
        mvprintw(dy + 1, dx + 2, "File:    ");
        std::string fp_disp = filepath;
        int fp_max = dw - 12;
        if ((int)fp_disp.size() > fp_max)
            fp_disp = "..." + fp_disp.substr(fp_disp.size() - fp_max + 3);
        mvprintw(dy + 1, dx + 11, "%s", fp_disp.c_str());

        // Preset label (no symbol — cycling is explained by the border hint)
        std::string label = std::string("[") + presets[preset_idx].label + "]";
        mvprintw(dy + 2, dx + 2, "Compile: %-11s", label.c_str());
        std::string cmd_disp = presets[preset_idx].cmd;
        int cmd_max = dw - 24;
        if ((int)cmd_disp.size() > cmd_max)
            cmd_disp = cmd_disp.substr(0, cmd_max - 3) + "...";
        attron(COLOR_PAIR(CP_SEL));
        std::string padded = cmd_disp;
        padded.resize(cmd_max, ' ');
        mvprintw(dy + 2, dx + 23, "%s", padded.c_str());
        attroff(COLOR_PAIR(CP_SEL));
        draw_hint();
        refresh();
    };

    draw_compile_row();
    nodelay(stdscr, FALSE);
    std::string compile;
    while (true) {
        int ch = getch();
        if (ch == 27) { nodelay(stdscr, TRUE); return; }
        if (ch == KEY_UP) {
            preset_idx = (preset_idx + (int)presets.size() - 1) % (int)presets.size();
            draw_compile_row();
        } else if (ch == KEY_DOWN) {
            preset_idx = (preset_idx + 1) % (int)presets.size();
            draw_compile_row();
        } else if (ch == '\n' || ch == '\r' || ch == KEY_ENTER) {
            // Restore bottom border and drop into free-text edit
            for (int c = dx + 1; c < dx + dw - 1; c++) mvaddch(dy + dh - 1, c, '-');
            mvprintw(dy + 2, dx + 2, "Compile: ");
            compile = input_line(dy + 2, dx + 11, fw, presets[preset_idx].cmd);
            break;
        }
    }
    nodelay(stdscr, TRUE);

    // ── Remaining fields ─────────────────────────────────────────────────────
    mvprintw(dy + 3, dx + 2, "Run:     ");
    std::string run = input_line(dy + 3, dx + 11, fw, default_run);
    mvprintw(dy + 4, dx + 2, "Ranks:   ");
    std::string ranks_str = input_line(dy + 4, dx + 11, 4, "1");
    int num_ranks = 1;
    try { num_ranks = std::max(1, std::stoi(ranks_str)); } catch (...) {}

    // Show companion files that will be transferred
    int extra_row = dy + 5;
    if (!companions.empty()) {
        mvprintw(extra_row, dx + 2, "Files:   ");
        std::string names;
        for (auto& c : companions) {
            if (!names.empty()) names += ", ";
            names += fs::path(c).filename().string();
        }
        int max_w = dw - 12;
        if ((int)names.size() > max_w) names = names.substr(0, max_w - 3) + "...";
        mvprintw(extra_row, dx + 11, "%s", names.c_str());
        extra_row++;
    }
    extra_row++; // blank line

    // Worker picker (only shown for single-rank jobs; MPI jobs auto-select)
    int picker_row = extra_row;
    if (num_ranks == 1) {
        mvprintw(picker_row, dx + 2, "Assign to:");
        mvprintw(picker_row + 1, dx + 4, "[0] Any idle worker (strategy decides)");
        for (int i = 0; i < (int)wkrs.size() && i < 6; i++) {
            std::string label = wkrs[i].display_name.empty()
                                ? wkrs[i].worker_id : wkrs[i].display_name;
            mvprintw(picker_row + 2 + i, dx + 4, "[%d] %-24s  %s  cap:%.3f",
                     i + 1,
                     pad(label, 24).c_str(),
                     pad(wkrs[i].state_str, 8).c_str(),
                     wkrs[i].capacity);
        }
    } else {
        mvprintw(picker_row, dx + 2, "MPI job: %d ranks - scheduler picks workers automatically",
                 num_ranks);
    }

    int prompt_row = num_ranks == 1
                     ? picker_row + 2 + std::min((int)wkrs.size(), 6)
                     : picker_row + 2;
    mvprintw(prompt_row, dx + 2, "Enter [0-%d] then [y] to submit, [Esc] cancel:",
             (int)wkrs.size());
    refresh();

    nodelay(stdscr, FALSE);
    int ch = getch();
    nodelay(stdscr, TRUE);

    if (ch == 27) return;  // Esc

    int wi = -1;  // -1 = any
    if (ch >= '1' && ch <= '9') {
        wi = ch - '1';
        if (wi >= (int)wkrs.size()) return;
    } else if (ch != '0' && ch != 'y' && ch != 'Y') {
        return;
    }

    // If they picked a number, wait for [y] confirm
    if (ch != 'y' && ch != 'Y') {
        mvprintw(prompt_row + 1, dx + 2, "[y] Confirm, [Esc] cancel");
        refresh();
        nodelay(stdscr, FALSE);
        int confirm = getch();
        nodelay(stdscr, TRUE);
        if (confirm != 'y' && confirm != 'Y') return;
    }

    std::string pinned = (wi >= 0 && wi < (int)wkrs.size())
                         ? wkrs[wi].worker_id : "";

    {
        std::lock_guard<std::mutex> lk(state_.mutex);
        state_.source_file     = filepath;
        state_.compile_cmd     = compile;
        state_.run_cmd         = run;
        state_.companion_files = companions;
        state_.num_ranks       = static_cast<uint32_t>(num_ranks);
    }
    cmds_.submit_job(pinned);
}

// ============================================================================
// dialog_force_assign — manually assign a queued job to a specific worker
// ============================================================================

void Tui::dialog_force_assign() {
    std::vector<WorkerDisplay> wkrs;
    std::vector<JobPtr>        jobs;
    {
        std::lock_guard<std::mutex> lk(state_.mutex);
        wkrs = state_.workers;
        jobs = state_.pending_jobs;
    }

    if (wkrs.empty() || jobs.empty()) {
        int dh = 5, dw = 50, dy, dx;
        draw_modal(dh, dw, dy, dx, "Force Assign");
        mvprintw(dy + 2, dx + 2, "No idle workers or pending jobs.");
        mvprintw(dy + 3, dx + 2, "[any key] Close");
        refresh();
        nodelay(stdscr, FALSE); getch(); nodelay(stdscr, TRUE);
        return;
    }

    int dh = std::min((int)jobs.size() + 7, 20), dw = 70, dy, dx;
    draw_modal(dh, dw, dy, dx, "Force Assign: Select Job");

    for (int i = 0; i < (int)jobs.size() && dy + 2 + i < dy + dh - 3; i++)
        mvprintw(dy + 2 + i, dx + 2, "[%d] %-12s  src:%s",
                 i + 1, jobs[i]->job_id.c_str(),
                 trunc(jobs[i]->source_file, 30).c_str());

    mvprintw(dy + dh - 2, dx + 2, "Enter job number [1-%d] or Esc:", (int)jobs.size());
    refresh();

    nodelay(stdscr, FALSE);
    echo(); curs_set(1);
    char jnum[8] = {};
    mvscanw(dy + dh - 2, dx + 33, (char*)"%4s", jnum);
    noecho(); curs_set(0);
    nodelay(stdscr, TRUE);

    int ji = atoi(jnum) - 1;
    if (ji < 0 || ji >= (int)jobs.size()) return;
    std::string job_id = jobs[ji]->job_id;

    draw_modal(dh, dw, dy, dx, "Force Assign: Select Worker");
    for (int i = 0; i < (int)wkrs.size() && dy + 2 + i < dy + dh - 3; i++) {
        std::string label = wkrs[i].display_name.empty()
                            ? wkrs[i].worker_id : wkrs[i].display_name;
        mvprintw(dy + 2 + i, dx + 2, "[%d] %-24s  %-12s  cap:%.3f",
                 i + 1, pad(label, 24).c_str(),
                 wkrs[i].state_str.c_str(), wkrs[i].capacity);
    }

    mvprintw(dy + dh - 2, dx + 2, "Enter worker number [1-%d] or Esc:", (int)wkrs.size());
    refresh();

    nodelay(stdscr, FALSE);
    echo(); curs_set(1);
    char wnum[8] = {};
    mvscanw(dy + dh - 2, dx + 37, (char*)"%4s", wnum);
    noecho(); curs_set(0);
    nodelay(stdscr, TRUE);

    int wi = atoi(wnum) - 1;
    if (wi < 0 || wi >= (int)wkrs.size()) return;

    cmds_.force_dispatch(job_id, wkrs[wi].worker_id);
}

// ============================================================================
// dialog_set_strategy
// ============================================================================

void Tui::dialog_set_strategy() {
    static const char* strategies[] = {
        "FirstAvailable", "CapacityWeighted", "RoundRobin", "Manual"
    };
    constexpr int N = 4;

    int dh = N + 5, dw = 40, dy, dx;
    draw_modal(dh, dw, dy, dx, "Select Strategy");

    for (int i = 0; i < N; i++)
        mvprintw(dy + 2 + i, dx + 2, "[%d] %s", i + 1, strategies[i]);

    mvprintw(dy + dh - 2, dx + 2, "Enter [1-%d] or Esc:", N);
    refresh();

    nodelay(stdscr, FALSE);
    int ch = getch();
    nodelay(stdscr, TRUE);

    int idx = ch - '1';
    if (idx >= 0 && idx < N)
        cmds_.set_strategy(strategies[idx]);
}

// ============================================================================
// dialog_edit_config — inline editor for system.conf
// ============================================================================

void Tui::dialog_edit_config() {
    // raw_idx links each editable field back to its exact line in raw_lines,
    // so fields with the same key in different sections save independently.
    struct Field { std::string key; std::string value; int raw_idx; };
    std::vector<Field>       fields;
    std::vector<std::string> raw_lines;

    std::ifstream in(conf_path_);
    if (in) {
        std::string line;
        while (std::getline(in, line)) {
            int idx = (int)raw_lines.size();
            raw_lines.push_back(line);
            if (line.empty() || line[0] == '#' || line[0] == '[') continue;
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            while (!key.empty() && key.back() == ' ') key.pop_back();
            std::string val = line.substr(eq + 1);
            auto hash = val.find('#');
            if (hash != std::string::npos) val = val.substr(0, hash);
            while (!val.empty() && (val.front()==' '||val.front()=='\t')) val.erase(0,1);
            while (!val.empty() && (val.back() ==' '||val.back() =='\t')) val.pop_back();
            fields.push_back({key, val, idx});
        }
    }

    if (fields.empty()) {
        int dh = 5, dw = 50, dy, dx;
        draw_modal(dh, dw, dy, dx, "Config");
        mvprintw(dy + 2, dx + 2, "Could not read: %s", conf_path_.c_str());
        mvprintw(dy + 3, dx + 2, "[any key]");
        refresh();
        nodelay(stdscr, FALSE); getch(); nodelay(stdscr, TRUE);
        return;
    }

    int sel = 0;
    int dh  = std::min((int)fields.size() + 5, 24);
    int dw  = 72;
    int dy, dx;

    while (true) {
        draw_modal(dh, dw, dy, dx, "Edit Config");
        int val_x = dx + 24;
        int val_w = dw - 26;

        for (int i = 0; i < (int)fields.size() && i < dh - 4; i++) {
            bool is_sel = (i == sel);
            if (is_sel) attron(A_BOLD);
            mvprintw(dy + 2 + i, dx + 2, "%-20s", fields[i].key.c_str());
            if (is_sel) attroff(A_BOLD);

            if (is_sel) attron(COLOR_PAIR(CP_SEL));
            std::string v = fields[i].value;
            v.resize(val_w, ' ');
            mvprintw(dy + 2 + i, val_x, "%s", v.c_str());
            if (is_sel) attroff(COLOR_PAIR(CP_SEL));
        }

        mvprintw(dy + dh - 2, dx + 2,
                 "[up/dn]Nav  [Enter]Edit  [a]Add Worker  [s]Save  [Esc]Cancel");
        refresh();

        nodelay(stdscr, FALSE);
        int ch = getch();
        nodelay(stdscr, TRUE);

        if (ch == 27) break;
        if (ch == KEY_UP   && sel > 0) sel--;
        if (ch == KEY_DOWN && sel < (int)fields.size() - 1) sel++;
        if (ch == '\n' || ch == '\r')
            fields[sel].value = input_line(dy + 2 + sel, val_x, val_w, fields[sel].value);

        if (ch == 'a' || ch == 'A') {
            dialog_add_worker();
            break;  // close editor; user can reopen to see and edit the new entry
        }

        if (ch == 's' || ch == 'S') {
            // Build index → updated line map, keyed by exact raw line position
            std::unordered_map<int, std::pair<std::string,std::string>> updates;
            for (const auto& f : fields)
                updates[f.raw_idx] = {f.key, f.value};

            std::ofstream out(conf_path_);
            for (int i = 0; i < (int)raw_lines.size(); i++) {
                auto it = updates.find(i);
                if (it != updates.end())
                    out << it->second.first << "=" << it->second.second << '\n';
                else
                    out << raw_lines[i] << '\n';
            }
            if (cmds_.reload_config) cmds_.reload_config();
            break;
        }
    }
}

// ============================================================================
// dialog_add_worker — append a new [section] to system.conf
// ============================================================================

void Tui::dialog_add_worker() {
    int dh = 12, dw = 62, dy, dx;
    draw_modal(dh, dw, dy, dx, "Add Worker");

    int lx = dx + 20, lw = dw - 22;

    mvprintw(dy + 2, dx + 2, "Name (section):");
    std::string name = input_line(dy + 2, lx, lw, "");
    if (name.empty()) return;

    mvprintw(dy + 3, dx + 2, "Host (IP):     ");
    std::string host = input_line(dy + 3, lx, lw, "");
    if (host.empty()) return;

    mvprintw(dy + 4, dx + 2, "User:          ");
    std::string user = input_line(dy + 4, lx, lw, "");
    if (user.empty()) return;

    mvprintw(dy + 6, dx + 2, "ssh_key_path:  ");
    std::string key = input_line(dy + 6, lx, lw, "~/.ssh/id_rsa");

    mvprintw(dy + 8, dx + 2, "[y] Add   [Esc/other] Cancel");
    refresh();

    nodelay(stdscr, FALSE);
    int ch = getch();
    nodelay(stdscr, TRUE);
    if (ch != 'y' && ch != 'Y') return;

    {
        std::ofstream out(conf_path_, std::ios::app);
        out << "\n[" << name << "]\n";
        out << "deploy_host=" << host << "\n";
        out << "deploy_user=" << user << "\n";
        if (key != "~/.ssh/id_rsa" && !key.empty())
            out << "ssh_key_path=" << key << "\n";
    }
    if (cmds_.reload_config) cmds_.reload_config();
}

// ============================================================================
// dialog_deploy_confirm — confirm before triggering a deploy subprocess
// ============================================================================

void Tui::dialog_deploy_confirm(const std::string& name, const std::string& ip) {
    if (!cmds_.deploy_worker) return;

    int dh = 7, dw = 60, dy, dx;
    draw_modal(dh, dw, dy, dx, "Deploy Worker");

    mvprintw(dy + 2, dx + 2, "Worker: %s", name.c_str());
    mvprintw(dy + 3, dx + 2, "Host:   %s", ip.c_str());
    mvprintw(dy + 5, dx + 2, "[y] Deploy   [Esc/n] Cancel");
    refresh();

    nodelay(stdscr, FALSE);
    int ch = getch();
    nodelay(stdscr, TRUE);

    if (ch == 'y' || ch == 'Y')
        cmds_.deploy_worker(name);
}

}  // namespace clustr

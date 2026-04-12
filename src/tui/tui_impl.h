#pragma once

// ============================================================================
// tui_impl.h — private helpers shared across tui_*.cpp files.
//
// Never include this outside of src/tui/.
// ============================================================================

#include <ncurses.h>
#include <string>
#include <ctime>
#include <chrono>
#include <cstdio>

namespace clustr {

// ── Color pair IDs ────────────────────────────────────────────────────────────

enum {
    CP_HEADER   = 1,  // white on blue  (header / status bar)
    CP_IDLE     = 2,  // green          (worker IDLE)
    CP_RUNNING  = 3,  // yellow         (worker RUNNING)
    CP_FAILED   = 4,  // red            (worker FAILED / OFFLINE)
    CP_SEL      = 5,  // black on white (selected row)
    CP_LOG_TS   = 6,  // dim white      (log timestamps)
    CP_SETUP    = 7,  // magenta        (compiling / file transfer / DEPLOYING)
    CP_DIM      = 8,  // white dim

    // Log line colors
    CP_LOG_HANDSHAKE = 9,   // yellow  — CONNECT, HELLO, GOODBYE, HEARTBEAT, READY
    CP_LOG_TRANSFER  = 10,  // cyan    — FILE_DATA, FILE_ACK, DEPLOY output
    CP_LOG_EXEC      = 11,  // magenta — COMPILE, COMPILED, DISPATCH, QUEUED, RUN
    CP_LOG_SPAWN     = 12,  // yellow bold — RUNNING, PROCESS_SPAWNED
    CP_LOG_SUCCESS   = 13,  // green   — TASK_RESULT SUCCESS, DEPLOY Complete
    CP_LOG_ERROR     = 14,  // red     — FAILED, ERROR, CANCEL, KILL
    CP_LOG_OUTPUT    = 15,  // cyan dim — "> " job output lines
};

// ── String helpers ────────────────────────────────────────────────────────────

inline std::string trunc(const std::string& s, int w) {
    if ((int)s.size() <= w) return s;
    return s.substr(0, w - 3) + "...";
}

inline std::string pad(std::string s, int w) {
    if ((int)s.size() > w) s.resize(w);
    else s.append(w - s.size(), ' ');
    return s;
}

inline std::string now_hms() {
    auto t = std::time(nullptr);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    return buf;
}

inline std::string elapsed_str(
    const std::chrono::steady_clock::time_point& start)
{
    if (start == std::chrono::steady_clock::time_point{}) return "        ";
    auto s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start).count();
    char buf[16];
    snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld",
             (long long)s / 3600,
             (long long)(s % 3600) / 60,
             (long long)s % 60);
    return buf;
}

// ── Color selectors ───────────────────────────────────────────────────────────

inline int state_color(const std::string& s) {
    if (s == "IDLE")     return CP_IDLE;
    if (s == "RUNNING")  return CP_RUNNING;
    if (s == "FAILED")   return CP_FAILED;
    if (s == "OFFLINE")  return CP_FAILED;
    if (s == "COMPILED" || s == "FILE_READY" ||
        s == "CONNECTED" || s == "DEPLOYING")
        return CP_SETUP;
    return CP_DIM;
}

// Return {color_pair, attribute} for a log line based on its [TAG] prefix.
inline std::pair<int,int> log_color(const std::string& text) {
    if (text.size() >= 3 && text[0] == '[') {
        auto close = text.find(']');
        if (close != std::string::npos) {
            std::string tag = text.substr(1, close - 1);

            if (tag == "CONNECT" || tag == "HELLO" || tag == "GOODBYE" ||
                tag == "HEARTBEAT" || tag == "READY" || tag == "DISCONNECT")
                return {CP_LOG_HANDSHAKE, 0};

            if (tag == "FILE_DATA" || tag == "FILE_ACK")
                return {CP_LOG_TRANSFER, 0};

            if (tag == "COMPILE" || tag == "COMPILED" ||
                tag == "DISPATCH" || tag == "QUEUED" || tag == "RUN")
                return {CP_LOG_EXEC, 0};

            if (tag == "RUNNING" || tag == "PROCESS_SPAWNED")
                return {CP_LOG_SPAWN, A_BOLD};

            if (tag == "TASK_RESULT") {
                if (text.find("SUCCESS") != std::string::npos)
                    return {CP_LOG_SUCCESS, A_BOLD};
                return {CP_LOG_ERROR, A_BOLD};
            }

            if (tag == "FAILED" || tag == "ERROR" ||
                tag == "KILL"   || tag == "CANCEL")
                return {CP_LOG_ERROR, A_BOLD};

            if (tag == "DEPLOY" ||
                (tag.size() > 7 && tag.substr(0, 7) == "DEPLOY:")) {
                if (text.find("Complete") != std::string::npos)
                    return {CP_LOG_SUCCESS, A_BOLD};
                if (text.find("FAILED") != std::string::npos ||
                    text.find("ERROR")  != std::string::npos)
                    return {CP_LOG_ERROR, A_BOLD};
                return {CP_LOG_TRANSFER, 0};
            }

            if (tag == "STATUS" || tag == "STRATEGY" || tag == "FORCE")
                return {CP_DIM, A_DIM};
        }
    }

    // Job output lines ("  > ")
    if (text.size() >= 4 && text[0] == ' ' && text[1] == ' ' &&
        text[2] == '>' && text[3] == ' ')
        return {CP_LOG_OUTPUT, A_DIM};

    return {CP_DIM, A_DIM};
}

}  // namespace clustr

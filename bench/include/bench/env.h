#pragma once

// bench/env.h - environment snapshot (captured once at process start).
//
// Honest benchmarking demands that every results file carries enough
// context to reproduce the measurement. Hardware, compiler, transport
// flags, and build type all change the numbers; if the file doesn't say
// which was used, next week's "regression" may just be a different -O
// flag.
//
// This is best-effort: missing fields are reported as "unknown" rather
// than failing. The benchmark itself must not depend on any of these.

#include "bench/json_writer.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#ifdef __APPLE__
#  include <sys/sysctl.h>
#  include <sys/utsname.h>
#endif
#ifdef __linux__
#  include <sys/utsname.h>
#  include <fstream>
#endif

namespace bench {

inline std::string shell_capture(const char* cmd) {
    std::array<char, 256> buf{};
    std::string out;
    FILE* p = popen(cmd, "r");
    if (!p) return "unknown";
    while (fgets(buf.data(), buf.size(), p)) out.append(buf.data());
    pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
        out.pop_back();
    return out.empty() ? std::string("unknown") : out;
}

inline std::string detect_cpu_model() {
#ifdef __APPLE__
    char buf[256] = {};
    std::size_t len = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0)
        return buf;
    return "unknown";
#elif defined(__linux__)
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("model name") == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos)
                return line.substr(pos + 2);
        }
    }
    return "unknown";
#else
    return "unknown";
#endif
}

inline std::string detect_os() {
#if defined(__APPLE__) || defined(__linux__)
    utsname u{};
    if (uname(&u) == 0)
        return std::string(u.sysname) + " " + u.release + " " + u.machine;
#endif
    return "unknown";
}

inline std::string detect_compiler() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
    return "unknown";
#endif
}

inline std::string now_iso8601() {
    std::time_t t = std::time(nullptr);
    std::tm tm_buf{};
#if defined(_WIN32)
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return buf;
}

inline ConfigMap capture_env() {
    ConfigMap m;
    m["timestamp_utc"]   = now_iso8601();
    m["hostname"]        = shell_capture("hostname");
    m["cpu_model"]       = detect_cpu_model();
    m["cpu_threads"]     = static_cast<std::int64_t>(std::thread::hardware_concurrency());
    m["os"]              = detect_os();
    m["compiler"]        = detect_compiler();
    m["git_sha"]         = shell_capture("git rev-parse --short HEAD 2>/dev/null");

#ifdef __OPTIMIZE__
    m["optimize"]        = true;
#else
    m["optimize"]        = false;
#endif
    return m;
}

}  // namespace bench

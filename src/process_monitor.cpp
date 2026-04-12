#include "process_monitor.h"
#include <signal.h>
#include <cstring>
#include <cstdio>
#include <string>

#ifdef __APPLE__
#include <libproc.h>
#endif

#ifdef __linux__
#include <fstream>
#include <sstream>
#endif

namespace clustr {

// ============================================================================
// query_process — platform-split implementation
// ============================================================================

ProcessStatusRespPayload query_process(uint32_t pid) {
    ProcessStatusRespPayload resp{};
    resp.pid         = pid;
    resp.cpu_percent = 0.0f;  // Accurate CPU% requires sampling over time;
                               // returning 0 for now — future improvement.
    resp.mem_bytes   = 0;
    resp.state       = static_cast<uint8_t>(ProcessState::UNKNOWN);

#ifdef __APPLE__
    struct proc_taskinfo info{};
    int ret = proc_pidinfo(static_cast<int>(pid),
                           PROC_PIDTASKINFO, 0,
                           &info, sizeof(info));
    if (ret <= 0) {
        resp.state = static_cast<uint8_t>(ProcessState::DEAD);
        return resp;
    }
    resp.mem_bytes = info.pti_resident_size;
    resp.state     = static_cast<uint8_t>(ProcessState::RUNNING);

#elif defined(__linux__)
    // ---- State from /proc/[pid]/stat ----
    std::string stat_path = "/proc/" + std::to_string(pid) + "/stat";
    FILE* stat_file = fopen(stat_path.c_str(), "r");
    if (!stat_file) {
        resp.state = static_cast<uint8_t>(ProcessState::DEAD);
        return resp;
    }

    int   pid_f  = 0;
    char  comm[256]{};
    char  state_c = 0;
    (void)fscanf(stat_file, "%d %255s %c", &pid_f, comm, &state_c);
    fclose(stat_file);

    switch (state_c) {
        case 'R': resp.state = static_cast<uint8_t>(ProcessState::RUNNING);  break;
        case 'S':
        case 'D': resp.state = static_cast<uint8_t>(ProcessState::SLEEPING); break;
        case 'T': resp.state = static_cast<uint8_t>(ProcessState::STOPPED);  break;
        case 'Z': resp.state = static_cast<uint8_t>(ProcessState::ZOMBIE);   break;
        default:  resp.state = static_cast<uint8_t>(ProcessState::UNKNOWN);  break;
    }

    // ---- Memory from /proc/[pid]/status ----
    std::string status_path = "/proc/" + std::to_string(pid) + "/status";
    std::ifstream status_file(status_path);
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            uint64_t kb = 0;
            sscanf(line.c_str(), "VmRSS: %lu kB", &kb);
            resp.mem_bytes = kb * 1024;
            break;
        }
    }
#endif

    return resp;
}

// ============================================================================
// kill_process
// ============================================================================

bool kill_process(uint32_t pid, uint8_t sig) {
    return ::kill(static_cast<pid_t>(pid), static_cast<int>(sig)) == 0;
}

// ============================================================================
// make_status_resp_msg
// ============================================================================

Message make_status_resp_msg(const ProcessStatusRespPayload& status, uint32_t msg_id) {
    Message msg;
    msg.type             = MessageType::PROCESS_STATUS_RESP;
    msg.message_id       = msg_id;
    msg.protocol_version = PROTOCOL_VERSION;
    msg.payload.resize(sizeof(ProcessStatusRespPayload));
    std::memcpy(msg.payload.data(), &status, sizeof(ProcessStatusRespPayload));
    return msg;
}

}  // namespace clustr

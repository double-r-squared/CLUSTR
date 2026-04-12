#pragma once

#include "protocol.h"

namespace clustr {

// ============================================================================
// Process Monitoring — Worker Side
// ============================================================================

// Query resource usage for a running process.
// Returns a populated ProcessStatusRespPayload.
// On macOS: uses proc_pidinfo() from <libproc.h>.
// On Linux: reads /proc/[pid]/stat and /proc/[pid]/status.
// If the process is not found, state is set to ProcessState::DEAD.
ProcessStatusRespPayload query_process(uint32_t pid);

// Send a POSIX signal to a process.
// Common values: 15 (SIGTERM) for graceful shutdown, 9 (SIGKILL) for immediate.
// Returns true on success, false if the process does not exist or permission denied.
bool kill_process(uint32_t pid, uint8_t sig);

// Build a PROCESS_STATUS_RESP message from a query result.
Message make_status_resp_msg(const ProcessStatusRespPayload& status, uint32_t msg_id);

}  // namespace clustr

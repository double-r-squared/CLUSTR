#pragma once

#include "protocol.h"
#include <functional>
#include <string>

namespace clustr {

// ============================================================================
// Remote Execution — Worker Side
// ============================================================================

// Run a command synchronously (used for compilation).
// Forks a child process, captures stdout and stderr via pipes, waits for
// completion or timeout, then returns an EXEC_RESULT message.
// The working_dir field in cmd is used as the child's cwd if non-empty.
// Never returns a dangling process — child is killed on timeout.
Message exec_sync(const ExecCmdPayload& cmd, uint32_t msg_id);

// Spawn a command in the background (used for running the compiled binary).
// Forks a child process, returns PROCESS_SPAWNED immediately with the PID.
// A detached thread waits for the child to exit and calls on_exit(exit_code).
// The caller is responsible for posting on_exit work back onto the IO thread
// if it needs to send network messages.
// on_exit receives (exit_code, captured_stdout+stderr).
Message spawn_async(const ExecCmdPayload& cmd,
                    uint32_t msg_id,
                    std::function<void(int32_t exit_code, std::string output)> on_exit);

}  // namespace clustr

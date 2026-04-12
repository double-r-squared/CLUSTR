#include "remote_exec.h"
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <thread>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>

namespace clustr {

// ============================================================================
// Internal helpers
// ============================================================================

// Split a shell command string into tokens for execvp.
// Handles space/tab delimiters. Does not handle quotes or escaping —
// commands are internally generated so this is sufficient.
static std::vector<std::string> tokenize(const std::string& cmd) {
    std::vector<std::string> tokens;
    std::string tok;
    for (char c : cmd) {
        if (c == ' ' || c == '\t') {
            if (!tok.empty()) { tokens.push_back(tok); tok.clear(); }
        } else {
            tok += c;
        }
    }
    if (!tok.empty()) tokens.push_back(tok);
    return tokens;
}

// Read all available bytes from a non-blocking fd into buf.
// Stops on EAGAIN/EWOULDBLOCK (no more data) or error.
static void drain(int fd, std::vector<uint8_t>& buf) {
    char tmp[4096];
    ssize_t n;
    while ((n = read(fd, tmp, sizeof(tmp))) > 0)
        buf.insert(buf.end(), tmp, tmp + n);
}

// Build an EXEC_RESULT message from collected output.
static Message build_exec_result(int32_t exit_code,
                                  const std::vector<uint8_t>& out,
                                  const std::vector<uint8_t>& err,
                                  uint32_t msg_id) {
    ExecResultPayload header{};
    header.exit_code   = exit_code;
    header.stdout_size = static_cast<uint32_t>(out.size());
    header.stderr_size = static_cast<uint32_t>(err.size());

    Message msg;
    msg.type             = MessageType::EXEC_RESULT;
    msg.message_id       = msg_id;
    msg.protocol_version = PROTOCOL_VERSION;
    msg.payload.resize(sizeof(ExecResultPayload) + out.size() + err.size());

    size_t off = 0;
    std::memcpy(msg.payload.data() + off, &header, sizeof(ExecResultPayload));
    off += sizeof(ExecResultPayload);
    if (!out.empty()) std::memcpy(msg.payload.data() + off, out.data(), out.size());
    off += out.size();
    if (!err.empty()) std::memcpy(msg.payload.data() + off, err.data(), err.size());

    return msg;
}

// ============================================================================
// exec_sync — run command, wait for completion, return EXEC_RESULT
// ============================================================================

Message exec_sync(const ExecCmdPayload& cmd, uint32_t msg_id) {
    std::string command(cmd.command);
    std::string working_dir(cmd.working_dir);
    uint32_t timeout_sec = cmd.timeout_seconds > 0 ? cmd.timeout_seconds : 300;

    auto tokens = tokenize(command);
    if (tokens.empty())
        throw std::runtime_error("exec_sync: empty command");

    // Create pipes for stdout and stderr
    int out_pipe[2], err_pipe[2];
    if (pipe(out_pipe) < 0 || pipe(err_pipe) < 0)
        throw std::runtime_error("exec_sync: pipe() failed");

    pid_t pid = fork();
    if (pid < 0) {
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        throw std::runtime_error("exec_sync: fork() failed");
    }

    if (pid == 0) {
        // ---- Child ----
        // Wire pipes to stdout/stderr
        close(out_pipe[0]);
        close(err_pipe[0]);
        dup2(out_pipe[1], STDOUT_FILENO);
        dup2(err_pipe[1], STDERR_FILENO);
        close(out_pipe[1]);
        close(err_pipe[1]);

        if (!working_dir.empty())
            (void)chdir(working_dir.c_str());

        std::vector<char*> argv;
        for (auto& t : tokens) argv.push_back(const_cast<char*>(t.c_str()));
        argv.push_back(nullptr);

        execvp(argv[0], argv.data());
        // execvp only returns on failure
        const char* err_msg = "execvp failed\n";
        (void)write(STDERR_FILENO, err_msg, strlen(err_msg));
        _exit(127);
    }

    // ---- Parent ----
    close(out_pipe[1]);
    close(err_pipe[1]);

    // Set read ends non-blocking so drain() doesn't block
    fcntl(out_pipe[0], F_SETFL, O_NONBLOCK);
    fcntl(err_pipe[0], F_SETFL, O_NONBLOCK);

    std::vector<uint8_t> stdout_buf, stderr_buf;
    int status = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);

    while (true) {
        struct pollfd fds[2];
        fds[0] = {out_pipe[0], POLLIN, 0};
        fds[1] = {err_pipe[0], POLLIN, 0};

        auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();

        if (remaining_ms <= 0) {
            // Timeout — kill child and report error
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            close(out_pipe[0]);
            close(err_pipe[0]);
            std::string msg = "Timed out after " + std::to_string(timeout_sec) + "s";
            std::vector<uint8_t> err_bytes(msg.begin(), msg.end());
            return build_exec_result(-1, {}, err_bytes, msg_id);
        }

        // Poll with a short cap so we check waitpid regularly
        int poll_ms = static_cast<int>(std::min<long>(remaining_ms, 100L));
        poll(fds, 2, poll_ms);

        if (fds[0].revents & POLLIN) drain(out_pipe[0], stdout_buf);
        if (fds[1].revents & POLLIN) drain(err_pipe[0], stderr_buf);

        // Check if child exited without blocking
        pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid) {
            // Drain any remaining output after exit
            drain(out_pipe[0], stdout_buf);
            drain(err_pipe[0], stderr_buf);
            break;
        }
    }

    close(out_pipe[0]);
    close(err_pipe[0]);

    int32_t exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return build_exec_result(exit_code, stdout_buf, stderr_buf, msg_id);
}

// ============================================================================
// spawn_async — fork and return immediately with PROCESS_SPAWNED
// ============================================================================

Message spawn_async(const ExecCmdPayload& cmd,
                    uint32_t msg_id,
                    std::function<void(int32_t, std::string)> on_exit) {
    std::string command(cmd.command);
    std::string working_dir(cmd.working_dir);
    std::string task_id(cmd.task_id);

    auto tokens = tokenize(command);
    if (tokens.empty())
        throw std::runtime_error("spawn_async: empty command");

    // Log file captures all job output so we can send it back to the scheduler.
    // Written to work_dir (same place as compiled binary) for easy cleanup.
    std::string log_path = (working_dir.empty() ? "/tmp" : working_dir)
                           + "/" + task_id + ".log";

    pid_t pid = fork();
    if (pid < 0)
        throw std::runtime_error("spawn_async: fork() failed");

    if (pid == 0) {
        // ---- Child ----
        int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) { dup2(devnull, STDIN_FILENO); close(devnull); }

        // Redirect stdout + stderr to the log file
        int logfd = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (logfd >= 0) {
            dup2(logfd, STDOUT_FILENO);
            dup2(logfd, STDERR_FILENO);
            close(logfd);
        }

        if (!working_dir.empty())
            (void)chdir(working_dir.c_str());

        std::vector<char*> argv;
        for (auto& t : tokens) argv.push_back(const_cast<char*>(t.c_str()));
        argv.push_back(nullptr);

        execvp(argv[0], argv.data());
        _exit(127);
    }

    // ---- Parent ----
    // Detach a waiter thread — it blocks on waitpid then fires the callback.
    // on_exit is responsible for posting back onto the ASIO io_context
    // before touching any network objects.
    std::thread([pid, log_path, on_exit]() {
        int status = 0;
        waitpid(pid, &status, 0);
        int32_t exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

        // Read captured output from log file
        std::string output;
        if (FILE* f = fopen(log_path.c_str(), "r")) {
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
                output.append(buf, n);
            fclose(f);
        }

        on_exit(exit_code, std::move(output));
    }).detach();

    // Build PROCESS_SPAWNED
    ProcessSpawnedPayload spawned{};
    std::strncpy(spawned.task_id, task_id.c_str(), sizeof(spawned.task_id) - 1);
    spawned.pid = static_cast<uint32_t>(pid);

    Message msg;
    msg.type             = MessageType::PROCESS_SPAWNED;
    msg.message_id       = msg_id;
    msg.protocol_version = PROTOCOL_VERSION;
    msg.payload.resize(sizeof(ProcessSpawnedPayload));
    std::memcpy(msg.payload.data(), &spawned, sizeof(ProcessSpawnedPayload));

    return msg;
}

}  // namespace clustr

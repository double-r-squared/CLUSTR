# CLAUDE.md — CLUSTR

## Project Overview

Distributed HPC cluster scheduler in C++20. Master-worker architecture with custom binary protocol over ASIO TCP. Minimal dependencies (ASIO only).

## Build & Run

```bash
# Build
mkdir -p build && cd build
cmake ..
cmake --build . --parallel

# Run scheduler (on master)
./scheduler <source_file> <compile_cmd> <run_cmd> [port]
# Example: ./scheduler job.cpp "g++ -O2 -o job job.cpp" "./job" 9999

# Deploy worker to remote (from master)
./scripts/deploy.sh [config_path]

# Run tests
./test_protocol [--verbose]
```

## Architecture

| Directory | Purpose |
|-----------|---------|
| `include/` | Headers (protocol, registry, config, file_transfer, remote_exec, process_monitor) |
| `src/` | Implementation matching headers |
| `server/` | Scheduler main() |
| `client/` | Worker main() |
| `tests/` | test_protocol only |
| `scripts/` | deploy.sh (SSH bootstrap) |

**Key constraint:** Credentials (SSH) only for initial deployment. After worker runs, all communication is TCP with no credentials.

## Core Data Flow

```
Scheduler                           Worker
    │                                   │
    ├────────── FILE_DATA ─────────────►│ (source + CRC32)
    │◄───────── FILE_ACK ───────────────┤
    ├────────── EXEC_CMD (compile) ────►│ (sync)
    │◄───────── EXEC_RESULT ────────────┤
    ├────────── EXEC_CMD (run) ────────►│ (async, detached)
    │◄───────── PROCESS_SPAWNED ────────┤ (PID)
    │                                   │ [job runs]
    │◄───────── TASK_RESULT ────────────┤
```

## State Machine (WorkerEntry)

```
CONNECTED → FILE_READY → COMPILED → RUNNING → IDLE
                ↓            ↓          ↓
              FAILED (any error, stderr captured)
```

## Key Conventions

- **Primary key:** `worker_id` (not IP) — IP can collide on restart
- **Work queue:** Holds `shared_ptr<WorkerEntry>` (prevents dangling)
- **Synchronous setup phase:** Worker proves it can compile + run before entering queue
- **Source transfer, not binary:** Heterogeneous nodes (ARM + x86) need native compilation
- **`working_dir` empty in EXEC_CMD:** Worker substitutes its configured `work_dir`

## Message Types (Binary Protocol, Big-Endian)

| Type | Direction | Payload |
|------|-----------|---------|
| `HELLO` (0x01) | Worker→Sched | 80 bytes (worker_id, version) |
| `CAPABILITY_REPORT` (0x03) | Worker→Sched | 308 bytes (hardware + benchmarks) |
| `FILE_DATA` (0x09) | Sched→Worker | filename + size + CRC32 + bytes |
| `FILE_ACK` (0x0A) | Worker→Sched | filename + success |
| `EXEC_CMD` (0x0B) | Sched→Worker | task_id + command + timeout + detach |
| `EXEC_RESULT` (0x0C) | Worker→Sched | exit_code + stdout + stderr |
| `PROCESS_SPAWNED` (0x0D) | Worker→Sched | task_id + PID |
| `PROCESS_KILL` (0x0E) | Sched→Worker | PID + signal |
| `PROCESS_STATUS_RESP` (0x10) | Worker→Sched | PID + CPU% + memory + state |

**Frame overhead:** 15 bytes (size + version + type + msg_id + CRC32)

## Configuration (`clustr.conf`)

Key-value with `#` comments. Loaded by both scheduler and worker.

| Key | Required | Default |
|-----|----------|---------|
| `scheduler_ip` | Yes | - |
| `scheduler_port` | No | 9999 |
| `work_dir` | No | `/tmp/clustr` |
| `deploy_user` | For deploy | - |
| `deploy_host` | For deploy | - |
| `ssh_key_path` | No | `~/.ssh/id_rsa` |

## Gotchas

1. **CRC32 mismatch** → connection closes; worker reconnects with backoff
2. **`spawn_async` callback** runs on background thread → must use `asio::post(io_context_, ...)` before touching network
3. **macOS needs `-lproc`** for `proc_pidinfo()` (already in CMakeLists.txt)
4. **CPU percentage is 0.0** — accurate measurement requires two samples over interval (not implemented)
5. **Worker ID auto-generated** unless specified in config
6. **Empty `working_dir` in EXEC_CMD** → worker uses its configured `work_dir`

## Testing

Only `test_protocol` exists. Tests CRC32, HELLO serialization, and capability detection round-trip. No unit tests for other modules yet.

## Future (Not Yet Implemented)

- Thread pool scheduler
- DAG task dependencies
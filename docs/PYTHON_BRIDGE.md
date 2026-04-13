# Phase 10: Python Bridge

## Overview

The Python bridge allows Python scripts to run as job payloads inside the existing MPI pipeline with zero protocol changes. A thin C++ harness wraps the Python script using the same `CLUSTR_MPI_MAIN` macro, same `jobs/` folder, same test infrastructure, and same stdout reporting conventions as any C++ job. From the TUI's perspective, a Python-backed job is indistinguishable from a native C++ job.

**Status**: Complete and tested
- Both transports (zero-copy, per-use) pass at 3 ranks
- Output format matches C++ `mpi_scatter_gather.cpp` exactly
- No new protocols, no frontend/backend changes required

---

## TUI Pipeline (Start to Finish)

This section documents the complete flow from picking a file in the TUI to seeing output — applicable to all jobs, not just Python.

### 1. File Browser Pick

User navigates the file browser on the Jobs page and selects a `.cpp` file (e.g., `jobs/python_scatter_gather.cpp`). This triggers `dialog_submit_job_file()`.

### 2. Submit Dialog (`src/tui/tui_dialogs.cpp`)

The dialog auto-generates compile and run commands from the file stem:

- **Compile presets** — 4 MPI transport/recv combos + 2 simple presets. Include paths point to `/var/tmp/clustr/include` and `/var/tmp/clustr/asio_include` (deployed worker paths).
- **Run command** — defaults to `./<stem>` (e.g., `./python_scatter_gather`)
- **Companion file detection** — scans the same directory for non-C++ files (`.py`, data files, etc.) and displays them in a "Files:" row. These will be bundled with the source.
- **Ranks** — user picks rank count. MPI jobs (>1) auto-select workers.

The dialog writes `source_file`, `compile_cmd`, `run_cmd`, `companion_files`, and `num_ranks` into `TuiState`, then calls `submit_job()`.

### 3. Job Queuing (`src/scheduler.cpp`)

`submit_job()` creates a `Job` object copying all fields from `TuiState` — including `companion_files` — pushes it into the job queue, and calls `try_schedule()`.

### 4. Dispatch

**Single-rank jobs** go through `dispatch_to()`. **MPI jobs** go through `dispatch_group()` which allocates N idle workers, sends `PEER_ROSTER` to each, waits for all `PEER_READY` responses, then sends files and compile commands.

### 5. File Transfer — Tarball Bundling (`src/file_transfer.cpp`)

This is the key mechanism that makes multi-file jobs work:

- **Pure C++ jobs** (no companions): `make_file_data_msg()` sends just the `.cpp` as a single `FILE_DATA` message — unchanged from the original protocol.
- **Jobs with companion files**: `make_bundle_msg()` runs `tar czf` to pack the `.cpp` + all companions into a single `<stem>_bundle.tar.gz`. This tarball is sent as **one** `FILE_DATA` message. One file, one ACK — no protocol deviation. Each file is added with `-C <absolute_dir> <basename>` so the archive contains flat basenames and each `-C` is unambiguous regardless of the working directory.

The worker receives the `FILE_DATA`, writes it to `work_dir/`. If the filename ends in `.tar.gz`, `handle_file_data()` automatically extracts it (`tar xzf -C work_dir`) and removes the archive. All files land flat in `work_dir/` where the compile command expects them.

```
Scheduler                              Worker
    │                                      │
    │  FILE_DATA (stem_bundle.tar.gz)      │
    │─────────────────────────────────────>│
    │                                      │  write to work_dir/
    │                                      │  detect .tar.gz
    │                                      │  tar xzf → flat files
    │                                      │  rm archive
    │              FILE_ACK                │
    │<─────────────────────────────────────│
    │                                      │
    │  EXEC_CMD (compile)                  │
    │─────────────────────────────────────>│
    │                                      │
```

### 6. Compile (`src/remote_exec.cpp`)

On `FILE_ACK` success, the scheduler sends `EXEC_CMD` with the compile command. The worker runs it via `exec_sync()`: fork, pipe stdout/stderr, `execvp`, wait. Returns `EXEC_RESULT` with exit code + captured output. Working dir is `/var/tmp/clustr/`.

### 7. Run

On compile success, the scheduler sends `EXEC_CMD` with the run command (`detach=1`). The worker runs it via `spawn_async()`: fork, redirect stdout+stderr to a log file (`dup2` both to one fd), `execvp`. A detached waiter thread calls `waitpid`, reads the log, and fires the `on_exit` callback.

### 8. Output Capture

All job output (stdout AND stderr) goes to a single log file: `work_dir/task_id.log`. On process exit, the waiter thread reads the log and sends output back to the scheduler as `TASK_RESULT`. The TUI renders it in the log panel.

### Key Constraints

- **Working dir is `/var/tmp/clustr/`** — all files land here, compile happens here, binary runs here.
- **Include paths are hardcoded** in presets to `/var/tmp/clustr/include` and `/var/tmp/clustr/asio_include`.
- **No env var injection** — the worker child inherits the worker daemon's environment. Custom env vars must be set another way (e.g., hardcoded in the binary, or resolved by convention).

### Relevant Source Files

| File | Role |
|------|------|
| `src/tui/tui_dialogs.cpp` | Submit dialog, compile presets, companion detection, worker picker |
| `src/tui/tui_input.cpp` | File browser, key handling |
| `src/scheduler.cpp` | Job queue, dispatch, send_compile, send_run |
| `src/file_transfer.cpp` | FILE_DATA/FILE_ACK, tarball bundling + extraction |
| `src/remote_exec.cpp` | `exec_sync` (compile), `spawn_async` (run) |
| `include/protocol.h` | ExecCmdPayload, FileDataPayload, message types |

---

## Architecture

### Design Principle: Drag-and-Drop

Adding a Python job requires only:

1. Write a Python script (pure function: read input file, compute, write output file)
2. Place it in `jobs/` alongside the C++ harness
3. Submit the `.cpp` from the TUI — companions are auto-detected and bundled

No changes to the MPI protocol, TUI, spawn infrastructure, or any shared code.

### Compute Data Flow (Per Rank)

```
                          C++ Harness (per rank)
                    +------------------------------+
                    |                              |
  scatter -------->|  write chunk to /tmp/*.bin   |
  (MPI network)    |          |                   |
                   |          v                   |
                   |  popen("python3 script       |
                   |    --input in.bin             |
                   |    --output out.bin           |
                   |    --rank R --size N          |
                   |    2>&1")                     |
                   |          |                   |
                   |    +-----v-----+             |
                   |    |  Python   |             |
                   |    |  script   |             |
                   |    |  (compute)|             |
                   |    +-----+-----+             |
                   |          |                   |
                   |  read result from /tmp/*.bin  |
                   |          |                   |
  gather <---------|  result back in memory       |
  (MPI network)    |                              |
                   +------------------------------+
```

### Why This Works

1. **No protocol changes**: The C++ harness handles all MPI communication (scatter, gather, barriers). Python never touches the network.
2. **Tarball bundling**: The scheduler packs `.cpp` + `.py` into one `.tar.gz`, sends as a single `FILE_DATA`. Worker extracts on receipt. One file, one ACK.
3. **Temp files for data**: Binary doubles pass through `/tmp/clustr_py_rankN_{in,out}.bin`. Avoids parsing overhead and preserves floating-point precision exactly.
4. **stdout for reporting**: `popen()` with `2>&1` merges Python's stdout and stderr into the C++ process's stdout. The TUI captures this identically to any C++ job's output.
5. **Script resolution**: The harness checks `CLUSTR_PYTHON_SCRIPT` env var first (used by the test harness), falls back to `example_python_job.py` in the current working dir (used by the TUI pipeline, where the tarball extracted the `.py` alongside the binary).

### Comparison with C++ Jobs

| Aspect | C++ Job (`mpi_scatter_gather.cpp`) | Python Bridge (`python_scatter_gather.cpp`) |
|--------|-------------------------------------|---------------------------------------------|
| Entry point | `CLUSTR_MPI_MAIN(mpi)` | `CLUSTR_MPI_MAIN(mpi)` |
| Scatter/Gather | `co_await mpi.scatter/gather` | `co_await mpi.scatter/gather` |
| Compute | Inline C++ loop | `popen("python3 script ...")` |
| Reporting | `std::cout << "[rank N] ..."` | `std::cout << "[rank N] ..."` + forwarded Python output |
| Data format | In-memory `std::vector<double>` | Temp binary files (8 bytes per double, little-endian) |
| File transfer | Single `.cpp` via `FILE_DATA` | `.cpp` + `.py` bundled as `.tar.gz` via `FILE_DATA` |
| Test harness | `tests/run_scatter_gather_local.sh` | `tests/run_python_bridge.sh` |
| TUI appearance | Identical | Identical |

---

## Files

### `jobs/python_scatter_gather.cpp` — C++ Harness

The harness follows the exact pattern of `mpi_scatter_gather.cpp`:

1. **Scatter**: Rank 0 builds `[1, 2, 3, 4, 5, 6]`, scatter distributes chunks
2. **Write to temp file**: Each rank writes its chunk as raw binary doubles to `/tmp/clustr_py_rankN_in.bin`
3. **Run Python**: `popen("python3 <script> --input <in> --output <out> --rank N --size S 2>&1")`
4. **Forward output**: All Python stdout/stderr lines are forwarded to `std::cout` via `fgets()` loop
5. **Read result**: Each rank reads the processed doubles from `/tmp/clustr_py_rankN_out.bin`
6. **Gather**: Results collected at rank 0

Script resolution:

```cpp
// 1. Check env var (test harness sets this)
const char* script_env = std::getenv("CLUSTR_PYTHON_SCRIPT");
if (script_env && script_env[0] != '\0') {
    script = script_env;
} else {
    // 2. Fall back to conventional name in working dir
    //    (TUI pipeline extracts .py from tarball here)
    script = "example_python_job.py";
}
```

### `jobs/example_python_job.py` — Example Compute Kernel

A pure function that doubles every value in the input:

```python
# Read binary doubles
with open(args.input, "rb") as f:
    raw = f.read()
values = list(struct.unpack(f"<{count}d", raw))

# Compute (swap this for numpy, scipy, ML inference, etc.)
result = [v * 2.0 for v in values]

# Write binary doubles
with open(args.output, "wb") as f:
    f.write(struct.pack(f"<{len(result)}d", *result))
```

The script knows nothing about MPI, rosters, or networking. It takes four arguments:
- `--input`: path to input binary file
- `--output`: path to write output binary file
- `--rank`: this rank's index (for future use)
- `--size`: total number of ranks (for future use)

### `tests/run_python_bridge.sh` — Test Script

Same structure as all other test scripts:
- Builds both transport variants (zero-copy and per-use)
- Generates roster files for 3 ranks on localhost
- Launches all ranks in parallel, waits for completion
- Shows per-rank logs on success, detailed failure logs on error
- Cleans up logs after a passing run (override with `KEEP_LOGS=1`)

Environment variables:
- `BASE_PORT`: starting port (default 18600)
- `KEEP_LOGS`: if set, preserve per-rank logs after a clean run

---

## Binary Format

Data passes between C++ and Python as raw arrays of IEEE 754 doubles in little-endian byte order:

| Field | Size | Description |
|-------|------|-------------|
| `values[0]` | 8 bytes | First double |
| `values[1]` | 8 bytes | Second double |
| ... | ... | ... |
| `values[N-1]` | 8 bytes | Last double |

No headers, no length prefix. The count is inferred from file size: `count = file_size / 8`.

C++ writes with `std::ofstream::write(reinterpret_cast<const char*>(data.data()), size * sizeof(double))`.
Python reads with `struct.unpack(f"<{count}d", raw)`.

---

## Expected Output

With `example_python_job.py` (doubling values) at 3 ranks:

```
[rank 0] Python script: example_python_job.py
[rank 0] Scattered chunk: 1 2
[rank 1] Scattered chunk: 3 4
[rank 2] Scattered chunk: 5 6
[rank 0] Python exited with code 0
[rank 0] Processed chunk: 2 4
[rank 1] Python exited with code 0
[rank 1] Processed chunk: 6 8
[rank 2] Python exited with code 0
[rank 2] Processed chunk: 10 12
[rank 0] Final result: 2 4 6 8 10 12
```

This matches the C++ `mpi_scatter_gather.cpp` output format exactly.

---

## Writing New Python Jobs

To add a new Python compute kernel:

1. **Create the script** in `jobs/`:
   ```python
   #!/usr/bin/env python3
   import argparse, struct

   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument("--input", required=True)
       parser.add_argument("--output", required=True)
       parser.add_argument("--rank", type=int, default=0)
       parser.add_argument("--size", type=int, default=1)
       args = parser.parse_args()

       # Read input doubles
       with open(args.input, "rb") as f:
           raw = f.read()
       count = len(raw) // 8
       values = list(struct.unpack(f"<{count}d", raw))

       # Your computation here
       result = [v ** 2 for v in values]  # example: square each value

       # Write output doubles
       with open(args.output, "wb") as f:
           f.write(struct.pack(f"<{len(result)}d", *result))

   if __name__ == "__main__":
       main()
   ```

2. **Place it in `jobs/`** alongside the C++ harness. The TUI auto-detects it as a companion file.

3. **Submit from the TUI** — select `python_scatter_gather.cpp`, the dialog shows the `.py` in "Files:", and both are bundled and sent to workers.

4. **Or run locally** with the test harness:
   ```bash
   export CLUSTR_PYTHON_SCRIPT="jobs/my_new_job.py"
   tests/run_python_bridge.sh
   ```

The contract is simple: read binary doubles from `--input`, compute, write binary doubles to `--output`. Everything else (MPI, networking, reporting) is handled by the C++ harness.

---

## Compilation

```bash
g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED \
    -DCLUSTR_TRANSPORT=1 -DCLUSTR_RECV=2 \
    -Iinclude -Ibuild/_deps/asio-src/asio/include \
    jobs/python_scatter_gather.cpp -o python_bridge -lpthread
```

---

## Limitations

1. **Latency**: Each rank spawns a Python process via `popen()`. Process startup adds ~20-50ms overhead per rank. Acceptable for batch workloads, not for tight inner loops.
2. **Data size**: Temp files pass through disk. For very large chunks, this could become a bottleneck. In practice, OS page cache keeps small-to-medium files in memory.
3. **Error reporting**: If the Python script crashes, the exit code propagates and the harness reports the failure. Python tracebacks are forwarded to stdout (via `2>&1`), which the TUI will display.
4. **Single data type**: Currently hardcoded to `double` arrays. Extending to other types requires changes to both the C++ harness and Python script's `struct` format string.

---

## Files Modified/Created

| File | Change | Type |
|------|--------|------|
| `jobs/python_scatter_gather.cpp` | C++ harness with popen + temp file I/O | New job |
| `jobs/example_python_job.py` | Example Python compute kernel | New job |
| `tests/run_python_bridge.sh` | Local test harness (both transports, 3 ranks) | New test |
| `include/file_transfer.h` | Added `make_bundle_msg()` declaration | Enhancement |
| `src/file_transfer.cpp` | Tarball bundling (scheduler) + extraction (worker) | Enhancement |
| `include/scheduler_strategy.h` | Added `companion_files` to `Job` struct | Enhancement |
| `include/tui.h` | Added `companion_files` to `TuiState` | Enhancement |
| `src/tui/tui_dialogs.cpp` | Auto-detect companions, show in dialog | Enhancement |
| `src/scheduler.cpp` | Use `make_bundle_msg` when companions exist | Enhancement |

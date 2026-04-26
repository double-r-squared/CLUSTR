#!/usr/bin/env bash
# tests/run_python_bridge.sh — validate Python bridge with scatter/gather.
#
# Builds python_scatter_gather.cpp, runs it at 3 ranks with
# example_python_job.py as the compute kernel.
#
# Env knobs:
#   BASE_PORT  starting port (default 18600)
#   KEEP_LOGS  if set, preserve per-rank logs after a clean run

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT_DIR/jobs/python_scatter_gather.cpp"
PY_SCRIPT="$ROOT_DIR/jobs/example_python_job.py"
BUILD_DIR="$ROOT_DIR/build/python_bridge"
ROSTER_DIR="$BUILD_DIR/rosters"
LOG_DIR="$BUILD_DIR/logs"
ASIO_INC="$ROOT_DIR/build/_deps/asio-src/asio/include"
INCLUDE_DIR="$ROOT_DIR/include"

BASE_PORT=${BASE_PORT:-18600}
HOST=127.0.0.1

if [ ! -f "$SRC" ]; then
    echo "[python_bridge] ERROR: source not found at $SRC" >&2
    exit 1
fi
if [ ! -f "$PY_SCRIPT" ]; then
    echo "[python_bridge] ERROR: python script not found at $PY_SCRIPT" >&2
    exit 1
fi
if [ ! -d "$ASIO_INC" ]; then
    echo "[python_bridge] ERROR: asio not found at $ASIO_INC" >&2
    echo "                      run cmake build at least once first" >&2
    exit 1
fi
if ! command -v python3 &>/dev/null; then
    echo "[python_bridge] ERROR: python3 not found in PATH" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR" "$ROSTER_DIR" "$LOG_DIR"

cleanup() {
    local pids
    pids=$(jobs -pr 2>/dev/null || true)
    if [ -n "$pids" ]; then
        kill $pids 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# -- 1. Build ---------------------------------------------------------------
for T in 1 2; do
    case "$T" in
        1) TNAME="zc";;
        2) TNAME="pu";;
    esac
    BIN="$BUILD_DIR/python_bridge_${TNAME}"
    echo "[python_bridge] Building ${TNAME} -> $BIN"
    g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED \
        -DCLUSTR_TRANSPORT=$T -DCLUSTR_RECV=2 \
        -I"$INCLUDE_DIR" -I"$ASIO_INC" \
        "$SRC" -o "$BIN" -lpthread

    # -- 2. Run at 3 ranks ---------------------------------------------------
    RANKS=3
    echo ""
    echo "========================================================================"
    echo "Running $TNAME @ $RANKS ranks"
    echo "========================================================================"

    RUN_NAME="${TNAME}_n${RANKS}"
    for r in $(seq 0 $((RANKS - 1))); do
        ROSTER="$ROSTER_DIR/${RUN_NAME}_rank${r}.conf"
        {
            echo "rank=$r"
            echo "size=$RANKS"
            echo "peer_port=$((BASE_PORT + r))"
            for p in $(seq 0 $((RANKS - 1))); do
                echo "peer.${p}=${HOST}:$((BASE_PORT + p))"
            done
        } > "$ROSTER"
    done

    declare -a PIDS=()
    for r in $(seq 0 $((RANKS - 1))); do
        ROSTER="$ROSTER_DIR/${RUN_NAME}_rank${r}.conf"
        LOG="$LOG_DIR/${RUN_NAME}_rank${r}.log"
        : > "$LOG"

        (
            export CLUSTR_MPI_ROSTER="$ROSTER"
            export CLUSTR_PYTHON_SCRIPT="$PY_SCRIPT"
            "$BIN" > "$LOG" 2>&1
        ) &
        PIDS+=($!)
    done

    FAIL=0
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "  [FAIL] rank $i exited nonzero"
            FAIL=1
        fi
    done
    unset PIDS

    if [ $FAIL -eq 1 ]; then
        echo "  ---- per-rank logs ----"
        for r in $(seq 0 $((RANKS - 1))); do
            echo "  -- rank $r --"
            sed 's/^/    /' "$LOG_DIR/${RUN_NAME}_rank${r}.log"
        done
        echo "[python_bridge] $RUN_NAME FAILED"
        exit 1
    else
        # Show all rank logs (rank 0 has the gather result, others have chunk info)
        for r in $(seq 0 $((RANKS - 1))); do
            sed 's/^/  /' "$LOG_DIR/${RUN_NAME}_rank${r}.log"
        done
        echo "[python_bridge] $RUN_NAME PASSED"
        if [ -z "${KEEP_LOGS:-}" ]; then
            rm -f "$LOG_DIR/${RUN_NAME}_rank"*.log
        fi
    fi
done

echo ""
echo "========================================================================"
echo "[python_bridge] All tests PASSED"
echo "========================================================================"

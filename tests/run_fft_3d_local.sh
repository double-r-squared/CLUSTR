#!/usr/bin/env bash
# tests/run_fft_3d_local.sh - Phase 8 local validation for the 3D pencil FFT.
#
# Builds fft_3d_test.cpp under all 4 preset corners and runs each build
# at 3, 4, and 6 ranks via localhost oversubscription.
#
# 64x64x64 complex<double> = 2 MB per rank - lightweight for local testing.
#
# Env knobs:
#   BASE_PORT  starting port (default 18300, uses up to +5 per run)
#   KEEP_LOGS  if set, preserve per-rank logs after a clean run
#   BENCH      if set, also build+run with -DCLUSTR_BENCHMARK and auto grid selection

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT_DIR/jobs/fft_3d_test.cpp"
BUILD_DIR="$ROOT_DIR/build/fft_3d_test"
ROSTER_DIR="$BUILD_DIR/rosters"
LOG_DIR="$BUILD_DIR/logs"
ASIO_INC="$ROOT_DIR/build/_deps/asio-src/asio/include"
INCLUDE_DIR="$ROOT_DIR/include"

BASE_PORT=${BASE_PORT:-18300}
HOST=127.0.0.1

if [ ! -f "$SRC" ]; then
    echo "[fft_3d_test] ERROR: source not found at $SRC" >&2
    exit 1
fi
if [ ! -d "$ASIO_INC" ]; then
    echo "[fft_3d_test] ERROR: asio not found at $ASIO_INC" >&2
    echo "                     run cmake build at least once first" >&2
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

# -- 1. Build under every (transport x recv) corner ----------------------
declare -a BINARIES=()
for T in 1 2; do
    for R in 2; do
        case "$T" in
            1) TNAME="zc";;
            2) TNAME="pu";;
        esac
        case "$R" in
            2) RNAME="central";;
        esac
        BIN="$BUILD_DIR/fft_3d_test_${TNAME}_${RNAME}"
        echo "[fft_3d_test] Building ${TNAME} / ${RNAME} -> $BIN"
        g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED \
            -DCLUSTR_TRANSPORT=$T -DCLUSTR_RECV=$R \
            -I"$INCLUDE_DIR" -I"$ASIO_INC" \
            "$SRC" -o "$BIN" -lpthread
        BINARIES+=("$BIN")
    done
done

echo "[fft_3d_test] Built ${#BINARIES[@]} binaries"

# -- 2. Run each binary at multiple rank counts ---------------------------
for BIN in "${BINARIES[@]}"; do
    BNAME=$(basename "$BIN")
    echo ""
    echo "========================================================================"
    echo "Running $BNAME"
    echo "========================================================================"

    for RANKS in 3 4 6; do
        echo ""
        echo "[fft_3d_test @ $RANKS ranks] Starting..."

        # Generate rosters for this rank count
        RUN_NAME="${BNAME}_n${RANKS}"
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

            # Launch rank process
            (
                export CLUSTR_MPI_ROSTER="$ROSTER"
                "$BIN" > "$LOG" 2>&1
            ) &
            PIDS+=($!)
        done

        # Wait for all ranks to finish
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
            echo "[fft_3d_test] $RUN_NAME FAILED"
            exit 1
        else
            sed 's/^/  /' "$LOG_DIR/${RUN_NAME}_rank0.log"
            echo "[fft_3d_test] $RUN_NAME PASSED"
            if [ -z "${KEEP_LOGS:-}" ]; then
                rm -f "$LOG_DIR/${RUN_NAME}_rank"*.log
            fi
        fi
    done
done

echo ""
echo "========================================================================"
echo "[fft_3d_test] All tests PASSED"
echo "========================================================================"

# -- 3. Optional benchmark pass (BENCH=1 tests/run_fft_3d_local.sh) -------
if [ -n "${BENCH:-}" ]; then
    echo ""
    echo "========================================================================"
    echo "[fft_3d_test] Building BENCHMARK binary (auto grid + CLUSTR_BENCHMARK)"
    echo "========================================================================"

    SRC_AUTO="$ROOT_DIR/jobs/fft_3d_test_auto.cpp"
    if [ ! -f "$SRC_AUTO" ]; then
        echo "[fft_3d_test] ERROR: $SRC_AUTO not found" >&2
        exit 1
    fi

    BIN_BENCH="$BUILD_DIR/fft_3d_test_bench"
    g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED \
        -DCLUSTR_TRANSPORT=1 -DCLUSTR_RECV=2 -DCLUSTR_BENCHMARK \
        -I"$INCLUDE_DIR" -I"$ASIO_INC" \
        "$SRC_AUTO" -o "$BIN_BENCH" -lpthread

    BNAME=$(basename "$BIN_BENCH")
    for RANKS in 3 4 6; do
        echo ""
        echo "[benchmark @ $RANKS ranks] Starting..."

        RUN_NAME="${BNAME}_n${RANKS}"
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
                "$BIN_BENCH" > "$LOG" 2>&1
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
            echo "[benchmark] $RUN_NAME FAILED"
            exit 1
        else
            sed 's/^/  /' "$LOG_DIR/${RUN_NAME}_rank0.log"
            echo "[benchmark] $RUN_NAME PASSED"
        fi
    done

    echo ""
    echo "========================================================================"
    echo "[fft_3d_test] All benchmarks PASSED"
    echo "========================================================================"
fi

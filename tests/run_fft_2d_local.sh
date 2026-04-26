#!/usr/bin/env bash
# tests/run_fft_2d_local.sh - Phase 7 local validation for the 2D slab FFT.
#
# Builds fft_2d_test.cpp under all 4 preset corners and runs each build
# at 2, 3, 4, and 8 ranks via localhost oversubscription.
#
# 256x256 complex<double> = 1 MB per rank - lightweight even at 8 procs.
#
# Env knobs:
#   BASE_PORT  starting port (default 18200, uses up to +7 per run)
#   KEEP_LOGS  if set, preserve per-rank logs after a clean run

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT_DIR/jobs/fft_2d_test.cpp"
BUILD_DIR="$ROOT_DIR/build/fft_2d_test"
ROSTER_DIR="$BUILD_DIR/rosters"
LOG_DIR="$BUILD_DIR/logs"
ASIO_INC="$ROOT_DIR/build/_deps/asio-src/asio/include"
INCLUDE_DIR="$ROOT_DIR/include"

BASE_PORT=${BASE_PORT:-18200}
HOST=127.0.0.1

if [ ! -f "$SRC" ]; then
    echo "[fft_2d_test] ERROR: source not found at $SRC" >&2
    exit 1
fi
if [ ! -d "$ASIO_INC" ]; then
    echo "[fft_2d_test] ERROR: asio not found at $ASIO_INC" >&2
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
    for R in 1 2; do
        case "$T" in
            1) TNAME="zc";;
            2) TNAME="pu";;
        esac
        case "$R" in
            1) RNAME="inline";;
            2) RNAME="central";;
        esac
        NAME="fft_2d_test_${TNAME}_${RNAME}"
        BIN="$BUILD_DIR/$NAME"
        echo "[fft_2d_test] build $NAME (TRANSPORT=$T RECV=$R)"
        g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED \
            -DCLUSTR_TRANSPORT=$T -DCLUSTR_RECV=$R \
            -I"$INCLUDE_DIR" -I"$ASIO_INC" \
            "$SRC" -o "$BIN" -lpthread
        BINARIES+=("$BIN")
    done
done

# -- 2. Run each build at multiple rank counts ---------------------------
overall_fail=0
for BIN in "${BINARIES[@]}"; do
    BNAME="$(basename "$BIN")"
    for NRANKS in 2 3 4 8; do
        RUN_NAME="${BNAME}_n${NRANKS}"
        echo "[fft_2d_test] run $RUN_NAME"

        # Generate rosters for this rank count.
        for r in $(seq 0 $((NRANKS-1))); do
            ROSTER="$ROSTER_DIR/${RUN_NAME}_rank${r}.conf"
            {
                echo "rank=$r"
                echo "size=$NRANKS"
                echo "peer_port=$((BASE_PORT + r))"
                for p in $(seq 0 $((NRANKS-1))); do
                    echo "peer.${p}=${HOST}:$((BASE_PORT + p))"
                done
            } > "$ROSTER"
        done

        declare -a pids=()
        for r in $(seq 0 $((NRANKS-1))); do
            ROSTER="$ROSTER_DIR/${RUN_NAME}_rank${r}.conf"
            LOG="$LOG_DIR/${RUN_NAME}_rank${r}.log"
            : > "$LOG"
            CLUSTR_MPI_ROSTER="$ROSTER" "$BIN" > "$LOG" 2>&1 &
            pids+=($!)
        done

        fail=0
        for i in "${!pids[@]}"; do
            if ! wait "${pids[$i]}"; then
                echo "  [FAIL] rank $i exited nonzero"
                fail=1
            fi
        done
        unset pids

        if [ $fail -eq 1 ]; then
            echo "  ---- per-rank logs ----"
            for r in $(seq 0 $((NRANKS-1))); do
                echo "  -- rank $r --"
                sed 's/^/    /' "$LOG_DIR/${RUN_NAME}_rank${r}.log"
            done
            overall_fail=1
            echo "[fft_2d_test] $RUN_NAME FAILED"
        else
            sed 's/^/  /' "$LOG_DIR/${RUN_NAME}_rank0.log"
            echo "[fft_2d_test] $RUN_NAME PASSED"
        fi
    done
done

if [ $overall_fail -eq 1 ]; then
    echo "[fft_2d_test] FAILED"
    exit 1
fi

if [ -z "${KEEP_LOGS:-}" ]; then
    rm -f "$LOG_DIR"/*.log "$ROSTER_DIR"/*.conf
fi

echo "[fft_2d_test] ALL BUILDS x ALL RANKS PASSED"

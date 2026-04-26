#!/usr/bin/env bash
# tests/run_redistribute_local.sh - Phase 6 local validation harness for
# RedistributePlan.
#
# Spawns 3 worker processes on 127.0.0.1, wired into a flat world via 3
# generated roster files. Builds the test under every preset corner
# (CLUSTR_TRANSPORT x CLUSTR_RECV) since redistribute only touches the
# world comm and works under both recv modes, unlike cart_sub. Runs each
# build in turn and fails on the first nonzero exit.
#
# Why 3 ranks:
#   The exit-criterion test uses global shape {9, 8}. With 3 ranks:
#     axis 0: 9/3 = 3  (uniform)
#     axis 1: 8/3 = 2, rem 2  -> ranks 0,1 get 3 cols, rank 2 gets 2
#   This exercises the off-by-one balanced-block path on one axis and the
#   uniform path on the other.
#
# Mirrors tests/run_cart_sub_local.sh so anyone familiar with that script
# will recognize the structure.
#
# Env knobs:
#   BASE_PORT  starting port for the 3-port range (default 18100)
#   KEEP_LOGS  if set, do not delete per-rank logs after a successful run

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT_DIR/jobs/redistribute_test.cpp"
BUILD_DIR="$ROOT_DIR/build/redistribute_test"
ROSTER_DIR="$BUILD_DIR/rosters"
LOG_DIR="$BUILD_DIR/logs"
ASIO_INC="$ROOT_DIR/build/_deps/asio-src/asio/include"
INCLUDE_DIR="$ROOT_DIR/include"

NRANKS=3
BASE_PORT=${BASE_PORT:-18100}
HOST=127.0.0.1

if [ ! -f "$SRC" ]; then
    echo "[redistribute_test] ERROR: source not found at $SRC" >&2
    exit 1
fi
if [ ! -d "$ASIO_INC" ]; then
    echo "[redistribute_test] ERROR: asio not found at $ASIO_INC" >&2
    echo "                          run cmake build at least once first" >&2
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

# -- 1. Generate per-rank rosters ----------------------------------------
for r in $(seq 0 $((NRANKS-1))); do
    ROSTER="$ROSTER_DIR/rank${r}.conf"
    {
        echo "rank=$r"
        echo "size=$NRANKS"
        echo "peer_port=$((BASE_PORT + r))"
        for p in $(seq 0 $((NRANKS-1))); do
            echo "peer.${p}=${HOST}:$((BASE_PORT + p))"
        done
    } > "$ROSTER"
done
echo "[redistribute_test] wrote $NRANKS rosters (ports ${BASE_PORT}..$((BASE_PORT+NRANKS-1)))"

# -- 2. Build under every (transport x recv) corner ----------------------
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
        NAME="redistribute_test_${TNAME}_${RNAME}"
        BIN="$BUILD_DIR/$NAME"
        echo "[redistribute_test] build $NAME (CLUSTR_TRANSPORT=$T CLUSTR_RECV=$R)"
        g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED \
            -DCLUSTR_TRANSPORT=$T -DCLUSTR_RECV=$R \
            -I"$INCLUDE_DIR" -I"$ASIO_INC" \
            "$SRC" -o "$BIN" -lpthread
        BINARIES+=("$BIN")
    done
done

# -- 3. Run each build ---------------------------------------------------
overall_fail=0
for BIN in "${BINARIES[@]}"; do
    NAME="$(basename "$BIN")"
    echo "[redistribute_test] run $NAME"

    declare -a pids=()
    for r in $(seq 0 $((NRANKS-1))); do
        ROSTER="$ROSTER_DIR/rank${r}.conf"
        LOG="$LOG_DIR/${NAME}_rank${r}.log"
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
            sed 's/^/    /' "$LOG_DIR/${NAME}_rank${r}.log"
        done
        overall_fail=1
        echo "[redistribute_test] $NAME FAILED"
    else
        sed 's/^/  /' "$LOG_DIR/${NAME}_rank0.log"
        echo "[redistribute_test] $NAME PASSED"
    fi
done

if [ $overall_fail -eq 1 ]; then
    echo "[redistribute_test] FAILED"
    exit 1
fi

if [ -z "${KEEP_LOGS:-}" ]; then
    rm -f "$LOG_DIR"/*.log
fi

echo "[redistribute_test] ALL BUILDS PASSED"

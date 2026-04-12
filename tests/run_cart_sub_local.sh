#!/usr/bin/env bash
# tests/run_cart_sub_local.sh - Phase 5D local validation harness for cart_sub.
#
# Spawns 6 worker processes on 127.0.0.1, wired into a 2x3 Cartesian grid via
# 6 generated roster files (one per rank). Each rank gets its own port in
# BASE_PORT..BASE_PORT+5. Builds the test under both CLUSTR_TRANSPORT modes
# against CLUSTR_RECV=CLUSTR_RECV_CENTRAL (the only mode that supports
# cart_sub) and runs each build in turn, failing on the first nonzero exit.
#
# Why this exists:
#   The cart_sub test needs 6 ranks to be meaningful (smallest grid that
#   exercises both axes with multiple cohorts is 2x2 = 4 ranks; we use 2x3
#   to also force size-2 and size-3 sub-comms in the same run). On a small
#   dev cluster you can't reach 6 machines, so we oversubscribe localhost
#   instead. The test source is identical to what would ship to the cluster
#   once 6+ machines are available.
#
# Env knobs:
#   BASE_PORT  starting port for the 6-port range (default 18000)
#   KEEP_LOGS  if set, do not delete per-rank logs after a successful run

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT_DIR/jobs/cart_sub_test.cpp"
BUILD_DIR="$ROOT_DIR/build/cart_sub_test"
ROSTER_DIR="$BUILD_DIR/rosters"
LOG_DIR="$BUILD_DIR/logs"
ASIO_INC="$ROOT_DIR/build/_deps/asio-src/asio/include"
INCLUDE_DIR="$ROOT_DIR/include"

NRANKS=6
BASE_PORT=${BASE_PORT:-18000}
HOST=127.0.0.1

if [ ! -f "$SRC" ]; then
    echo "[cart_sub_test] ERROR: source not found at $SRC" >&2
    exit 1
fi
if [ ! -d "$ASIO_INC" ]; then
    echo "[cart_sub_test] ERROR: asio not found at $ASIO_INC" >&2
    echo "                       run cmake build at least once first" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR" "$ROSTER_DIR" "$LOG_DIR"

# Kill any orphan child if the script is interrupted mid-run.
cleanup() {
    local pids
    pids=$(jobs -pr 2>/dev/null || true)
    if [ -n "$pids" ]; then
        kill $pids 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── 1. Generate per-rank rosters ─────────────────────────────────────────
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
echo "[cart_sub_test] wrote $NRANKS rosters (ports ${BASE_PORT}..$((BASE_PORT+NRANKS-1)))"

# ── 2. Build under each transport (CENTRAL recv only) ────────────────────
declare -a BINARIES=()
for T in 1 2; do
    case "$T" in
        1) NAME="cart_sub_test_zc";;
        2) NAME="cart_sub_test_pu";;
    esac
    BIN="$BUILD_DIR/$NAME"
    echo "[cart_sub_test] build $NAME (CLUSTR_TRANSPORT=$T CLUSTR_RECV=2)"
    g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED \
        -DCLUSTR_TRANSPORT=$T -DCLUSTR_RECV=2 \
        -I"$INCLUDE_DIR" -I"$ASIO_INC" \
        "$SRC" -o "$BIN" -lpthread
    BINARIES+=("$BIN")
done

# ── 3. Run each build ────────────────────────────────────────────────────
overall_fail=0
for BIN in "${BINARIES[@]}"; do
    NAME="$(basename "$BIN")"
    echo "[cart_sub_test] run $NAME"

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
        echo "[cart_sub_test] $NAME FAILED"
    else
        # Print rank 0's log (carries the phase-by-phase progress lines).
        sed 's/^/  /' "$LOG_DIR/${NAME}_rank0.log"
        echo "[cart_sub_test] $NAME PASSED"
    fi
done

if [ $overall_fail -eq 1 ]; then
    echo "[cart_sub_test] FAILED"
    exit 1
fi

# Logs are normally noise after a clean run; keep them around if asked.
if [ -z "${KEEP_LOGS:-}" ]; then
    rm -f "$LOG_DIR"/*.log
fi

echo "[cart_sub_test] ALL BUILDS PASSED"

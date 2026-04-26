#!/usr/bin/env bash
# bench_all.sh - build and run the full clustr benchmark matrix.
#
# Outputs JSONL files under bench/results/. Safe to re-run; files are
# appended, and the plot script (scripts/plot_bench.py) prefers the latest
# record per (bench, config) tuple.
#
# Environment knobs:
#   BENCH_WARMUP        default 10
#   BENCH_ITERATIONS    default 100
#   BENCH_SAMPLES       if set, include raw samples in JSON (bigger files)
#   BENCH_SKIP          comma list: p2p,collectives,fft,transfer,openmpi
#
# This script never touches production paths (no SSH, no deploy). Runs
# all benches as localhost oversubscription, same pattern as
# tests/run_fft_3d_local.sh.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/bench"
RESULTS_DIR="$ROOT_DIR/bench/results"
ASIO_INC="$ROOT_DIR/build/_deps/asio-src/asio/include"
INC="$ROOT_DIR/include"
BENCH_INC="$ROOT_DIR/bench/include"
BASE_PORT=${BASE_PORT:-19100}
HOST=127.0.0.1

mkdir -p "$BUILD_DIR" "$RESULTS_DIR" "$BUILD_DIR/rosters" "$BUILD_DIR/logs"

if [ ! -d "$ASIO_INC" ]; then
    echo "[bench_all] ERROR: asio headers missing at $ASIO_INC" >&2
    echo "            run 'cmake -B build && cmake --build build' at least once." >&2
    exit 1
fi

SKIP="${BENCH_SKIP:-}"
should_run() {
    case ",$SKIP," in
        *",$1,"*) return 1 ;;
        *)        return 0 ;;
    esac
}

# -- Compile a bench binary with given flags ----------------------------------
compile_mpi() {
    local src="$1"
    local out="$2"
    shift 2
    echo "[bench_all] Building $(basename "$out")..."
    g++ -std=c++20 -O2 -DASIO_STANDALONE -DASIO_NO_DEPRECATED \
        -DCLUSTR_TRANSPORT=1 -DCLUSTR_RECV=2 \
        "$@" \
        -I"$INC" -I"$BENCH_INC" -I"$ASIO_INC" \
        "$src" -o "$out" -lpthread
}

compile_sync() {
    local src="$1"
    local out="$2"
    echo "[bench_all] Building $(basename "$out") (sync)..."
    # Sync benches need the scheduler-side sources (protocol, file_transfer).
    g++ -std=c++20 -O2 \
        -I"$INC" -I"$BENCH_INC" \
        "$src" \
        "$ROOT_DIR/src/file_transfer.cpp" \
        "$ROOT_DIR/src/protocol.cpp" \
        -o "$out"
}

# -- Launch N ranks of a binary as one benchmark run --------------------------
run_ranks() {
    local bin="$1"
    local ranks="$2"
    local tag="$3"

    local run_name="$(basename "$bin")_${tag}_n${ranks}"
    # Write a roster file for each rank
    for r in $(seq 0 $((ranks - 1))); do
        local roster="$BUILD_DIR/rosters/${run_name}_rank${r}.conf"
        {
            echo "rank=$r"
            echo "size=$ranks"
            echo "peer_port=$((BASE_PORT + r))"
            for p in $(seq 0 $((ranks - 1))); do
                echo "peer.${p}=${HOST}:$((BASE_PORT + p))"
            done
        } > "$roster"
    done

    local pids=()
    for r in $(seq 0 $((ranks - 1))); do
        local roster="$BUILD_DIR/rosters/${run_name}_rank${r}.conf"
        local log="$BUILD_DIR/logs/${run_name}_rank${r}.log"
        : > "$log"
        (
            export CLUSTR_MPI_ROSTER="$roster"
            "$bin" > "$log" 2>&1
        ) &
        pids+=($!)
    done

    local fail=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            fail=1
        fi
    done

    if [ $fail -ne 0 ]; then
        echo "[bench_all] FAIL: $run_name"
        for r in $(seq 0 $((ranks - 1))); do
            echo "-- rank $r --"
            sed 's/^/   /' "$BUILD_DIR/logs/${run_name}_rank${r}.log"
        done
        return 1
    fi
    # Echo rank 0 output (useful status lines)
    sed 's/^/  /' "$BUILD_DIR/logs/${run_name}_rank0.log"
    BASE_PORT=$((BASE_PORT + ranks + 2))
    return 0
}

# ----------------------------------------------------------------------------
# PHASE B2 — P2P pingpong + bandwidth (2 ranks)
# ----------------------------------------------------------------------------
if should_run p2p; then
    echo ""
    echo "================================================================"
    echo "PHASE B2 — P2P microbenchmarks"
    echo "================================================================"
    compile_mpi "$ROOT_DIR/bench/mpi/bench_p2p_pingpong.cpp" \
                "$BUILD_DIR/bench_p2p_pingpong"
    compile_mpi "$ROOT_DIR/bench/mpi/bench_p2p_bandwidth.cpp" \
                "$BUILD_DIR/bench_p2p_bandwidth"

    : > "$RESULTS_DIR/p2p_pingpong.jsonl"
    : > "$RESULTS_DIR/p2p_bandwidth.jsonl"

    run_ranks "$BUILD_DIR/bench_p2p_pingpong"  2 "zc_central"
    run_ranks "$BUILD_DIR/bench_p2p_bandwidth" 2 "zc_central"

    # Pack/unpack transport variant for comparison
    compile_mpi "$ROOT_DIR/bench/mpi/bench_p2p_pingpong.cpp" \
                "$BUILD_DIR/bench_p2p_pingpong_pu" \
                -UCLUSTR_TRANSPORT -DCLUSTR_TRANSPORT=2
    run_ranks "$BUILD_DIR/bench_p2p_pingpong_pu" 2 "pu_central"
fi

# ----------------------------------------------------------------------------
# PHASE B3 — Collectives (sweep over ranks)
# ----------------------------------------------------------------------------
if should_run collectives; then
    echo ""
    echo "================================================================"
    echo "PHASE B3 — Collectives"
    echo "================================================================"

    declare -A OP_MAP=(
        [bcast]=1 [reduce]=2 [allreduce]=3
        [scatter]=4 [gather]=5 [barrier]=6
    )

    for op in bcast reduce allreduce scatter gather barrier; do
        : > "$RESULTS_DIR/coll_${op}.jsonl"
        compile_mpi "$ROOT_DIR/bench/mpi/bench_collectives.cpp" \
                    "$BUILD_DIR/bench_coll_${op}" \
                    -DBENCH_OP=${OP_MAP[$op]}
        for ranks in 2 4 8; do
            run_ranks "$BUILD_DIR/bench_coll_${op}" "$ranks" "zc_central"
        done
    done
fi

# ----------------------------------------------------------------------------
# PHASE B4 — FFT strong + weak scaling
# ----------------------------------------------------------------------------
if should_run fft; then
    echo ""
    echo "================================================================"
    echo "PHASE B4 — FFT 3D scaling"
    echo "================================================================"
    compile_mpi "$ROOT_DIR/bench/fft/bench_fft_3d.cpp" \
                "$BUILD_DIR/bench_fft_3d"
    : > "$RESULTS_DIR/fft_3d.jsonl"
    : > "$RESULTS_DIR/fft_3d_steps.jsonl"

    # Strong scaling: fixed 48^3 across rank counts
    BENCH_FFT_N=48 BENCH_FFT_TYPE=strong \
        run_ranks "$BUILD_DIR/bench_fft_3d" 2 "strong_48"
    BENCH_FFT_N=48 BENCH_FFT_TYPE=strong \
        run_ranks "$BUILD_DIR/bench_fft_3d" 4 "strong_48"
    BENCH_FFT_N=48 BENCH_FFT_TYPE=strong \
        run_ranks "$BUILD_DIR/bench_fft_3d" 6 "strong_48"

    # Weak scaling: 24^3 per rank baseline, scale up with cbrt(ranks)
    BENCH_FFT_N=24 BENCH_FFT_TYPE=weak \
        run_ranks "$BUILD_DIR/bench_fft_3d" 2 "weak"
    BENCH_FFT_N=24 BENCH_FFT_TYPE=weak \
        run_ranks "$BUILD_DIR/bench_fft_3d" 4 "weak"
    BENCH_FFT_N=24 BENCH_FFT_TYPE=weak \
        run_ranks "$BUILD_DIR/bench_fft_3d" 6 "weak"
fi

# ----------------------------------------------------------------------------
# PHASE B5 — File transfer + tarball bundle throughput (in-process)
# ----------------------------------------------------------------------------
if should_run transfer; then
    echo ""
    echo "================================================================"
    echo "PHASE B5 — File transfer"
    echo "================================================================"
    compile_sync "$ROOT_DIR/bench/transfer/bench_file_transfer.cpp" \
                 "$BUILD_DIR/bench_file_transfer"
    : > "$RESULTS_DIR/file_transfer.jsonl"
    "$BUILD_DIR/bench_file_transfer"
fi

# ----------------------------------------------------------------------------
# PHASE B6 — OpenMPI baseline (optional)
# ----------------------------------------------------------------------------
if should_run openmpi; then
    echo ""
    echo "================================================================"
    echo "PHASE B6 — OpenMPI baseline"
    echo "================================================================"
    if command -v mpicxx >/dev/null 2>&1 && command -v mpirun >/dev/null 2>&1; then
        : > "$RESULTS_DIR/ompi_pingpong.jsonl"
        : > "$RESULTS_DIR/ompi_bcast.jsonl"

        mpicxx -std=c++20 -O2 -I"$BENCH_INC" \
            "$ROOT_DIR/bench/openmpi/bench_ompi_pingpong.cpp" \
            -o "$BUILD_DIR/bench_ompi_pingpong"
        mpirun --oversubscribe -n 2 "$BUILD_DIR/bench_ompi_pingpong" || true

        mpicxx -std=c++20 -O2 -I"$BENCH_INC" \
            "$ROOT_DIR/bench/openmpi/bench_ompi_bcast.cpp" \
            -o "$BUILD_DIR/bench_ompi_bcast"
        for ranks in 2 4 8; do
            mpirun --oversubscribe -n "$ranks" "$BUILD_DIR/bench_ompi_bcast" || true
        done
    else
        echo "[bench_all] mpicxx/mpirun not found — skipping OpenMPI baseline"
        echo "[bench_all] install with: brew install open-mpi  (macOS)"
        echo "[bench_all]              apt install openmpi-bin libopenmpi-dev  (Linux)"
    fi
fi

echo ""
echo "================================================================"
echo "[bench_all] Done. Results in $RESULTS_DIR/"
ls -la "$RESULTS_DIR"/*.jsonl 2>/dev/null || true
echo "================================================================"
echo "Next: run   python3 scripts/plot_bench.py"
echo "      or read docs/BENCHMARKS.md for interpretation"

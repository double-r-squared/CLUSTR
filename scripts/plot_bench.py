#!/usr/bin/env python3
"""plot_bench.py - render matplotlib plots from bench/results/*.jsonl.

Reads every JSONL file written by the bench binaries and produces a set of
PNG plots under bench/plots/. Each plot is produced from the MOST RECENT
record per (bench, config) tuple so that re-running bench_all.sh simply
refreshes the figures instead of overlaying stale runs.

Reading rule (applied across every bench):
  - Files are append-only JSONL (one JSON object per line).
  - We dedupe by the full config dict, keeping the LAST record. This way
    a partial re-run of one config doesn't invalidate earlier unrelated
    configs, but re-running the same config replaces the old number.

What gets plotted:
  p2p_pingpong       -> latency vs bytes (log-log), zc vs pu vs openmpi
  p2p_bandwidth      -> GiB/s vs bytes, zc vs pu
  coll_*             -> latency vs ranks, per op
  fft_3d             -> p50 seconds vs ranks (strong + weak overlaid)
  fft_3d_steps       -> stacked bar of per-step time at each rank count
  file_transfer      -> throughput MiB/s vs bytes (log-x) + bundle time vs companions

All plots use percentiles (p50 as the line, p99 as an optional errorband)
rather than mean, because tail matters and mean hides it in distributed
systems. See docs/BENCHMARKS.md for interpretation guidance.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib",
          file=sys.stderr)
    sys.exit(1)


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "bench" / "results"
PLOTS_DIR = ROOT / "bench" / "plots"


# ---- Loading ---------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read one JSONL file, dedupe by config dict (last wins)."""
    if not path.exists():
        return []
    by_cfg: Dict[str, Dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = json.dumps(rec.get("config", {}), sort_keys=True)
            by_cfg[key] = rec
    return list(by_cfg.values())


def load_all() -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not RESULTS_DIR.exists():
        return out
    for p in sorted(RESULTS_DIR.glob("*.jsonl")):
        recs = load_jsonl(p)
        if recs:
            out[p.stem] = recs
    return out


# ---- Helpers ---------------------------------------------------------------

def ns_to_us(ns: float) -> float:
    return ns / 1e3

def ns_to_ms(ns: float) -> float:
    return ns / 1e6

def ns_to_s(ns: float) -> float:
    return ns / 1e9

def stats_p50(rec: Dict[str, Any]) -> float:
    return float(rec.get("stats", {}).get("p50_ns", 0.0))

def stats_p99(rec: Dict[str, Any]) -> float:
    return float(rec.get("stats", {}).get("p99_ns", 0.0))

def cfg(rec: Dict[str, Any], key: str, default: Any = None) -> Any:
    return rec.get("config", {}).get(key, default)


# ---- Plot functions --------------------------------------------------------

def plot_pingpong(results: Dict[str, List[Dict[str, Any]]]):
    """Latency vs message size, clustr zc/pu vs OpenMPI (if present)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    series = []
    if "p2p_pingpong" in results:
        # Split by transport config if multiple variants were run.
        by_transport: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
        for r in results["p2p_pingpong"]:
            bytes_ = cfg(r, "bytes")
            transport = cfg(r, "transport", "zc")
            if bytes_ is None:
                continue
            # one-way = RTT/2 (bench is ping+pong)
            by_transport[f"clustr_{transport}"].append(
                (int(bytes_), stats_p50(r) / 2.0, stats_p99(r) / 2.0))
        for label, pts in by_transport.items():
            series.append((label, sorted(pts)))

    if "ompi_pingpong" in results:
        pts = []
        for r in results["ompi_pingpong"]:
            bytes_ = cfg(r, "bytes")
            if bytes_ is None:
                continue
            pts.append((int(bytes_), stats_p50(r) / 2.0, stats_p99(r) / 2.0))
        if pts:
            series.append(("openmpi", sorted(pts)))

    if not series:
        plt.close(fig)
        return

    for label, pts in series:
        xs = [p[0] for p in pts]
        p50 = [ns_to_us(p[1]) for p in pts]
        p99 = [ns_to_us(p[2]) for p in pts]
        line, = ax.plot(xs, p50, marker="o", label=f"{label} (p50)")
        ax.plot(xs, p99, marker="x", linestyle="--",
                color=line.get_color(), alpha=0.5, label=f"{label} (p99)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("One-way latency (us)")
    ax.set_title("P2P ping-pong: one-way latency vs message size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p2p_pingpong.png", dpi=120)
    plt.close(fig)
    print(f"  wrote bench/plots/p2p_pingpong.png")


def plot_bandwidth(results: Dict[str, List[Dict[str, Any]]]):
    if "p2p_bandwidth" not in results:
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    by_transport: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for r in results["p2p_bandwidth"]:
        bytes_ = cfg(r, "bytes")
        burst = cfg(r, "burst", 1)
        transport = cfg(r, "transport", "zc")
        if bytes_ is None:
            continue
        # total_bytes / p50_seconds, both sides see same amount, so report
        # sender-side goodput (bytes * burst / time).
        p50_s = ns_to_s(stats_p50(r))
        if p50_s <= 0:
            continue
        gbps = (int(bytes_) * int(burst)) / p50_s / (1 << 30)
        by_transport[f"clustr_{transport}"].append((int(bytes_), gbps))

    if not by_transport:
        plt.close(fig)
        return

    for label, pts in by_transport.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=label)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("Goodput (GiB/s)")
    ax.set_title("P2P streaming bandwidth")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p2p_bandwidth.png", dpi=120)
    plt.close(fig)
    print(f"  wrote bench/plots/p2p_bandwidth.png")


def plot_collectives(results: Dict[str, List[Dict[str, Any]]]):
    ops = []
    for key in results:
        if key.startswith("coll_"):
            ops.append(key[5:])
    if not ops:
        return
    ops.sort()

    fig, ax = plt.subplots(figsize=(9, 5))
    for op in ops:
        recs = results[f"coll_{op}"]
        # group by ranks, pick mid-size bytes (or the only size for barrier)
        by_ranks: Dict[int, float] = {}
        for r in recs:
            ranks = int(cfg(r, "ranks", 0))
            bytes_ = cfg(r, "bytes")
            if ranks == 0:
                continue
            # pick the 4 KiB bucket when available, or any single-size op
            if bytes_ is not None and int(bytes_) not in (4096, 0):
                continue
            by_ranks[ranks] = ns_to_us(stats_p50(r))

        if not by_ranks:
            continue
        xs = sorted(by_ranks.keys())
        ys = [by_ranks[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=op)

    # Also overlay OpenMPI bcast baseline for the same pattern, if present.
    if "ompi_bcast" in results:
        by_ranks: Dict[int, float] = {}
        for r in results["ompi_bcast"]:
            ranks = int(cfg(r, "ranks", 0))
            bytes_ = cfg(r, "bytes")
            if ranks == 0:
                continue
            if bytes_ is not None and int(bytes_) != 4096:
                continue
            by_ranks[ranks] = ns_to_us(stats_p50(r))
        if by_ranks:
            xs = sorted(by_ranks.keys())
            ys = [by_ranks[x] for x in xs]
            ax.plot(xs, ys, marker="s", linestyle="--", label="openmpi_bcast")

    ax.set_xlabel("Ranks")
    ax.set_ylabel("p50 latency (us)")
    ax.set_title("Collective latency (4 KiB payload) vs rank count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "collectives.png", dpi=120)
    plt.close(fig)
    print(f"  wrote bench/plots/collectives.png")


def plot_fft_scaling(results: Dict[str, List[Dict[str, Any]]]):
    if "fft_3d" not in results:
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    # Split by 'type' config (strong / weak).
    by_type: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for r in results["fft_3d"]:
        ranks = cfg(r, "ranks")
        type_ = cfg(r, "type", "strong")
        if ranks is None:
            continue
        by_type[type_].append((int(ranks), ns_to_s(stats_p50(r))))

    for label, pts in by_type.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=label)

    ax.set_xlabel("Ranks")
    ax.set_ylabel("p50 time per FFT (s)")
    ax.set_title("Parallel 3D FFT scaling")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fft_scaling.png", dpi=120)
    plt.close(fig)
    print(f"  wrote bench/plots/fft_scaling.png")


def plot_fft_breakdown(results: Dict[str, List[Dict[str, Any]]]):
    if "fft_3d_steps" not in results:
        return
    # Expect records with config: {ranks, type, step} -> p50_ns
    fig, ax = plt.subplots(figsize=(9, 5))

    # Group by step name -> list of (ranks, p50_ms) for the 'strong' series
    by_step: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    all_ranks = set()
    for r in results["fft_3d_steps"]:
        step = cfg(r, "step")
        ranks = cfg(r, "ranks")
        type_ = cfg(r, "type", "strong")
        if step is None or ranks is None or type_ != "strong":
            continue
        by_step[step].append((int(ranks), ns_to_ms(stats_p50(r))))
        all_ranks.add(int(ranks))

    if not by_step or not all_ranks:
        plt.close(fig)
        return

    rank_list = sorted(all_ranks)
    step_names = sorted(by_step.keys())
    bottoms = [0.0] * len(rank_list)
    for step in step_names:
        m = {rk: tm for rk, tm in by_step[step]}
        heights = [m.get(rk, 0.0) for rk in rank_list]
        ax.bar([str(rk) for rk in rank_list], heights,
               bottom=bottoms, label=step)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xlabel("Ranks")
    ax.set_ylabel("p50 time (ms)")
    ax.set_title("FFT step breakdown (strong scaling)")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fft_breakdown.png", dpi=120)
    plt.close(fig)
    print(f"  wrote bench/plots/fft_breakdown.png")


def plot_file_transfer(results: Dict[str, List[Dict[str, Any]]]):
    if "file_transfer" not in results:
        return
    # Two plots: throughput vs bytes (for make_file_data_msg + handle_file_data),
    # and bundle time vs companion count.
    single_recs = [r for r in results["file_transfer"]
                   if r.get("bench") in ("make_file_data_msg", "handle_file_data")]
    bundle_recs = [r for r in results["file_transfer"]
                   if r.get("bench") in ("make_bundle_msg", "bundle_roundtrip")]

    if single_recs:
        fig, ax = plt.subplots(figsize=(8, 5))
        by_bench: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for r in single_recs:
            bytes_ = cfg(r, "bytes")
            p50_s = ns_to_s(stats_p50(r))
            if not bytes_ or p50_s <= 0:
                continue
            mibps = (int(bytes_) / (1024.0 * 1024.0)) / p50_s
            by_bench[r["bench"]].append((int(bytes_), mibps))

        for label, pts in by_bench.items():
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", label=label)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("File size (bytes)")
        ax.set_ylabel("Throughput (MiB/s)")
        ax.set_title("File transfer throughput (scheduler/worker, in-process)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "file_transfer_throughput.png", dpi=120)
        plt.close(fig)
        print(f"  wrote bench/plots/file_transfer_throughput.png")

    if bundle_recs:
        fig, ax = plt.subplots(figsize=(8, 5))
        by_bench: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for r in bundle_recs:
            comp = cfg(r, "companion_count")
            if comp is None:
                continue
            by_bench[r["bench"]].append((int(comp), ns_to_ms(stats_p50(r))))

        for label, pts in by_bench.items():
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", label=label)

        ax.set_xlabel("Companion file count")
        ax.set_ylabel("p50 time (ms)")
        ax.set_title("Tarball bundle time vs companion count (4 KiB each)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "file_transfer_bundles.png", dpi=120)
        plt.close(fig)
        print(f"  wrote bench/plots/file_transfer_bundles.png")


# ---- Entry ------------------------------------------------------------------

def main() -> int:
    results = load_all()
    if not results:
        print(f"no JSONL records found under {RESULTS_DIR}", file=sys.stderr)
        print(f"run: bash scripts/bench_all.sh", file=sys.stderr)
        return 1

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"plot_bench: {sum(len(v) for v in results.values())} records across "
          f"{len(results)} files")

    plot_pingpong(results)
    plot_bandwidth(results)
    plot_collectives(results)
    plot_fft_scaling(results)
    plot_fft_breakdown(results)
    plot_file_transfer(results)

    print(f"done. figures in {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

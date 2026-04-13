#!/usr/bin/env python3
"""Example Python job — doubles every value in the input.

This is the Python equivalent of `for (double& v : chunk) v *= 2.0;`
from mpi_scatter_gather.cpp.

The script is a pure function: read binary input, compute, write binary output.
It knows nothing about MPI, rosters, or networking. The C++ harness handles
all of that.

Usage (called by python_scatter_gather.cpp, not directly):
    python3 example_python_job.py --input /tmp/in.bin --output /tmp/out.bin --rank 0 --size 3
"""

import argparse
import struct
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to input binary (doubles)")
    parser.add_argument("--output", required=True, help="path to write output binary (doubles)")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--size", type=int, default=1)
    args = parser.parse_args()

    # Read input — raw array of doubles (8 bytes each, little-endian)
    with open(args.input, "rb") as f:
        raw = f.read()

    count = len(raw) // 8
    values = list(struct.unpack(f"<{count}d", raw))

    # ── Compute ─────────────────────────────────────────────────────────────
    # This is where real work goes. Swap this for numpy, scipy, ML inference,
    # whatever. The contract is: read input doubles, write output doubles.
    result = [v * 2.0 for v in values]

    # Write output — same format
    with open(args.output, "wb") as f:
        f.write(struct.pack(f"<{len(result)}d", *result))


if __name__ == "__main__":
    main()

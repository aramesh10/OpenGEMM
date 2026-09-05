#!/usr/bin/env python3
"""Ablate configurations for one (dtype, shape) and store the winner.

    CUDA_VISIBLE_DEVICES=3 python scripts/tune.py --dtype bf16 --shape 8192
    8192 8192
    CUDA_VISIBLE_DEVICES=3 python scripts/tune.py --dtype e4m3 --all

Pin a GPU. The winner lands in configs.json, which gemm() dispatches from,
so the next call runs it; no rebuild. gemm() and emit_kernel() run this
sweep themselves for a shape with no entry, and --all re-tunes every shape
in shapes.jsonc.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opengemm import DTYPES  # noqa: E402
from opengemm.python import bench  # noqa: E402
from opengemm.python.build import extension  # noqa: E402
from opengemm.python.tune import tune  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", required=True, choices=sorted(DTYPES))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"))
    group.add_argument("--all", action="store_true",
                       help="every shape in shapes.jsonc")
    args = parser.parse_args()
    ext = extension(DTYPES[args.dtype].impl)
    for m, n, k in ([tuple(args.shape)] if args.shape else bench.load_shapes()):
        try:
            tune(args.dtype, m, n, k, ext)
        except RuntimeError as exc:
            print(exc, flush=True)


if __name__ == "__main__":
    main()

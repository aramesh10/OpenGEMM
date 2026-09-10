#!/usr/bin/env python3
"""Emit a standalone kernel for one (dtype, shape).

    python scripts/emit_kernel.py --dtype e4m3 --shape 4096 4096 4096
    python scripts/emit_kernel.py --dtype nvfp4 --shape 4096 4096 4096 --file emitted/nvfp4_4k.cu

Without --file the pair is named <dtype>_<M>_<N>_<K>. A shape with no stored
configuration is tuned and stored first (pin a GPU with CUDA_VISIBLE_DEVICES).
Dtypes: bf16 f16 tf32 s8 u8 e4m3 e5m2 e3m2 e2m3 e2m1 e4m3xe5m2 nvfp4 mxfp8 mxfp4.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opengemm import DTYPES, emit_kernel
from opengemm.python.dtypes import elems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", required=True, choices=sorted(DTYPES))
    parser.add_argument("--shape", nargs=3, type=int, required=True, metavar=("M", "N", "K"))
    parser.add_argument("--file", default=None, help="write <file>.cu and <file>.cuh")
    args = parser.parse_args()
    m, n, k = args.shape
    keys = ("atype", "sftype") if DTYPES[args.dtype].impl == "smm" else ("atype", "btype")
    names = dict(zip(keys, elems(args.dtype)))
    emit_kernel(m=m, n=n, k=k, file=args.file, **names)


if __name__ == "__main__":
    main()

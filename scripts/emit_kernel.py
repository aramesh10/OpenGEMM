#!/usr/bin/env python3
"""Emit a standalone kernel for one (dtype, shape).

    python scripts/emit_kernel.py --dtype e4m3 --shape 4096 4096 4096
    python scripts/emit_kernel.py --dtype nvfp4 --shape 4096 4096 4096 --file
    emitted/nvfp4_4k.cu

Without --file the .cuh and .cu are printed. A shape with no stored
configuration is tuned and stored first (pin a GPU with CUDA_VISIBLE_DEVICES).
Dtypes: bf16 f16 tf32 s8 u8 e4m3 e5m2 e3m2 e2m3 e2m1 e4m3xe5m2 nvfp4 mxfp8
mxfp4.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from opengemm import DTYPES, emit_kernel


def meta_operands(dtype, m, n, k):
    """Return meta tensors with the dtypes and extents these operands arrive
    in.
    """
    d = DTYPES[dtype]
    if d.impl == "smm":
        a = torch.empty((m, d.k_extent(k)), dtype=d.elem_dtype, device="meta")
        b = torch.empty((n, d.k_extent(k)), dtype=d.elem_dtype, device="meta")
        sf = torch.empty((0,), dtype=d.sf_dtype, device="meta")
        return a, b, sf, sf, {}
    a = torch.empty((m, d.k_extent(k)), dtype=d.torch_dtype, device="meta")
    b = torch.empty((n, d.k_extent(k)), dtype=d.torch_dtype_b, device="meta")
    return a, b, None, None, {"atype": d.elem_a, "btype": d.elem_b}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", required=True, choices=sorted(DTYPES))
    parser.add_argument("--shape", nargs=3, type=int, required=True,
                        metavar=("M", "N", "K"))
    parser.add_argument("--file", default=None,
                        help="write <file>.cu and <file>.cuh instead of printing")
    args = parser.parse_args()
    a, b, sfa, sfb, names = meta_operands(args.dtype, *args.shape)
    emit_kernel(a, b, sfa, sfb, file=args.file, **names)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Correctness of gemm() for every dtype on a few shapes, aligned and ragged.
A shape with no stored configuration is tuned and stored on first use, so
the first run of a fresh clone takes a while; later runs are lookups.

    CUDA_VISIBLE_DEVICES=3 python scripts/test.py
    CUDA_VISIBLE_DEVICES=3 python scripts/test.py --dtype nvfp4 --emit
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import opengemm as og
from opengemm import DTYPES
from opengemm.python import bench

SHAPES = [(2048, 2048, 2048), (128, 128, 128), (1, 8192, 8192),
          (300, 520, 768), (77, 33, 128), (2048, 2048, 384)]


def check(name, m, n, k, emit, workdir):
    dtype = DTYPES[name]
    entry = bench.make_inputs(m, n, k, dtype)
    if dtype.impl == "smm":
        operands, names = entry[:4], {}
    else:
        operands = entry
        names = ({"atype": dtype.elem_a, "btype": dtype.elem_b}
                 if entry[0].dtype == torch.uint8 else {})
    if emit:
        file = workdir / f"{name}_{m}_{n}_{k}.cu"
        og.emit_kernel(*operands, file=str(file), **names)
        out = og.run_kernel(str(file), *operands)
    else:
        out = og.gemm(*operands, **names)
    torch.cuda.synchronize()
    rtol, atol = dtype.tolerance(k)
    torch.testing.assert_close(out, bench.reference(entry, dtype),
                               rtol=rtol, atol=atol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default=None)
    parser.add_argument("--emit", action="store_true",
                        help="go through emit_kernel + run_kernel instead")
    args = parser.parse_args()
    names = [args.dtype] if args.dtype else list(DTYPES)
    failed = 0
    with tempfile.TemporaryDirectory(prefix="opengemm_test_") as tmp:
        for name in names:
            for m, n, k in SHAPES:
                sub_byte = DTYPES[name].impl == "smm" or DTYPES[name].bits < 8
                if sub_byte and k % 128:
                    continue
                try:
                    check(name, m, n, k, args.emit, Path(tmp))
                    print(f"ok    {name:10s} {m}x{n}x{k}", flush=True)
                except Exception as exc:
                    failed += 1
                    print(f"FAIL  {name:10s} {m}x{n}x{k}: "
                          f"{str(exc).splitlines()[0][:120]}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

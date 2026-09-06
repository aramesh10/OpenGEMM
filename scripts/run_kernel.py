#!/usr/bin/env python3
"""Compile an emitted kernel, check it against the reference, and time it
against cuBLAS under the same protocol benchmark.py uses.

    python scripts/run_kernel.py emitted/e4m3_4096_4096_4096.cu
"""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from opengemm import DTYPES, run_kernel
from opengemm.python import bench
from opengemm.python.emit import parse_tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="the .cu written by emit_kernel")
    args = parser.parse_args()
    impl, name, m, n, k, _ = parse_tag(Path(args.file).read_text())
    dtype = DTYPES[name]

    rotation = bench.make_input_set(m, n, k, dtype)
    operands = 2 if impl == "mm" else 4

    def launch(entry):
        return run_kernel(args.file, *entry[:operands], out=entry[-1])

    out = rotation[0][-1]
    out.fill_(float("nan") if out.dtype.is_floating_point else 12345)
    launch(rotation[0])
    torch.cuda.synchronize()
    expected = bench.reference(rotation[0], dtype)
    rtol, atol = dtype.tolerance(k)
    try:
        torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
    except Exception as exc:
        print(f"INCORRECT {name} {m}x{n}x{k}")
        raise SystemExit(str(exc).splitlines()[0][:200])
    print(f"CORRECT   {name} {m}x{n}x{k}  (rtol {rtol:.2e}, atol {atol:.2e})")

    cycle = itertools.cycle(rotation)
    once = lambda: launch(next(cycle))
    warmup, iterations = bench.report_plan(once)
    us = bench.timed(once, warmup, iterations)
    line = f"kernel {us:8.2f}us  {2 * m * n * k / us / 1e6:8.1f} TFLOP/s"
    baseline, why = bench.baseline_for(rotation, dtype)
    if baseline is None:
        print(line + f"   (no cuBLAS baseline: {why})")
    else:
        base = bench.timed(baseline, warmup, iterations)
        print(line + f"   cuBLAS {base:8.2f}us   {base / us:.3f}x")


if __name__ == "__main__":
    main()

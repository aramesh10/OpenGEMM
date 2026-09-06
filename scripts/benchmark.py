#!/usr/bin/env python3
"""Benchmark every dtype against cuBLAS across the full shape suite, using
each shape's best (tuned) kernel configuration.

    CUDA_VISIBLE_DEVICES=3 python scripts/benchmark.py                      #
    everything
    CUDA_VISIBLE_DEVICES=3 python scripts/benchmark.py --dtype bf16 e4m3 nvfp4
    CUDA_VISIBLE_DEVICES=3 python scripts/benchmark.py --shape 4096 4096 4096
    CUDA_VISIBLE_DEVICES=3 python scripts/benchmark.py --quick               # fast,
    less precise
    CUDA_VISIBLE_DEVICES=3 python scripts/benchmark.py --tune-missing        # tune
    untuned shapes first

For every (dtype, shape), this replays the configuration `tune.py` stored as
the fastest for that pair - the same one `gemm()` dispatches - and times it
against torch's own kernel for that dtype (cuBLAS, or `torch._scaled_mm` /
`torch._int_mm` where that is what backs it) with both sides going through
the identical harness in `opengemm.bench`. A shape with no stored entry is
skipped by default, since there is no "best config" to report for it; pass
--tune-missing to tune and store it first (a few minutes each), then measure
it like the rest.

Pin a GPU with CUDA_VISIBLE_DEVICES - the full sweep is 14 dtypes x 68
shapes and an unpinned run lands on device 0 alongside anything else there.
Full-precision timing (the default) takes on the order of an hour for the
whole suite; --quick uses the ablation-length window instead, in minutes.

Exits non-zero if anything came back INCORRECT or failed to launch, so this
doubles as a regression test: `scripts/test.py` checks correctness on a
handful of shapes per dtype, this checks performance on every tuned one.

Results are written as JSON to --output (default
results/perf_<gpu>_<date>.json).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from opengemm import DTYPES  # noqa: E402
from opengemm.python import bench  # noqa: E402
from opengemm.python.build import ROOT  # noqa: E402
from opengemm.python.tune import resolve_config  # noqa: E402

LOW_SPEEDUP = 0.95
BAD_SPEEDUP = 0.90


def cuda_healthy():
    """Return whether the CUDA context still works.

    A launch failure surfaces asynchronously, often on a later unrelated call,
    and leaves the context dead for the rest of the process; a real synchronize
    is the authoritative check.
    """
    try:
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def row_for(dtype_name, m, n, k, config, quick):
    """Measure one (dtype, shape) against cuBLAS.

    A launch or measurement failure is recorded as status "error" rather than
    raised, so one bad shape does not end the sweep; `row["fatal"]` marks a
    failure that took the CUDA context with it, which no later shape in this
    process can recover from.

    Returns:
        The result row, as written to the JSON output.
    """
    dtype = DTYPES[dtype_name]
    row = {"dtype": dtype_name, "m": m, "n": n, "k": k}
    try:
        buffers = bench.make_input_set(m, n, k, dtype)
    except Exception as exc:
        row["status"] = "error"
        row["note"] = f"could not build inputs: {str(exc).splitlines()[0][:120]}"
        if not cuda_healthy():
            row["fatal"] = True
        return row

    try:
        wrong = bench.correctness_error(buffers, config, dtype, m, n, k)
        if wrong:
            row["status"] = "incorrect"
            row["note"] = wrong
        else:
            kernel = bench.runner(buffers, config, dtype, m, n, k)
            baseline, why = bench.baseline_for(buffers, dtype)
            plan = bench.plan if quick else bench.report_plan
            warmup, iterations = plan(kernel)
            us = bench.timed(kernel, warmup, iterations)
            row["kernel_us"] = round(us, 3)
            row["tflops"] = round(2 * m * n * k / us / 1e6, 1)
            if baseline is None:
                row["status"] = "no_baseline"
                row["note"] = why
            else:
                base = bench.timed(baseline, warmup, iterations)
                row["cublas_us"] = round(base, 3)
                row["speedup"] = round(base / us, 4)
                row["status"] = "ok"
    except Exception as exc:
        row["status"] = "error"
        row["note"] = str(exc).splitlines()[0][:160]

    del buffers
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    if not cuda_healthy():
        row["status"] = "error"
        row["fatal"] = True
        row["note"] = (row.get("note") + "; " if row.get("note") else "") + \
            "CUDA context is unusable after this shape (unspecified launch " \
            "failure) - the sweep stops here; rerun in a fresh process"
    return row


def print_row(row):
    m, n, k = row["m"], row["n"], row["k"]
    if row["status"] in ("error", "incorrect"):
        print(f"{m:>7}{n:>7}{k:>7}  {row['status'].upper():>10}  {row['note']}",
              flush=True)
        return
    cublas = f"{row['cublas_us']:>11.2f}u" if "cublas_us" in row else f"{'-':>12}"
    speedup = f"{row['speedup']:>8.3f}x" if "speedup" in row else f"{'-':>9}"
    note = ""
    if row["status"] == "no_baseline":
        note = f"  (no baseline: {row['note']})"
    print(f"{m:>7}{n:>7}{k:>7}{cublas}{row['kernel_us']:>10.2f}u"
          f"{row['tflops']:>10.1f}{speedup}{note}", flush=True)


def summarize(rows):
    by_dtype = {}
    for row in rows:
        by_dtype.setdefault(row["dtype"], []).append(row)
    summary = {}
    for name, dtype_rows in by_dtype.items():
        speedups = [r["speedup"] for r in dtype_rows if "speedup" in r]
        summary[name] = {
            "shapes": len(dtype_rows),
            "ok": sum(r["status"] == "ok" for r in dtype_rows),
            "no_baseline": sum(r["status"] == "no_baseline" for r in dtype_rows),
            "incorrect": sum(r["status"] == "incorrect" for r in dtype_rows),
            "error": sum(r["status"] == "error" for r in dtype_rows),
            "min_speedup": min(speedups) if speedups else None,
            "median_speedup": sorted(speedups)[len(speedups) // 2] if speedups else None,
            "max_speedup": max(speedups) if speedups else None,
            "below_0.95x": sum(s < LOW_SPEEDUP for s in speedups),
            "below_0.90x": sum(s < BAD_SPEEDUP for s in speedups),
        }
    return summary


def print_summary(summary):
    print(f"\n{'dtype':<11}{'shapes':>7}{'ok':>5}{'noBase':>8}"
          f"{'bad':>5}{'min':>8}{'median':>8}{'max':>8}{'<0.95x':>8}{'<0.90x':>8}")
    for name in sorted(summary):
        s = summary[name]
        bad = s["incorrect"] + s["error"]
        fmt = lambda v: f"{v:>8.3f}" if v is not None else f"{'-':>8}"
        print(f"{name:<11}{s['shapes']:>7}{s['ok']:>5}"
              f"{s['no_baseline']:>8}{bad:>5}{fmt(s['min_speedup'])}"
              f"{fmt(s['median_speedup'])}{fmt(s['max_speedup'])}"
              f"{s['below_0.95x']:>8}{s['below_0.90x']:>8}")


def output_path(explicit):
    if explicit:
        return Path(explicit)
    gpu = re.sub(r"\W+", "-", torch.cuda.get_device_name(0)).strip("-")
    return ROOT / "results" / f"perf_{gpu}_{bench.env_stamp()['date']}.json"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", nargs="+", choices=sorted(DTYPES), default=None,
                        help="default: every dtype")
    parser.add_argument("--shape", nargs=3, type=int, default=None,
                        metavar=("M", "N", "K"), help="default: shapes.jsonc")
    parser.add_argument("--tune-missing", action="store_true",
                        help="tune and store shapes with no stored configuration "
                             "first, then measure them too")
    parser.add_argument("--quick", action="store_true",
                        help="ablation-length timing window, not the reported one")
    parser.add_argument("--output", default=None,
                        help="JSON results path (default results/perf_<gpu>_<date>.json)")
    parser.add_argument("--resume", action="store_true",
                        help="skip (dtype, shape) pairs already in --output and "
                             "extend it, instead of measuring everything again - "
                             "for continuing after the sweep aborts")
    args = parser.parse_args()

    dtype_names = args.dtype or sorted(DTYPES)
    shapes = [tuple(args.shape)] if args.shape else bench.load_shapes()

    path = output_path(args.output)
    rows = []
    if args.resume:
        if path.exists():
            rows = json.loads(path.read_text())["rows"]
            print(f"resuming from {path}: {len(rows)} shape(s) already measured")
        else:
            print(f"--resume given but {path} does not exist; starting fresh")
    done = {(r["dtype"], r["m"], r["n"], r["k"]) for r in rows}

    aborted = False
    for name in dtype_names:
        if aborted:
            break
        dtype = DTYPES[name]
        configs = bench.load_configs(dtype.impl, name)

        print(f"\n{bench.env_stamp()['gpu']}  dtype {name}")
        print(f"{'M':>7}{'N':>7}{'K':>7}{'cuBLAS':>12}{'kernel':>11}"
              f"{'TFLOP/s':>10}{'speedup':>9}")
        for m, n, k in shapes:
            if (name, m, n, k) in done:
                continue
            config = configs.get((name, m, n, k))
            if config is None:
                if not (args.tune_missing or args.shape):
                    continue
                try:
                    config = resolve_config(name, m, n, k)
                except (ValueError, RuntimeError) as exc:
                    print(f"{m:>7}{n:>7}{k:>7}  {'SKIPPED':>10}  "
                          f"{str(exc).splitlines()[0][:120]}", flush=True)
                    continue
            row = row_for(name, m, n, k, config, args.quick)
            print_row(row)
            rows.append(row)
            if row.get("fatal"):
                print(f"\naborting the sweep: {row['note']}", flush=True)
                aborted = True
                break
            time.sleep(0.5)

    summary = summarize(rows)
    print_summary(summary)

    problems = [r for r in rows if r["status"] in ("incorrect", "error")]
    if problems:
        print(f"\n{len(problems)} shape(s) failed:")
        for r in problems:
            print(f"  {r['dtype']:<10} {r['m']}x{r['n']}x{r['k']}: "
                  f"{r['status']} - {r['note']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"env": bench.env_stamp(), "rows": rows,
                                "summary": summary, "aborted": aborted},
                               indent=1) + "\n")
    print(f"\nwrote {path}")

    raise SystemExit(1 if problems or aborted else 0)


if __name__ == "__main__":
    main()

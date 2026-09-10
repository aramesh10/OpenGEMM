"""Inputs, references, torch baselines and the timing harness the tuner and the scripts share.

Timing rotates through enough buffer copies to exceed twice L2, so no
iteration reads its predecessor's cache lines, and times the kernel and
cuBLAS through the same function.
"""

import collections
import datetime
import itertools
import os
import statistics
from typing import NamedTuple

import torch

from .dtypes import DTYPES, elems, k_of, sf_block
from .quant import SMALL_FLOATS, pack, quantize, round_tf32, to_blocked, unpack

REPORT_WARMUP = 10_000
REPORT_ITERATIONS = 4_000
WARMUP_CAP_S = 20.0
WINDOW_CAP_S = 4.0
LAUNCH_CHUNK = 256
LAUNCH_DEPTH = 2

# (rtol, atol) at K = 4096, scaled up with sqrt(K / 4096) past it.
TOLERANCE = {
    "s8": (0.0, 0.0),
    "u8": (0.0, 0.0),
    "nvfp4": (2.0**-6, 2.0**-5),
    "mxfp8": (2.0**-6, 2.0**-5),
    "mxfp4": (2.0**-6, 2.0**-5),
}
DEFAULT_TOLERANCE = (1.0e-3, 2.0e-2)
INT_RANGE = {"s8": (-127, 128), "u8": (0, 256)}

# torch's kernel for each format, or why there is none.
BASELINE = {
    "bf16": "mm",
    "f16": "mm",
    "tf32": "mm_tf32",
    "s8": "int_mm",
    "e4m3": "scaled_mm",
    "e4m3xe5m2": "scaled_mm",
    "nvfp4": "scaled_mm_blocked",
    "mxfp8": "scaled_mm_blocked",
}
NO_BASELINE = {
    "u8": "torch._int_mm takes signed operands only",
    "e5m2": "torch._scaled_mm refuses e5m2 x e5m2",
    "e3m2": "torch has no float6 dtype",
    "e2m3": "torch has no float6 dtype",
    "e2m1": "cuBLAS exposes no dense fp4 GEMM",
    "mxfp4": "torch._scaled_mm refuses e2m1 operands with e8m0 block scales",
}


class Inputs(NamedTuple):
    """One rotation of operands, with the output the kernel writes into."""

    a: torch.Tensor
    b: torch.Tensor
    sfa: torch.Tensor = None  # blocked scales, what the kernel reads
    sfb: torch.Tensor = None
    sfa_raw: torch.Tensor = None  # (rows, K / block) scales, what the reference reads
    sfb_raw: torch.Tensor = None
    out: torch.Tensor = None


def tolerance(name, k):
    """`(rtol, atol)` for a reduction of length `k`."""
    rtol, atol = TOLERANCE.get(name, DEFAULT_TOLERANCE)
    scale = max(1.0, (k / 4096) ** 0.5)
    return rtol * scale, atol * scale


def make_inputs(m, n, k, name, seed=0, device="cuda"):
    """Build the operands for one GEMM; `seed` makes each rotation differ."""
    d = DTYPES[name]
    generator = torch.Generator(device=device).manual_seed(seed)
    if d.out is torch.int32:
        low, high = INT_RANGE[name]
        a = torch.randint(low, high, (m, k), generator=generator, device=device, dtype=torch.int32)
        b = torch.randint(low, high, (n, k), generator=generator, device=device, dtype=torch.int32)
        return Inputs(a.to(d.a).contiguous(), b.to(d.b).contiguous())
    a = torch.rand((m, k), generator=generator, device=device) * 2 - 1
    b = torch.rand((n, k), generator=generator, device=device) * 2 - 1
    if d.impl == "smm":
        a, sfa = quantize(a.to(torch.bfloat16), name)
        b, sfb = quantize(b.to(torch.bfloat16), name)
        return Inputs(a, b, to_blocked(sfa), to_blocked(sfb), sfa, sfb)
    if name == "tf32":
        a, b = round_tf32(a), round_tf32(b)
    if d.bits < 8:
        return Inputs(pack(a, name), pack(b, name))
    return Inputs(a.to(d.a).contiguous(), b.to(d.b).contiguous())


def out_buffer(m, n, name, device="cuda"):
    return torch.empty((m, n), device=device, dtype=DTYPES[name].out)


def make_input_set(m, n, k, name, seed=0, cap_bytes=48 << 30):
    """Enough input rotations to exceed twice L2, each with its own output."""
    d = DTYPES[name]
    iteration_bytes = (m * k + n * k) * d.bits // 8 + m * n * d.out.itemsize
    if d.impl == "smm":
        iteration_bytes += (m * k + n * k) // sf_block(d)
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    rotations = max(2, min(-(-2 * l2_bytes // iteration_bytes), cap_bytes // iteration_bytes))
    return [
        make_inputs(m, n, k, name, seed + i)._replace(out=out_buffer(m, n, name))
        for i in range(rotations)
    ]


def _values(t, elem, k):
    """A packed operand as float32 values."""
    if elem in SMALL_FLOATS:
        return unpack(t.view(torch.uint8), elem, k)
    return t.float()


def reference(entry, name):
    """The reference product: fp32, or exact for the integer types, row-major."""
    d = DTYPES[name]
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if d.impl == "smm":
            elem = elems(name)[0]
            block = sf_block(d)
            k = k_of(d, entry.a.size(1))
            a = _values(entry.a, elem, k) * entry.sfa_raw.float().repeat_interleave(block, dim=1)
            b = _values(entry.b, elem, k) * entry.sfb_raw.float().repeat_interleave(block, dim=1)
            return (a @ b.t()).to(torch.bfloat16)
        if d.out is torch.int32:
            return torch.mm(entry.a.double(), entry.b.double().t()).to(torch.int32)
        k = k_of(d, entry.a.size(1))
        a, b = _values(entry.a, name, k), _values(entry.b, name, k)
        return torch.mm(a, b.t()).to(d.out)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def runner(buffers, bound):
    """A zero-argument callable that launches the bound kernel on the next rotation each call."""
    lib, g = bound
    cycle = itertools.cycle(buffers)

    def call():
        r = next(cycle)
        lib.launch(g, r.a, r.b, r.sfa, r.sfb, r.out)
        return r.out

    return call


def correctness_error(buffers, bound, name, k):
    """None if the bound kernel reproduces the reference on the first rotation,
    else the first line of the mismatch."""
    rtol, atol = tolerance(name, k)
    try:
        torch.testing.assert_close(
            runner(buffers[:1], bound)(), reference(buffers[0], name), rtol=rtol, atol=atol
        )
        return None
    except Exception as exc:
        return str(exc).splitlines()[0][:90]


def baseline_for(buffers, name):
    """torch's kernel for this format over the same rotations.

    Returns:
        `(callable, None)`, or `(None, reason)` when torch has nothing to compare against.
    """
    kind = BASELINE.get(name)
    if kind is None:
        return None, NO_BASELINE[name]
    d = DTYPES[name]
    if kind == "scaled_mm_blocked":
        cycle = itertools.cycle([(r.a, r.b.t(), r.sfa, r.sfb, r.out) for r in buffers])

        def call():
            a, b, sfa, sfb, c = next(cycle)
            return torch._scaled_mm(a, b, sfa, sfb, out_dtype=torch.bfloat16, out=c)

    elif kind == "int_mm":
        outs = [torch.empty(r.out.shape, device="cuda", dtype=torch.int32) for r in buffers]
        cycle = itertools.cycle([(r.a, r.b.t(), o) for r, o in zip(buffers, outs)])

        def call():
            a, b, o = next(cycle)
            return torch._int_mm(a, b, out=o)

    elif kind == "scaled_mm":
        unit = torch.ones((), device="cuda", dtype=torch.float32)
        cycle = itertools.cycle([(r.a, r.b.t(), r.out) for r in buffers])

        def call():
            a, b, c = next(cycle)
            return torch._scaled_mm(a, b, scale_a=unit, scale_b=unit, out_dtype=d.out, out=c)

    else:
        tf32 = kind == "mm_tf32"
        cycle = itertools.cycle([(r.a, r.b.t(), r.out) for r in buffers])

        def call():
            a, b, c = next(cycle)
            previous = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = tf32
            try:
                if tf32:
                    return torch.mm(a, b, out=c)
                return torch.mm(a, b, out=c, out_dtype=d.out)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous

    try:
        call()
        torch.cuda.synchronize()
        return call, None
    except Exception as exc:
        return None, str(exc).splitlines()[0][:110]


def elapsed_ms(fn, iterations):
    """Milliseconds `iterations` calls of `fn` take, by CUDA events."""
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    pending = collections.deque()
    for i in range(iterations):
        fn()
        if i % LAUNCH_CHUNK == LAUNCH_CHUNK - 1:
            marker = torch.cuda.Event()
            marker.record()
            pending.append(marker)
            if len(pending) > LAUNCH_DEPTH:
                pending.popleft().synchronize()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def per_iteration_ms(fn, probe=10):
    """A quick estimate of one call's milliseconds."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    return max(elapsed_ms(fn, probe) / probe, 1e-4)


def plan(fn, warmup_s=0.3, window_ms=20, min_iterations=50, max_window_s=0.15):
    """`(warmup, iterations)` for ranking candidates inside a sweep."""
    ms = per_iteration_ms(fn)
    affordable = int(max_window_s * 1000 / ms) + 1
    warmup = max(3, int(warmup_s * 1000 / ms))
    iterations = max(int(window_ms / ms) + 1, min(min_iterations, affordable))
    return warmup, iterations


def report_plan(fn, warmup_s=1.0, window_ms=300):
    """`(warmup, iterations)` for a number that gets reported."""
    ms = per_iteration_ms(fn)
    warmup = max(REPORT_WARMUP, int(warmup_s * 1000 / ms))
    iterations = max(REPORT_ITERATIONS, int(window_ms / ms) + 1)
    warmup = max(8, min(warmup, int(WARMUP_CAP_S * 1000 / ms)))
    iterations = max(8, min(iterations, int(WINDOW_CAP_S * 1000 / ms)))
    return warmup, iterations


def warm(fn, iterations):
    for i in range(iterations):
        fn()
        if i % 64 == 63:
            torch.cuda.synchronize()
    torch.cuda.synchronize()


def timed(fn, warmup, iterations, repeats=5):
    """Median microseconds per call over `repeats` windows."""
    warm(fn, warmup)
    return statistics.median(elapsed_ms(fn, iterations) * 1000 / iterations for _ in range(repeats))


def env_stamp():
    """The GPU, device pin, torch version and date, for stored measurements."""
    return {
        "gpu": torch.cuda.get_device_name(0),
        "device": os.environ.get("CUDA_VISIBLE_DEVICES", "unpinned"),
        "torch": torch.__version__,
        "date": datetime.date.today().isoformat(),
    }

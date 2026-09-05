"""Inputs, references and the timing harness the scripts share.

Timing follows NVIDIA's GEMM measurement methodology: rotate through enough
buffer copies to exceed twice L2, so no iteration reads its predecessor's
cache lines; warm up for at least 10,000 iterations and profile for 4,000,
more where a shape is short; and time the kernel and cuBLAS through the
same function.
"""

import collections
import datetime
import itertools
import json
import os
import re
import statistics
from pathlib import Path

import torch

from .build import PACKAGE, SRC
from .dtypes import DTYPES, ELEM_INDEX, quantize, to_blocked

REPORT_WARMUP = 10_000
REPORT_ITERATIONS = 4_000
WARMUP_CAP_S = 20.0
WINDOW_CAP_S = 4.0
LAUNCH_CHUNK = 256
LAUNCH_DEPTH = 2

# configs.json keys in the order the extension's launcher() takes them.
CONFIG_KEYS = {
    "mm": (("use_2cta", bool), ("output_n", int), ("use_clc", bool),
           ("supergroup", int), ("swap_ab", bool), ("k_pad", -1),
           ("epi_direct", 0), ("epi_hold", 1), ("cluster_n", 1),
           ("stages", 0), ("epi_double", 0), ("split_k", 1),
           ("l2_promo", 0), ("block_m", 128), ("cluster_m", 0),
           ("cluster_k", 1), ("walk", 0)),
    "smm": (("use_2cta", bool), ("output_n", int), ("use_clc", bool),
            ("supergroup", int), ("swap_ab", bool), ("epi_direct", 0),
            ("epi_trade", 0), ("deep_stages", 0), ("cluster_m", 0),
            ("cluster_n", 1), ("cluster_k", 1), ("persistent", 1)),
}


def load_shapes():
    """Return the distinct (M, N, K) triples of shapes.jsonc, in file order."""
    seen, shapes = set(), []
    text = re.sub(r"//[^\n]*", "", (PACKAGE / "shapes.jsonc").read_text())
    for m, n, k in json.loads(text):
        if (m, n, k) not in seen:
            seen.add((m, n, k))
            shapes.append((m, n, k))
    return shapes



def make_inputs(m, n, k, dtype, seed=0, device="cuda"):
    """Build the operands for one GEMM.

    Args:
        m: Rows of A and C.
        n: Rows of B and columns of C.
        k: Reduction length in elements.
        dtype: An entry of `DTYPES`.
        seed: Generator seed, so rotations differ.
        device: Where the tensors live.

    Returns:
        Dense: `(a, b)`. Block-scaled: `(a, b, sfa_blocked, sfb_blocked, sfa,
        sfb)`, the last two being the unblocked scales the reference needs.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    if dtype.impl == "smm":
        a = torch.rand((m, k), generator=generator, device=device) * 2 - 1
        b = torch.rand((n, k), generator=generator, device=device) * 2 - 1
        a_packed, sfa = quantize(a.to(torch.bfloat16), dtype)
        b_packed, sfb = quantize(b.to(torch.bfloat16), dtype)
        return (a_packed.view(dtype.elem_dtype), b_packed.view(dtype.elem_dtype),
                to_blocked(sfa), to_blocked(sfb), sfa, sfb)
    if dtype.out_dtype is torch.int32:
        low, high = dtype.int_range
        a = torch.randint(low, high, (m, k), generator=generator,
                          device=device, dtype=torch.int32)
        b = torch.randint(low, high, (n, k), generator=generator,
                          device=device, dtype=torch.int32)
        return (a.to(dtype.torch_dtype).contiguous(),
                b.to(dtype.torch_dtype_b).contiguous())
    a = torch.rand((m, k), generator=generator, device=device) * 2 - 1
    b = torch.rand((n, k), generator=generator, device=device) * 2 - 1
    if dtype.quantize is not None:
        a, b = dtype.quantize(a), dtype.quantize(b)
    if dtype.packer is not None:
        return dtype.packer(a), dtype.packer(b)
    return (a.to(dtype.torch_dtype).contiguous(),
            b.to(dtype.torch_dtype_b).contiguous())


def out_buffer(m, n, dtype, device="cuda"):
    """Return an output buffer laid out the way the kernel writes it."""
    if dtype.impl == "smm":
        return torch.empty((m, n), device=device, dtype=torch.bfloat16)
    return torch.empty_strided((m, n), (1, m), device=device,
                               dtype=dtype.out_dtype)


def make_input_set(m, n, k, dtype, seed=0, cap_bytes=48 << 30):
    """Build enough input rotations to exceed twice L2, each with its own
    output.

    Args:
        m: Rows of A and C.
        n: Rows of B and columns of C.
        k: Reduction length in elements.
        dtype: An entry of `DTYPES`.
        seed: Seed of the first rotation; later ones count up from it.
        cap_bytes: Upper bound on the bytes of all rotations together.

    Returns:
        A list of `make_inputs` tuples, each with an output buffer appended.
    """
    out_bytes = 2 if dtype.impl == "smm" else torch.empty(
        (), dtype=dtype.out_dtype).element_size()
    bits = 4 if dtype.impl == "smm" and dtype.elem == "e2m1" else \
        8 if dtype.impl == "smm" else dtype.bits
    iteration_bytes = (m * k + n * k) * bits // 8 + m * n * out_bytes
    if dtype.impl == "smm":
        iteration_bytes += (m * k + n * k) // dtype.block
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    # Twice L2 of rotations defeats the cache; the cap bounds a huge shape.
    rotations = max(2, min(-(-2 * l2_bytes // iteration_bytes),
                           cap_bytes // iteration_bytes))
    return [make_inputs(m, n, k, dtype, seed + i) + (out_buffer(m, n, dtype),)
            for i in range(rotations)]


def reference(entry, dtype):
    """Compute the reference product in fp32, or exactly for the integer types.

    Args:
        entry: One rotation from `make_inputs`.
        dtype: An entry of `DTYPES`.

    Returns:
        `(M, N)` in the kernel's output dtype, row-major.
    """
    a, b = entry[0], entry[1]
    previous = torch.backends.cuda.matmul.allow_tf32
    # The reference must not round through tf32.
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if dtype.impl == "smm":
            sfa, sfb = entry[4], entry[5]
            k = dtype.per_byte * a.size(1)
            a = dtype.unpacker(a.view(torch.uint8), k) \
                * sfa.float().repeat_interleave(dtype.block, dim=1)
            b = dtype.unpacker(b.view(torch.uint8), k) \
                * sfb.float().repeat_interleave(dtype.block, dim=1)
            return (a @ b.t()).to(torch.bfloat16)
        if dtype.out_dtype is torch.int32:
            return torch.mm(a.double(), b.double().t()).to(torch.int32)
        if dtype.unpacker is not None:
            k = a.shape[1] * 8 // dtype.bits
            a, b = dtype.unpacker(a, k), dtype.unpacker(b, k)
        return torch.mm(a.float(), b.float().t()).to(dtype.out_dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous



def config_args(config, impl):
    """Return `config` as the positional arguments the extension's launcher()
    takes.

    Raises:
        KeyError: If `config` has a key the launcher does not take.
    """
    keys = CONFIG_KEYS[impl]
    unknown = sorted(set(config) - {key for key, _ in keys})
    if unknown:
        raise KeyError(f"config has unknown keys: {unknown}")
    return tuple(cast(config[key]) if isinstance(cast, type)
                 else int(config.get(key, cast)) for key, cast in keys)


def runner(ext, buffers, config, dtype):
    """Return a zero-argument callable that launches `config` on the next
    rotation each call.
    """
    cycle = itertools.cycle(buffers)
    if dtype.impl == "smm":
        launch = ext.launcher(*config_args(config, "smm"))

        def call():
            a, b, sfa, sfb, _, _, c = next(cycle)
            return launch(a, b, sfa, sfb, out=c)
        return call

    launch = ext.launcher(*config_args(config, "mm"),
                          ELEM_INDEX[dtype.elem_a], ELEM_INDEX[dtype.elem_b])

    def call():
        a, b, c = next(cycle)
        return launch(a, b, out=c)
    return call


def correctness_error(ext, buffers, config, dtype, k):
    """Return None if `config` reproduces the reference on the first rotation,
    else the first line of the mismatch.
    """
    rtol, atol = dtype.tolerance(k)
    try:
        torch.testing.assert_close(runner(ext, buffers[:1], config, dtype)(),
                                   reference(buffers[0], dtype),
                                   rtol=rtol, atol=atol)
        return None
    except Exception as exc:
        return str(exc).splitlines()[0][:90]



def baseline_for(buffers, dtype):
    """Return torch's kernel for this dtype over the same rotations.

    Returns:
        `(callable, None)`, or `(None, reason)` when torch has nothing to
        compare
        against.
    """
    if dtype.impl == "smm":
        if dtype.baseline is False:
            return None, dtype.baseline_unavailable
        cycle = itertools.cycle([(a, b.t(), sfa, sfb, c)
                                 for a, b, sfa, sfb, _, _, c in buffers])

        def call():
            a, b, sfa, sfb, c = next(cycle)
            return torch._scaled_mm(a, b, sfa, sfb,
                                    out_dtype=torch.bfloat16, out=c)
    elif dtype.baseline is None:
        return None, dtype.baseline_unavailable
    elif dtype.baseline == "int_mm":
        m, n = buffers[0][-1].shape
        outs = [torch.empty((m, n), device="cuda", dtype=torch.int32)
                for _ in buffers]
        cycle = itertools.cycle([(a, b.t(), o)
                                 for (a, b, _), o in zip(buffers, outs)])

        def call():
            a, b, o = next(cycle)
            return torch._int_mm(a, b, out=o)
    elif dtype.baseline == "scaled_mm":
        unit = torch.ones((), device="cuda", dtype=torch.float32)
        cycle = itertools.cycle([(a, b.t(), c) for a, b, c in buffers])

        def call():
            a, b, c = next(cycle)
            return torch._scaled_mm(a, b, scale_a=unit, scale_b=unit,
                                    out_dtype=dtype.out_dtype, out=c)
    else:
        tf32 = dtype.baseline == "mm_tf32"
        cycle = itertools.cycle([(a, b.t(), c) for a, b, c in buffers])

        def call():
            a, b, c = next(cycle)
            previous = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = tf32
            try:
                if tf32:
                    return torch.mm(a, b, out=c)
                return torch.mm(a, b, out=c, out_dtype=dtype.out_dtype)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
    try:
        call()
        torch.cuda.synchronize()
        return call, None
    except Exception as exc:
        return None, str(exc).splitlines()[0][:110]



def elapsed_ms(fn, iterations):
    """Return the milliseconds `iterations` calls of `fn` take, by CUDA events.
    """
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    pending = collections.deque()
    for i in range(iterations):
        fn()
        # Bound how far the host runs ahead without ever letting the queue
        # drain.
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
    """Return a quick estimate of one call's milliseconds."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    return max(elapsed_ms(fn, probe) / probe, 1e-4)


def plan(fn, warmup_s=0.3, window_ms=20, min_iterations=50, max_window_s=0.15):
    """Return `(warmup, iterations)` for ranking candidates inside a sweep."""
    ms = per_iteration_ms(fn)
    affordable = int(max_window_s * 1000 / ms) + 1
    return (max(3, int(warmup_s * 1000 / ms)),
            max(int(window_ms / ms) + 1, min(min_iterations, affordable)))


def report_plan(fn, warmup_s=1.0, window_ms=300):
    """Return `(warmup, iterations)` for a number that gets reported."""
    ms = per_iteration_ms(fn)
    warmup = max(REPORT_WARMUP, int(warmup_s * 1000 / ms))
    iterations = max(REPORT_ITERATIONS, int(window_ms / ms) + 1)
    return (max(8, min(warmup, int(WARMUP_CAP_S * 1000 / ms))),
            max(8, min(iterations, int(WINDOW_CAP_S * 1000 / ms))))


def warm(fn, iterations):
    for i in range(iterations):
        fn()
        # A periodic sync keeps a long warmup from queueing unboundedly.
        if i % 64 == 63:
            torch.cuda.synchronize()
    torch.cuda.synchronize()


def timed(fn, warmup, iterations, repeats=5):
    """Return the median microseconds per call over `repeats` windows."""
    warm(fn, warmup)
    us = [elapsed_ms(fn, iterations) * 1000 / iterations
          for _ in range(repeats)]
    return statistics.median(us)


def env_stamp():
    """Return the GPU, device pin, torch version and date, for stored
    measurements.
    """
    return {"gpu": torch.cuda.get_device_name(0),
            "device": os.environ.get("CUDA_VISIBLE_DEVICES", "unpinned"),
            "torch": torch.__version__,
            "date": datetime.date.today().isoformat()}



# New tunings are written here, not into the package. Set OPENGEMM_CONFIGS to
# put the directory somewhere other than the working directory.
TUNED_DIR = "opengemm-configs"
TUNED_FILE = "tuned_configs.json"


def tuned_path():
    """Return the path of the local store new tunings are written to."""
    directory = os.environ.get("OPENGEMM_CONFIGS")
    return (Path(directory) if directory else Path.cwd() / TUNED_DIR) / TUNED_FILE


def config_path(impl):
    """Return the path of the configurations shipped with the package."""
    return SRC / impl / "configs.json"


def entry_key(entry):
    return (entry["dtype"], entry["m"], entry["n"], entry["k"])


def _entries(path):
    return json.loads(path.read_text())["entries"] if path.exists() else []


def tuned_entries(impl=None):
    """Return the local store's entries, optionally one implementation's."""
    return [e for e in _entries(tuned_path())
            if impl is None or DTYPES[e["dtype"]].impl == impl]


def stored_entries(impl):
    """Return every entry for `impl`, the local store taking precedence over
    the configurations shipped with the package.
    """
    entries = {entry_key(e): e for e in _entries(config_path(impl))}
    entries.update({entry_key(e): e for e in tuned_entries(impl)})
    return list(entries.values())


def load_configs(impl, dtype=None):
    """Return `{(dtype, m, n, k): config}` for the stored entries with a
    configuration, optionally one dtype's.
    """
    return {entry_key(e): e["config"] for e in stored_entries(impl)
            if e.get("config") is not None
            and (dtype is None or e["dtype"] == dtype)}


def stored_config(dtype, m, n, k):
    """Return the stored configuration for a (dtype name, shape).

    The local store is consulted first, then the shipped configurations.

    Returns:
        The configuration; None if untuned; or a string saying why the shape
        cannot be served.
    """
    key = (dtype, m, n, k)
    for entries in (tuned_entries(DTYPES[dtype].impl),
                    _entries(config_path(DTYPES[dtype].impl))):
        for e in entries:
            if entry_key(e) == key:
                return e["config"] if e.get("config") is not None \
                    else e.get("unimplementable", "recorded as unimplementable")
    return None


def write_entry(impl, entry):
    """Store `entry` in the local store, replacing any entry for the same
    (dtype, shape).

    Entries stay sorted by dtype and then shapes.jsonc order. Concurrent tuners
    share the file, so writers are serialized and the file replaced atomically.

    Returns:
        The path written.
    """
    path = tuned_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / ".tuned_configs.lock"
    with lock.open("w") as lock_file:
        os.lockf(lock_file.fileno(), os.F_LOCK, 0)
        store = (json.loads(path.read_text()) if path.exists()
                 else {"arch": "sm_100", "entries": []})
        key = entry_key(entry)
        shape_order = {s: i for i, s in enumerate(load_shapes())}
        dtype_order = {name: i for i, name in enumerate(DTYPES)}
        entries = [e for e in store["entries"] if entry_key(e) != key]
        entries.append(entry)
        store["entries"] = sorted(
            entries,
            key=lambda e: (dtype_order.get(e["dtype"], 1 << 30),
                           shape_order.get((e["m"], e["n"], e["k"]), 1 << 30)))
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(store, indent=1) + "\n")
        temporary.replace(path)
    return path

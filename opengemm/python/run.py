"""Compile and run a kernel emit_kernel wrote."""

import ctypes
import os
import time
from pathlib import Path

from .api import check_operands, output
from .build import SHARED_FLAGS, nvcc
from .capi import raw_stream
from .dtypes import DTYPES
from .emit import parse_tag
from .log import log

_loaded = {}

def build(source):
    """Compile `source` to a shared library beside it, reusing one newer than the source and header."""
    source = Path(source).resolve()
    header = source.with_suffix(".cuh")
    library = source.with_suffix(".so")
    newest = max(p.stat().st_mtime for p in (source, header) if p.exists())
    if library.exists() and library.stat().st_mtime >= newest:
        return library
    started = time.perf_counter()
    nvcc([source], library, *SHARED_FLAGS)
    log(f"built {library.name} in {time.perf_counter() - started:.1f} s")
    return library

def entry_point(source):
    """`(callable, impl, dtype, m, n, k)` for an emitted source, compiled and
    loaded on first use and again when it changes."""
    mtime = os.stat(source).st_mtime
    cached = _loaded.get(source)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    path = Path(source).resolve()
    impl, dtype, m, n, k, entry = parse_tag(path.read_text())
    function = getattr(ctypes.CDLL(str(build(path))), entry)
    function.argtypes = [ctypes.c_void_p] * (4 if impl == "mm" else 6)
    function.restype = None
    _loaded[source] = (mtime, (function, impl, dtype, m, n, k))
    return _loaded[source][1]

def run_kernel(file, a, b, sfa=None, sfb=None, out=None):
    """Run the kernel emitted to `file` on these tensors, on the current CUDA stream.

    Compiles on first use and caches the .so beside the source.

    Args:
        file: The .cu path `emit_kernel` wrote.
        a: `(M, K)` operand in the dtype the kernel was emitted for.
        b: `(N, K)` operand.
        sfa: Blocked scales of `a`; block-scaled kernels only.
        sfb: Blocked scales of `b`.
        out: Row-major output to write into.

    Returns:
        `(M, N)`, row-major: float32 or int32 for dense, bfloat16 for
        block-scaled.
    """
    call, impl, name, m, n, k = entry_point(file)
    d = DTYPES[name]
    check_operands(d, m, n, k, a, b, sfa, sfb)
    out = output(d, m, n, out, a.device)
    operands = (a, b, sfa, sfb, out) if impl == "smm" else (a, b, out)
    call(*(t.data_ptr() for t in operands), ctypes.c_void_p(raw_stream(a.get_device())))
    return out

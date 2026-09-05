import ctypes
import os
import subprocess
import time
from pathlib import Path

import torch

from .build import ARCH
from .dtypes import DTYPES
from .emit import parse_tag
from .log import log

NVCC = ["nvcc", "-O3", "-std=c++20", ARCH, "--expt-relaxed-constexpr",
        "-diag-suppress", "68,2361", "-shared", "-Xcompiler", "-fPIC",
        "-lcuda"]
_loaded = {}


def build(source):
    """Compile `source` to a shared library beside it, reusing one newer than
    the source and header.
    """
    source = Path(source).resolve()
    header = source.with_suffix(".cuh")
    library = source.with_suffix(".so")
    newest = max(p.stat().st_mtime for p in (source, header) if p.exists())
    if library.exists() and library.stat().st_mtime >= newest:
        return library
    started = time.perf_counter()
    result = subprocess.run(NVCC + [str(source), "-o", str(library)],
                            capture_output=True, text=True, cwd=source.parent)
    if result.returncode != 0:
        raise RuntimeError("nvcc failed:\n" + (result.stderr or result.stdout))
    log(f"built {library.name} in {time.perf_counter() - started:.1f} s")
    return library


def entry_point(source):
    """Return `(callable, impl, dtype, m, n, k)` for an emitted source,
    compiled and loaded on first use and again when it changes.
    """
    mtime = os.stat(source).st_mtime
    cached = _loaded.get(source)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    path = Path(source).resolve()
    impl, dtype, m, n, k, entry = parse_tag(path.read_text())
    library = build(path)
    handle = ctypes.CDLL(str(library))
    function = getattr(handle, entry)
    function.argtypes = [ctypes.c_void_p] * (4 if impl == "mm" else 6)
    function.restype = None
    _loaded[source] = (mtime, (function, impl, dtype, m, n, k))
    return _loaded[source][1]


def _raw_stream(device):
    try:
        # The private accessor skips building a Stream object on every call.
        return torch._C._cuda_getCurrentRawStream(device.index or 0)
    except AttributeError:
        return torch.cuda.current_stream(device).cuda_stream


def _check(t, name, shape, dtype):
    if t.shape != torch.Size(shape) or t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype} {list(shape)}, got {t.dtype} "
                         f"{list(t.shape)}: the kernel is compiled for one "
                         f"shape and dtype")
    if not (t.is_cuda and t.is_contiguous()):
        raise ValueError(f"{name} must be a contiguous CUDA tensor")


def run_kernel(file, a, b, sfa=None, sfb=None, out=None):
    """Run the kernel emitted to `file` on these tensors, on the current CUDA
    stream.

    Compiles on first use and caches the .so beside the source.

    Args:
        file: The .cu path `emit_kernel` wrote.
        a: `(M, K)` operand in the dtype the kernel was emitted for.
        b: `(N, K)` operand.
        sfa: Blocked scales of `a`; block-scaled kernels only.
        sfb: Blocked scales of `b`.
        out: Output to write into, laid out as the kernel writes it.

    Returns:
        `(M, N)` float32 or int32 with strides `(1, M)` for dense, bfloat16
        row-major for block-scaled.

    Raises:
        ValueError: If a tensor does not match the shape and dtype the kernel
            was compiled for.
    """
    call, impl, dtype, m, n, k = entry_point(file)
    d = DTYPES[dtype]
    if impl == "mm":
        _check(a, "a", (m, d.k_extent(k)), d.torch_dtype)
        _check(b, "b", (n, d.k_extent(k)), d.torch_dtype_b)
        if sfa is not None or sfb is not None:
            raise ValueError(f"{dtype} is a dense kernel; it takes no scales")
        if out is None:
            out = torch.empty_strided((m, n), (1, m), device=a.device,
                                      dtype=d.out_dtype)
        elif (out.shape != (m, n) or out.stride() != (1, m)
              or out.dtype != d.out_dtype):
            raise ValueError(f"out must be {d.out_dtype} [{m}, {n}] with "
                             f"strides (1, {m})")
        args = (a.data_ptr(), b.data_ptr(), out.data_ptr())
    else:
        if sfa is None or sfb is None:
            raise ValueError(f"{dtype} is block-scaled; pass sfa and sfb")
        _check(a, "a", (m, d.k_extent(k)), d.elem_dtype)
        _check(b, "b", (n, d.k_extent(k)), d.elem_dtype)
        for name, t in (("sfa", sfa), ("sfb", sfb)):
            if not (t.is_cuda and t.is_contiguous() and t.dtype == d.sf_dtype):
                raise ValueError(f"{name} must be a contiguous {d.sf_dtype} "
                                 f"CUDA tensor in the 128x4 blocked layout")
        if out is None:
            out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        elif (out.shape != (m, n) or not out.is_contiguous()
              or out.dtype != torch.bfloat16):
            raise ValueError(f"out must be contiguous bf16 [{m}, {n}]")
        args = (a.data_ptr(), b.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
                out.data_ptr())
    call(*args, ctypes.c_void_p(_raw_stream(a.device)))
    return out

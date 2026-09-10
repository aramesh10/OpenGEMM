"""C[M, N] = A[M, K] @ B[N, K].T on B200."""

import torch

from . import capi
from .configs import resolve_config
from .dtypes import DTYPES, dtype_name, extent, k_of, sf_block

_bound = {}  # (dtype name, m, n, k) -> (Library, OgGemm)


def bind(name, m, n, k, device):
    """Resolve the kernel for a shape and memoize it, tuning first when nothing is stored."""
    with torch.cuda.device(device):  # a sweep, if one runs, happens on the operands' device
        config = resolve_config(name, m, n, k)
    _bound[(name, m, n, k)] = capi.bind(name, m, n, k, config)
    return _bound[(name, m, n, k)]


def scale_bytes(d, rows, k):
    """Bytes of a scale tensor in the 128x4 blocked layout."""
    return (-(-rows // 128) * 128) * (-(-k // (sf_block(d) * 4)) * 4)


def check_operands(d, m, n, k, a, b, sfa, sfb):
    """Raise unless the operands are what a kernel for this format and shape reads."""
    cols = extent(d, k)
    for name, t, rows, dtype in (("a", a, m, d.a), ("b", b, n, d.b)):
        if not (t.is_cuda and t.dim() == 2 and t.is_contiguous()):
            raise TypeError(f"{name} must be a contiguous 2-D CUDA tensor")
        if t.dtype is not dtype or t.shape != (rows, cols):
            raise TypeError(
                f"{name} must be {dtype} [{rows}, {cols}], got {t.dtype} {list(t.shape)}"
            )
    if b.device != a.device:
        raise ValueError(f"a is on {a.device} and b on {b.device}")
    if d.impl == "mm":
        if sfa is not None or sfb is not None:
            raise TypeError("a dense format takes no scales")
        return
    if sfa is None or sfb is None:
        raise TypeError("a block-scaled format takes both sfa and sfb")
    for name, t, rows in (("sfa", sfa, m), ("sfb", sfb, n)):
        if not (t.is_cuda and t.is_contiguous() and t.dtype is d.sf and t.device == a.device):
            raise TypeError(
                f"{name} must be a contiguous {d.sf} tensor on {a.device}, in the 128x4 blocked layout"
            )
        want = scale_bytes(d, rows, k)
        if t.numel() < want:
            raise ValueError(
                f"{name} must hold {want} bytes of the 128x4 blocked scale layout, got {t.numel()}"
            )


def output(d, m, n, out, device):
    """The tensor C is written to: `out` checked, or a new row-major one."""
    if out is None:
        return torch.empty((m, n), device=device, dtype=d.out)
    if not (out.is_contiguous() and out.shape == (m, n) and out.dtype is d.out and out.device == device):
        raise ValueError(f"out must be a contiguous {d.out} [{m}, {n}] on {device}")
    if out.data_ptr() % 16:
        raise ValueError("out must start on a 16-byte boundary")
    return out


def gemm(a, b, sfa=None, sfb=None, out=None, atype=None, btype=None):
    """Compute C[M, N] = A[M, K] @ B[N, K].T.

    Args:
        a: `(M, K)` operand.
        b: `(N, K)` operand.
        sfa: Scales of `a` in the 128x4 blocked layout `torch._scaled_mm`
            takes (`to_blocked` builds it); block-scaled formats only.
        sfb: Scales of `b`, likewise.
        out: Output to write into instead of allocating one.
        atype: Element name for a `uint8` operand: "u8", "e3m2", "e2m3" or "e2m1".
        btype: Element name for `b`; defaults to `atype` when the dtypes match.

    Returns:
        `(M, N)`, row-major: float32 for dense formats, int32 for s8 and u8,
        bfloat16 for block-scaled ones.
    """
    name = dtype_name(a, b, sfa, atype, btype)
    d = DTYPES[name]
    m, n, k = a.size(0), b.size(0), k_of(d, a.size(1))
    check_operands(d, m, n, k, a, b, sfa, sfb)
    lib, g = _bound.get((name, m, n, k)) or bind(name, m, n, k, a.device)
    out = output(d, m, n, out, a.device)
    lib.launch(g, a, b, sfa, sfb, out)
    return out

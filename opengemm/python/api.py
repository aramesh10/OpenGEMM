"""C[M, N] = A[M, K] @ B[N, K].T on B200."""

import torch

from .bench import config_args
from .build import extension
from .dtypes import DENSE_OF_TORCH, DTYPES, ELEM_INDEX, SCALED_OF_TORCH
from .tune import resolve_config

# Keyed by (dtype name, M, N, K).
_launchers = {}


def _dense_elem(t, name, which):
    if name is not None:
        if name not in ELEM_INDEX:
            raise ValueError(f"{which}type={name!r}; have {list(ELEM_INDEX)}")
        return name
    elem = DENSE_OF_TORCH.get(t.dtype)
    if elem is None:
        raise TypeError(
            f"{which} is {t.dtype}, which names no element: pass "
            f"{which}type= (uint8 carries u8, e3m2, e2m3 and e2m1 alike; "
            f"the rest are inferred)")
    return elem


def dtype_name(a, b, sfa=None, atype=None, btype=None):
    """Return the dtype name for these operands, as configs.json spells it.

    Args:
        a: Left operand.
        b: Right operand.
        sfa: Scales of `a` for a block-scaled call, else None.
        atype: Element name for a `uint8` operand: `"u8"`, `"e3m2"`, `"e2m3"`
            or `"e2m1"`.
        btype: Element name for `b`; defaults to `atype` when the dtypes
            match.

    Returns:
        A name such as `"bf16"`, `"e4m3xe5m2"` or `"nvfp4"`.

    Raises:
        TypeError: If the dtypes name no element or no block-scaled format.
    """
    if sfa is not None:
        if (a.dtype, sfa.dtype) not in SCALED_OF_TORCH:
            raise TypeError(
                f"{a.dtype} operands with {sfa.dtype} scales is not a "
                f"format; have { {v: k for k, v in SCALED_OF_TORCH.items()} }")
        return SCALED_OF_TORCH[(a.dtype, sfa.dtype)]
    ea = _dense_elem(a, atype, "a")
    eb = _dense_elem(b, btype or (atype if b.dtype == a.dtype else None), "b")
    return ea if ea == eb else f"{ea}x{eb}"


def launcher(dtype, m, n, k):
    """Return the launcher bound to the stored configuration for a shape.

    A shape with no stored configuration is tuned first and the winner stored.

    Args:
        dtype: Dtype name as configs.json spells it.
        m: Rows of A and C.
        n: Rows of B and columns of C.
        k: Reduction length in elements, not bytes.

    Returns:
        A callable taking the operands and an optional `out`.
    """
    key = (dtype, m, n, k)
    launch = _launchers.get(key)
    if launch is None:
        d = DTYPES[dtype]
        config = resolve_config(dtype, m, n, k)
        args = config_args(config, d.impl)
        if d.impl == "mm":
            args += (ELEM_INDEX[d.elem_a], ELEM_INDEX[d.elem_b])
        launch = _launchers[key] = extension(d.impl).launcher(*args)
    return launch


def gemm(a, b, sfa=None, sfb=None, out=None, atype=None, btype=None):
    """Compute C[M, N] = A[M, K] @ B[N, K].T.

    Both operands are row-major with K innermost. Every call runs a measured
    configuration from configs.json; a (dtype, shape) with no entry is tuned on
    its first call, a few minutes on the GPU, and the winner is stored. The
    first call builds the extension.

    Dense: the element is inferred from the dtype (bfloat16, float16, float32
    computed as tf32, int8, float8_e4m3fn, float8_e5m2). uint8 holds u8, e3m2,
    e2m3 or e2m1, fp6 and fp4 packed densely along K with K a multiple of 128,
    and is named with `atype` and `btype`. Mixed e4m3 x e5m2 is supported.

    Block-scaled: nvfp4 is float4_e2m1fn_x2 operands with float8_e4m3fn scales
    per 16, mxfp8 is float8_e4m3fn with float8_e8m0fnu per 32, mxfp4 is
    float4_e2m1fn_x2 with float8_e8m0fnu per 32.

    Args:
        a: `(M, K)` operand.
        b: `(N, K)` operand.
        sfa: Scales of `a` in the 128x4 blocked layout `torch._scaled_mm` takes
            (`to_blocked` builds it); block-scaled calls only.
        sfb: Scales of `b`, likewise.
        out: Output to write into instead of allocating one.
        atype: Element name for a `uint8` operand.
        btype: Element name for `b`; defaults to `atype` when the dtypes
            match.

    Returns:
        Dense: `(M, N)` float32, or int32 for int8, with column-major strides
        `(1, M)`, which is how the accumulator leaves tensor memory; call
        `.contiguous()` for row-major. Block-scaled: `(M, N)` bfloat16,
        row-major.
    """
    if (sfa is None) != (sfb is None):
        raise ValueError("sfa and sfb go together: both or neither")
    dtype = dtype_name(a, b, sfa, atype, btype)
    d = DTYPES[dtype]
    key = (dtype, a.size(0), b.size(0), d.k_values(a.size(1)))
    launch = _launchers.get(key)
    if launch is None:
        if not a.is_cuda:
            raise TypeError("a and b must be CUDA tensors")
        # A sweep allocates and times on the current device; make it the
        # operands'.
        with torch.cuda.device(a.device):
            launch = launcher(*key)
    if sfa is None:
        return launch(a, b, out=out)
    return launch(a, b, sfa, sfb, out=out)

"""The formats gemm() serves, as torch sees them.

Everything about the kernel side, the element names, the tile alignment, the
scale blocks, lives in the C registry; Python reads element names out of
registry.cuh only to name a format from names.
"""

import re
from pathlib import Path
from typing import NamedTuple

import torch
from torch import (
    bfloat16,
    float16,
    float32,
    float4_e2m1fn_x2,
    float8_e4m3fn,
    float8_e5m2,
    float8_e8m0fnu,
    int8,
    int32,
    uint8,
)

SRC = Path(__file__).resolve().parents[1] / "src"

class Dtype(NamedTuple):
    impl: str  # "mm" dense, "smm" block-scaled
    a: torch.dtype  # of a
    b: torch.dtype  # of b
    bits: int  # per value; under 8 means packed along K
    out: torch.dtype  # of c
    sf: torch.dtype  # of the scales; None for dense

# fmt: off
DTYPES = {
    #                  impl   a                 b                 bits out       sf
    "bf16":      Dtype("mm",  bfloat16,         bfloat16,         16,  float32,  None),
    "f16":       Dtype("mm",  float16,          float16,          16,  float32,  None),
    "tf32":      Dtype("mm",  float32,          float32,          32,  float32,  None),
    "s8":        Dtype("mm",  int8,             int8,             8,   int32,    None),
    "u8":        Dtype("mm",  uint8,            uint8,            8,   int32,    None),
    "e4m3":      Dtype("mm",  float8_e4m3fn,    float8_e4m3fn,    8,   float32,  None),
    "e5m2":      Dtype("mm",  float8_e5m2,      float8_e5m2,      8,   float32,  None),
    "e3m2":      Dtype("mm",  uint8,            uint8,            6,   float32,  None),
    "e2m3":      Dtype("mm",  uint8,            uint8,            6,   float32,  None),
    "e2m1":      Dtype("mm",  uint8,            uint8,            4,   float32,  None),
    "e4m3xe5m2": Dtype("mm",  float8_e4m3fn,    float8_e5m2,      8,   float32,  None),
    "nvfp4":     Dtype("smm", float4_e2m1fn_x2, float4_e2m1fn_x2, 4,   bfloat16, float8_e4m3fn),
    "mxfp8":     Dtype("smm", float8_e4m3fn,    float8_e4m3fn,    8,   bfloat16, float8_e8m0fnu),
    "mxfp4":     Dtype("smm", float4_e2m1fn_x2, float4_e2m1fn_x2, 4,   bfloat16, float8_e8m0fnu),
}
# fmt: on

DENSE_OF_TORCH = {
    d.a: name for name, d in DTYPES.items() if d.impl == "mm" and d.a is d.b and d.a is not uint8
}
SCALED_OF_TORCH = {(d.a, d.sf): name for name, d in DTYPES.items() if d.impl == "smm"}

_ELEMS = {}

def elems(name):
    """The element names of a format as registry.cuh spells them.

    `(a, b)` for a dense format, `(elem, sf)` for a block-scaled one.
    """
    if not _ELEMS:
        for impl in ("mm", "smm"):
            text = (SRC / impl / "registry.cuh").read_text()
            for found in re.finditer(r"X\((\w+), (\w+), (\w+)\)", text):
                _ELEMS[found.group(1)] = (found.group(2), found.group(3))
    return _ELEMS[name]

def _dense(t, which):
    name = DENSE_OF_TORCH.get(t.dtype)
    if name is None:
        raise TypeError(
            f"{which} is {t.dtype}, which names no element: pass {which}type= "
            f"(uint8 carries u8, e3m2, e2m3 and e2m1 alike; the rest are inferred)"
        )
    return name

def dtype_name(a=None, b=None, sfa=None, atype=None, btype=None, sftype=None):
    """The DTYPES key for these operands, or for these element names.

    Args:
        a: `(M, K)` operand, or None to name the format from `atype` alone.
        b: `(N, K)` operand.
        sfa: Scales of `a`, for a block-scaled call.
        atype: Element of `a`: "bf16", "e4m3", "e2m1" and the rest of the
            dense names. Required for a uint8 operand.
        btype: Element of `b`; defaults to `atype` when the dtypes match.
        sftype: Scale element, "ue4m3" or "ue8m0", for a block-scaled format
            named without operands.

    Raises:
        TypeError: If the operands or the names pick out no format.
    """
    if sfa is not None:
        name = SCALED_OF_TORCH.get((a.dtype, sfa.dtype))
        if name is None:
            raise TypeError(
                f"{a.dtype} operands with {sfa.dtype} scales is not a format; "
                f"have {SCALED_OF_TORCH}"
            )
        return name
    if sftype is not None:
        for name, d in DTYPES.items():
            if d.impl == "smm" and elems(name) == (atype, sftype):
                return name
        raise TypeError(f"atype={atype!r} with sftype={sftype!r} is not a format")
    if a is not None:
        atype = atype or _dense(a, "a")
        btype = btype or (atype if b.dtype == a.dtype else _dense(b, "b"))
    if atype is None:
        raise TypeError("without operands, pass atype= (and sftype= for a block-scaled format)")
    name = atype if btype in (None, atype) else f"{atype}x{btype}"
    if name not in DTYPES:
        raise TypeError(f"{name} is not a format this build serves; have {list(DTYPES)}")
    return name

def extent(d, k):
    """The row extent of a `(rows, K)` operand as torch sees it."""
    return k * d.bits // 8 if d.bits < 8 else k

def k_of(d, extent):
    """K in values for a row extent as torch sees it; the inverse of `extent`."""
    return extent * 8 // d.bits if d.bits < 8 else extent

def sf_block(d):
    """Values per scale of a block-scaled format: 16 for ue4m3 scales, 32 for ue8m0."""
    return 16 if d.sf is float8_e4m3fn else 32

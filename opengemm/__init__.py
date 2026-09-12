"""opengemm: open GEMM kernels for NVIDIA B200 (sm_100a) in plain CUDA.

import opengemm as og
c = og.gemm(a, b)                      # dense
c = og.gemm(a, b, sfa, sfb)            # block-scaled
og.emit_kernel(a, b, file="k.cu")      # a standalone .cu/.cuh pair
og.run_kernel("k.cu", a, b)            # compile and run it
"""

from .python.api import gemm
from .python.build import prebuild
from .python.dtypes import DTYPES, dtype_name
from .python.emit import emit_kernel
from .python.quant import pack, quantize, to_blocked, unpack
from .python.run import run_kernel

__all__ = [
    "gemm",
    "prebuild",
    "emit_kernel",
    "run_kernel",
    "dtype_name",
    "DTYPES",
    "quantize",
    "to_blocked",
    "pack",
    "unpack",
]
__version__ = "0.1.4"

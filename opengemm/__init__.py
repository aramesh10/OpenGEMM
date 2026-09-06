"""opengemm: open GEMM kernels for NVIDIA B200 (sm_100a) in plain CUDA.

    import opengemm as og
    c = og.gemm(a, b)                      # dense
    c = og.gemm(a, b, sfa, sfb)            # block-scaled
    og.emit_kernel(a, b, file="k.cu")      # a standalone .cu/.cuh pair
    og.run_kernel("k.cu", a, b)            # compile and run it
"""

from .python.api import dtype_name, gemm
from .python.build import prebuild
from .python.dtypes import DTYPES, pack_e2m1, quantize, to_blocked, unpack_e2m1
from .python.emit import emit_kernel
from .python.run import run_kernel

__all__ = ["gemm", "prebuild", "emit_kernel", "run_kernel", "dtype_name",
           "DTYPES", "quantize", "to_blocked", "pack_e2m1", "unpack_e2m1"]
__version__ = "0.1.0"

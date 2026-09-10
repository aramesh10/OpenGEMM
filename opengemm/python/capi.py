"""The compiled libraries, through the C ABI src/common/abi.h declares."""

import ctypes
import threading
from ctypes import POINTER, byref, c_char, c_char_p, c_int32, c_void_p

import torch

from . import build
from .dtypes import DTYPES

ABI_VERSION = 3
OG_OK = 0
OG_ERR_NO_KERNEL = -2

class OgGemm(ctypes.Structure):
    """One GEMM as C sees it; the fields mirror OgGemm in src/common/abi.h."""

    # fmt: off
    _fields_ = [
        ("m", c_int32), ("n", c_int32), ("k", c_int32),
        ("dtype", c_char * 16),
        ("use_2cta", c_int32), ("output_n", c_int32), ("use_clc", c_int32), ("swap_ab", c_int32),
        ("cluster_m", c_int32), ("cluster_n", c_int32), ("cluster_k", c_int32),
        ("block_m", c_int32), ("stages", c_int32), ("epi_hold", c_int32),
        ("epi_double", c_int32), ("epi_direct", c_int32),
        ("epi_trade", c_int32), ("deep_stages", c_int32),
        ("supergroup", c_int32), ("split_k", c_int32), ("walk", c_int32), ("l2_promo", c_int32),
        ("persistent", c_int32),
        ("row", c_int32),
    ]
    # fmt: on

assert ctypes.sizeof(OgGemm) == 108

KERNEL_KEYS = {
    "mm": (
        "use_2cta",
        "output_n",
        "use_clc",
        "swap_ab",
        "cluster_m",
        "cluster_n",
        "cluster_k",
        "block_m",
        "stages",
        "epi_hold",
        "epi_double",
        "epi_direct",
        "split_k",
    ),
    "smm": (
        "use_2cta",
        "output_n",
        "use_clc",
        "swap_ab",
        "cluster_m",
        "cluster_n",
        "cluster_k",
        "epi_trade",
        "deep_stages",
    ),
}
LAUNCH_KEYS = {
    "mm": ("supergroup", "walk", "l2_promo"),
    "smm": ("supergroup", "epi_direct", "persistent"),
}


def raw_stream(device):
    """The current CUDA stream on device index `device`, as a raw handle."""
    try:
        return torch._C._cuda_getCurrentRawStream(device)
    except AttributeError:
        return torch.cuda.current_stream(device).cuda_stream


def scale_bytes(d, rows, k):
    """Bytes of a scale tensor in the 128x4 blocked layout."""
    return (-(-rows // 128) * 128) * (-(-k // (d.block * 4)) * 4)


class Library:
    """One compiled library: its six entry points, ABI checked at load."""

    def __init__(self, impl):
        self.impl = impl
        self.path = build.library_path(impl)
        lib = ctypes.CDLL(str(self.path))
        prefix = f"og_{impl}_"
        self._abi = lib[prefix + "abi"]
        self._abi.restype = c_int32
        self._error = lib[prefix + "error"]
        self._error.restype = c_char_p
        self._count = lib[prefix + "kernels"]
        self._count.restype = c_int32
        self._kernel = lib[prefix + "kernel"]
        self._kernel.argtypes = [c_int32, POINTER(OgGemm)]
        self._bind = lib[prefix + "bind"]
        self._bind.argtypes = [POINTER(OgGemm)]
        self._launch = lib[prefix + "launch"]
        self._launch.argtypes = [
            POINTER(OgGemm),
            c_void_p,
            c_void_p,
            c_void_p,
            c_void_p,
            c_void_p,
            c_int32,
            c_void_p,
        ]
        found = self._abi()
        if found != ABI_VERSION:
            raise RuntimeError(
                f"{impl} library {self.path} is ABI version {found}, this build speaks "
                f"{ABI_VERSION}; delete it and let it rebuild, or set OPENGEMM_JIT=1"
            )
        self._kernels = None

    def error(self):
        return self._error().decode()

    def kernels(self):
        """Every kernel this build compiled, as configs.json dicts with a `dtype` key."""
        if self._kernels is None:
            rows = []
            for i in range(self._count()):
                g = OgGemm()
                self._kernel(i, byref(g))
                row = {"dtype": g.dtype.decode()}
                for key in KERNEL_KEYS[self.impl]:
                    row[key] = getattr(g, key)
                rows.append(row)
            self._kernels = rows
        return self._kernels

    def spell(self, row):
        return " ".join(f"{key}={int(row[key])}" for key in KERNEL_KEYS[self.impl])

    def bind(self, name, m, n, k, config):
        """Resolve a configs.json entry for a shape to a compiled kernel.

        Returns the OgGemm to launch with. Do not edit it afterwards: the row
        it holds was resolved for exactly these fields.
        """
        knobs = {key: int(config[key]) for key in KERNEL_KEYS[self.impl] + LAUNCH_KEYS[self.impl]}
        g = OgGemm(m=m, n=n, k=k, dtype=name.encode(), **knobs)
        status = self._bind(byref(g))
        if status == OG_OK:
            return g
        message = self.error()
        if status == OG_ERR_NO_KERNEL:
            rows = [self.spell(row) for row in self.kernels() if row["dtype"] == name]
            message += (
                f"\n\nthis build compiles these {name} kernels "
                f"(src/{self.impl}/registry.cuh):\n  " + "\n  ".join(rows)
            )
        raise RuntimeError(message)

    def launch(self, g, a, b, sfa, sfb, out):
        """Enqueue the bound kernel on the current stream of the operands' device."""
        device = a.get_device()
        status = self._launch(
            byref(g),
            a.data_ptr(),
            b.data_ptr(),
            0 if sfa is None else sfa.data_ptr(),
            0 if sfb is None else sfb.data_ptr(),
            out.data_ptr(),
            device,
            raw_stream(device),
        )
        if status != OG_OK:
            raise RuntimeError(self.error())

_libraries, _lock = {}, threading.Lock()

def library(impl):
    """The loaded library for `impl`, loading it on first use."""
    with _lock:
        if impl not in _libraries:
            _libraries[impl] = Library(impl)
    return _libraries[impl]

def bind(name, m, n, k, config):
    """Return `(library, OgGemm)` for a configuration on a shape."""
    lib = library(DTYPES[name].impl)
    return lib, lib.bind(name, m, n, k, config)

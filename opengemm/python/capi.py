"""The compiled kernels, bound through ctypes.

Each library exports the C surface `src/<impl>/capi.h` declares and links no
libtorch and no libpython, so one build serves every torch and every Python
version. The work around the launch -- checking the operands, allocating the
output, padding an unaligned K -- is done here rather than in C++, where
dtypes.py already holds the element metadata it needs.
"""

import ctypes
import threading

import torch

from . import build
from .dtypes import DTYPES, ELEMS
from .policy import mm_policy, smm_policy

OG_OK = 0
OG_ERR_NO_POLICY = -1

ABI_VERSION = {"mm": 1, "smm": 1}
# Which enumeration each registry column names, so a row reads back in the
# vocabulary the rest of the package speaks.
ENUM_COLUMNS = {"mm": {"elem_a": "elem", "elem_b": "elem"},
                "smm": {"elem": "elem", "sf": "sf"}}

K_PAD_AUTO, K_PAD_COPY = -1, 1


def _fields(names):
    return [(name, ctypes.c_int32) for name in names]


class MmConfig(ctypes.Structure):
    """`OgMmConfig` of src/mm/capi.h."""
    _fields_ = _fields((
        "elem_a", "elem_b", "cta_group", "block_m", "mma_n", "stages",
        "swap_ab", "epi_hold", "epi_mode", "epi_direct", "use_clc", "split_k",
        "rm", "rn", "rk", "supergroup", "walk", "l2_promo", "splits"))


class MmShape(ctypes.Structure):
    """`OgMmShape` of src/mm/capi.h."""
    _fields_ = _fields(("m", "n", "k", "a_pitch", "b_pitch"))


class SmmConfig(ctypes.Structure):
    """`OgSmmConfig` of src/smm/capi.h."""
    _fields_ = _fields((
        "elem_a", "elem_b", "elem_sf", "cta_group", "mma_n", "swap_ab",
        "epi_trade", "deep", "use_clc", "rm", "rn", "rk", "supergroup",
        "epi_direct", "persistent"))


class SmmShape(ctypes.Structure):
    """`OgSmmShape` of src/smm/capi.h."""
    _fields_ = _fields(("m", "n", "k"))


STRUCTS = {"mm": (MmConfig, MmShape), "smm": (SmmConfig, SmmShape)}


def raw_stream(device):
    """Return the current CUDA stream on `device` as a raw handle."""
    try:
        # The private accessor skips building a Stream object on every call.
        return torch._C._cuda_getCurrentRawStream(device.index or 0)
    except AttributeError:
        return torch.cuda.current_stream(device).cuda_stream


class Library:
    """One compiled library, with its prototypes declared and ABI checked."""

    def __init__(self, impl):
        self.impl = impl
        self.path = build.library_path(impl)
        self.handle = ctypes.CDLL(str(self.path))
        config_type, shape_type = STRUCTS[impl]
        self.config_type, self.shape_type = config_type, shape_type
        p = f"og_{impl}_"
        for name, restype in (("abi_version", ctypes.c_int32),
                              ("config_bytes", ctypes.c_int32),
                              ("shape_bytes", ctypes.c_int32),
                              ("elem_names", ctypes.c_char_p),
                              ("registry_fields", ctypes.c_char_p),
                              ("registry_rows", ctypes.c_int32),
                              ("registry_cols", ctypes.c_int32),
                              ("registry_data",
                               ctypes.POINTER(ctypes.c_int32)),
                              ("error", ctypes.c_char_p)):
            entry = getattr(self.handle, p + name)
            entry.restype, entry.argtypes = restype, []
            setattr(self, name, entry)
        if impl == "smm":
            self.sf_names = self.handle.og_smm_sf_names
            self.sf_names.restype, self.sf_names.argtypes = ctypes.c_char_p, []
        pointers = 5 if impl == "mm" else 7
        self.launch = getattr(self.handle, p + "launch")
        self.launch.restype = ctypes.c_int32
        self.launch.argtypes = ([ctypes.POINTER(config_type),
                                 ctypes.POINTER(shape_type)]
                                + [ctypes.c_void_p] * (pointers - 2)
                                + [ctypes.c_int32, ctypes.c_void_p])
        self._check()

    def _check(self):
        """Fail at load rather than at launch when the library and this module
        disagree about the ABI.
        """
        where = f"{self.impl} library {self.path}"
        found, want = self.abi_version(), ABI_VERSION[self.impl]
        if found != want:
            raise RuntimeError(
                f"{where} is ABI version {found}, this build speaks {want}; "
                f"delete it and let it rebuild, or set OPENGEMM_JIT=1")
        for label, size, struct in (("config", self.config_bytes(),
                                     self.config_type),
                                    ("shape", self.shape_bytes(),
                                     self.shape_type)):
            if size != ctypes.sizeof(struct):
                raise RuntimeError(
                    f"{where} has a {size}-byte {label} struct; this module "
                    f"describes {ctypes.sizeof(struct)} bytes")
        if self.impl == "mm":
            names = self.elem_names().decode().split(",")
            if names != list(ELEMS):
                raise RuntimeError(
                    f"{where} names elements {names}; dtypes.ELEMS is "
                    f"{list(ELEMS)}. One of the two was edited alone.")

    def names(self, which):
        """Return the enumerators of `"elem"` or `"sf"`, in ordinal order."""
        entry = self.sf_names if which == "sf" else self.elem_names
        return entry().decode().split(",")

    def registry(self):
        """Return every configuration this build compiles, as dicts.

        Elements read back as names rather than ordinals, because the block-
        scaled enumerations do not share an order with the dense one.
        """
        fields = self.registry_fields().decode().split(",")
        rows, cols = self.registry_rows(), self.registry_cols()
        flat = ctypes.cast(self.registry_data(),
                           ctypes.POINTER(ctypes.c_int32 * (rows * cols)))
        flat = flat.contents
        enums = {column: self.names(which)
                 for column, which in ENUM_COLUMNS[self.impl].items()}
        out = []
        for row in range(rows):
            at = row * cols
            entry = dict(zip(fields, flat[at:at + cols]))
            for column, spelling in enums.items():
                entry[column] = spelling[entry[column]]
            out.append(entry)
        return out

    def unmatched(self, config):
        """Return the message for a config the registry does not hold."""
        wanted = " ".join(f"{key}={value}" for key, value in config.items())
        lines = sorted({" ".join(f"{k}={v}" for k, v in row.items())
                        for row in self.registry()})
        return ("no compiled configuration matches this request:\n  wanted "
                + wanted + "\n\nthis build compiles exactly these "
                "configurations (registry.cuh):\n"
                + "".join(f"  {line}\n" for line in lines))


_libraries, _lock = {}, threading.Lock()


def library(impl):
    """Return the loaded library for `impl`, loading it on first use."""
    with _lock:
        if impl not in _libraries:
            _libraries[impl] = Library(impl)
    return _libraries[impl]


def registry(impl):
    """Return every configuration `impl`'s build compiles, as dicts."""
    return library(impl).registry()


class MmLauncher:
    """A dense configuration bound to a shape, then launched many times.

    Everything the shape and the configuration decide is settled here, so a
    launch is a handful of checks and one call.
    """

    def __init__(self, dtype, config, m, n, k):
        self.d = d = DTYPES[dtype]
        self.lib = library("mm")
        self.dtype, self.m, self.n, self.k = dtype, m, n, k
        policy, bound = mm_policy(dtype, config, k)
        self.policy = policy
        self.config = MmConfig(
            elem_a=ELEMS.index(policy["elem_a"]),
            elem_b=ELEMS.index(policy["elem_b"]),
            **{key: int(value) for key, value in policy.items()
               if key not in ("elem_a", "elem_b")},
            supergroup=int(bound["supergroup"]), walk=int(bound["walk"]),
            l2_promo=int(bound["l2_promo"]), splits=int(bound["splits"]))
        # Padding rescues an unaligned pitch by widening the row and leaving
        # the tail unread; that cannot rescue fp6 or fp4, whose expanding
        # tensor maps require the true K to be a multiple of 128.
        unaligned = k % d.k_align != 0
        if unaligned and d.bits < 8:
            raise ValueError(
                f"K = {k} is not a multiple of {d.k_align}, which the "
                f"expanding tensor map for {policy['elem_a']} requires")
        k_pad = config.get("k_pad", K_PAD_AUTO)
        self.pad_to = 0
        if unaligned:
            if not (unaligned if k_pad == K_PAD_AUTO else k_pad == K_PAD_COPY):
                raise ValueError(
                    f"K = {k} is not a multiple of {d.k_align} and the pad_k "
                    f"variant was not requested")
            self.pad_to = -(-k // d.k_align) * d.k_align
        pitch = self.pad_to or k
        self.shape = MmShape(m=m, n=n, k=k, a_pitch=pitch, b_pitch=pitch)
        self.extent = d.k_extent(k)
        # The kernel accumulates into C rather than overwriting it whenever the
        # work is split, so C has to arrive zeroed.
        self.zero_out = bound["splits"] > 1 or policy["rk"] > 1

    def _operand(self, t, name, rows, dtype):
        if not (t.is_cuda and t.dim() == 2 and t.is_contiguous()):
            raise TypeError(f"{name} must be a contiguous 2-D CUDA tensor")
        if t.dtype is not dtype or t.shape != (rows, self.extent):
            raise TypeError(
                f"{name} must be {dtype} [{rows}, {self.extent}], got "
                f"{t.dtype} {list(t.shape)}")
        if self.pad_to:
            wide = torch.empty((rows, self.d.k_extent(self.pad_to)),
                               dtype=dtype, device=t.device)
            wide[:, :self.extent] = t
            return wide
        return t

    def __call__(self, a, b, out=None):
        a = self._operand(a, "a", self.m, self.d.torch_dtype)
        b = self._operand(b, "b", self.n, self.d.torch_dtype_b)
        strides = (1, self.m)
        if out is None:
            out = torch.empty_strided((self.m, self.n), strides,
                                      device=a.device, dtype=self.d.out_dtype)
        elif (out.shape != (self.m, self.n) or out.stride() != strides
                or out.dtype is not self.d.out_dtype):
            raise ValueError(
                f"out must be {self.d.out_dtype} [{self.m}, {self.n}] with "
                f"strides {strides}")
        if self.zero_out:
            out.zero_()
        status = self.lib.launch(
            ctypes.byref(self.config), ctypes.byref(self.shape),
            a.data_ptr(), b.data_ptr(), out.data_ptr(), a.device.index or 0,
            ctypes.c_void_p(raw_stream(a.device)))
        if status != OG_OK:
            raise RuntimeError(self.lib.unmatched(self.policy)
                               if status == OG_ERR_NO_POLICY
                               else self.lib.error().decode())
        return out


class SmmLauncher:
    """A block-scaled configuration bound to a shape."""

    def __init__(self, dtype, config, m, n, k):
        self.d = d = DTYPES[dtype]
        self.lib = library("smm")
        self.dtype, self.m, self.n, self.k = dtype, m, n, k
        policy, bound = smm_policy(dtype, config)
        self.policy = policy
        elems = self.lib.names("elem")
        self.config = SmmConfig(
            elem_a=elems.index(policy["elem_a"]),
            elem_b=elems.index(policy["elem_b"]),
            elem_sf=self.lib.names("sf").index(policy["elem_sf"]),
            **{key: int(value) for key, value in policy.items()
               if not key.startswith("elem")},
            supergroup=int(bound["supergroup"]),
            epi_direct=int(bound["epi_direct"]),
            persistent=int(bound["persistent"]))
        if k % d.block:
            raise ValueError(f"K = {k} is not a multiple of the "
                             f"{d.block}-wide scale block")
        self.shape = SmmShape(m=m, n=n, k=k)
        self.extent = d.k_extent(k)
        self.scale_bytes = {rows: self._scale_bytes(rows) for rows in (m, n)}

    def _scale_bytes(self, rows):
        """Bytes of a scale tensor in the 128x4 blocked layout."""
        return (-(-rows // 128) * 128) * (-(-self.k // (self.d.block * 4)) * 4)

    def _operand(self, t, name, rows):
        if not (t.is_cuda and t.dim() == 2 and t.is_contiguous()):
            raise TypeError(f"{name} must be a contiguous 2-D CUDA tensor")
        if t.dtype is not self.d.elem_dtype or t.shape != (rows, self.extent):
            raise TypeError(
                f"{name} must be {self.d.elem_dtype} [{rows}, {self.extent}], "
                f"got {t.dtype} {list(t.shape)}")

    def _scales(self, t, name, rows):
        if not (t.is_cuda and t.is_contiguous()
                and t.dtype is self.d.sf_dtype):
            raise TypeError(f"{name} must be a contiguous {self.d.sf_dtype} "
                            f"CUDA tensor in the 128x4 blocked layout")
        want = self.scale_bytes[rows]
        if t.numel() < want:
            raise ValueError(f"{name} must hold {want} bytes of the 128x4 "
                             f"blocked scale layout for this shape, got "
                             f"{t.numel()}")

    def __call__(self, a, b, sfa, sfb, out=None):
        self._operand(a, "a", self.m)
        self._operand(b, "b", self.n)
        self._scales(sfa, "sfa", self.m)
        self._scales(sfb, "sfb", self.n)
        if out is None:
            out = torch.empty((self.m, self.n), device=a.device,
                              dtype=self.d.out_dtype)
        elif (out.shape != (self.m, self.n) or not out.is_contiguous()
                or out.dtype is not self.d.out_dtype):
            raise ValueError(f"out must be a contiguous {self.d.out_dtype} "
                             f"[{self.m}, {self.n}] tensor")
        status = self.lib.launch(
            ctypes.byref(self.config), ctypes.byref(self.shape), a.data_ptr(),
            b.data_ptr(), sfa.data_ptr(), sfb.data_ptr(), out.data_ptr(),
            a.device.index or 0, ctypes.c_void_p(raw_stream(a.device)))
        if status != OG_OK:
            raise RuntimeError(self.lib.unmatched(self.policy)
                               if status == OG_ERR_NO_POLICY
                               else self.lib.error().decode())
        return out


def launcher(dtype, config, m, n, k):
    """Return a launcher for `config` bound to this dtype and shape."""
    kind = SmmLauncher if DTYPES[dtype].impl == "smm" else MmLauncher
    return kind(dtype, config, m, n, k)

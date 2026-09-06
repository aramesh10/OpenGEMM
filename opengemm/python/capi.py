"""The compiled kernels, bound through ctypes.

Each library exports the C surface `src/<impl>/capi.h` declares and links no
libtorch and no libpython, so one build serves every torch and every Python
version. Checking the operands, allocating the output and padding an unaligned
K happen here, where dtypes.py already holds the element metadata they need.
"""

import ctypes
import threading

import torch

from . import build
from .dtypes import DTYPES
from .policy import mm_policy, mm_row, smm_policy, smm_row

OG_OK = 0

ABI_VERSION = {"mm": 2, "smm": 2}
# Operand pointers, then int32 arguments, that og_<impl>_launch() takes between
# the row it dispatches on and the stream it enqueues to.
LAUNCH_ABI = {"mm": (3, 10), "smm": (5, 7)}

K_PAD_AUTO, K_PAD_COPY = -1, 1


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
        self._rows = self._index = None
        p = f"og_{impl}_"
        for name, restype in (("abi_version", ctypes.c_int32),
                              ("enums", ctypes.c_char_p),
                              ("registry_fields", ctypes.c_char_p),
                              ("registry_rows", ctypes.c_int32),
                              ("registry_data",
                               ctypes.POINTER(ctypes.c_int32)),
                              ("error", ctypes.c_char_p)):
            entry = getattr(self.handle, p + name)
            entry.restype, entry.argtypes = restype, []
            setattr(self, name, entry)
        pointers, integers = LAUNCH_ABI[impl]
        self.launch = getattr(self.handle, p + "launch")
        self.launch.restype = ctypes.c_int32
        self.launch.argtypes = ([ctypes.c_int32]
                                + [ctypes.c_void_p] * pointers
                                + [ctypes.c_int32] * integers
                                + [ctypes.c_void_p])
        self.fields = self.registry_fields().decode().split(",")
        # Which columns hold an enumerator, and how this build spells each.
        self.spellings = {column: names.split(",") for column, names in
                          (pair.split("=")
                           for pair in self.enums().decode().split(";"))}
        self._check()

    def _check(self):
        """Fail at load rather than at launch when the library and this module
        disagree about the ABI.
        """
        found, want = self.abi_version(), ABI_VERSION[self.impl]
        if found != want:
            raise RuntimeError(
                f"{self.impl} library {self.path} is ABI version {found}, "
                f"this build speaks {want}; delete it and let it rebuild, or "
                f"set OPENGEMM_JIT=1")

    def _load(self):
        """Read the registry once, as rows and as a lookup from row to index.

        Elements read back as names rather than ordinals, because the block-
        scaled enumerations do not share an order with the dense one.
        """
        if self._rows is None:
            rows, cols = self.registry_rows(), len(self.fields)
            flat = ctypes.cast(self.registry_data(),
                               ctypes.POINTER(ctypes.c_int32 * (rows * cols)))
            self._rows = []
            for row in range(rows):
                at = row * cols
                entry = dict(zip(self.fields, flat.contents[at:at + cols]))
                for column, spelling in self.spellings.items():
                    entry[column] = spelling[entry[column]]
                self._rows.append(entry)
            self._index = {self._key(row): at
                           for at, row in enumerate(self._rows)}
        return self._rows

    def registry(self):
        """Return every configuration this build compiles, as dicts.

        A row's position is the row `launch` dispatches on.
        """
        return [dict(row) for row in self._load()]

    def _key(self, row):
        """Return `row` in field order, with the non-enumerated columns as the
        ints the registry holds, so a bool and a 1 compare equal.
        """
        return tuple(row[f] if f in self.spellings else int(row[f])
                     for f in self.fields)

    def _spell(self, row):
        """Return `row` as `field=value` text, in the registry's vocabulary."""
        return " ".join(f"{f}={v}" for f, v in zip(self.fields,
                                                   self._key(row)))

    def row_index(self, want):
        """Return the row of the kernel `want` names, for `launch`.

        Args:
            want: A registry row as `opengemm.python.policy` spells it.

        Raises:
            RuntimeError: If this build compiled no such kernel, listing the
                ones it did.
        """
        rows = self._load()
        found = self._index.get(self._key(want))
        if found is None:
            raise RuntimeError(
                "no compiled configuration matches this request:\n  wanted "
                + self._spell(want) + "\n\nthis build compiles exactly these "
                f"configurations (src/{self.impl}/registry.cuh):\n"
                + "".join(f"  {self._spell(row)}\n" for row in rows))
        return found


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
    """A dense configuration bound to a shape, then launched many times."""

    def __init__(self, dtype, config, m, n, k):
        self.d = d = DTYPES[dtype]
        self.lib = library("mm")
        self.m, self.n = m, n
        policy, bound = mm_policy(dtype, config, k)
        self.row = self.lib.row_index(mm_row(policy))
        # Padding widens the row and leaves the tail unread, which cannot
        # rescue fp6 or fp4: their expanding tensor maps require the true K.
        unaligned = k % d.k_align != 0
        if unaligned and d.bits < 8:
            raise ValueError(
                f"K = {k} is not a multiple of {d.k_align}, which the "
                f"expanding tensor map for {policy['elem_a']} requires")
        self.pad_to = 0
        if unaligned:
            if config.get("k_pad", K_PAD_AUTO) not in (K_PAD_AUTO,
                                                       K_PAD_COPY):
                raise ValueError(
                    f"K = {k} is not a multiple of {d.k_align} and the pad_k "
                    f"variant was not requested")
            self.pad_to = -(-k // d.k_align) * d.k_align
        pitch = self.pad_to or k
        self.args = tuple(int(value) for value in
                          (m, n, k, pitch, pitch, bound["supergroup"],
                           bound["walk"], bound["splits"], bound["l2_promo"]))
        self.extent = d.k_extent(k)
        # A split kernel accumulates into C rather than overwriting it.
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
            self.row, a.data_ptr(), b.data_ptr(), out.data_ptr(), *self.args,
            a.device.index or 0, ctypes.c_void_p(raw_stream(a.device)))
        if status != OG_OK:
            raise RuntimeError(self.lib.error().decode())
        return out


class SmmLauncher:
    """A block-scaled configuration bound to a shape."""

    def __init__(self, dtype, config, m, n, k):
        self.d = d = DTYPES[dtype]
        self.lib = library("smm")
        self.m, self.n, self.k = m, n, k
        policy, bound = smm_policy(dtype, config)
        self.row = self.lib.row_index(smm_row(policy))
        if k % d.block:
            raise ValueError(f"K = {k} is not a multiple of the "
                             f"{d.block}-wide scale block")
        self.args = tuple(int(value) for value in
                          (m, n, k, bound["supergroup"], bound["epi_direct"],
                           bound["persistent"]))
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
            self.row, a.data_ptr(), b.data_ptr(), sfa.data_ptr(),
            sfb.data_ptr(), out.data_ptr(), *self.args, a.device.index or 0,
            ctypes.c_void_p(raw_stream(a.device)))
        if status != OG_OK:
            raise RuntimeError(self.lib.error().decode())
        return out


def launcher(dtype, config, m, n, k):
    """Return a launcher for `config` bound to this dtype and shape."""
    kind = SmmLauncher if DTYPES[dtype].impl == "smm" else MmLauncher
    return kind(dtype, config, m, n, k)

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
LAUNCH_ABI = {"mm": (3, 10), "smm": (5, 7)}

K_PAD_AUTO, K_PAD_COPY = -1, 1

ENTRY_POINTS = (("abi_version", ctypes.c_int32),
                ("enums", ctypes.c_char_p),
                ("registry_fields", ctypes.c_char_p),
                ("registry_rows", ctypes.c_int32),
                ("registry_data", ctypes.POINTER(ctypes.c_int32)),
                ("error", ctypes.c_char_p))


def raw_stream(device):
    """Return the current CUDA stream on `device` as a raw handle.

    The private accessor skips building a Stream object on every call.
    """
    try:
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
        prefix = f"og_{impl}_"
        for name, restype in ENTRY_POINTS:
            entry = getattr(self.handle, prefix + name)
            entry.restype, entry.argtypes = restype, []
            setattr(self, name, entry)
        pointers, integers = LAUNCH_ABI[impl]
        self.launch = getattr(self.handle, prefix + "launch")
        self.launch.restype = ctypes.c_int32
        self.launch.argtypes = ([ctypes.c_int32]
                                + [ctypes.c_void_p] * pointers
                                + [ctypes.c_int32] * integers
                                + [ctypes.c_void_p])
        self.fields = self.registry_fields().decode().split(",")
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

    def _decode(self, values):
        """Return one registry row as a dict, its enumerators spelled out.

        Elements read back as names, not ordinals: the block-scaled
        enumerations do not share an order with the dense one.
        """
        row = dict(zip(self.fields, values))
        for column, spelling in self.spellings.items():
            row[column] = spelling[row[column]]
        return row

    def _load(self):
        """Read the registry once, as rows and as a lookup from row to index."""
        if self._rows is None:
            rows, cols = self.registry_rows(), len(self.fields)
            flat = ctypes.cast(self.registry_data(),
                               ctypes.POINTER(ctypes.c_int32 * (rows * cols)))
            values = flat.contents
            self._rows = [self._decode(values[at:at + cols])
                          for at in range(0, rows * cols, cols)]
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


class Launcher:
    """One compiled configuration bound to a dtype and a shape.

    A subclass names its `impl` and resolves `self.row` and `self.args`; the
    operand, output and launch steps both surfaces share are here.
    """

    impl = None

    def __init__(self, dtype, config, m, n, k):
        self.d = DTYPES[dtype]
        self.lib = library(self.impl)
        self.m, self.n, self.k = m, n, k
        self.extent = self.d.k_extent(k)

    def _operand(self, t, name, rows, dtype):
        """Check `t` is the [rows, extent] operand the kernel reads."""
        if not (t.is_cuda and t.dim() == 2 and t.is_contiguous()):
            raise TypeError(f"{name} must be a contiguous 2-D CUDA tensor")
        if t.dtype is not dtype or t.shape != (rows, self.extent):
            raise TypeError(
                f"{name} must be {dtype} [{rows}, {self.extent}], got "
                f"{t.dtype} {list(t.shape)}")
        return t

    def _out(self, out, device, strides=None):
        """Return C, allocated when `out` is None and checked when it is not.

        `strides` is the layout this kernel writes; None means contiguous.
        """
        shape = (self.m, self.n)
        if out is None:
            if strides is None:
                return torch.empty(shape, device=device,
                                   dtype=self.d.out_dtype)
            return torch.empty_strided(shape, strides, device=device,
                                       dtype=self.d.out_dtype)
        laid_out = (out.is_contiguous() if strides is None
                    else out.stride() == strides)
        if not (laid_out and out.shape == shape
                and out.dtype is self.d.out_dtype):
            layout = "contiguous" if strides is None else f"strides {strides}"
            raise ValueError(f"out must be {self.d.out_dtype} "
                             f"[{self.m}, {self.n}], {layout}")
        return out

    def _launch(self, device, *pointers):
        """Enqueue the kernel, raising the library's message on a failure."""
        status = self.lib.launch(self.row, *pointers, *self.args,
                                 device.index or 0,
                                 ctypes.c_void_p(raw_stream(device)))
        if status != OG_OK:
            raise RuntimeError(self.lib.error().decode())


class MmLauncher(Launcher):
    """A dense configuration bound to a shape, then launched many times."""

    impl = "mm"

    def __init__(self, dtype, config, m, n, k):
        super().__init__(dtype, config, m, n, k)
        policy, bound = mm_policy(dtype, config, k)
        self.row = self.lib.row_index(mm_row(policy))
        self.pad_to = self._padded_k(config, policy)
        pitch = self.pad_to or k
        self.args = tuple(int(value) for value in
                          (m, n, k, pitch, pitch, bound["supergroup"],
                           bound["walk"], bound["splits"], bound["l2_promo"]))
        self.zero_out = bound["splits"] > 1 or policy["rk"] > 1

    def _padded_k(self, config, policy):
        """Return the K to widen the operands to, or 0 when K already fits.

        Padding widens the row and leaves the tail unread, which cannot rescue
        fp6 or fp4: their expanding tensor maps require the true K.
        """
        d, k = self.d, self.k
        if k % d.k_align == 0:
            return 0
        if d.bits < 8:
            raise ValueError(
                f"K = {k} is not a multiple of {d.k_align}, which the "
                f"expanding tensor map for {policy['elem_a']} requires")
        if config.get("k_pad", K_PAD_AUTO) not in (K_PAD_AUTO, K_PAD_COPY):
            raise ValueError(
                f"K = {k} is not a multiple of {d.k_align} and the pad_k "
                f"variant was not requested")
        return -(-k // d.k_align) * d.k_align

    def _operand(self, t, name, rows, dtype):
        """Check `t`, then widen it into a padded buffer when K needs one."""
        t = super()._operand(t, name, rows, dtype)
        if not self.pad_to:
            return t
        wide = torch.empty((rows, self.d.k_extent(self.pad_to)), dtype=dtype,
                           device=t.device)
        wide[:, :self.extent] = t
        return wide

    def __call__(self, a, b, out=None):
        a = self._operand(a, "a", self.m, self.d.torch_dtype)
        b = self._operand(b, "b", self.n, self.d.torch_dtype_b)
        out = self._out(out, a.device, strides=(1, self.m))
        if self.zero_out:
            out.zero_()
        self._launch(a.device, a.data_ptr(), b.data_ptr(), out.data_ptr())
        return out


class SmmLauncher(Launcher):
    """A block-scaled configuration bound to a shape."""

    impl = "smm"

    def __init__(self, dtype, config, m, n, k):
        super().__init__(dtype, config, m, n, k)
        policy, bound = smm_policy(dtype, config)
        self.row = self.lib.row_index(smm_row(policy))
        if k % self.d.block:
            raise ValueError(f"K = {k} is not a multiple of the "
                             f"{self.d.block}-wide scale block")
        self.args = tuple(int(value) for value in
                          (m, n, k, bound["supergroup"], bound["epi_direct"],
                           bound["persistent"]))
        self.scale_bytes = {rows: self._scale_bytes(rows) for rows in (m, n)}

    def _scale_bytes(self, rows):
        """Bytes of a scale tensor in the 128x4 blocked layout."""
        return (-(-rows // 128) * 128) * (-(-self.k // (self.d.block * 4)) * 4)

    def _scales(self, t, name, rows):
        """Check `t` holds this operand's scales in the blocked layout."""
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
        self._operand(a, "a", self.m, self.d.elem_dtype)
        self._operand(b, "b", self.n, self.d.elem_dtype)
        self._scales(sfa, "sfa", self.m)
        self._scales(sfb, "sfb", self.n)
        out = self._out(out, a.device)
        self._launch(a.device, a.data_ptr(), b.data_ptr(), sfa.data_ptr(),
                     sfb.data_ptr(), out.data_ptr())
        return out


def launcher(dtype, config, m, n, k):
    """Return a launcher for `config` bound to this dtype and shape."""
    kind = SmmLauncher if DTYPES[dtype].impl == "smm" else MmLauncher
    return kind(dtype, config, m, n, k)

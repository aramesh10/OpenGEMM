"""CUTLASS stand-in baselines for the formats torch has no kernel for.

`opengemm.bench.baseline_for` covers every format torch can run, which is where
cuBLAS (or `_scaled_mm` / `_int_mm`) comes from. For u8, e5m2, e3m2, e2m3, e2m1
and mxfp4 it has nothing, so `scripts/benchmark.py` falls back to the CUTLASS
GEMMs in `cutlass_ref.cu`, compiled here on first use against a CUTLASS 4.x
checkout and cached next to the kernel libraries.

Point `OPENGEMM_CUTLASS` (or `CUTLASS_PATH`) at that checkout; without one the
formats keep reporting "no baseline" exactly as they did before.

Each format is compiled in a few tile shapes, since a single hard-coded tile
would make CUTLASS look slow for the wrong reason. `baselines_for` hands back
all of them and the caller reports whichever wins.
"""

import ctypes
import hashlib
import itertools
import os
import re
from ctypes import c_char_p, c_int32, c_void_p
from pathlib import Path

from opengemm.python import build, capi
from opengemm.python.dtypes import DTYPES, k_of

ABI_VERSION = 1
MIN_VERSION = (4, 0)
SOURCE = Path(__file__).resolve().parent / "cutlass_ref.cu"
CANDIDATES = ("OPENGEMM_CUTLASS", "CUTLASS_PATH", "CUTLASS_DIR", "CUTLASS_HOME")

_library = None
_unavailable = None


def _version(root):
    """`(major, minor)` of the CUTLASS checkout at `root`, or None if it is not one."""
    header = root / "include" / "cutlass" / "version.h"
    try:
        text = header.read_text()
    except OSError:
        return None
    found = {
        part: int(m.group(1))
        for part in ("MAJOR", "MINOR")
        if (m := re.search(rf"#define CUTLASS_{part} (\d+)", text))
    }
    if len(found) != 2:
        return None
    return found["MAJOR"], found["MINOR"]


def checkout():
    """The CUTLASS checkout to build against, or None with the reason in `_unavailable`."""
    global _unavailable
    roots = [Path(os.environ[name]) for name in CANDIDATES if os.environ.get(name)]
    roots += [build.ROOT / "third_party" / "cutlass", Path.home() / "cutlass"]
    for root in roots:
        version = _version(root)
        if version is None:
            continue
        if version < MIN_VERSION:  # keep looking; a later candidate may be new enough
            _unavailable = (
                f"CUTLASS at {root} is {version[0]}.{version[1]}, "
                f"sm_100a needs {MIN_VERSION[0]}.{MIN_VERSION[1]}+"
            )
            continue
        _unavailable = None
        return root
    if _unavailable is None:
        _unavailable = "set OPENGEMM_CUTLASS to a CUTLASS checkout for a baseline"
    return None


def _build(root):
    """Compile `cutlass_ref.cu` against the checkout at `root`, cached by content."""
    includes = [f"-I{root / 'include'}", f"-I{root / 'tools' / 'util' / 'include'}"]
    digest = hashlib.blake2s(digest_size=8)
    digest.update(SOURCE.read_bytes())
    digest.update(str(_version(root)).encode())
    digest.update((build.nvcc_version() or "no nvcc").encode())
    digest.update(" ".join(build.NVCC_FLAGS + build.LIBRARY_FLAGS).encode())
    out = build.CACHE / digest.hexdigest() / "libcutlass_ref.so"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cutlass_ref] building the CUTLASS baselines against {root}", flush=True)
    print(f"[cutlass_ref] compiles once then cached at {out.parent}", flush=True)
    print("[cutlass_ref] may take a few minutes...", flush=True)
    staging = out.with_suffix(f".{os.getpid()}.tmp")
    try:
        build.nvcc([SOURCE], staging, *build.LIBRARY_FLAGS, *includes)
    except RuntimeError:
        staging.unlink(missing_ok=True)
        raise
    os.replace(staging, out)
    return out


class Library:
    """The compiled CUTLASS baselines, through the C ABI `cutlass_ref.cu` declares."""

    def __init__(self, path):
        lib = ctypes.CDLL(str(path))
        self._abi = lib.ogref_abi
        self._abi.restype = c_int32
        self._error = lib.ogref_error
        self._error.restype = c_char_p
        self._configs = lib.ogref_configs
        self._configs.restype = c_int32
        self._dtype = lib.ogref_config_dtype
        self._dtype.restype, self._dtype.argtypes = c_char_p, [c_int32]
        self._tile = lib.ogref_config_tile
        self._tile.restype, self._tile.argtypes = c_char_p, [c_int32]
        self._launch = lib.ogref_launch
        self._launch.restype = c_int32
        self._launch.argtypes = [c_int32] * 4 + [c_void_p] * 6
        found = self._abi()
        if found != ABI_VERSION:
            raise RuntimeError(
                f"CUTLASS baseline library {path} is ABI version {found}, "
                f"this build speaks {ABI_VERSION}; delete it and let it rebuild"
            )

    def configs(self, name):
        """`(index, tile)` for every configuration compiled for format `name`."""
        return [
            (i, self._tile(i).decode())
            for i in range(self._configs())
            if self._dtype(i).decode() == name
        ]

    def launch(self, index, entry, k):
        """Enqueue one configuration on the current stream of the operands' device."""
        m, n = entry.out.shape
        status = self._launch(
            index,
            m,
            n,
            k,
            entry.a.data_ptr(),
            entry.b.data_ptr(),
            0 if entry.sfa is None else entry.sfa.data_ptr(),
            0 if entry.sfb is None else entry.sfb.data_ptr(),
            entry.out.data_ptr(),
            capi.raw_stream(entry.a.get_device()),
        )
        if status != 0:
            raise RuntimeError(self._error().decode())


def library():
    """The loaded CUTLASS baseline library, or None when there is no checkout to build from."""
    global _library, _unavailable
    if _library is None and _unavailable is None:
        root = checkout()
        if root is not None:
            try:
                _library = Library(_build(root))
            except Exception as exc:
                _unavailable = str(exc).splitlines()[0][:160]
    return _library


def reason():
    """Why there is no CUTLASS baseline, once `library()` has come back None."""
    return _unavailable or "no CUTLASS baseline"


def baselines_for(buffers, name):
    """Every CUTLASS configuration for `name`, as `(tile, callable)` over the same rotations.

    The callables have the signature `bench.timed` wants, and each returns the
    output it wrote, so the caller can check it against the reference.
    """
    lib = library()
    if lib is None:
        return []
    k = k_of(DTYPES[name], buffers[0].a.shape[1])
    out = []
    for index, tile in lib.configs(name):
        cycle = itertools.cycle(buffers)

        def call(index=index, cycle=cycle, k=k):
            entry = next(cycle)
            lib.launch(index, entry, k)
            return entry.out

        out.append((tile, call))
    return out

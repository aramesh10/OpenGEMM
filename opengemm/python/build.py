"""Where the compiled kernel libraries come from.

A wheel ships them, already built. A source checkout, an edited kernel or
OPENGEMM_JIT=1 compiles them with nvcc instead -- one translation unit per
impl, covering every kernel its registry.cuh names -- and caches the result
under OPENGEMM_CACHE, keyed by a digest of the sources it was built from.
"""

import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path

from .log import log

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
SRC = PACKAGE / "src"
ARCH = "-gencode=arch=compute_100a,code=sm_100a"
NVCC_FLAGS = ["-O3", ARCH, "--expt-relaxed-constexpr", "-std=c++20",
              "--split-compile=0", "-diag-suppress", "68,2361"]

# A wheel ships the compiled libraries here; a source checkout has nothing.
LIB = PACKAGE / "lib"
CACHE = Path(os.environ.get("OPENGEMM_CACHE") or
             Path(os.environ.get("XDG_CACHE_HOME",
                                 Path.home() / ".cache")) / "opengemm")
# The library links no torch and no libpython, and nvcc links the CUDA runtime
# statically, so what it needs at load is the driver and nothing else. Hidden
# visibility keeps the inline helpers in the two host_utils.cuh files, which
# share names and not signatures, from colliding; capi.h opts the entry points
# back out.
LIBRARY_FLAGS = ["-shared", "-Xcompiler", "-fPIC",
                 "-Xcompiler", "-fvisibility=hidden"]


def source_hash(impl):
    """Return a digest of the sources `impl`'s library is built from.

    Content, not mtime: pip rewrites mtimes on reinstall, so a timestamp says a
    rebuild is due when nothing changed.
    """
    digest = hashlib.blake2s(digest_size=8)
    for path in sorted((SRC / impl).glob("*.cu*")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compile_library(impl, out):
    """Compile `impl`'s C surface to the shared library `out`.

    Args:
        impl: `"mm"` for the dense kernels, `"smm"` for the block-scaled ones.
        out: Where to write the library; the parent must exist.

    Raises:
        RuntimeError: If nvcc fails, with its diagnostics.
    """
    src = SRC / impl
    # Written beside the target and moved into place, so a reader never opens a
    # half-written library and two builders cannot interleave.
    staging = out.with_suffix(f".{os.getpid()}.tmp")
    command = (["nvcc"] + NVCC_FLAGS + LIBRARY_FLAGS + [f"-I{src}"]
               + [str(src / f"{impl}_capi.cu"), "-o", str(staging), "-lcuda"])
    stubs = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    stubs = stubs / "targets" / "x86_64-linux" / "lib" / "stubs"
    if stubs.is_dir():
        # Linking against the stub lets a machine with no driver, such as a
        # wheel builder, still produce a library; the SONAME is the real one.
        command += [f"-L{stubs}"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        staging.unlink(missing_ok=True)
        raise RuntimeError(f"nvcc failed building the {impl} library:\n"
                           + (result.stderr or result.stdout))
    os.replace(staging, out)
    return out


class _Ticker:
    """Print the elapsed seconds while nvcc runs, which prints nothing itself.

    nvcc is silent for the whole compile, so a caller with no notice cannot
    tell a slow build from a hang and reaches for ctrl-C, which leaves the
    build lock behind and makes the next run wait on it forever.
    """

    def __init__(self, message, every=15):
        self.message, self.every, self.done = message, every, threading.Event()
        self.thread = threading.Thread(target=self._tick, daemon=True)

    def _tick(self):
        started = time.perf_counter()
        while not self.done.wait(self.every):
            elapsed = time.perf_counter() - started
            log(f"{self.message}, {elapsed:.0f} s so far")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        self.thread.join(timeout=1)


def library_path(impl):
    """Return the path of `impl`'s compiled library, building it when needed.

    A wheel ships the library and this returns it untouched. A source checkout,
    an edited kernel or `OPENGEMM_JIT=1` falls through to nvcc, whose result is
    cached under `OPENGEMM_CACHE` keyed by the sources it was built from.

    Args:
        impl: `"mm"` or `"smm"`.

    Returns:
        The path of a library exporting the surface `capi.h` declares.
    """
    override = os.environ.get(f"OPENGEMM_LIB_{impl.upper()}")
    if override:
        return Path(override)
    name = f"libopengemm_{impl}.so"
    shipped = LIB / name
    digest = source_hash(impl)
    if not os.environ.get("OPENGEMM_JIT") and shipped.exists():
        stamp = LIB / "stamp.json"
        try:
            import json
            recorded = json.loads(stamp.read_text())["sources"][impl]
        except Exception:
            recorded = None
        # A stamp that disagrees means someone edited a kernel in an installed
        # copy; build rather than run something else than the source says.
        if recorded in (None, digest):
            return shipped
    out = CACHE / digest / name
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"building the {impl} library: one nvcc over every kernel "
        f"{SRC / impl / 'registry.cuh'} names, about half a minute. The "
        f"result is cached under {CACHE}.")
    started = time.perf_counter()
    with _Ticker(f"compiling the {impl} library"):
        compile_library(impl, out)
    log(f"built the {impl} library in {time.perf_counter() - started:.0f} s")
    return out


def prebuild(impls=("mm", "smm")):
    """Resolve every library ahead of the first `gemm` call, building any that
    a wheel did not ship.

        python -m opengemm

    Args:
        impls: Which libraries to resolve; both by default.

    Returns:
        `(impl, path)` for each, in the order given.
    """
    return [(impl, library_path(impl)) for impl in impls]

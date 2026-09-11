"""Where the compiled kernel libraries come from: a wheel ships them, and a
source checkout, an edited kernel or OPENGEMM_JIT=1 compiles them with nvcc
into a cache keyed by a digest of the sources and of the compiler that
turns them into a library.
"""

import functools
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from .log import log

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
SRC = PACKAGE / "src"
LIB = PACKAGE / "lib"
CACHE = Path(
    os.environ.get("OPENGEMM_CACHE")
    or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "opengemm"
)

ARCH = "-gencode=arch=compute_100a,code=sm_100a"
NVCC_FLAGS = [
    "-O3",
    ARCH,
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "--split-compile=0",
    "-diag-suppress",
    "68,2361",
]
SHARED_FLAGS = ["-shared", "-Xcompiler", "-fPIC"]
LIBRARY_FLAGS = SHARED_FLAGS + ["-Xcompiler", "-fvisibility=hidden"]

def nvcc(inputs, out, *flags):
    """Run nvcc over `inputs` to produce `out`.

    Raises:
        RuntimeError: If nvcc fails, with its diagnostics.
    """
    command = ["nvcc", *NVCC_FLAGS, *flags, *map(str, inputs), "-o", str(out), "-lcuda"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("nvcc failed:\n" + (result.stderr or result.stdout))

@functools.lru_cache(maxsize=None)
def nvcc_version():
    """The nvcc release line, or None when nvcc is not on the path."""
    try:
        out = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return next((line.strip() for line in out.splitlines() if "release" in line), "unknown")

def source_hash(impl):
    """A digest of the sources `impl`'s library is built from; content, not mtime, since pip rewrites mtimes."""
    digest = hashlib.blake2s(digest_size=8)
    for directory in (SRC / "common", SRC / impl):
        for path in sorted(p for p in directory.iterdir() if p.suffix in (".cu", ".cuh", ".h")):
            digest.update(str(path.relative_to(SRC)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()

def build_hash(impl):
    """The cache key for `impl`'s library: its sources, plus the compiler and the
    flags that turn them into one, so a toolkit upgrade or a flag edit does not
    reuse a library built by something else. Include paths are left out; they
    name locations, not content.
    """
    digest = hashlib.blake2s(digest_size=8)
    digest.update(source_hash(impl).encode())
    digest.update((nvcc_version() or "no nvcc").encode())
    digest.update(" ".join(NVCC_FLAGS + LIBRARY_FLAGS).encode())
    return digest.hexdigest()

def compile_library(impl, out):
    """Compile `impl`'s C surface to the shared library `out`; the parent must exist."""
    src = SRC / impl
    staging = out.with_suffix(f".{os.getpid()}.tmp")
    flags = LIBRARY_FLAGS + [f"-I{SRC}", f"-I{src}"]
    stubs = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "targets/x86_64-linux/lib/stubs"
    if stubs.is_dir():
        flags.append(f"-L{stubs}")
    try:
        nvcc([src / "capi.cu"], staging, *flags)
    except RuntimeError:
        staging.unlink(missing_ok=True)
        raise
    os.replace(staging, out)
    return out

def library_path(impl):
    """The path of `impl`'s compiled library: the one a wheel shipped, else one built by nvcc and cached under OPENGEMM_CACHE by build digest."""
    override = os.environ.get(f"OPENGEMM_LIB_{impl.upper()}")
    if override:
        return Path(override)
    name = f"libopengemm_{impl}.so"
    shipped = LIB / name
    if not os.environ.get("OPENGEMM_JIT") and shipped.exists():
        try:
            stamp = json.loads((LIB / "stamp.json").read_text())
        except Exception:
            stamp = {}
        # A shipped library is a binary someone else's nvcc already produced, so
        # only the sources and the arch it was built for have to still match.
        sources_match = stamp.get("sources", {}).get(impl) in (None, source_hash(impl))
        if sources_match and stamp.get("arch", ARCH) == ARCH:
            return shipped
    out = CACHE / build_hash(impl) / name
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"Building the {impl} extension")
    log(f"Compiles once then cached at {out.parent}")
    log("May take up to 2 minutes...")
    started = time.perf_counter()
    compile_library(impl, out)
    log(f"Built the {impl} extension in {time.perf_counter() - started:.0f}s")
    return out

def prebuild(impls=("mm", "smm")):
    """Resolve every library ahead of the first gemm() call, building any a wheel did not ship.

    Returns:
        `(impl, path)` for each.
    """
    return [(impl, library_path(impl)) for impl in impls]
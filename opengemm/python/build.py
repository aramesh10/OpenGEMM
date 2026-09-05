"""JIT build of the torch extensions.

torch caches a build under ~/.cache/torch_extensions and rebuilds it when a
source changes; the first load after a clone compiles every kernel
registry.cuh names, which takes a few minutes.
"""

import os
import time
from pathlib import Path

from .log import log

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
SRC = PACKAGE / "src"
ARCH = "-gencode=arch=compute_100a,code=sm_100a"
NVCC_FLAGS = ["-O3", ARCH, "--expt-relaxed-constexpr", "-std=c++20",
              "-diag-suppress", "68,2361"]

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
_loaded = {}


def _built_before(name):
    """Return whether torch's cache holds a build of `name`; a changed source
    still rebuilds.
    """
    try:
        # Private to torch; without it, skip the notice rather than the build.
        from torch.utils.cpp_extension import _get_build_directory
        return any(Path(_get_build_directory(name, False)).glob(f"{name}*.so"))
    except Exception:
        return True


def extension(impl):
    """Load the `mm` or `smm` extension, building it first when needed.

    Args:
        impl: `"mm"` for the dense kernel, `"smm"` for the block-scaled one.

    Returns:
        The extension module: `registry()`, `launcher()` and the `Launcher`
        class.
    """
    if impl not in _loaded:
        from torch.utils.cpp_extension import load
        name = f"opengemm_{impl}"
        if not _built_before(name):
            log(f"building the {impl} extension; a few minutes, then torch "
                f"caches it")
        src = SRC / impl
        started = time.perf_counter()
        _loaded[impl] = load(
            name=name, sources=[str(src / f"{impl}.cu")],
            extra_cuda_cflags=NVCC_FLAGS + [f"-I{src}"],
            extra_cflags=["-O3"], extra_ldflags=["-lcuda"],
            verbose=bool(os.environ.get("OPENGEMM_VERBOSE")))
        seconds = time.perf_counter() - started
        # A cached load takes a second or two; anything longer was a build.
        if seconds > 10:
            log(f"built the {impl} extension in {seconds:.0f} s")
    return _loaded[impl]

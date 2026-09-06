"""Wheel build: compile the kernels once, here, rather than on every machine.

The compiled libraries link no libtorch and no libpython, so the wheel is
`py3-none-<platform>`: one build serves every PyTorch and every Python 3.
nvcc is needed to build it; a GPU is not. Set OPENGEMM_SKIP_KERNELS=1 to make
a wheel without them, which falls back to compiling at first use.
"""

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # setuptools older than 70.1 leaves this to the wheel pkg
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# Loaded by path under a stand-in package, not imported: importing
# `opengemm.python.build` would run the package __init__, which imports torch,
# and a wheel builder has no torch installed. build.py and the log module it
# reaches for depend on nothing outside the standard library, so a package with
# only a __path__ is enough to resolve the relative import.
_HERE = Path(__file__).parent
_package = types.ModuleType("_opengemm_build")
_package.__path__ = [str(_HERE / "opengemm" / "python")]
sys.modules["_opengemm_build"] = _package
_spec = importlib.util.spec_from_file_location(
    "_opengemm_build.build", _HERE / "opengemm" / "python" / "build.py")
og_build = importlib.util.module_from_spec(_spec)
sys.modules["_opengemm_build.build"] = og_build
_spec.loader.exec_module(og_build)

IMPLS = ("mm", "smm")


def nvcc_version():
    """Return the nvcc release line, or None when nvcc is not on the path."""
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return next((line.strip() for line in out.splitlines()
                 if "release" in line), "unknown")


class build_py(_build_py):
    """Compile the kernel libraries into the package being built."""

    def run(self):
        super().run()
        if os.environ.get("OPENGEMM_SKIP_KERNELS"):
            print("opengemm: OPENGEMM_SKIP_KERNELS set; the wheel ships "
                  "sources and compiles at first use")
            return
        out = Path(self.build_lib) / "opengemm" / "lib"
        out.mkdir(parents=True, exist_ok=True)
        for impl in IMPLS:
            print(f"opengemm: compiling the {impl} library")
            og_build.compile_library(impl, out / f"libopengemm_{impl}.so")
        # The digest lets an installed copy notice that its kernels were edited
        # and rebuild rather than run something else than the sources say.
        (out / "stamp.json").write_text(json.dumps({
            "abi": 1,
            "arch": og_build.ARCH,
            "nvcc": nvcc_version(),
            "sources": {impl: og_build.source_hash(impl) for impl in IMPLS},
        }, indent=1) + "\n")


class bdist_wheel(_bdist_wheel):
    """Tag the wheel by platform but not by Python: there is no CPython ABI in
    it, only CUDA code and the driver.
    """

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = bool(os.environ.get("OPENGEMM_SKIP_KERNELS"))

    def get_tag(self):
        python, abi, platform = super().get_tag()
        if os.environ.get("OPENGEMM_SKIP_KERNELS"):
            return python, abi, platform
        return "py3", "none", platform


setup(cmdclass={"build_py": build_py, "bdist_wheel": bdist_wheel})

"""Build the kernel libraries ahead of time: ``python -m opengemm``."""

from .python.build import LIB, prebuild
from .python.log import log

if __name__ == "__main__":
    for impl, path in prebuild():
        where = "shipped with the wheel" if path.parent == LIB else "built"
        log(f"the {impl} library is ready ({where}): {path}")

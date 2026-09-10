"""The stored configurations: what configs.json ships, what tuning adds."""

import json
import os
import re
from pathlib import Path

from .build import PACKAGE, SRC
from .dtypes import DTYPES
from .log import log

TUNED_DIR = "opengemm-configs"
TUNED_FILE = "tuned_configs.json"

def load_shapes():
    """The distinct (M, N, K) triples of shapes.jsonc, in file order."""
    text = re.sub(r"//[^\n]*", "", (PACKAGE / "shapes.jsonc").read_text())
    return list(dict.fromkeys(tuple(shape) for shape in json.loads(text)))

def tuned_path():
    """The local store new tunings are written to."""
    directory = os.environ.get("OPENGEMM_CONFIGS")
    return (Path(directory) if directory else Path.cwd() / TUNED_DIR) / TUNED_FILE

def config_path(impl):
    """The configurations shipped with the package."""
    return SRC / impl / "configs.json"

def entry_key(entry):
    return (entry["dtype"], entry["m"], entry["n"], entry["k"])

def _entries(path):
    return json.loads(path.read_text())["entries"] if path.exists() else []

def tuned_entries(impl=None):
    """The local store's entries, optionally one implementation's."""
    return [e for e in _entries(tuned_path()) if impl is None or DTYPES[e["dtype"]].impl == impl]

def stored_entries(impl):
    """Every entry for `impl`, the local store taking precedence over the shipped ones."""
    entries = {entry_key(e): e for e in _entries(config_path(impl))}
    entries.update({entry_key(e): e for e in tuned_entries(impl)})
    return list(entries.values())

def load_configs(impl, dtype=None):
    """`{(dtype, m, n, k): config}` for the stored entries with a configuration."""
    return {
        entry_key(e): e["config"]
        for e in stored_entries(impl)
        if e.get("config") is not None and (dtype is None or e["dtype"] == dtype)
    }

def stored_entry(name, m, n, k):
    """The stored entry for a (dtype, shape), the local store first; None when untuned."""
    key = (name, m, n, k)
    impl = DTYPES[name].impl
    for entries in (tuned_entries(impl), _entries(config_path(impl))):
        for e in entries:
            if entry_key(e) == key:
                return e
    return None

def resolve_config(name, m, n, k):
    """The configuration for a (dtype, shape): the stored one, else the winner
    of a sweep run now and stored, a few minutes on the GPU.

    Raises:
        ValueError: If the shape is recorded as unimplementable.
    """
    entry = stored_entry(name, m, n, k)
    if entry is None:
        from .tune import tune

        log(
            f"no stored configuration for {name} {m}x{n}x{k}; tuning it now, a few minutes on the GPU"
        )
        return tune(name, m, n, k)
    if entry.get("config") is None:
        why = entry.get("unimplementable", "recorded as unimplementable")
        raise ValueError(f"{name} {m}x{n}x{k} cannot be served: {why}")
    return entry["config"]

def write_entry(impl, entry):
    """Store `entry` in the local store, replacing any entry for the same (dtype, shape); serialized and atomic, sorted by dtype then shapes.jsonc order. Returns the path written."""
    path = tuned_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / ".tuned_configs.lock"
    with lock.open("w") as lock_file:
        os.lockf(lock_file.fileno(), os.F_LOCK, 0)
        store = json.loads(path.read_text()) if path.exists() else {"arch": "sm_100", "entries": []}
        shape_order = {shape: i for i, shape in enumerate(load_shapes())}
        dtype_order = {name: i for i, name in enumerate(DTYPES)}
        entries = [e for e in store["entries"] if entry_key(e) != entry_key(entry)]
        entries.append(entry)
        store["entries"] = sorted(
            entries,
            key=lambda e: (
                dtype_order.get(e["dtype"], 1 << 30),
                shape_order.get((e["m"], e["n"], e["k"]), 1 << 30),
            ),
        )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(store, indent=1) + "\n")
        temporary.replace(path)
    return path
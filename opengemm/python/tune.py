"""Ablation tuning for one (dtype, shape): every compiled kernel of the format,
crossed with the launch-time axes, checked and timed; the fastest wins, or
within 1.5% the plainer one, and is stored for gemm() to dispatch from.
"""

import itertools

from . import bench, capi
from .configs import write_entry
from .dtypes import DTYPES
from .log import log

TIE = 0.015
MIN_SPLIT_K_TILES = 16

def supergroups(num_n):
    """The supergroup widths worth trying for `num_n` N-tiles."""
    return sorted(w for w in {1, 2, 4, 8, 16, num_n} if 1 <= w <= max(num_n, 1))

def split_options(k_tiles):
    """The split-K counts a reduction of `k_tiles` tiles can afford."""
    afford = k_tiles // MIN_SPLIT_K_TILES
    return (1,) + tuple(s for s in (2, 4, 8) if s <= afford)

def mm_candidates(name, m, n, k):
    """The dense candidate space: every compiled kernel of this format that
    fits the shape, crossed with split-K, supergroup and L2 promotion."""
    k_tiles = -(-k // 128)
    splits = split_options(k_tiles)
    space = []
    for e in capi.library("mm").kernels():
        if e["dtype"] != name:
            continue
        mma_n = m if e["swap_ab"] else n
        if e["cluster_k"] > k_tiles:
            continue
        if e["swap_ab"] and min(e["output_n"], 128) > -(-m // 8) * 8:
            continue
        entry_splits = [s for s in splits if (s > 1) == (e["split_k"] > 1)]
        if not entry_splits:
            continue
        config = {key: value for key, value in e.items() if key != "dtype"}
        for flag in ("use_2cta", "use_clc", "swap_ab"):
            config[flag] = bool(config[flag])
        config["walk"] = 0
        num_n = -(-mma_n // e["output_n"])
        axes = itertools.product(entry_splits, supergroups(num_n), (0, 2))
        for split_k, supergroup, l2_promo in axes:
            space.append(dict(config, split_k=split_k, supergroup=supergroup, l2_promo=l2_promo))
    return space

def smm_candidates(name, m, n, k):
    """The block-scaled candidate space: every compiled kernel of this format
    that fits the shape, crossed with supergroup and persistence."""
    block_k = 1024 // DTYPES[name].bits
    space = []
    for e in capi.library("smm").kernels():
        if e["dtype"] != name:
            continue
        mma_n = m if e["swap_ab"] else n
        if e["output_n"] % 128 and e["output_n"] < mma_n:
            continue
        if e["use_2cta"] and k < block_k:
            continue
        config = {key: value for key, value in e.items() if key != "dtype"}
        for flag in ("use_2cta", "use_clc", "swap_ab"):
            config[flag] = bool(config[flag])
        num_n = -(-mma_n // e["output_n"])
        persistents = (1,) if e["use_clc"] else (1, 0)
        for supergroup, persistent in itertools.product(supergroups(num_n), persistents):
            space.append(dict(config, supergroup=supergroup, persistent=persistent, epi_direct=0))
    return space

def cluster_ctas(config):
    group = 2 if config["use_2cta"] else 1
    return (config["cluster_m"] or group) * config["cluster_n"] * config["cluster_k"]

def simplicity(config):
    """A sort key under which the plainer configuration wins a tie."""
    return (
        config["swap_ab"],
        config["use_clc"],
        config.get("epi_double", 0),
        config.get("epi_trade", 0),
        config.get("deep_stages", 0),
        0 if config.get("split_k", 1) <= 1 else config["split_k"],
        1 if config.get("block_m", 128) != 128 else 0,
        cluster_ctas(config),
        config.get("epi_direct", 0),
        config["output_n"],
        int(config["use_2cta"]),
        config.get("epi_hold", 1),
        0 if config.get("persistent", 1) else 1,
    )

def label(config):
    """The one-line label for a configuration, as the sweep prints it."""
    group = 2 if config["use_2cta"] else 1
    text = (
        f"v{group} n{config['output_n']:<3} sg{config['supergroup']:<2} "
        f"clc{int(config['use_clc'])} swap{int(config['swap_ab'])}"
    )
    for key in (
        "stages",
        "epi_double",
        "split_k",
        "epi_trade",
        "deep_stages",
        "epi_direct",
        "l2_promo",
    ):
        if config.get(key):
            text += f" {key}{config[key]}"
    if config.get("block_m") == 64:
        text += " bm64"
    if not config.get("persistent", 1):
        text += " onepass"
    if cluster_ctas(config) > group:
        text += f" c{config['cluster_m'] or group}x{config['cluster_n']}x{config['cluster_k']}"
    return text

def tune(name, m, n, k):
    """Sweep one (dtype name, shape), store the winner and return it.

    Raises:
        RuntimeError: If no candidate is correct, or K is one the format cannot take.
    """
    impl = DTYPES[name].impl
    buffers = bench.make_input_set(m, n, k, name)
    space = (mm_candidates if impl == "mm" else smm_candidates)(name, m, n, k)
    log(f"tuning {name} {m}x{n}x{k} on {bench.env_stamp()['gpu']}: {len(space)} candidates")

    ranked, rejected = [], 0
    for config in space:
        bound = capi.bind(name, m, n, k, config)
        if bench.correctness_error(buffers, bound, name, k):
            rejected += 1
            continue
        kernel = bench.runner(buffers, bound)
        warmup, iterations = bench.plan(kernel)
        ranked.append((bench.timed(kernel, warmup, iterations, repeats=3), config))
    if not ranked:
        raise RuntimeError(f"no correct candidate for {name} {m}x{n}x{k}")

    ranked.sort(key=lambda row: row[0])
    fastest = ranked[0][0]
    within = [row for row in ranked if row[0] <= fastest * (1 + TIE)]
    best = min(within, key=lambda row: simplicity(row[1]))[1]

    kernel = bench.runner(buffers, capi.bind(name, m, n, k, best))
    baseline, why = bench.baseline_for(buffers, name)
    warmup, iterations = bench.report_plan(kernel)
    us = bench.timed(kernel, warmup, iterations)
    base = bench.timed(baseline, warmup, iterations) if baseline else None

    log(f"best: {label(best)}  ({rejected} candidates failed correctness)")
    measured = {"us": round(us, 3), "cublas_us": base and round(base, 3), "env": bench.env_stamp()}
    path = write_entry(
        impl, {"dtype": name, "m": m, "n": n, "k": k, "config": best, "measured": measured}
    )
    if base:
        log(f"kernel {us:.2f}us  cuBLAS {base:.2f}us  {base / us:.3f}x; stored in {path}")
    else:
        log(f"kernel {us:.2f}us  no baseline: {why}; stored in {path}")
    return best

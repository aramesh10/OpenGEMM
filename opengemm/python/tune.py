"""Ablation tuning for one (dtype, shape).

Every kernel the build compiles is a candidate, crossed with the launch-time
axes. Each is checked for correctness and timed; the fastest wins, or within
1.5% the plainer one. The winner is re-measured against cuBLAS and written
to configs.json, which `gemm()` dispatches from.
"""
from . import bench
from .build import extension
from .dtypes import DTYPES, ELEM_INDEX
from .log import log

TIE = 0.015
MIN_SPLIT_K_TILES = 16


def supergroups(num_n):
    """Return the supergroup widths worth trying for `num_n` N-tiles."""
    return sorted(w for w in {1, 2, 4, 8, 16, num_n} if 1 <= w <= max(num_n, 1))


def split_options(k_tiles):
    """Return the split-K counts a reduction of `k_tiles` tiles can afford."""
    afford = k_tiles // MIN_SPLIT_K_TILES
    return (1,) + tuple(s for s in (2, 4, 8) if s <= afford)


def unique(rows):
    seen, out = set(), []
    for e in rows:
        key = tuple(sorted(e.items()))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def mm_candidates(ext, m, n, k, dtype):
    """Return the dense candidate space for a shape: every compiled kernel of
    this element pair that fits it, crossed with split-K, supergroup and L2
    promotion.
    """
    want = (ELEM_INDEX[dtype.elem_a], ELEM_INDEX[dtype.elem_b])
    k_tiles = -(-k // 128)
    splits = split_options(k_tiles)
    space = []
    for e in unique(ext.registry()):
        if (e["elem_a"], e["elem_b"]) != want:
            continue
        mma_n = m if e["swap_ab"] else n
        if e["cluster_k"] > k_tiles:
            continue
        # A swapped kernel tiles M with output_n; one wider than M, rounded to
        # the 8 rows a box takes, is wasted work.
        if e["swap_ab"] and min(e["output_n"], 128) > -(-m // 8) * 8:
            continue
        # The accumulating epilogue is compiled in, so a kernel is either
        # split-K or not.
        entry_splits = [s for s in splits
                        if (s > 1) == bool(e["splits_expected"])]
        if not entry_splits:
            continue
        config = {key: value for key, value in e.items()
                  if key not in ("splits_expected", "elem_a", "elem_b")}
        for flag in ("use_2cta", "use_clc", "swap_ab"):
            config[flag] = bool(config[flag])
        config["k_pad"] = -1
        config["walk"] = 0
        num_n = -(-mma_n // e["output_n"])
        for split_k in entry_splits:
            for supergroup in supergroups(num_n):
                for l2_promo in (0, 2):
                    space.append(dict(config, split_k=split_k,
                                      supergroup=supergroup,
                                      l2_promo=l2_promo))
    return space


def smm_candidates(ext, m, n, k, dtype):
    """Return the block-scaled candidate space for a shape: every compiled
    kernel of this format that fits it, crossed with supergroup and
    persistence.
    """
    want = next((elem, sf) for name, elem, sf in ext.formats()
                if name == dtype.name)
    block_k = 1024 // (4 if dtype.elem == "e2m1" else 8)
    space = []
    for e in unique(ext.registry()):
        if (e["elem"], e["sf"]) != want:
            continue
        mma_n = m if e["swap_ab"] else n
        # Narrow tiles are for shapes they cover in one go.
        if e["output_n"] % 128 and e["output_n"] < mma_n:
            continue
        # A 2-CTA tile needs a whole K block.
        if e["use_2cta"] and k < block_k:
            continue
        config = {key: value for key, value in e.items()
                  if key not in ("elem", "sf")}
        for flag in ("use_2cta", "use_clc", "swap_ab"):
            config[flag] = bool(config[flag])
        num_n = -(-mma_n // e["output_n"])
        # Cluster launch control is itself persistent.
        persistents = (1,) if e["use_clc"] else (1, 0)
        for supergroup in supergroups(num_n):
            for persistent in persistents:
                space.append(dict(config, supergroup=supergroup,
                                  persistent=persistent, epi_direct=0))
    return space


def cluster_ctas(config):
    group = 2 if config["use_2cta"] else 1
    return ((config.get("cluster_m") or group)
            * (config.get("cluster_n") or 1) * (config.get("cluster_k") or 1))


def simplicity(config):
    """Return a sort key under which the plainer configuration wins a tie."""
    return (config["swap_ab"], config["use_clc"],
            config.get("epi_double", 0), config.get("epi_trade", 0),
            config.get("deep_stages", 0),
            0 if (config.get("split_k") or 1) <= 1 else config["split_k"],
            1 if config.get("block_m", 128) != 128 else 0,
            cluster_ctas(config), config.get("epi_direct", 0),
            config["output_n"], int(config["use_2cta"]),
            config.get("epi_hold", 1), 0 if config.get("persistent", 1) else 1)


def label(config):
    """Return the one-line label for a configuration, as the sweep prints it.
    """
    group = 2 if config["use_2cta"] else 1
    text = (f"v{group} n{config['output_n']:<3} sg{config['supergroup']:<2} "
            f"clc{int(config['use_clc'])} swap{int(config['swap_ab'])}")
    for key in ("stages", "epi_double", "split_k", "epi_trade", "deep_stages",
                "epi_direct", "l2_promo"):
        if config.get(key):
            text += f" {key}{config[key]}"
    if config.get("block_m") == 64:
        text += " bm64"
    if not config.get("persistent", 1):
        text += " onepass"
    if cluster_ctas(config) > group:
        text += (f" c{config.get('cluster_m') or group}"
                 f"x{config.get('cluster_n') or 1}"
                 f"x{config.get('cluster_k') or 1}")
    return text


def tune(name, m, n, k, ext=None):
    """Sweep one (dtype name, shape), store the winner and return it.

    Args:
        name: Dtype name.
        m: Rows of A and C.
        n: Rows of B and columns of C.
        k: Reduction length in elements.
        ext: The built extension; loaded on demand when None.

    Returns:
        The winning configuration, as stored in configs.json.

    Raises:
        RuntimeError: If no candidate is correct.
    """
    dtype = DTYPES[name]
    ext = ext or extension(dtype.impl)
    buffers = bench.make_input_set(m, n, k, dtype)
    space = (mm_candidates if dtype.impl == "mm" else smm_candidates)(
        ext, m, n, k, dtype)
    log(f"tuning {name} {m}x{n}x{k} on {bench.env_stamp()['gpu']}: "
        f"{len(space)} candidates")

    ranked, rejected = [], 0
    for config in space:
        if bench.correctness_error(ext, buffers, config, dtype, k):
            rejected += 1
            continue
        kernel = bench.runner(ext, buffers, config, dtype)
        warmup, iterations = bench.plan(kernel)
        ranked.append((bench.timed(kernel, warmup, iterations, repeats=3),
                       config))
    if not ranked:
        raise RuntimeError(f"no correct candidate for {name} {m}x{n}x{k}")

    ranked.sort(key=lambda row: row[0])
    fastest = ranked[0][0]
    best = min((row for row in ranked if row[0] <= fastest * (1 + TIE)),
               key=lambda row: simplicity(row[1]))[1]

    kernel = bench.runner(ext, buffers, best, dtype)
    baseline, why = bench.baseline_for(buffers, dtype)
    warmup, iterations = bench.report_plan(kernel)
    us = bench.timed(kernel, warmup, iterations)
    base = bench.timed(baseline, warmup, iterations) if baseline else None

    log(f"best: {label(best)}  ({rejected} candidates failed correctness)")
    path = bench.write_entry(
        dtype.impl, {"dtype": name, "m": m, "n": n, "k": k, "config": best,
                     "measured": {"us": round(us, 3),
                                  "cublas_us": base and round(base, 3),
                                  "env": bench.env_stamp()}})
    log(f"kernel {us:.2f}us  "
        + (f"cuBLAS {base:.2f}us  {base / us:.3f}x" if base
           else f"no baseline: {why}")
        + f"; stored in {path}")
    return best


def resolve_config(dtype, m, n, k):
    """Return the configuration for a (dtype name, shape): the stored one, else
    the one a sweep finds now and stores.

    Raises:
        ValueError: If the shape is recorded as unimplementable.
    """
    stored = bench.stored_config(dtype, m, n, k)
    if isinstance(stored, str):
        raise ValueError(f"{dtype} {m}x{n}x{k} cannot be served: {stored}")
    if stored is not None:
        return stored
    log(f"no stored configuration for {dtype} {m}x{n}x{k}; tuning it now, "
        f"a few minutes on the GPU")
    return tune(dtype, m, n, k)

"""The configuration a shape runs, in the vocabulary the kernel is compiled for.

configs.json and the sweep speak the harness vocabulary: use_2cta where the
instruction says cta_group, output_n where it says mma_n, epi_double where it
says epi_mode, and cluster_m for the cluster's whole M extent. This module is
the one place the two meet, so an emitted kernel and a launched one cannot
disagree about what a stored configuration means.
"""

from .dtypes import DTYPES

CONFIG_KEYS = {
    "mm": ("use_2cta", "output_n", "use_clc", "supergroup", "swap_ab",
           "k_pad", "epi_direct", "epi_hold", "cluster_n", "stages",
           "epi_double", "split_k", "l2_promo", "block_m", "cluster_m",
           "cluster_k", "walk"),
    "smm": ("use_2cta", "output_n", "use_clc", "supergroup", "swap_ab",
            "epi_direct", "epi_trade", "deep_stages", "cluster_m",
            "cluster_n", "cluster_k", "persistent"),
}


def check(config, impl):
    """Raise if `config` names a key `impl`'s kernels do not take.

    Raises:
        KeyError: With the offending keys.
    """
    unknown = sorted(set(config) - set(CONFIG_KEYS[impl]))
    if unknown:
        raise KeyError(f"config has unknown keys: {unknown}")


def splits_for(epi_mode, requested, k):
    """Return the split count a config runs, clamped to the tiles K offers.

    The kernel is specialized on whether it accumulates, so the compiled flag
    has to follow the count after the clamp rather than the request before it.
    """
    if epi_mode == 1 or requested <= 1:
        return 1
    return min(requested, -(-k // 128))


def mm_row(policy):
    """Return `policy` as the registry row src/mm/launch.cuh exports.

    The kernel is templated on the hardware names; the registry reads back in
    the harness ones, and that is the side a stored configuration is matched to
    a compiled kernel on.
    """
    return {"elem_a": policy["elem_a"],
            "elem_b": policy["elem_b"],
            "use_2cta": policy["cta_group"] == 2,
            "block_m": policy["block_m"],
            "output_n": policy["mma_n"],
            "stages": policy["stages"],
            "swap_ab": policy["swap_ab"],
            "epi_hold": policy["epi_hold"],
            "epi_double": policy["epi_mode"],
            "epi_direct": policy["epi_direct"],
            "use_clc": policy["use_clc"],
            "splits_expected": policy["split_k"],
            "cluster_m": policy["cta_group"] * policy["rm"],
            "cluster_n": policy["rn"],
            "cluster_k": policy["rk"]}


def smm_row(policy):
    """Return `policy` as the registry row src/smm/launch.cuh exports."""
    return {"elem": policy["elem_a"],
            "sf": policy["elem_sf"],
            "use_2cta": policy["cta_group"] == 2,
            "output_n": policy["mma_n"],
            "swap_ab": policy["swap_ab"],
            "epi_trade": policy["epi_trade"],
            "deep_stages": policy["deep"],
            "use_clc": policy["use_clc"],
            "cluster_m": policy["cta_group"] * policy["rm"],
            "cluster_n": policy["rn"],
            "cluster_k": policy["rk"]}


def mm_policy(dtype, config, k):
    """Return `(policy, launch)` for a dense configuration.

    Args:
        dtype: Dtype name as configs.json spells it.
        config: A configs.json configuration.
        k: Reduction length in values, not bytes.

    Returns:
        `policy`, the fields the kernel is templated on with the elements as
        names; and `launch`, the arguments read at launch time.
    """
    check(config, "mm")
    d = DTYPES[dtype]
    group = 2 if config["use_2cta"] else 1
    cluster_m = config.get("cluster_m") or 0
    epi_mode = config.get("epi_double", 0)
    splits = splits_for(epi_mode, config.get("split_k", 0), k)
    policy = {
        "elem_a": d.elem_a,
        "elem_b": d.elem_b,
        "cta_group": group,
        "block_m": config.get("block_m") or 128,
        "mma_n": config["output_n"],
        "stages": config.get("stages", 0),
        "swap_ab": bool(config["swap_ab"]),
        "epi_hold": config.get("epi_hold", 1),
        "epi_mode": epi_mode,
        "epi_direct": bool(config.get("epi_direct", 0)),
        "use_clc": bool(config["use_clc"]),
        "split_k": splits > 1,
        "rm": max((cluster_m // group) if cluster_m > 0 else 1, 1),
        "rn": max(config.get("cluster_n") or 1, 1),
        "rk": max(config.get("cluster_k") or 1, 1),
    }
    launch = {"supergroup": config["supergroup"],
              "l2_promo": config.get("l2_promo", 0),
              "splits": splits,
              "walk": config.get("walk", 0)}
    return policy, launch


def smm_policy(dtype, config):
    """Return `(policy, launch)` for a block-scaled configuration.

    Args:
        dtype: Format name as configs.json spells it.
        config: A configs.json configuration.

    Returns:
        `policy` and `launch` as `mm_policy` returns them; the block-scaled
        kernel carries no split count, walk or L2 promotion, and reads
        epi_direct and persistent at launch.
    """
    check(config, "smm")
    d = DTYPES[dtype]
    group = 2 if config["use_2cta"] else 1
    cluster_m = config.get("cluster_m") or group
    policy = {
        "elem_a": d.elem,
        "elem_b": d.elem,
        "elem_sf": d.sf,
        "cta_group": group,
        "mma_n": config["output_n"],
        "swap_ab": bool(config["swap_ab"]),
        "epi_trade": config.get("epi_trade", 0),
        "deep": bool(config.get("deep_stages", 0)),
        "use_clc": bool(config["use_clc"]),
        "rm": max(cluster_m // group, 1),
        "rn": max(config.get("cluster_n") or 1, 1),
        "rk": max(config.get("cluster_k") or 1, 1),
    }
    launch = {"supergroup": config["supergroup"],
              "epi_direct": config.get("epi_direct", 0),
              "persistent": config.get("persistent", 1)}
    return policy, launch

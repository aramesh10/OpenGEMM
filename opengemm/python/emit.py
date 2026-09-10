"""Emit a standalone kernel specialized to one (dtype, shape).

The pair compiles against nothing but CUDA: the shared headers are inlined,
every knob is a compile-time constant, and every branch the configuration
decides is folded away.
"""

import re
import subprocess
import tempfile
from pathlib import Path


from . import fold
from .build import SRC, nvcc
from .configs import resolve_config
from .dtypes import DTYPES, dtype_name, elems, k_of
from .log import log

MARK = "// opengemm"

HEADER_INCLUDES = ["#include <cudaTypedefs.h>"]


def drop_blocks(text, markers, replace=None):
    """Remove the top-level constructs whose first line starts with one of `markers`, brace-balanced; `replace` gives text to put in a construct's place."""
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        marker = next((m for m in markers if lines[i].startswith(m)), None)
        if marker is None:
            out.append(lines[i])
            i += 1
            continue
        while out and out[-1].startswith("template <"):
            out.pop()
        if replace and marker in replace:
            out.append(replace[marker])
        depth, opened = 0, False
        while i < len(lines):
            depth += lines[i].count("{") - lines[i].count("}")
            opened = opened or "{" in lines[i]
            i += 1
            if opened and depth <= 0:
                break
        while i < len(lines) and not lines[i].strip():
            i += 1
    return "\n".join(out)


def elem_traits(src, used):
    """ElemTraits specialized to the elements this kernel uses, read out of types.cuh's ELEM_SPEC."""
    text = (src / "types.cuh").read_text()
    order = re.search(r"enum class Elem \{([^}]*)\}", text).group(1)
    order = [name.strip() for name in order.split(",") if name.strip()]
    body = re.search(r"constexpr ElemSpec ELEM_SPEC\[\] = \{(.*?)\n\};", text, re.S).group(1)
    rows = re.findall(r"\{([^{}]*)\}", body)
    if len(rows) != len(order):
        raise RuntimeError("ELEM_SPEC and the Elem enum disagree in length")

    out = ["template <Elem E>\nstruct ElemTraits;\n"]
    for name in dict.fromkeys(used):
        value_bits, container_bits, global_bits, fmt, kind, tmap = [
            field.strip() for field in rows[order.index(name)].split(",")
        ]
        out.append(
            f"template <>\nstruct ElemTraits<Elem::{name}> {{\n"
            f"    static constexpr int value_bits = {value_bits};\n"
            f"    static constexpr int container_bits = {container_bits};\n"
            f"    static constexpr int global_bits = {global_bits};\n"
            f"    static constexpr int format = {fmt};\n"
            f"    static constexpr Kind kind = {kind};\n"
            f"    static constexpr CUtensorMapDataType tmap = {tmap};\n"
            "};\n"
        )
    return "\n".join(out)


def strip_static_asserts(text):
    """Remove the static_asserts; this kernel's Policy is already known good."""
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith("static_assert("):
            depth = 0
            while i < len(lines):
                depth += lines[i].count("(") - lines[i].count(")")
                i += 1
                if depth <= 0:
                    break
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def pin_policy(text, values):
    """Substitute the Policy away: every `P.field` becomes the literal it was going to be."""
    text = re.sub(r"^template <Policy P>\n", "", text, flags=re.M)
    text = text.replace("Geom<P>", "Geom")
    for name, literal in values.items():
        text = re.sub(rf"\bP\.{name}\b", literal, text)
    text = re.sub(r"\btrue \? ([^;\n]*?) : [^;\n]*?;", r"\1;", text)
    text = re.sub(r"\bfalse \? [^;\n]*? : ([^;\n]*?);", r"\1;", text)
    return text


GEOM_MEMBER = re.compile(r"^\s*static constexpr\s+(\w+)\s+(\w+)\s*=", re.M)


def geom_members(text):
    body = re.search(r"^struct Geom \{(.*?)^\};", text, re.S | re.M).group(1)
    return [(m.group(1), m.group(2)) for m in GEOM_MEMBER.finditer(body)]


def run_probe(header_text, declarations, names, workdir):
    """Ask nvcc what each of `names` evaluates to, as `{last :: component: int}`."""
    (workdir / "geom.cuh").write_text(header_text)
    prints = "\n".join(f'  printf("{name}=%lld\\n", (long long)({name}));' for name in names)
    (workdir / "probe.cu").write_text(
        '#include <cstdio>\n#include "geom.cuh"\n\n'
        + declarations
        + f"\nint main() {{\n{prints}\n  return 0;\n}}\n"
    )
    nvcc([workdir / "probe.cu"], workdir / "probe")
    run = subprocess.run([str(workdir / "probe")], capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError("the probe did not run:\n" + run.stderr)
    values = {}
    for line in run.stdout.splitlines():
        name, _, value = line.partition("=")
        values[name.rsplit("::", 1)[-1]] = int(value)
    missing = [n.rsplit("::", 1)[-1] for n in names if n.rsplit("::", 1)[-1] not in values]
    if missing:
        raise RuntimeError(f"the probe did not report {missing}")
    return values


def evaluate_geom(header_text, members, workdir):
    """Every Geom member by value, from the compiler that owns the definition."""
    return run_probe(header_text, "", [f"Geom::{n}" for _, n in members], workdir)


def kernel_body(text, kernel):
    start = text.index(f"void {kernel}(")
    params = text.index("(", start)
    brace = text.index("{", fold.match_bracket(text, params))
    return brace + 1, fold.match_bracket(text, brace)


def evaluate_locals(header_text, kernel, workdir):
    """The kernel body's own constexpr locals by value, so branches on them fold too."""
    lo, hi = kernel_body(header_text, kernel)
    seen, decls = {}, []
    for kind, name, expr in re.findall(
        r"constexpr\s+([\w:]+)\s+(\w+)\s*=\s*(.*?);", header_text[lo:hi], re.S
    ):
        expr = " ".join(expr.split())
        if name in seen:
            if seen[name] != expr:
                raise RuntimeError(f"{kernel} declares {name} twice, differently")
            continue
        seen[name] = expr
        decls.append(f"constexpr {kind} {name} = {expr};")
    if not decls:
        return {}
    return run_probe(
        header_text,
        "namespace probe {\n" + "\n".join(decls) + "\n}\n",
        [f"probe::{name}" for name in seen],
        workdir,
    )


def enum_names(text, enum):
    body = re.search(rf"enum class {enum} \{{([^}}]*)\}}", text).group(1)
    return [name.strip() for name in body.split(",") if name.strip()]


def tmap_names(src, dtype, impl, acc_type):
    """The CUtensorMapDataType enumerators for the operands and accumulator, by name."""
    if impl != "mm":
        return {}
    text = (src / "types.cuh").read_text()
    order = enum_names(text, "Elem")
    body = re.search(r"constexpr ElemSpec ELEM_SPEC\[\] = \{(.*?)\n\};", text, re.S).group(1)
    rows = re.findall(r"\{([^{}]*)\}", body)
    elem_a, _, elem_b = dtype.partition("x")
    elem_b = elem_b or elem_a
    pick = lambda e: rows[order.index(e)].split(",")[5].strip()
    return {
        "tmap_a": pick(elem_a),
        "tmap_b": pick(elem_b),
        "tmap_c": (
            "CU_TENSOR_MAP_DATA_TYPE_INT32"
            if acc_type == "int"
            else "CU_TENSOR_MAP_DATA_TYPE_FLOAT32"
        ),
    }


def flatten_geom(text, members, values, tmaps, acc_type):
    """Replace struct Geom with the constants it evaluates to."""
    lines = []
    for kind, name in members:
        if name in tmaps:
            lines.append(f"constexpr CUtensorMapDataType G_{name} = {tmaps[name]};")
        elif re.search(rf"enum class {kind} \{{", text):
            names = enum_names(text, kind)
            lines.append(f"constexpr {kind} G_{name} = {kind}::{names[values[name]]};")
        elif kind == "bool":
            lines.append(f"constexpr bool G_{name} = {'true' if values[name] else 'false'};")
        else:
            lines.append(f"constexpr {kind} G_{name} = {values[name]};")
    text = drop_blocks(text, ("struct Geom",))
    anchor = "// ---- the tuned configuration, as compile-time constants ----"
    assert anchor in text, "config constant block not found"
    enums = re.findall(r"^enum class \w+ \{[^}]*\};", text, re.M)
    for declaration in enums:
        text = text.replace(declaration + "\n", "", 1)
    text = text.replace(anchor, anchor + "\n" + "\n".join(enums + lines), 1)
    text = re.sub(r"^\s*using G = Geom;\n", "", text, flags=re.M)
    text = re.sub(r"^\s*using ACC = typename G::acc_t;\n", "", text, flags=re.M)
    text = re.sub(r"typename (?:G|Geom)::acc_t", acc_type, text)
    text = re.sub(r"\b(?:G|Geom)::(\w+)", r"G_\1", text)
    text = re.sub(r"\bACC\b", acc_type, text)
    return text


def expand_clc(text):
    """Replace the CLC aggregate with the four values it holds."""
    found = re.search(r"^struct (Clc|CLC) \{", text, re.M)
    if not found:
        return text
    name = found.group(1)
    text = drop_blocks(text, (f"struct {name}",))

    params = (
        "int clc_arrived, int clc_finished, int clc_ready,\n"
        "                          uint4 *clc_handles"
    )
    before = text
    text = text.replace(f"const {name} &clc", params)
    assert text != before, "CLC parameters not found"

    for field in ("arrived", "finished", "ready", "handles"):
        text = text.replace(f"clc.{field}", f"clc_{field}")

    build = re.search(
        rf"( *)const {name} clc = \{{clc_base,\s*\n"
        r"\s*clc_base \+ CLC_DEPTH \* MBAR,\s*\n"
        r"\s*clc_base \+ 2 \* CLC_DEPTH \* MBAR,\s*\n"
        r"\s*clc_handles\};",
        text,
    )
    assert build, "CLC construction not found"
    pad = build.group(1)
    text = (
        text[: build.start()]
        + (
            f"{pad}const int clc_arrived  = clc_base;\n"
            f"{pad}const int clc_finished = clc_base + CLC_DEPTH * MBAR;\n"
            f"{pad}const int clc_ready    = clc_base + 2 * CLC_DEPTH * MBAR;"
        )
        + text[build.end() :]
    )

    args = "clc_arrived, clc_finished, clc_ready, clc_handles"
    text, calls = re.subn(r"\b(clc_issue|next_tile)<([^>]*)>\(clc, ", rf"\1<\2>({args}, ", text)
    assert calls >= 2, f"expected CLC call sites, rewrote {calls}"
    return text


def flatten_enums(text):
    """Replace the kernel's own enums with integer constants; the CUDA driver's are left alone."""
    for name, body in re.findall(r"^enum class (\w+) \{([^}]*)\};", text, re.M):
        members = [x.strip() for x in body.split(",") if x.strip()]
        constants = "\n".join(
            f"constexpr int {name}_{member} = {index};" for index, member in enumerate(members)
        )
        text = re.sub(rf"^enum class {name} \{{[^}}]*\}};", constants, text, flags=re.M)
        text = re.sub(rf"\b{name}::(\w+)", rf"{name}_\1", text)
        text = re.sub(rf"\b{name}\b(?!_)", "int", text)

    def unnamed(match):
        rows = [
            row.strip().rstrip(",")
            for row in match.group(1).splitlines()
            if row.strip().rstrip(",")
        ]
        return "\n".join(f"constexpr int {row};" for row in rows)

    return re.sub(r"^enum : int \{\n(.*?)^\};", unnamed, text, flags=re.M | re.S)


def take_includes(text):
    kept = [line for line in text.splitlines() if not line.strip().startswith("#include <")]
    return "\n".join(kept).strip("\n")


def drop_alias_layer(text):
    """Remove the kernel's local re-declarations of the config, `BLOCK_M = G_block_m` and the rest."""
    lines = text.splitlines()
    opens = [
        i for i, line in enumerate(lines) if re.match(r"^(void )?\w*_?gemm_kernel\(", line.strip())
    ]
    if not opens:
        return text
    begin = opens[0]

    alias, kept = {}, []
    pattern = re.compile(r"^\s*constexpr\s+\w+\s+(.+);\s*$")
    for index, line in enumerate(lines):
        found = pattern.match(line) if index > begin else None
        if found and "G_" in line:
            pairs = [part.split("=") for part in found.group(1).split(",")]
            if all(
                len(pair) == 2
                and pair[1].strip().startswith("G_")
                and pair[1].strip().isidentifier()
                for pair in pairs
            ):
                for name, value in pairs:
                    alias[name.strip()] = value.strip()
                continue
        kept.append(line)

    head = "\n".join(kept[:begin])
    body = "\n".join(kept[begin:])
    for name, value in alias.items():
        body = re.sub(rf"\b{name}\b", value, body)
    return head + "\n" + body


def bind_launch_args(text):
    """Turn the kernel's runtime launch arguments into the constants the header states."""
    before = text
    text = re.sub(
        r"int m, int n, int k,\s*\n\s*int supergroup, int walk, "
        r"int splits\) \{",
        "int m, int n, int k) {",
        text,
    )
    text = re.sub(r"^\s*int supergroup, int walk,\n", "", text, flags=re.M)
    text = text.replace(
        "tile_coords(spatial, num_m_groups, num_n_groups, supergroup, walk,",
        "tile_coords(spatial, num_m_groups, num_n_groups,",
    )
    text = re.sub(
        r"int m, int n, int k,\s*\n\s*int supergroup,\s*\n\s*"
        r"int epi_direct\) \{",
        "int m, int n, int k) {",
        text,
    )
    assert text != before, "no kernel signature matched"
    text = re.sub(r"\bsupergroup\b", "SUPERGROUP", text)
    text = re.sub(r"\bwalk\b", "WALK", text)
    text = re.sub(r"\bsplits\b", "SPLITS", text)
    text = re.sub(r"\bepi_direct\b", "EPI_DIRECT", text)
    return text


def function_extent(text, start):
    depth, i, opened = 0, start, False
    while i < len(text):
        if text[i] == "{":
            depth += 1
            opened = True
        elif text[i] == "}":
            depth -= 1
            if opened and depth == 0:
                return start, i + 2
        i += 1
    return start, len(text)


def strip_comments(text):
    """Remove every C++ comment, leaving string literals alone."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote, out_start = c, i
            i += 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
            i += 1
            out.append(text[out_start:i])
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
        else:
            out.append(c)
            i += 1
    lines = [line.rstrip() for line in "".join(out).splitlines()]
    kept, blank = [], False
    for line in lines:
        if line:
            kept.append(line)
            blank = False
        elif not blank and kept:
            kept.append("")
            blank = True
    return "\n".join(kept).strip("\n") + "\n"


def rename_template_params(text):
    """Give every remaining template parameter a trailing underscore, so the constants can take the plain uppercase names."""
    while True:
        for found in re.finditer(r"^template <([^>]*)>\n", text, re.M):
            names = [x.split()[-1] for x in found.group(1).split(",")]
            if all(name.endswith("_") for name in names):
                continue
            start, stop = function_extent(text, found.start())
            span = text[start:stop]
            for name in names:
                if not name.endswith("_"):
                    span = re.sub(rf"\b{name}\b", name + "_", span)
            text = text[:start] + span + text[stop:]
            break
        else:
            return text


def plain_constant_names(text):
    """Rename `G_block_m` to `BLOCK_M`, `Kind_f8f6f4` to `KIND_F8F6F4`, and so on."""
    renames = {}
    for name in set(re.findall(r"\b(G_\w+)\b", text)):
        renames[name] = name[2:].upper()
    for prefix in ("Elem", "Kind", "Acc", "SElem", "SFElem", "SKind"):
        for name in set(re.findall(rf"\b({prefix}_\w+)\b", text)):
            renames[name] = name.upper()
    taken = set(re.findall(r"\b[A-Z][A-Z0-9_]*\b", text)) - set(renames)
    for new_name in {n for n in renames.values() if n in taken}:
        assert new_name + "_" not in taken, f"cannot move {new_name} aside"
        text = re.sub(rf"\b{new_name}\b", new_name + "_", text)
    for old, new in sorted(renames.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"\b{old}\b", new, text)
    return text


def inline(path):
    """A header's text without its include guard and local includes."""
    out = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "#pragma once" or stripped.startswith('#include "'):
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


def cpp_bool(value):
    return "true" if value else "false"


def splits_for(epi_double, split_k, k):
    """The split count a configuration runs, clamped to the K tiles."""
    if epi_double == 1 or split_k <= 1:
        return 1
    return min(split_k, -(-k // 128))


def policy_literals(name, config, k):
    """A configs.json entry as the C++ literals `pin_policy` substitutes for `P.<field>`, and the launch-time constants; mirrors policy_of in src/<impl>/registry.cuh."""
    group = 2 if config["use_2cta"] else 1
    cluster = {
        "rm": max(config["cluster_m"] // group, 1),
        "rn": max(config["cluster_n"], 1),
        "rk": max(config["cluster_k"], 1),
    }
    if DTYPES[name].impl == "mm":
        elem_a, elem_b = elems(name)
        splits = splits_for(config["epi_double"], config["split_k"], k)
        policy = {
            "elem_a": f"Elem::{elem_a}",
            "elem_b": f"Elem::{elem_b}",
            "cta_group": group,
            "block_m": config["block_m"],
            "mma_n": config["output_n"],
            "stages": config["stages"],
            "swap_ab": bool(config["swap_ab"]),
            "epi_hold": config["epi_hold"],
            "epi_mode": config["epi_double"],
            "epi_direct": bool(config["epi_direct"]),
            "use_clc": bool(config["use_clc"]),
            "split_k": splits > 1,
            **cluster,
        }
        constants = {
            "SUPERGROUP": config["supergroup"],
            "L2_PROMO": config["l2_promo"],
            "SPLITS": splits,
            "WALK": config["walk"],
        }
    else:
        elem, sf = elems(name)
        policy = {
            "elem_a": f"SElem::{elem}",
            "elem_b": f"SElem::{elem}",
            "elem_sf": f"SFElem::{sf}",
            "cta_group": group,
            "mma_n": config["output_n"],
            "swap_ab": bool(config["swap_ab"]),
            "epi_trade": config["epi_trade"],
            "deep": bool(config["deep_stages"]),
            "use_clc": bool(config["use_clc"]),
            **cluster,
        }
        constants = {
            "SUPERGROUP": config["supergroup"],
            "EPI_DIRECT": config["epi_direct"],
            "PERSISTENT": config["persistent"],
        }
    literals = {key: cpp_bool(v) if isinstance(v, bool) else str(v) for key, v in policy.items()}
    return literals, constants


MM_LAUNCH = """
extern "C" void {name}(const void *a, const void *b, void *c,
                       cudaStream_t stream) {{
  using G = Geom;{zero_c}

  // This library carries its own CUDA runtime state, whose current device
  // starts at 0, so a caller on any other one has to be followed. The operands
  // say where that is; the stream cannot, since the default stream is 0 on
  // every device. The attributes below are per device too.
  int device = 0;
  cudaPointerAttributes where{{}};
  if (cudaPointerGetAttributes(&where, a) == cudaSuccess)
    device = where.device;
  int current = 0;
  cudaGetDevice(&current);
  if (current != device)
    cudaSetDevice(device);

  static thread_local uint64_t configured = 0;
  const uint64_t seen = uint64_t{{1}} << (device & 63);
  if (!(configured & seen)) {{
    configured |= seen;
    if constexpr (G::cluster_ctas > 8)
      cudaFuncSetAttribute(mm_gemm_kernel,
                           cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mm_gemm_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         G::smem_bytes);
  }}

  void *pa = const_cast<void *>(G::swap_ab ? b : a);
  void *pb = const_cast<void *>(G::swap_ab ? a : b);
  constexpr int mma_m = G::swap_ab ? N : M;
  constexpr int mma_n = G::swap_ab ? M : N;
  constexpr CUtensorMapL2promotion l2 =
      (L2_PROMO == 2)   ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B
    : (L2_PROMO == 1) ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B
                      : CU_TENSOR_MAP_L2_PROMOTION_NONE;

  CUtensorMap a_tmap, b_tmap, c_tmap;
  init_ab_tmap(&a_tmap, pa, mma_m, K, G::block_m, G::block_k, G::tmap_a,
               G::global_bits_a, 0, l2);
  init_ab_tmap(&b_tmap, pb, mma_n, K, G::block_n, G::block_k, G::tmap_b,
               G::global_bits_b, 0, l2);
  constexpr int c_cols      = (N + 3) & ~3;
  constexpr int c_tile_rows = G::swap_ab ? G::c_tma_n : G::block_m;
  constexpr int c_tile_cols = G::swap_ab   ? G::block_m
                            : G::vec_stage ? G::store_n
                                           : G::c_tma_n;
  constexpr CUtensorMapSwizzle c_swizzle = G::vec_stage
                                         ? CU_TENSOR_MAP_SWIZZLE_128B
                                         : CU_TENSOR_MAP_SWIZZLE_NONE;
  init_c_tmap(&c_tmap, c, M, c_cols, c_tile_rows, c_tile_cols, G::tmap_c,
              c_swizzle);

  constexpr int tiles_m = (mma_m + G::tile_m - 1) / G::tile_m;
  constexpr int tiles_n = (mma_n + G::mma_n - 1) / G::mma_n;
  constexpr int logical_clusters =
      ((tiles_m + G::rm - 1) / G::rm) * ((tiles_n + G::rn - 1) / G::rn)
      * SPLITS;

  cudaLaunchConfig_t cfg = {{}};
  cfg.blockDim = dim3(G::threads, 1, 1);
  cfg.dynamicSmemBytes = G::smem_bytes;
  cfg.stream = stream;

  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = G::cluster_m;
  attrs[0].val.clusterDim.y = G::rn;
  attrs[0].val.clusterDim.z = G::rk;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;

  int launch_clusters = logical_clusters;
  if constexpr (!G::use_clc) {{
    static const int resident = [&] {{
      int blocks = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, mm_gemm_kernel, G::threads, G::smem_bytes);
      int sms = 0, device = 0;
      cudaGetDevice(&device);
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
      const int ctas = blocks * sms;
      const int per_cluster = G::cluster_m * G::rn * G::rk;
      return (ctas / per_cluster) > 0 ? (ctas / per_cluster) : 1;
    }}();
    launch_clusters = logical_clusters < resident ? logical_clusters
                                                  : resident;
  }}
  cfg.gridDim = dim3(launch_clusters * G::cluster_m, G::rn, G::rk);

  cudaLaunchKernelEx(&cfg, mm_gemm_kernel, a_tmap, b_tmap, c_tmap,
                     static_cast<typename G::acc_t *>(c), mma_m, mma_n, K);
}}
"""

SMM_LAUNCH = """
extern "C" void {name}(const void *a, const void *b, const void *sfa,
                       const void *sfb, void *c, cudaStream_t stream) {{
  using G = Geom;{zero_c}
  // This library carries its own CUDA runtime state, whose current device
  // starts at 0, so a caller on any other one has to be followed. The operands
  // say where that is; the stream cannot, since the default stream is 0 on
  // every device. The attributes below are per device too.
  int device = 0;
  cudaPointerAttributes where{{}};
  if (cudaPointerGetAttributes(&where, a) == cudaSuccess)
    device = where.device;
  int current = 0;
  cudaGetDevice(&current);
  if (current != device)
    cudaSetDevice(device);

  static thread_local uint64_t configured = 0;
  const uint64_t seen = uint64_t{{1}} << (device & 63);
  if (!(configured & seen)) {{
    configured |= seen;
    if constexpr (G::cluster_ctas > 8)
      cudaFuncSetAttribute(smm_gemm_kernel,
                           cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(smm_gemm_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         G::smem_bytes);
  }}

  void *pa  = const_cast<void *>(G::swap_ab ? b : a);
  void *pb  = const_cast<void *>(G::swap_ab ? a : b);
  void *psa = const_cast<void *>(G::swap_ab ? sfb : sfa);
  void *psb = const_cast<void *>(G::swap_ab ? sfa : sfb);
  constexpr int mma_m = G::swap_ab ? N : M;
  constexpr int mma_n = G::swap_ab ? M : N;

  const int sf_n_blocks = (G::cta_group == 1) ? G::sf_n_blocks : 1;
  constexpr int c_cols      = (N + 7) & ~7;
  constexpr int c_tile_rows = G::swap_ab ? G::c_tma_n : G::block_m;
  constexpr int c_tile_cols = G::swap_ab ? G::block_m
                            : G::vec_stage ? G::store_n
                                           : G::c_tma_n;
  constexpr CUtensorMapSwizzle c_swizzle = G::vec_stage
                                         ? CU_TENSOR_MAP_SWIZZLE_128B
                                         : CU_TENSOR_MAP_SWIZZLE_NONE;

  CUtensorMap a_tmap, b_tmap, c_tmap, sfa_tmap, sfb_tmap;
  init_ab_tmap(&a_tmap, pa, mma_m, K, G::block_m, G::block_k, G::elem_bits);
  init_ab_tmap(&b_tmap, pb, mma_n, K, G::block_n, G::block_k, G::elem_bits);
  init_sf_tmap(&sfa_tmap, psa, mma_m, K, G::block_k, 1, G::sf_block);
  init_sf_tmap(&sfb_tmap, psb, mma_n, K, G::block_k, sf_n_blocks, G::sf_block);
  init_c_tmap(&c_tmap, c, M, c_cols, c_tile_rows, c_tile_cols, c_swizzle);

  constexpr int tiles =
      ((mma_m + G::mma_m - 1) / G::mma_m + G::mc_m - 1) / G::mc_m
    * (((mma_n + G::mma_n - 1) / G::mma_n + G::mc_n - 1) / G::mc_n);

  int clusters = tiles;
  if (PERSISTENT && !G::use_clc) {{
    static const int wave = [] {{
      int blocks = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, smm_gemm_kernel, G::threads, G::smem_bytes);
      int sms = 0, device = 0;
      cudaGetDevice(&device);
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
      return (blocks * sms) / G::cluster_ctas;
    }}();
    if (wave > 0 && wave < clusters)
      clusters = wave;
  }}

  cudaLaunchAttribute attrs[2] = {{}};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = G::cluster_x;
  attrs[0].val.clusterDim.y = G::rn;
  attrs[0].val.clusterDim.z = G::rk;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t cfg = {{}};
  cfg.gridDim          = dim3(clusters * G::cluster_x, G::rn, G::rk);
  cfg.blockDim         = dim3(G::threads, 1, 1);
  cfg.dynamicSmemBytes = G::smem_bytes;
  cfg.stream           = stream;
  cfg.attrs            = attrs;
  cfg.numAttrs         = 2;

  cudaLaunchKernelEx(&cfg, smm_gemm_kernel, a_tmap, b_tmap, c_tmap,
                     sfa_tmap, sfb_tmap, static_cast<uint16_t *>(c),
                     mma_m, mma_n, K);
}}
"""


def emit(dtype, m, n, k, config, stem):
    """`(source, header)` text for one (dtype, shape, config); `stem` is the header's file stem, which the source includes."""
    impl = DTYPES[dtype].impl
    src = SRC / impl
    common = SRC / "common"
    entry = f"{impl}_{dtype}_{m}_{n}_{k}"

    fields, extra = policy_literals(dtype, config, k)
    DEAD = ("struct Policy",)
    if impl == "mm":
        launch = MM_LAUNCH
        types = drop_blocks(
            inline(src / "types.cuh"),
            DEAD + ("struct ElemSpec", "constexpr ElemSpec ELEM_SPEC", "template <Elem E>"),
            replace={"template <Elem E>": elem_traits(src, elems(dtype))},
        )
    else:
        launch = SMM_LAUNCH
        types = drop_blocks(inline(src / "types.cuh"), DEAD)

    shared_body = take_includes(inline(common / "ptx.cuh"))
    types_body = take_includes(types)
    device_body = take_includes(inline(src / "ptx.cuh") + "\n\n" + inline(common / "clc.cuh"))
    kernel_body = take_includes(inline(src / "kernel.cuh"))
    host_body = take_includes(inline(common / "tmap.cuh") + "\n\n" + inline(src / "tmap.cuh"))

    constants = "\n".join(f"constexpr int {key} = {int(value)};" for key, value in extra.items())
    staged = (
        "#pragma once\n\n"
        + "".join(f"{line}\n" for line in HEADER_INCLUDES)
        + "\n// ---- the tuned configuration, as compile-time constants ----\n"
        f"constexpr int M = {m};\nconstexpr int N = {n};\n"
        f"constexpr int K = {k};\n{constants}\n\n"
        + pin_policy(strip_static_asserts(shared_body), fields)
        + "\n\n"
        + pin_policy(strip_static_asserts(types_body), fields)
        + "\n\n"
        + pin_policy(strip_static_asserts(device_body), fields)
        + "\n\n"
        + pin_policy(strip_static_asserts(kernel_body), fields)
        + "\n"
    )

    members = geom_members(staged)
    with tempfile.TemporaryDirectory(prefix="opengemm_emit_") as tmp:
        values = evaluate_geom(staged, members, Path(tmp))
    acc_type = "int" if values.get("acc", 0) == 1 else "float"
    tmaps = tmap_names(src, dtype, impl, acc_type)
    staged = flatten_geom(staged, members, values, tmaps, acc_type)
    staged = drop_blocks(
        staged, ("struct KindOf", "template <Elem E>", "template <>", "template <SElem")
    )
    staged = expand_clc(staged)
    staged = flatten_enums(staged)
    staged = drop_alias_layer(staged)
    staged = bind_launch_args(staged)

    args = (
        "const void *a, const void *b, void *c, cudaStream_t stream"
        if impl == "mm"
        else "const void *a, const void *b, const void *sfa, const void *sfb,\n"
        "                       void *c, cudaStream_t stream"
    )
    accumulates = int(extra.get("SPLITS", 1)) > 1 or int(values.get("rk", 1)) > 1
    zero_c = (
        ("\n  cudaMemsetAsync(c, 0, (size_t)M * N * sizeof(%s), stream);" % acc_type)
        if accumulates
        else ""
    )
    body = (
        host_body
        + "\n\n"
        + f'extern "C" void {entry}({args});\n'
        + launch.format(name=entry, zero_c=zero_c)
    )
    body = re.sub(r"^\s*using G = Geom;\n", "", body, flags=re.M)
    body = re.sub(r"typename (?:G|Geom)::acc_t", acc_type, body)
    body = re.sub(r"\b(?:G|Geom)::(\w+)", r"G_\1", body)
    staged = rename_template_params(staged)
    combined = plain_constant_names(staged + "\n\x00\n" + body)
    header, source = combined.split("\n\x00\n")
    header = strip_comments(header)
    source = strip_comments(f'#include "{stem}.cuh"\n\n' + source)

    kernel = f"{impl}_gemm_kernel"
    with tempfile.TemporaryDirectory(prefix="opengemm_emit_") as tmp:
        known = evaluate_locals(header, kernel, Path(tmp))
    header, source = fold.fold_all(header, source, known, {kernel, entry})
    for label, text in (("header", header), ("source", source)):
        left = re.search(r".*if constexpr.*", text)
        if left:
            raise RuntimeError(f"a branch survived folding in the {label}: {left.group(0).strip()}")

    tag = (
        f"{MARK} {impl} {dtype} {m} {n} {k} {entry}\n"
        f"// config: "
        + " ".join(f"{key}={int(value)}" for key, value in sorted(config.items()))
        + "\n"
    )
    return tag + source, tag + header


def parse_tag(text):
    """`(impl, dtype, m, n, k, entry)` from an emitted file's first line."""
    first = text.split("\n", 1)[0]
    if not first.startswith(MARK):
        raise ValueError(
            "not an opengemm kernel: the first line should read "
            f"'{MARK} <impl> <dtype> <M> <N> <K> <entry>'"
        )
    impl, dtype, m, n, k, entry = first[len(MARK) :].split()
    return impl, dtype, int(m), int(n), int(k), entry


def emit_kernel(
    a=None,
    b=None,
    sfa=None,
    sfb=None,
    file=None,
    atype=None,
    btype=None,
    m=None,
    n=None,
    k=None,
    sftype=None,
):
    """Emit a standalone CUDA kernel for one dtype and shape.

    Takes operands, or the shape and element names on their own:

        emit_kernel(a, b, file="k")
        emit_kernel(m=1024, n=1024, k=1024, atype="bf16")
        emit_kernel(m=1024, n=1024, k=1024, atype="e4m3", btype="e5m2")
        emit_kernel(m=1024, n=1024, k=1024, atype="e2m1", sftype="ue4m3")

    Only shapes and dtypes are read, so meta tensors work. The configuration is
    the stored one for the (dtype, shape) or, when there is none, the winner of
    a tuning sweep run first and stored, a few minutes on the GPU.

    Args:
        a: `(M, K)` operand, or a meta tensor of that shape and dtype; None to
            give `m`, `n`, `k` and the element names instead.
        b: `(N, K)` operand.
        sfa: Scales of `a`, for a block-scaled kernel; only the dtype is read.
        sfb: Scales of `b`.
        file: Stem to write `<file>.cu` and `<file>.cuh` to; with None, the
            kernel names itself `<dtype>_<M>_<N>_<K>`.
        atype: Element of `a`: "bf16", "e4m3", "e2m1" and the rest. Names a
            `uint8` operand, or `a` itself without operands.
        btype: Element of `b`; defaults to `atype`.
        m: Rows of A and C; operands give it instead.
        n: Rows of B and columns of C.
        k: Reduction length, in values rather than the packed extent.
        sftype: Scale element, "ue4m3" or "ue8m0", which with `atype` picks
            the block-scaled format: e2m1 x ue4m3 is nvfp4, e4m3 x ue8m0 is
            mxfp8, e2m1 x ue8m0 is mxfp4.

    Returns:
        `(source, header)` text. The entry point is `extern "C" void
        mm_<dtype>_<M>_<N>_<K>(a, b, c, stream)` for dense and
        `smm_<dtype>_<M>_<N>_<K>(a, b, sfa, sfb, c, stream)` for block-scaled;
        `run_kernel` compiles and calls it.
    """
    name = dtype_name(a, b, sfa, atype, btype, sftype)
    if a is not None:
        m, n, k = a.size(0), b.size(0), k_of(DTYPES[name], a.size(1))
    elif None in (m, n, k):
        raise ValueError("without operands, pass m, n and k")
    config = resolve_config(name, m, n, k)
    return write_kernel(name, config, m, n, k, file)


def write_kernel(name, config, m, n, k, file=None):
    """Write the .cu and .cuh pair for one configuration; returns `(source, header)` text."""
    file = Path(file) if file else Path(f"{name}_{m}_{n}_{k}")
    source, header = emit(name, m, n, k, config, file.stem)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.with_suffix(".cuh").write_text(header)
    file.with_suffix(".cu").write_text(source)
    log(f"wrote {file.with_suffix('.cu')} and {file.with_suffix('.cuh')}")
    return source, header

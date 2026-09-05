// Host side of the block-scaled kernel: format inference, the launch, and the
// torch extension surface. Which configuration a shape runs is decided in
// Python from configs.json; this file runs the one it is handed.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "host_utils.cuh"
#include "smm.cuh"
#include "registry.cuh"

namespace {

// A format is an element type and a scale type. Which of the three
// block-scaled MMA kinds serves it is derived from the pair, so a caller
// cannot name a pair the hardware does not have.
struct Format {
  const char *name;
  SElem  elem;
  SFElem sf;
  at::ScalarType elem_dtype;
  at::ScalarType sf_dtype;
};

constexpr Format FORMATS[] = {
    {"nvfp4", SElem::e2m1, SFElem::ue4m3, at::kFloat4_e2m1fn_x2, at::kFloat8_e4m3fn},
    {"mxfp8", SElem::e4m3, SFElem::ue8m0, at::kFloat8_e4m3fn,    at::kFloat8_e8m0fnu},
    {"mxfp4", SElem::e2m1, SFElem::ue8m0, at::kFloat4_e2m1fn_x2, at::kFloat8_e8m0fnu},
};

const Format &format_of(at::ScalarType elem, at::ScalarType sf) {
  for (const Format &f : FORMATS)
    if (f.elem_dtype == elem && f.sf_dtype == sf)
      return f;
  TORCH_CHECK(false, "no block-scaled format takes ", elem, " operands with ",
              sf, " scales; have nvfp4 (float4_e2m1fn_x2 x float8_e4m3fn), "
              "mxfp8 (float8_e4m3fn x float8_e8m0fnu), mxfp4 "
              "(float4_e2m1fn_x2 x float8_e8m0fnu)");
}

template <Policy P>
void configure_kernel() {
  static const bool done = [] {
    cudaFuncSetAttribute(smm_gemm_kernel<P>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         Geom<P>::smem_bytes);
    if constexpr (Geom<P>::cluster_ctas > 8)
      cudaFuncSetAttribute(smm_gemm_kernel<P>,
                           cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    return true;
  }();
  (void)done;
}

struct Operands {
  void *a, *b, *sfa, *sfb;
  int   m, n;
};

template <Policy P>
Operands mma_operands(void *a, void *b, void *sfa, void *sfb, int m, int n) {
  if constexpr (Geom<P>::swap_ab)
    return {b, a, sfb, sfa, n, m};
  else
    return {a, b, sfa, sfb, m, n};
}

template <Policy P>
void launch_cfg(void *a, void *b, void *sfa, void *sfb, void *c, int m, int n,
                int k, int supergroup, int epi_direct, int persistent) {
  using G = Geom<P>;
  configure_kernel<P>();

  const Operands op = mma_operands<P>(a, b, sfa, sfb, m, n);
  const int sf_n_blocks = (G::cta_group == 1) ? G::sf_n_blocks : 1;
  const int c_cols      = (n + 7) & ~7;
  const int c_tile_rows = G::swap_ab ? G::c_tma_n : G::block_m;
  const int c_tile_cols = G::swap_ab ? G::block_m
                        : G::vec_stage ? G::store_n
                                       : G::c_tma_n;
  const CUtensorMapSwizzle c_swizzle = G::vec_stage
                                     ? CU_TENSOR_MAP_SWIZZLE_128B
                                     : CU_TENSOR_MAP_SWIZZLE_NONE;

  CUtensorMap a_tmap, b_tmap, c_tmap, sfa_tmap, sfb_tmap;
  init_ab_tmap(&a_tmap, op.a, op.m, k, G::block_m, G::block_k, G::elem_bits);
  init_ab_tmap(&b_tmap, op.b, op.n, k, G::block_n, G::block_k, G::elem_bits);
  init_sf_tmap(&sfa_tmap, op.sfa, op.m, k, G::block_k, 1, G::sf_block);
  init_sf_tmap(&sfb_tmap, op.sfb, op.n, k, G::block_k, sf_n_blocks, G::sf_block);
  init_c_tmap(&c_tmap, c, m, c_cols, c_tile_rows, c_tile_cols, c_swizzle);

  const int64_t tiles = ceil_div(ceil_div(op.m, G::mma_m), G::mc_m)
                      * ceil_div(ceil_div(op.n, G::mma_n), G::mc_n);
  int clusters = static_cast<int>(tiles);
  if (persistent && !G::use_clc)
    clusters = std::min(clusters, wave_clusters(G::cluster_ctas));

  cudaLaunchAttribute attrs[2] = {};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = G::cluster_x;
  attrs[0].val.clusterDim.y = G::rn;
  attrs[0].val.clusterDim.z = G::rk;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim          = dim3(clusters * G::cluster_x, G::rn, G::rk);
  cfg.blockDim         = dim3(G::threads, 1, 1);
  cfg.dynamicSmemBytes = G::smem_bytes;
  cfg.stream           = c10::cuda::getCurrentCUDAStream();
  cfg.attrs            = attrs;
  cfg.numAttrs         = 2;

  cudaLaunchKernelEx(&cfg, smm_gemm_kernel<P>, a_tmap, b_tmap, c_tmap,
                     sfa_tmap, sfb_tmap, static_cast<uint16_t *>(c),
                     op.m, op.n, k, supergroup, epi_direct);
}

template <Policy... Ps, typename... Args>
bool launch_from_registry(smm_registry::List<Ps...>, const Policy &want,
                          Args... args) {
  return (... || (Ps == want ? (launch_cfg<Ps>(args...), true) : false));
}

using Field = std::pair<const char *, int64_t>;

// The harness names a configuration differently from the hardware: use_2cta
// where the instruction says cta_group, output_n where it says mma_n, and
// cluster_m for the cluster's whole M extent.
std::vector<Field> policy_fields(const Policy &p) {
  return {{"elem", static_cast<int>(p.elem_a)},
          {"sf", static_cast<int>(p.elem_sf)},
          {"use_2cta", p.cta_group == 2},
          {"output_n", p.mma_n},
          {"swap_ab", p.swap_ab},
          {"epi_trade", p.epi_trade},
          {"deep_stages", p.deep},
          {"use_clc", p.use_clc},
          {"cluster_m", p.cta_group * p.rm},
          {"cluster_n", p.rn},
          {"cluster_k", p.rk}};
}

std::string describe(const Policy &p) {
  std::ostringstream text;
  const char *separator = "";
  for (const Field &field : policy_fields(p)) {
    text << separator << field.first << "=" << field.second;
    separator = " ";
  }
  return text.str();
}

template <Policy... Ps>
std::vector<std::string> registry_text(smm_registry::List<Ps...>) {
  std::vector<std::string> lines;
  for (const std::string &line : {describe(Ps)...})
    if (std::find(lines.begin(), lines.end(), line) == lines.end())
      lines.push_back(line);
  return lines;
}

template <Policy P>
std::unordered_map<std::string, int64_t> registry_entry() {
  std::unordered_map<std::string, int64_t> fields;
  for (const Field &field : policy_fields(P))
    fields[field.first] = field.second;
  return fields;
}

template <Policy... Ps>
std::vector<std::unordered_map<std::string, int64_t>>
registry_fields(smm_registry::List<Ps...>) {
  return {registry_entry<Ps>()...};
}

// Bytes of a scale tensor in the 128x4 blocked layout for `rows` rows of K.
int64_t scale_bytes(int64_t rows, int64_t k, int block) {
  return ceil_div(rows, 128) * 128 * ceil_div(k, int64_t{block} * 4) * 4;
}

torch::Tensor run(const torch::Tensor &a, const torch::Tensor &b,
                  const torch::Tensor &sfa, const torch::Tensor &sfb,
                  const Format &f, const Config &cfg,
                  const c10::optional<torch::Tensor> &out) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && sfa.is_cuda() && sfb.is_cuda(),
              "a, b, sfa and sfb must be CUDA tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2-D");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() &&
                  sfa.is_contiguous() && sfb.is_contiguous(),
              "a, b, sfa and sfb must be contiguous");
  TORCH_CHECK(a.size(1) == b.size(1), "a is [M, ", a.size(1), "] but b is [N, ",
              b.size(1), "]");
  const int m = a.size(0);
  const int n = b.size(0);
  const int k = a.size(1) * per_byte_of(f.elem);
  const int block = sf_block_of(f.sf);
  TORCH_CHECK(k % block == 0, "K = ", k, " is not a multiple of the ", block,
              "-wide scale block");
  TORCH_CHECK(sfa.numel() >= scale_bytes(m, k, block) &&
                  sfb.numel() >= scale_bytes(n, k, block),
              "sfa and sfb must hold the 128x4 blocked scale layout: ",
              scale_bytes(m, k, block), " and ", scale_bytes(n, k, block),
              " bytes for this shape");

  torch::Tensor c;
  if (out.has_value()) {
    c = *out;
    TORCH_CHECK(c.scalar_type() == at::kBFloat16 && c.dim() == 2 &&
                    c.size(0) == m && c.size(1) == n && c.is_contiguous(),
                "out must be a contiguous bf16 [", m, ", ", n, "] tensor");
  } else {
    c = torch::empty({m, n}, torch::dtype(torch::kBFloat16).device(a.device()));
  }

  const bool launched = launch_from_registry(
      smm_registry::All{}, cfg.policy, a.data_ptr(), b.data_ptr(),
      sfa.data_ptr(), sfb.data_ptr(), c.data_ptr(), m, n, k, cfg.supergroup,
      cfg.epi_direct, cfg.persistent);
  if (!launched) {
    std::ostringstream text;
    text << "no compiled configuration matches this request:\n  wanted "
         << describe(cfg.policy)
         << "\n\nthis build compiles exactly these configurations "
            "(registry.cuh):\n";
    for (const std::string &line : registry_text(smm_registry::All{}))
      text << "  " << line << "\n";
    TORCH_CHECK(false, text.str());
  }
  return c;
}

// A configuration resolved once, then launched many times. The format is
// left open and read off the tensors it is handed.
struct Launcher {
  Config cfg;
  torch::Tensor call(const torch::Tensor &a, const torch::Tensor &b,
                     const torch::Tensor &sfa, const torch::Tensor &sfb,
                     const c10::optional<torch::Tensor> &out) const {
    const Format &f = format_of(a.scalar_type(), sfa.scalar_type());
    Config bound = cfg;
    bound.policy.elem_a  = f.elem;
    bound.policy.elem_b  = f.elem;
    bound.policy.elem_sf = f.sf;
    return run(a, b, sfa, sfb, f, bound, out);
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  // The tuning surface: what this build compiles, the formats it serves,
  // and a way to run one configuration.
  mod.def("registry", []() { return registry_fields(smm_registry::All{}); },
          "every configuration this build compiles, as data");
  mod.def("formats", []() {
            std::vector<std::tuple<std::string, int64_t, int64_t>> rows;
            for (const Format &f : FORMATS)
              rows.emplace_back(f.name, static_cast<int64_t>(f.elem),
                                static_cast<int64_t>(f.sf));
            return rows;
          },
          "(name, elem, sf) per block-scaled format, elem and sf as the "
          "registry() rows carry them");

  pybind11::class_<Launcher>(mod, "Launcher")
      .def("__call__", &Launcher::call, pybind11::arg("a"), pybind11::arg("b"),
           pybind11::arg("sfa"), pybind11::arg("sfb"),
           pybind11::arg("out") = pybind11::none());

  mod.def("launcher",
          [](int64_t use_2cta, int64_t output_n, bool use_clc,
             int64_t supergroup, bool swap_ab, int64_t epi_direct,
             int64_t epi_trade, int64_t deep_stages, int64_t cluster_m,
             int64_t cluster_n, int64_t cluster_k, int64_t persistent) {
            const int cta_group = use_2cta ? 2 : 1;
            Config cfg{};
            cfg.policy = Policy{SElem::e2m1, SElem::e2m1, SFElem::ue4m3,
                                cta_group, static_cast<int>(output_n),
                                swap_ab, static_cast<int>(epi_trade),
                                deep_stages != 0, use_clc,
                                std::max<int>(static_cast<int>(cluster_m) / cta_group, 1),
                                std::max<int>(static_cast<int>(cluster_n), 1),
                                std::max<int>(static_cast<int>(cluster_k), 1)};
            cfg.supergroup = static_cast<int>(supergroup);
            cfg.epi_direct = static_cast<int>(epi_direct);
            cfg.persistent = persistent != 0;
            return Launcher{cfg};
          },
          "a configuration bound once, so a launch costs five arguments");
}

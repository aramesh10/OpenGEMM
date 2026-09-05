// Host side of the dense kernel: element inference, the launch, and the torch
// extension surface. Which configuration a shape runs is decided in Python
// from configs.json; this file runs the one it is handed.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mm.cuh"
#include "host_utils.cuh"
#include "registry.cuh"

namespace {

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// The host's view of an element: how it arrives from torch, and what to call
// it. Its width and MMA kind are not repeated here; they come from
// types.cuh's ELEM_SPEC, which the kernel derives its geometry from.
struct ElemInfo {
  Elem elem;
  at::ScalarType scalar_type;
  const char *name;

  Kind kind() const { return ELEM_SPEC[static_cast<int>(elem)].kind; }
  // The format's own width in DRAM: 6 for the fp6 pair, 4 for fp4.
  int bits() const { return ELEM_SPEC[static_cast<int>(elem)].value_bits; }
  int bytes() const { return ELEM_SPEC[static_cast<int>(elem)].container_bits / 8; }
  // The K multiple a row must be for TMA to address it: 16 bytes' worth for
  // the byte-aligned types, and a flat 128 values for the sub-byte ones,
  // whose expanding tensor maps require it of globalDim[0].
  int k_align() const { return bits() < 8 ? 128 : 16 * 8 / bits(); }
  // K in values from a row's extent. fp6 and fp4 arrive as uint8 holding
  // several values per byte, and 6 does not divide 8, so this is per row.
  int k_values(int64_t row_extent) const {
    return bits() < 8 ? static_cast<int>(row_extent) * 8 / bits()
                      : static_cast<int>(row_extent);
  }
};

// fp6 and fp4 have no torch dtype, so they arrive as uint8 densely packed;
// uint8 also carries u8, so those four are named by the caller, not inferred.
constexpr ElemInfo ELEMS[] = {
    {Elem::bf16, at::kBFloat16, "bf16"},
    {Elem::f16, at::kHalf, "f16"},
    {Elem::tf32, at::kFloat, "tf32"},
    {Elem::s8, at::kChar, "s8"},
    {Elem::u8, at::kByte, "u8"},
    {Elem::e4m3, at::kFloat8_e4m3fn, "e4m3"},
    {Elem::e5m2, at::kFloat8_e5m2, "e5m2"},
    {Elem::e3m2, at::kByte, "e3m2"},
    {Elem::e2m3, at::kByte, "e2m3"},
    {Elem::e2m1, at::kByte, "e2m1"},
};

const ElemInfo &elem_info(Elem e) { return ELEMS[static_cast<int>(e)]; }

Elem elem_from_index(int64_t index) {
  TORCH_CHECK(index >= 0 && index < static_cast<int64_t>(std::size(ELEMS)),
              "element index ", index, " out of range");
  return ELEMS[index].elem;
}

enum : int { K_PAD_AUTO = -1, K_PAD_COPY = 1 };

using Field = std::pair<const char *, int64_t>;

// The harness names a configuration differently from the hardware: use_2cta
// where the instruction says cta_group, output_n where it says mma_n,
// epi_double where it says epi_mode, and cluster_m for the cluster's whole M
// extent. This is the one place the two vocabularies meet.
std::vector<Field> policy_fields(const Policy &p) {
  return {{"elem_a", static_cast<int>(p.elem_a)},
          {"elem_b", static_cast<int>(p.elem_b)},
          {"use_2cta", p.cta_group == 2},
          {"block_m", p.block_m},
          {"output_n", p.mma_n},
          {"stages", p.stages},
          {"swap_ab", p.swap_ab},
          {"epi_hold", p.epi_hold},
          {"epi_double", p.epi_mode},
          {"epi_direct", p.epi_direct},
          {"use_clc", p.use_clc},
          {"split_k", p.split_k},
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

Config normalized(const Config &requested, int k_tiles) {
  Config c = requested;
  if (c.policy.block_m <= 0) c.policy.block_m = 128;
  if (c.policy.rm <= 0)      c.policy.rm = 1;
  if (c.policy.rn <= 0)      c.policy.rn = 1;
  if (c.policy.rk <= 0)      c.policy.rk = 1;
  c.splits = (c.policy.epi_mode == 1 || requested.splits <= 1)
                 ? 1
                 : std::min(requested.splits, k_tiles);
  // The kernel is specialized on whether it accumulates, so the compiled
  // flag has to follow the count after it is clamped.
  c.policy.split_k = c.splits > 1;
  return c;
}

template <typename Kern>
int max_active_clusters(Kern kern, dim3 cluster, dim3 block, int smem,
                        int grid_clusters, cudaStream_t stream = nullptr) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid_clusters * cluster.x, cluster.y, cluster.z);
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cluster.x;
  attr[0].val.clusterDim.y = cluster.y;
  attr[0].val.clusterDim.z = cluster.z;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int active = 0;
  cudaOccupancyMaxActiveClusters(&active, kern, &cfg);
  return active;
}

// One-time per instantiation: the driver has to be told this kernel wants
// more than the default 48 KB of shared memory, and a cluster of more than 8
// CTAs is non-portable and has to be opted into.
template <Policy P>
void configure_kernel() {
  static const bool done = [] {
    if constexpr (Geom<P>::cluster_ctas > 8)
      cudaFuncSetAttribute(mm_gemm_kernel<P>,
                           cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(mm_gemm_kernel<P>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         Geom<P>::smem_bytes);
    return true;
  }();
  (void)done;
}

// swap_ab is applied on the host: the kernel always computes the same
// untransposed product, so what changes is which pointer, extent and row
// pitch it is handed.
struct Operands {
  void *a, *b;
  int   m, n, a_pitch, b_pitch;
};

template <Policy P>
Operands mma_operands(void *a, void *b, int m, int n, int a_pitch,
                      int b_pitch) {
  if constexpr (Geom<P>::swap_ab)
    return {b, a, n, m, b_pitch, a_pitch};
  else
    return {a, b, m, n, a_pitch, b_pitch};
}

CUtensorMapL2promotion l2_promotion(int l2_promo) {
  return (l2_promo == 2) ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B
       : (l2_promo == 1) ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B
                         : CU_TENSOR_MAP_L2_PROMOTION_NONE;
}

void check_tmap(CUresult status, const char *which) {
  if (status == CUDA_SUCCESS)
    return;
  const char *name = nullptr;
  cuGetErrorName(status, &name);
  TORCH_CHECK(false, "cuTensorMapEncodeTiled failed for ", which, ": ",
              name ? name : "unknown error");
}

template <Policy P>
void launch_cfg(void *a, void *b, void *c, int m, int n, int k, int a_pitch,
                int b_pitch, int supergroup, int walk, int splits,
                int l2_promo) {
  using G = Geom<P>;
  constexpr int SMEM = G::smem_bytes;
  auto kern = mm_gemm_kernel<P>;
  configure_kernel<P>();

  const Operands op = mma_operands<P>(a, b, m, n, a_pitch, b_pitch);
  const int mma_m = op.m;
  const int mma_n = op.n;
  const CUtensorMapL2promotion l2 = l2_promotion(l2_promo);

  CUtensorMap a_tmap, b_tmap, c_tmap;
  check_tmap(init_ab_tmap(&a_tmap, op.a, mma_m, k, G::block_m, G::block_k,
                          G::tmap_a, G::global_bits_a, op.a_pitch, l2), "A");
  check_tmap(init_ab_tmap(&b_tmap, op.b, mma_n, k, G::block_n, G::block_k,
                          G::tmap_b, G::global_bits_b, op.b_pitch, l2), "B");
  if constexpr (G::swap_ab)
    init_c_tmap(&c_tmap, c, mma_m, (mma_n + 7) & ~7, G::block_m, G::c_tma_n,
                G::tmap_c);
  else
    init_c_tmap(&c_tmap, c, mma_n, (mma_m + 7) & ~7, G::c_tma_n, G::block_m,
                G::tmap_c, CU_TENSOR_MAP_SWIZZLE_NONE);

  const int64_t tiles_m = ceil_div(mma_m, G::tile_m);
  const int64_t tiles_n = ceil_div(mma_n, G::mma_n);
  const int logical_clusters = static_cast<int>(
      ceil_div(tiles_m, G::rm) * ceil_div(tiles_n, G::rn) * splits);

  cudaLaunchConfig_t cfg = {};
  cfg.blockDim = dim3(G::threads, 1, 1);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = c10::cuda::getCurrentCUDAStream();

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
  if constexpr (!G::use_clc) {
    static const int resident =
        max_active_clusters(kern, dim3(G::cluster_m, G::rn, G::rk),
                            cfg.blockDim, SMEM, 4096, cfg.stream);
    launch_clusters = std::min(logical_clusters, resident);
  }
  cfg.gridDim = dim3(launch_clusters * G::cluster_m, G::rn, G::rk);

  cudaLaunchKernelEx(&cfg, kern, a_tmap, b_tmap, c_tmap,
                     static_cast<typename G::acc_t *>(c), mma_m, mma_n, k,
                     supergroup, walk, splits);
}

template <Policy... Ps, typename... Args>
bool launch_from_registry(mm_registry::List<Ps...>, const Policy &want,
                          Args... args) {
  return (... || (Ps == want ? (launch_cfg<Ps>(args...), true) : false));
}

template <Policy... Ps>
std::vector<std::string> registry_text(mm_registry::List<Ps...>) {
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
  fields["splits_expected"] = fields["split_k"];
  fields.erase("split_k");
  return fields;
}

template <Policy... Ps>
std::vector<std::unordered_map<std::string, int64_t>>
registry_fields(mm_registry::List<Ps...>) {
  return {registry_entry<Ps>()...};
}

torch::Tensor run(const torch::Tensor &a, const torch::Tensor &b,
                  const Config &requested,
                  const c10::optional<torch::Tensor> &out_opt) {
  const ElemInfo &info_a = elem_info(requested.policy.elem_a);
  const ElemInfo &info_b = elem_info(requested.policy.elem_b);
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2-D");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
              "a and b must be contiguous, K innermost");
  TORCH_CHECK(a.scalar_type() == info_a.scalar_type,
              "a is ", a.scalar_type(), " but ", info_a.name, " arrives as ",
              info_a.scalar_type);
  TORCH_CHECK(b.scalar_type() == info_b.scalar_type,
              "b is ", b.scalar_type(), " but ", info_b.name, " arrives as ",
              info_b.scalar_type);
  // kind::i8 accumulates in s32; every other kind in f32.
  const at::ScalarType OUTPUT =
      (info_a.kind() == Kind::i8) ? at::kInt : at::kFloat;

  const int m = a.size(0);
  const int n = b.size(0);
  // K in values, not extents: a packed operand's extent is bytes.
  const int k = info_a.k_values(a.size(1));
  TORCH_CHECK(info_b.k_values(b.size(1)) == k, "a is [M, K=", k,
              "] but b is [N, K=", info_b.k_values(b.size(1)), "]");

  const int K_ALIGN = std::max(info_a.k_align(), info_b.k_align());
  const bool k_unaligned = (k % K_ALIGN != 0);
  // Padding rescues an unaligned pitch by widening the row and leaving the
  // tail unread: globalDim stays at the true K and TMA zero-fills past it.
  // That cannot rescue fp6 or fp4, whose expanding tensor maps require
  // globalDim[0] itself to be a multiple of 128.
  TORCH_CHECK((info_a.bits() >= 8 && info_b.bits() >= 8) || !k_unaligned,
              "K = ", k, " is not a multiple of ", K_ALIGN, ", which the "
              "expanding tensor map for ", info_a.name, " requires");
  const bool pad_k = (requested.k_pad == K_PAD_AUTO)
                         ? k_unaligned
                         : (requested.k_pad == K_PAD_COPY);
  TORCH_CHECK(pad_k || !k_unaligned, "K = ", k, " is not a multiple of ",
              K_ALIGN, " and the pad_k variant was not requested");
  torch::Tensor a_use = a, b_use = b;
  if (pad_k && k_unaligned) {
    const int k_padded = static_cast<int>(ceil_div(k, K_ALIGN)) * K_ALIGN;
    a_use = torch::empty({m, k_padded}, a.options());
    b_use = torch::empty({n, k_padded}, b.options());
    a_use.narrow(1, 0, k).copy_(a);
    b_use.narrow(1, 0, k).copy_(b);
  }

  const int k_tiles = static_cast<int>(ceil_div(k, 128));
  const Config conf = normalized(requested, k_tiles);

  torch::Tensor out;
  if (out_opt.has_value()) {
    out = *out_opt;
    TORCH_CHECK(out.scalar_type() == OUTPUT && out.dim() == 2 &&
                    out.size(0) == m && out.size(1) == n &&
                    out.stride(0) == 1 && out.stride(1) == m,
                "out must be a ", OUTPUT, " [", m, ", ", n,
                "] tensor with strides (1, ", m, ")");
  } else {
    out = torch::empty_strided({m, n}, {1, m},
                               torch::dtype(OUTPUT).device(a.device()));
  }
  if (conf.splits > 1 || conf.policy.rk > 1)
    out.zero_();

  const bool launched = launch_from_registry(
      mm_registry::All{}, conf.policy, a_use.data_ptr(), b_use.data_ptr(),
      out.data_ptr(), m, n, k,
      // The pitch is a row stride in elements, converted back from a packed
      // operand's byte extent.
      info_a.k_values(a_use.size(1)), info_b.k_values(b_use.size(1)),
      conf.supergroup, conf.walk, conf.splits, conf.l2_promo);
  if (!launched) {
    std::ostringstream text;
    text << "no compiled configuration matches this request:\n  wanted "
         << describe(conf.policy)
         << "\n\nthis build compiles exactly these configurations "
            "(registry.cuh):\n";
    for (const std::string &line : registry_text(mm_registry::All{}))
      text << "  " << line << "\n";
    TORCH_CHECK(false, text.str());
  }
  return out;
}

// A configuration resolved once, then launched many times: the explicit
// path converts nineteen arguments on every call, which on a host-bound
// shape is most of the call.
struct Launcher {
  Config conf;
  torch::Tensor call(const torch::Tensor &a, const torch::Tensor &b,
                     const c10::optional<torch::Tensor> &out) const {
    return run(a, b, conf, out);
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  // The tuning surface: what this build compiles, and a way to run one
  // configuration.
  mod.def("registry", []() { return registry_fields(mm_registry::All{}); },
          "every configuration this build compiles, as data");

  pybind11::class_<Launcher>(mod, "Launcher")
      .def("__call__", &Launcher::call, pybind11::arg("a"),
           pybind11::arg("b"), pybind11::arg("out") = pybind11::none());

  mod.def("launcher",
          [](int64_t use_2cta, int64_t output_n, bool use_clc,
             int64_t supergroup, bool swap_ab, int64_t k_pad,
             int64_t epi_direct, int64_t epi_hold, int64_t cluster_n,
             int64_t stages, int64_t epi_double, int64_t split_k,
             int64_t l2_promo, int64_t block_m, int64_t cluster_m,
             int64_t cluster_k, int64_t walk, int64_t elem_a,
             int64_t elem_b) {
            Config conf{};
            const int group = use_2cta ? 2 : 1;
            conf.policy = Policy{
                elem_from_index(elem_a),
                elem_from_index(elem_b < 0 ? elem_a : elem_b),
                group,
                static_cast<int>(block_m),
                static_cast<int>(output_n),
                static_cast<int>(stages),
                swap_ab,
                static_cast<int>(epi_hold),
                static_cast<int>(epi_double),
                epi_direct != 0,
                use_clc,
                split_k > 1,
                cluster_m > 0 ? static_cast<int>(cluster_m) / group : 1,
                static_cast<int>(cluster_n),
                static_cast<int>(cluster_k)};
            conf.splits     = static_cast<int>(split_k);
            conf.supergroup = static_cast<int>(supergroup);
            conf.walk       = static_cast<int>(walk);
            conf.l2_promo   = static_cast<int>(l2_promo);
            conf.k_pad      = static_cast<int>(k_pad);
            return Launcher{conf};
          },
          "a configuration bound once, so a launch costs three arguments");
}

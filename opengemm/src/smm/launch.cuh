// Host side of the block-scaled kernel: the registry as data, and the launch.
// Nothing here knows about torch or Python, so this header compiles into a
// library any caller can bind. smm.cu wraps it in the surface capi.h declares.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>

#include "capi.h"
#include "host_utils.cuh"
#include "registry.cuh"
#include "smm.cuh"
#include "types.cuh"

namespace opengemm_smm {

// One buffer per thread, read back through og_smm_error(). Only a driver or
// runtime failure fills it; a configuration that names no compiled kernel says
// so through its status, and the caller has the registry to explain it with.
inline thread_local char error_text[256];

inline void set_error(const char *what, const char *detail) {
  snprintf(error_text, sizeof error_text, "%s: %s", what,
           detail ? detail : "unknown error");
}

// The harness names a configuration differently from the hardware: use_2cta
// where the instruction says cta_group, output_n where it says mma_n, and
// cluster_m for the cluster's whole M extent. REGISTRY_FIELDS names the
// columns in the order policy_row writes them.
inline constexpr int REGISTRY_COLS = 11;
inline constexpr const char *REGISTRY_FIELDS =
    "elem,sf,use_2cta,output_n,swap_ab,epi_trade,deep_stages,use_clc,"
    "cluster_m,cluster_n,cluster_k";
inline constexpr const char *ELEM_NAMES =
    "e4m3,e5m2,e3m2,e2m3,e2m1_c8,e2m1";
inline constexpr const char *SF_NAMES = "ue4m3,ue8m0";

constexpr std::array<int32_t, REGISTRY_COLS> policy_row(const Policy &p) {
  return {static_cast<int32_t>(p.elem_a),
          static_cast<int32_t>(p.elem_sf),
          p.cta_group == 2,
          p.mma_n,
          p.swap_ab,
          p.epi_trade,
          p.deep,
          p.use_clc,
          p.cta_group * p.rm,
          p.rn,
          p.rk};
}

// Folded at compile time, so the table costs the build nothing.
template <Policy... Ps>
constexpr auto make_registry(smm_registry::List<Ps...>) {
  std::array<int32_t, sizeof...(Ps) * REGISTRY_COLS> table{};
  int at = 0;
  for (const auto &row : {policy_row(Ps)...})
    for (int32_t value : row) table[at++] = value;
  return table;
}

inline constexpr auto REGISTRY = make_registry(smm_registry::All{});
inline constexpr int REGISTRY_ROWS =
    static_cast<int>(REGISTRY.size()) / REGISTRY_COLS;

// The count of commas plus one, so a field name added without a column (or the
// reverse) is a build error rather than a misread table.
constexpr int name_count(const char *text) {
  int names = 1;
  for (const char *at = text; *at; ++at)
    if (*at == ',') ++names;
  return names;
}
static_assert(name_count(REGISTRY_FIELDS) == REGISTRY_COLS,
              "REGISTRY_FIELDS names a different number of columns than "
              "policy_row writes");

inline Policy policy_of(const OgSmmConfig &c) {
  return Policy{static_cast<SElem>(c.elem_a),
                static_cast<SElem>(c.elem_b),
                static_cast<SFElem>(c.elem_sf),
                c.cta_group,
                c.mma_n,
                c.swap_ab != 0,
                c.epi_trade,
                c.deep != 0,
                c.use_clc != 0,
                c.rm,
                c.rn,
                c.rk};
}

// Once per instantiation per device: both attributes are per device, so a
// process that launches on a second one has to set them again there.
template <Policy P>
void configure_kernel(int device) {
  static std::atomic<uint64_t> configured{0};
  const uint64_t bit = uint64_t{1} << (device & 63);
  if (configured.fetch_or(bit) & bit)
    return;
  cudaFuncSetAttribute(smm_gemm_kernel<P>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       Geom<P>::smem_bytes);
  if constexpr (Geom<P>::cluster_ctas > 8)
    cudaFuncSetAttribute(smm_gemm_kernel<P>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
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
int launch_cfg(void *a, void *b, void *sfa, void *sfb, void *c, int m, int n,
               int k, int supergroup, int epi_direct, int persistent,
               int device, cudaStream_t stream) {
  using G = Geom<P>;
  configure_kernel<P>(device);

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
  init_sf_tmap(&sfb_tmap, op.sfb, op.n, k, G::block_k, sf_n_blocks,
               G::sf_block);
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
  cfg.stream           = stream;
  cfg.attrs            = attrs;
  cfg.numAttrs         = 2;

  const cudaError_t launched = cudaLaunchKernelEx(
      &cfg, smm_gemm_kernel<P>, a_tmap, b_tmap, c_tmap, sfa_tmap, sfb_tmap,
      static_cast<uint16_t *>(c), op.m, op.n, k, supergroup, epi_direct);
  if (launched != cudaSuccess) {
    set_error("cudaLaunchKernelEx", cudaGetErrorString(launched));
    return OG_ERR_LAUNCH;
  }
  return OG_OK;
}

// Returns whether the registry held the policy; `status` carries how the
// launch went for the one that matched.
template <Policy... Ps, typename... Args>
bool launch_from_registry(smm_registry::List<Ps...>, const Policy &want,
                          int &status, Args... args) {
  return (... || (Ps == want ? (status = launch_cfg<Ps>(args...), true)
                             : false));
}

}  // namespace opengemm_smm

// Host side of the dense kernel: the registry as data, and the launch.
// Nothing here knows about torch or Python, so this header compiles into a
// library any caller can bind. mm.cu wraps it in the surface capi.h declares.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>

#include "capi.h"
#include "device_utils.cuh"
#include "host_utils.cuh"
#include "mm.cuh"
#include "registry.cuh"
#include "types.cuh"

namespace opengemm_mm {

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// One buffer per thread, read back through og_mm_error(). Only a driver or
// runtime failure fills it; a configuration that names no compiled kernel says
// so through its status, and the caller has the registry to explain it with.
inline thread_local char error_text[256];

inline void set_error(const char *what, const char *detail) {
  snprintf(error_text, sizeof error_text, "%s: %s", what,
           detail ? detail : "unknown error");
}

// The harness names a configuration differently from the hardware: use_2cta
// where the instruction says cta_group, output_n where it says mma_n,
// epi_double where it says epi_mode, and cluster_m for the cluster's whole M
// extent. This is the one place the two vocabularies meet, and REGISTRY_FIELDS
// names the columns in the order policy_row writes them.
inline constexpr int REGISTRY_COLS = 15;
inline constexpr const char *REGISTRY_FIELDS =
    "elem_a,elem_b,use_2cta,block_m,output_n,stages,swap_ab,epi_hold,"
    "epi_double,epi_direct,use_clc,splits_expected,cluster_m,cluster_n,"
    "cluster_k";
inline constexpr const char *ELEM_NAMES =
    "bf16,f16,tf32,s8,u8,e4m3,e5m2,e3m2,e2m3,e2m1";

constexpr std::array<int32_t, REGISTRY_COLS> policy_row(const Policy &p) {
  return {static_cast<int32_t>(p.elem_a),
          static_cast<int32_t>(p.elem_b),
          p.cta_group == 2,
          p.block_m,
          p.mma_n,
          p.stages,
          p.swap_ab,
          p.epi_hold,
          p.epi_mode,
          p.epi_direct,
          p.use_clc,
          p.split_k,
          p.cta_group * p.rm,
          p.rn,
          p.rk};
}

// Folded at compile time, so the table costs the build nothing.
template <Policy... Ps>
constexpr auto make_registry(mm_registry::List<Ps...>) {
  std::array<int32_t, sizeof...(Ps) * REGISTRY_COLS> table{};
  int at = 0;
  for (const auto &row : {policy_row(Ps)...})
    for (int32_t value : row) table[at++] = value;
  return table;
}

inline constexpr auto REGISTRY = make_registry(mm_registry::All{});
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

inline Policy policy_of(const OgMmConfig &c) {
  return Policy{static_cast<Elem>(c.elem_a),
                static_cast<Elem>(c.elem_b),
                c.cta_group,
                c.block_m,
                c.mma_n,
                c.stages,
                c.swap_ab != 0,
                c.epi_hold,
                c.epi_mode,
                c.epi_direct != 0,
                c.use_clc != 0,
                c.split_k != 0,
                c.rm,
                c.rn,
                c.rk};
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

// Once per instantiation per device: the driver has to be told this kernel
// wants more than the default 48 KB of shared memory, and a cluster of more
// than 8 CTAs is non-portable and has to be opted into. Both attributes are
// per device, so a process that launches on a second one has to set them
// again there; a single flag would configure whichever device happened to be
// current first and leave the rest to fail the launch.
template <Policy P>
void configure_kernel(int device) {
  static std::atomic<uint64_t> configured{0};
  const uint64_t bit = uint64_t{1} << (device & 63);
  if (configured.fetch_or(bit) & bit)
    return;
  if constexpr (Geom<P>::cluster_ctas > 8)
    cudaFuncSetAttribute(mm_gemm_kernel<P>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaFuncSetAttribute(mm_gemm_kernel<P>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       Geom<P>::smem_bytes);
}

// swap_ab is applied on the host: the kernel always computes the same
// untransposed product, so what changes is which pointer, extent and row pitch
// it is handed.
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

inline CUtensorMapL2promotion l2_promotion(int l2_promo) {
  return (l2_promo == 2) ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B
       : (l2_promo == 1) ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B
                         : CU_TENSOR_MAP_L2_PROMOTION_NONE;
}

inline int check_tmap(CUresult status, const char *which) {
  if (status == CUDA_SUCCESS)
    return OG_OK;
  const char *name = nullptr;
  cuGetErrorName(status, &name);
  set_error(which, name);
  return OG_ERR_TENSORMAP;
}

template <Policy P>
int launch_cfg(void *a, void *b, void *c, int m, int n, int k, int a_pitch,
               int b_pitch, int supergroup, int walk, int splits, int l2_promo,
               int device, cudaStream_t stream) {
  using G = Geom<P>;
  constexpr int SMEM = G::smem_bytes;
  auto kern = mm_gemm_kernel<P>;
  configure_kernel<P>(device);

  const Operands op = mma_operands<P>(a, b, m, n, a_pitch, b_pitch);
  const int mma_m = op.m;
  const int mma_n = op.n;
  const CUtensorMapL2promotion l2 = l2_promotion(l2_promo);

  CUtensorMap a_tmap, b_tmap, c_tmap;
  if (const int bad = check_tmap(
          init_ab_tmap(&a_tmap, op.a, mma_m, k, G::block_m, G::block_k,
                       G::tmap_a, G::global_bits_a, op.a_pitch, l2),
          "cuTensorMapEncodeTiled failed for A"))
    return bad;
  if (const int bad = check_tmap(
          init_ab_tmap(&b_tmap, op.b, mma_n, k, G::block_n, G::block_k,
                       G::tmap_b, G::global_bits_b, op.b_pitch, l2),
          "cuTensorMapEncodeTiled failed for B"))
    return bad;
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
  if constexpr (!G::use_clc) {
    static const int resident =
        max_active_clusters(kern, dim3(G::cluster_m, G::rn, G::rk),
                            cfg.blockDim, SMEM, 4096, cfg.stream);
    launch_clusters = std::min(logical_clusters, resident);
  }
  cfg.gridDim = dim3(launch_clusters * G::cluster_m, G::rn, G::rk);

  const cudaError_t launched = cudaLaunchKernelEx(
      &cfg, kern, a_tmap, b_tmap, c_tmap,
      static_cast<typename G::acc_t *>(c), mma_m, mma_n, k, supergroup, walk,
      splits);
  if (launched != cudaSuccess) {
    set_error("cudaLaunchKernelEx", cudaGetErrorString(launched));
    return OG_ERR_LAUNCH;
  }
  return OG_OK;
}

// Returns whether the registry held the policy; `status` carries how the
// launch went for the one that matched.
template <Policy... Ps, typename... Args>
bool launch_from_registry(mm_registry::List<Ps...>, const Policy &want,
                          int &status, Args... args) {
  return (... || (Ps == want ? (status = launch_cfg<Ps>(args...), true)
                             : false));
}

}  // namespace opengemm_mm

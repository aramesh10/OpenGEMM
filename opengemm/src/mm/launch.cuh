#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "capi.h"
#include "common/launch.cuh"
#include "common/registry.cuh"
#include "kernel.cuh"
#include "ptx.cuh"
#include "registry.cuh"
#include "tmap.cuh"
#include "types.cuh"

namespace opengemm_mm {

using opengemm::ceil_div;
using opengemm::error_text;
using opengemm::set_error;

inline constexpr int REGISTRY_COLS = 15;
inline constexpr const char *REGISTRY_FIELDS =
    "elem_a,elem_b,use_2cta,block_m,output_n,stages,swap_ab,epi_hold,"
    "epi_double,epi_direct,use_clc,splits_expected,cluster_m,cluster_n,"
    "cluster_k";
#define OG_MM_ELEM_NAMES "bf16,f16,tf32,s8,u8,e4m3,e5m2,e3m2,e2m3,e2m1"
inline constexpr const char *ENUMS =
    "elem_a=" OG_MM_ELEM_NAMES ";elem_b=" OG_MM_ELEM_NAMES;
#undef OG_MM_ELEM_NAMES

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

inline constexpr auto REGISTRY =
    opengemm::make_table<REGISTRY_COLS>(policy_row, mm_registry::All{});
inline constexpr int REGISTRY_ROWS =
    static_cast<int>(REGISTRY.size()) / REGISTRY_COLS;

static_assert(opengemm::name_count(REGISTRY_FIELDS) == REGISTRY_COLS,
              "REGISTRY_FIELDS names a different number of columns than "
              "policy_row writes");

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
  opengemm::configure_kernel<mm_gemm_kernel<P>, G>(device);

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

using LaunchFn = int (*)(void *, void *, void *, int, int, int, int, int, int,
                         int, int, int, int, cudaStream_t);

template <Policy... Ps>
constexpr auto make_launchers(opengemm::List<Ps...>) {
  return std::array<LaunchFn, sizeof...(Ps)>{&launch_cfg<Ps>...};
}

inline constexpr auto LAUNCHERS = make_launchers(mm_registry::All{});
static_assert(static_cast<int>(LAUNCHERS.size()) == REGISTRY_ROWS,
              "the launcher table and the registry disagree on how many "
              "kernels this build compiles");

}

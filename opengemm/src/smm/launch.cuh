// Host side of the block-scaled kernel: the registry as data, and the launch.
// Nothing here knows about torch or Python; capi.cu wraps it in the surface
// capi.h declares.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "capi.h"
#include "common/launch.cuh"
#include "common/registry.cuh"
#include "kernel.cuh"
#include "registry.cuh"
#include "tmap.cuh"
#include "types.cuh"

namespace opengemm_smm {

using opengemm::ceil_div;
using opengemm::error_text;
using opengemm::set_error;

// The registry reads back in the harness vocabulary, not the hardware's:
// use_2cta for cta_group, output_n for mma_n, cluster_m for the cluster's
// whole M extent. REGISTRY_FIELDS names the columns policy_row writes.
inline constexpr int REGISTRY_COLS = 11;
inline constexpr const char *REGISTRY_FIELDS =
    "elem,sf,use_2cta,output_n,swap_ab,epi_trade,deep_stages,use_clc,"
    "cluster_m,cluster_n,cluster_k";
// Every enumerated registry column and its spellings, so a caller decodes a
// row without a table of its own.
inline constexpr const char *ENUMS =
    "elem=e4m3,e5m2,e3m2,e2m3,e2m1_c8,e2m1;sf=ue4m3,ue8m0";

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

inline constexpr auto REGISTRY =
    opengemm::make_table<REGISTRY_COLS>(policy_row, smm_registry::All{});
inline constexpr int REGISTRY_ROWS =
    static_cast<int>(REGISTRY.size()) / REGISTRY_COLS;

static_assert(opengemm::name_count(REGISTRY_FIELDS) == REGISTRY_COLS,
              "REGISTRY_FIELDS names a different number of columns than "
              "policy_row writes");

inline int sm_count() {
  static const int count = [] {
    int device = 0, n = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device);
    return n;
  }();
  return count;
}

// Clusters resident at once. One CTA fills an SM - the kernel takes 159-222 KB
// of the SM's 228 KB and all 512 TMEM columns - so this is arithmetic rather
// than an occupancy query. Checked against cudaOccupancyMaxActiveClusters for
// every cluster size the registry builds (1 and 2 CTAs) and for the smallest
// and largest shared-memory footprints: identical in every case. A cluster of
// more than 2 CTAs would need that checked again.
inline int wave_clusters(int cluster_ctas) {
  return sm_count() / cluster_ctas;
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
  opengemm::configure_kernel<smm_gemm_kernel<P>, G>(device);

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

// One launcher per registry row, in the order make_table wrote them, so a row
// read out of REGISTRY is the kernel it selects here.
using LaunchFn = int (*)(void *, void *, void *, void *, void *, int, int,
                         int, int, int, int, int, cudaStream_t);

template <Policy... Ps>
constexpr auto make_launchers(opengemm::List<Ps...>) {
  return std::array<LaunchFn, sizeof...(Ps)>{&launch_cfg<Ps>...};
}

inline constexpr auto LAUNCHERS = make_launchers(smm_registry::All{});
static_assert(static_cast<int>(LAUNCHERS.size()) == REGISTRY_ROWS,
              "the launcher table and the registry disagree on how many "
              "kernels this build compiles");

}  // namespace opengemm_smm

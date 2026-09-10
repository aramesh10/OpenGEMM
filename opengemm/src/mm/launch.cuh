#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>

#include "capi.h"
#include "kernel.cuh"
#include "ptx.cuh"
#include "registry.cuh"
#include "tmap.cuh"
#include "types.cuh"

#include "common/launch.cuh"

namespace opengemm_mm {

using opengemm::ceil_div;
using opengemm::check_tmap;
using opengemm::error_text;
using opengemm::fail;
using mm_registry::Named;
using mm_registry::POLICIES;

inline constexpr int KERNELS = static_cast<int>(POLICIES.size());

// The configuration an OgGemm asks for, as error text.
inline void spell(char *out, size_t cap, const OgGemm &g) {
  snprintf(out, cap,
           "use_2cta=%d output_n=%d use_clc=%d swap_ab=%d cluster_m=%d "
           "cluster_n=%d cluster_k=%d block_m=%d stages=%d epi_hold=%d "
           "epi_double=%d epi_direct=%d split_k=%d",
           g.use_2cta, g.output_n, g.use_clc, g.swap_ab, g.cluster_m,
           g.cluster_n, g.cluster_k, g.block_m, g.stages, g.epi_hold,
           g.epi_double, g.epi_direct, g.split_k);
}

// Resolve an OgGemm to the compiled kernel it names: check the dtype and K,
// clamp the split count, and find the Policy in the registry.
inline int bind(OgGemm *g) {
  const Named *d = mm_registry::named(g->dtype);
  if (!d)
    return fail(OG_ERR_BIND, "dtype %.16s names no element pair this build compiles", g->dtype);
  if (g->m <= 0 || g->n <= 0 || g->k <= 0)
    return fail(OG_ERR_BIND, "m, n and k must be positive, got %d, %d, %d", g->m, g->n, g->k);
  // TMA addresses 16 bytes of a row at a time, and the expanding maps for
  // the sub-byte elements need a flat 128 values.
  const int bits = ELEM_SPEC[static_cast<int>(d->a)].value_bits;
  const int multiple = bits < 8 ? 128 : 128 / bits;
  if (g->k % multiple)
    return fail(OG_ERR_BIND, "K = %d is not a multiple of %d, which the tensor map for %s requires",
                g->k, multiple, d->name);
  g->split_k = mm_registry::splits_for(*g);
  const Policy want = mm_registry::policy_of(*g, *d);
  for (int i = 0; i < KERNELS; ++i)
    if (POLICIES[i] == want) {
      g->row = i;
      return OG_OK;
    }
  char spelled[200];
  spell(spelled, sizeof spelled, *g);
  return fail(OG_ERR_NO_KERNEL, "no compiled %s kernel matches %s", d->name, spelled);
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

struct Operands {
  void *a, *b;
  int   m, n;
};

template <Policy P>
Operands mma_operands(void *a, void *b, int m, int n) {
  if constexpr (Geom<P>::swap_ab)
    return {b, a, n, m};
  else
    return {a, b, m, n};
}

inline CUtensorMapL2promotion l2_promotion(int l2_promo) {
  return (l2_promo == 2) ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B
       : (l2_promo == 1) ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B
                         : CU_TENSOR_MAP_L2_PROMOTION_NONE;
}

template <Policy P>
int launch_cfg(const OgGemm *g, void *a, void *b, void *, void *, void *c,
               int device, cudaStream_t stream) {
  using G = Geom<P>;
  auto kern = mm_gemm_kernel<P>;
  opengemm::configure_kernel<mm_gemm_kernel<P>, G>(device);

  const Operands op = mma_operands<P>(a, b, g->m, g->n);
  const int k = g->k;
  const CUtensorMapL2promotion l2 = l2_promotion(g->l2_promo);

  CUtensorMap a_tmap, b_tmap, c_tmap;
  if (const int bad = check_tmap(
          init_ab_tmap(&a_tmap, op.a, op.m, k, G::block_m, G::block_k,
                       G::tmap_a, G::global_bits_a, 0, l2), "A"))
    return bad;
  if (const int bad = check_tmap(
          init_ab_tmap(&b_tmap, op.b, op.n, k, G::block_n, G::block_k,
                       G::tmap_b, G::global_bits_b, 0, l2), "B"))
    return bad;
  // C is row-major [m, n]; swap_ab stages it transposed, so its box is too.
  constexpr int c_tile_rows = G::swap_ab ? G::c_tma_n : G::block_m;
  constexpr int c_tile_cols = G::swap_ab   ? G::block_m
                            : G::vec_stage ? G::store_n
                                           : G::c_tma_n;
  constexpr CUtensorMapSwizzle c_swizzle = G::vec_stage
                                         ? CU_TENSOR_MAP_SWIZZLE_128B
                                         : CU_TENSOR_MAP_SWIZZLE_NONE;
  if (const int bad = check_tmap(
          init_c_tmap(&c_tmap, c, g->m, (g->n + 3) & ~3, c_tile_rows,
                      c_tile_cols, G::tmap_c, c_swizzle), "C"))
    return bad;

  const int splits = g->split_k;
  const int64_t tiles_m = ceil_div(op.m, G::tile_m);
  const int64_t tiles_n = ceil_div(op.n, G::mma_n);
  const int logical_clusters = static_cast<int>(
      ceil_div(tiles_m, G::rm) * ceil_div(tiles_n, G::rn) * splits);

  cudaLaunchConfig_t cfg = {};
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
  if constexpr (!G::use_clc) {
    // Measured once per process: every device is taken to be the same GPU.
    static const int resident =
        max_active_clusters(kern, dim3(G::cluster_m, G::rn, G::rk),
                            cfg.blockDim, G::smem_bytes, 4096, cfg.stream);
    launch_clusters = std::min(logical_clusters, resident);
  }
  cfg.gridDim = dim3(launch_clusters * G::cluster_m, G::rn, G::rk);

  if constexpr (G::acc_store) {   // split-K and cluster-K add into C
    const cudaError_t zeroed = cudaMemsetAsync(
        c, 0, sizeof(typename G::acc_t) * static_cast<size_t>(g->m) * g->n, stream);
    if (zeroed != cudaSuccess)
      return fail(OG_ERR_LAUNCH, "cudaMemsetAsync: %s", cudaGetErrorString(zeroed));
  }

  const cudaError_t launched = cudaLaunchKernelEx(
      &cfg, kern, a_tmap, b_tmap, c_tmap,
      static_cast<typename G::acc_t *>(c), op.m, op.n, k, g->supergroup,
      g->walk, splits);
  if (launched != cudaSuccess)
    return fail(OG_ERR_LAUNCH, "cudaLaunchKernelEx: %s", cudaGetErrorString(launched));
  return OG_OK;
}

using LaunchFn = int (*)(const OgGemm *, void *, void *, void *, void *, void *,
                         int, cudaStream_t);

template <Policy... Ps>
constexpr auto make_launchers(opengemm::List<Ps...>) {
  return std::array<LaunchFn, sizeof...(Ps)>{&launch_cfg<Ps>...};
}

inline constexpr auto LAUNCHERS = make_launchers(mm_registry::All{});
static_assert(LAUNCHERS.size() == POLICIES.size());

}

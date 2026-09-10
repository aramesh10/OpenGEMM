#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>

#include "capi.h"
#include "kernel.cuh"
#include "registry.cuh"
#include "tmap.cuh"
#include "types.cuh"

#include "common/launch.cuh"

namespace opengemm_smm {

using opengemm::ceil_div;
using opengemm::check_tmap;
using opengemm::error_text;
using opengemm::fail;
using smm_registry::Named;
using smm_registry::POLICIES;

inline constexpr int KERNELS = static_cast<int>(POLICIES.size());

// The configuration an OgGemm asks for, as error text.
inline void spell(char *out, size_t cap, const OgGemm &g) {
  snprintf(out, cap,
           "use_2cta=%d output_n=%d use_clc=%d swap_ab=%d cluster_m=%d "
           "cluster_n=%d cluster_k=%d epi_trade=%d deep_stages=%d",
           g.use_2cta, g.output_n, g.use_clc, g.swap_ab, g.cluster_m,
           g.cluster_n, g.cluster_k, g.epi_trade, g.deep_stages);
}

// Resolve an OgGemm to the compiled kernel it names: check the dtype and K,
// and find the Policy in the registry.
inline int bind(OgGemm *g) {
  const Named *d = smm_registry::named(g->dtype);
  if (!d)
    return fail(OG_ERR_BIND, "dtype %.16s names no block-scaled format this build compiles", g->dtype);
  if (g->m <= 0 || g->n <= 0 || g->k <= 0)
    return fail(OG_ERR_BIND, "m, n and k must be positive, got %d, %d, %d", g->m, g->n, g->k);
  const int block = sf_block_of(d->sf);
  if (g->k % block)
    return fail(OG_ERR_BIND, "K = %d is not a multiple of the %d-wide scale block of %s",
                g->k, block, d->name);
  const Policy want = smm_registry::policy_of(*g, *d);
  for (int i = 0; i < KERNELS; ++i)
    if (POLICIES[i] == want) {
      g->row = i;
      return OG_OK;
    }
  char spelled[160];
  spell(spelled, sizeof spelled, *g);
  return fail(OG_ERR_NO_KERNEL, "no compiled %s kernel matches %s", d->name, spelled);
}

inline int sm_count() {
  // Read once per process: every device is taken to be the same GPU.
  static const int count = [] {
    int device = 0, n = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device);
    return n;
  }();
  return count;
}

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
int launch_cfg(const OgGemm *g, void *a, void *b, void *sfa, void *sfb, void *c,
               int device, cudaStream_t stream) {
  using G = Geom<P>;
  opengemm::configure_kernel<smm_gemm_kernel<P>, G>(device);

  const Operands op = mma_operands<P>(a, b, sfa, sfb, g->m, g->n);
  const int k = g->k;
  const int sf_n_blocks = (G::cta_group == 1) ? G::sf_n_blocks : 1;
  const int c_cols      = (g->n + 7) & ~7;
  const int c_tile_rows = G::swap_ab ? G::c_tma_n : G::block_m;
  const int c_tile_cols = G::swap_ab ? G::block_m
                        : G::vec_stage ? G::store_n
                                       : G::c_tma_n;
  const CUtensorMapSwizzle c_swizzle = G::vec_stage
                                     ? CU_TENSOR_MAP_SWIZZLE_128B
                                     : CU_TENSOR_MAP_SWIZZLE_NONE;

  CUtensorMap a_tmap, b_tmap, c_tmap, sfa_tmap, sfb_tmap;
  if (const int bad = check_tmap(
          init_ab_tmap(&a_tmap, op.a, op.m, k, G::block_m, G::block_k, G::elem_bits), "A"))
    return bad;
  if (const int bad = check_tmap(
          init_ab_tmap(&b_tmap, op.b, op.n, k, G::block_n, G::block_k, G::elem_bits), "B"))
    return bad;
  if (const int bad = check_tmap(
          init_sf_tmap(&sfa_tmap, op.sfa, op.m, k, G::block_k, 1, G::sf_block), "SFA"))
    return bad;
  if (const int bad = check_tmap(
          init_sf_tmap(&sfb_tmap, op.sfb, op.n, k, G::block_k, sf_n_blocks, G::sf_block), "SFB"))
    return bad;
  if (const int bad = check_tmap(
          init_c_tmap(&c_tmap, c, g->m, c_cols, c_tile_rows, c_tile_cols, c_swizzle), "C"))
    return bad;

  const int64_t tiles = ceil_div(ceil_div(op.m, G::mma_m), G::mc_m)
                      * ceil_div(ceil_div(op.n, G::mma_n), G::mc_n);
  int clusters = static_cast<int>(tiles);
  if (g->persistent && !G::use_clc)
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
      static_cast<uint16_t *>(c), op.m, op.n, k, g->supergroup, g->epi_direct);
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

inline constexpr auto LAUNCHERS = make_launchers(smm_registry::All{});
static_assert(LAUNCHERS.size() == POLICIES.size());

}

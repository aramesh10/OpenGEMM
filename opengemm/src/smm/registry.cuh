#pragma once

#include <algorithm>
#include <cstring>

#include "common/abi.h"
#include "common/registry.cuh"
#include "types.cuh"

namespace smm_registry {

using opengemm::List;

// Every block-scaled format this library compiles, by its configs.json name.
#define OPENGEMM_FORMATS(X)                                                   \
  X(nvfp4, e2m1, ue4m3) X(mxfp8, e4m3, ue8m0) X(mxfp4, e2m1, ue8m0)

#define OPENGEMM_FACTORY(NAME, ELEM, SF)                                      \
  constexpr Policy NAME(int cta_group, int mma_n = 256, bool swap_ab = false, \
                        int epi_trade = 0, bool deep = false,                 \
                        bool use_clc = false, int rm = 1, int rn = 1,         \
                        int rk = 1) {                                         \
    return Policy{SElem::ELEM, SElem::ELEM, SFElem::SF, cta_group, mma_n,     \
                  swap_ab, epi_trade, deep, use_clc, rm, rn, rk};             \
  }
OPENGEMM_FORMATS(OPENGEMM_FACTORY)
#undef OPENGEMM_FACTORY

struct Named {
  const char *name;
  SElem elem;
  SFElem sf;
};
#define OPENGEMM_NAMED(NAME, ELEM, SF) {#NAME, SElem::ELEM, SFElem::SF},
inline constexpr Named NAMES[] = {OPENGEMM_FORMATS(OPENGEMM_NAMED)};
#undef OPENGEMM_NAMED
#undef OPENGEMM_FORMATS

using All = List<
    nvfp4(1, 8), nvfp4(1, 8, false, 0, true), nvfp4(1, 8, true),
    nvfp4(1, 8, true, 0, true), nvfp4(1, 16), nvfp4(1, 16, false, 0, true),
    nvfp4(1, 16, true), nvfp4(1, 16, true, 0, true), nvfp4(1, 32),
    nvfp4(1, 32, false, 0, true), nvfp4(1, 32, true),
    nvfp4(1, 32, true, 0, true), nvfp4(1, 64), nvfp4(1, 64, false, 0, true),
    nvfp4(1, 64, true), nvfp4(1, 64, true, 0, true), nvfp4(1, 128),
    nvfp4(1, 128, false, 0, true), nvfp4(1, 256), nvfp4(1, 256, false, 1),
    nvfp4(2, 8), nvfp4(2, 8, false, 0, false, true),
    nvfp4(2, 8, false, 0, true), nvfp4(2, 16),
    nvfp4(2, 16, false, 0, false, true), nvfp4(2, 16, false, 0, true),
    nvfp4(2, 32), nvfp4(2, 32, false, 0, false, true),
    nvfp4(2, 32, false, 0, true), nvfp4(2, 64),
    nvfp4(2, 64, false, 0, false, true), nvfp4(2, 64, false, 0, true),
    nvfp4(2, 128), nvfp4(2, 128, false, 0, false, true),
    nvfp4(2, 128, false, 0, true), nvfp4(2, 256),
    nvfp4(2, 256, false, 0, false, true), nvfp4(2, 256, false, 1),
    nvfp4(2, 256, false, 2),
    mxfp8(1, 8), mxfp8(1, 8, true), mxfp8(1, 16), mxfp8(1, 16, true),
    mxfp8(1, 32), mxfp8(1, 32, true), mxfp8(1, 64), mxfp8(1, 64, true),
    mxfp8(1, 128), mxfp8(1, 256), mxfp8(2, 128), mxfp8(2, 256),
    mxfp8(2, 256, false, 0, false, true),
    mxfp4(1, 8), mxfp4(1, 8, true), mxfp4(1, 16), mxfp4(1, 16, true),
    mxfp4(1, 32), mxfp4(1, 32, true), mxfp4(1, 64), mxfp4(1, 64, true),
    mxfp4(1, 128), mxfp4(1, 256), mxfp4(2, 128), mxfp4(2, 256),
    mxfp4(2, 256, false, 0, false, true)>;

inline constexpr auto POLICIES = opengemm::to_array(All{});

inline const Named *named(const char (&dtype)[16]) {
  for (const Named &d : NAMES)
    if (std::strncmp(d.name, dtype, sizeof dtype) == 0)
      return &d;
  return nullptr;
}

inline const Named *named(SElem elem, SFElem sf) {
  for (const Named &d : NAMES)
    if (d.elem == elem && d.sf == sf)
      return &d;
  return nullptr;
}

// A configs.json entry in the kernel's vocabulary.
inline Policy policy_of(const OgGemm &g, const Named &d) {
  const int group = g.use_2cta ? 2 : 1;
  return Policy{d.elem,
                d.elem,
                d.sf,
                group,
                g.output_n,
                g.swap_ab != 0,
                g.epi_trade,
                g.deep_stages != 0,
                g.use_clc != 0,
                std::max<int>(g.cluster_m / group, 1),
                std::max<int>(g.cluster_n, 1),
                std::max<int>(g.cluster_k, 1)};
}

// The inverse of policy_of: the configs.json entry that binds to a compiled
// kernel; the launch knobs are the caller's.
inline void describe(const Policy &p, OgGemm *g) {
  *g = OgGemm{};
  if (const Named *d = named(p.elem_a, p.elem_sf))
    std::strncpy(g->dtype, d->name, sizeof g->dtype - 1);
  g->use_2cta    = p.cta_group == 2;
  g->output_n    = p.mma_n;
  g->use_clc     = p.use_clc;
  g->swap_ab     = p.swap_ab;
  g->cluster_m   = p.cta_group * p.rm;
  g->cluster_n   = p.rn;
  g->cluster_k   = p.rk;
  g->epi_trade   = p.epi_trade;
  g->deep_stages = p.deep;
}

}

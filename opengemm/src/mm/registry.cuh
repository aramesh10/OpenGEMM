#pragma once

#include <algorithm>
#include <cstring>

#include "common/abi.h"
#include "common/registry.cuh"
#include "types.cuh"

namespace mm_registry {

using opengemm::List;

// Every element pair this library compiles, by its configs.json name.
#define OPENGEMM_PAIRS(X)                                                    \
  X(bf16, bf16, bf16) X(f16, f16, f16) X(tf32, tf32, tf32) X(s8, s8, s8)     \
  X(u8, u8, u8) X(e4m3, e4m3, e4m3) X(e5m2, e5m2, e5m2) X(e3m2, e3m2, e3m2) \
  X(e2m3, e2m3, e2m3) X(e2m1, e2m1, e2m1) X(e4m3xe5m2, e4m3, e5m2)

#define OPENGEMM_FACTORY(NAME, A, B)                                          \
  constexpr Policy NAME(int cta_group, int block_m = 128, int mma_n = 256,    \
                        int stages = 0, bool swap_ab = false,                 \
                        int epi_hold = 1, int epi_mode = 0,                   \
                        bool epi_direct = false, bool use_clc = false,        \
                        bool split_k = false, int rm = 1, int rn = 1,         \
                        int rk = 1) {                                         \
    return Policy{Elem::A, Elem::B, cta_group, block_m, mma_n, stages,        \
                  swap_ab, epi_hold, epi_mode, epi_direct, use_clc, split_k,  \
                  rm, rn, rk};                                                \
  }
OPENGEMM_PAIRS(OPENGEMM_FACTORY)
#undef OPENGEMM_FACTORY

struct Named {
  const char *name;
  Elem a, b;
};
#define OPENGEMM_NAMED(NAME, A, B) {#NAME, Elem::A, Elem::B},
inline constexpr Named NAMES[] = {OPENGEMM_PAIRS(OPENGEMM_NAMED)};
#undef OPENGEMM_NAMED
#undef OPENGEMM_PAIRS

using All = List<
    bf16(1, 128, 8, 0, true), bf16(1, 128, 16, 0, true),
    bf16(1, 128, 24, 0, true), bf16(1, 128, 32, 0, true), bf16(1, 128, 128),
    bf16(1, 128, 128, 0, false, 1, 0, true), bf16(1, 128, 256),
    bf16(2, 128, 64), bf16(2, 128, 64, 0, false, 1, 0, false, true),
    bf16(2, 128, 64, 0, false, 1, 0, true), bf16(2, 128, 128),
    bf16(2, 128, 128, 0, false, 1, 0, false, false, true),
    bf16(2, 128, 128, 0, false, 1, 0, true), bf16(2, 128, 192),
    bf16(2, 128, 256), bf16(2, 128, 256, 0, false, 1, 0, false, false, true),
    bf16(2, 128, 256, 0, false, 1, 0, false, true), bf16(2, 128, 256, 8),
    bf16(2, 128, 256, 8, false, 1, 0, false, false, true),
    f16(1, 128, 8, 0, true), f16(1, 128, 16, 0, true),
    f16(1, 128, 24, 0, true), f16(1, 128, 32, 0, true), f16(1, 128, 128),
    f16(1, 128, 256), f16(2, 128, 128),
    f16(2, 128, 128, 0, false, 1, 0, false, false, true),
    f16(2, 128, 128, 0, false, 1, 1), f16(2, 128, 128, 9, false, 2, 1),
    f16(2, 128, 256), f16(2, 128, 256, 0, false, 1, 0, false, false, true),
    f16(2, 128, 256, 0, false, 1, 0, false, true),
    f16(2, 128, 256, 0, false, 1, 1), f16(2, 128, 256, 8),
    f16(2, 128, 256, 8, false, 1, 0, false, false, true),
    f16(2, 128, 256, 9, false, 1, 1),
    tf32(1, 128, 8, 0, true), tf32(1, 128, 16, 0, true),
    tf32(1, 128, 24, 0, true), tf32(1, 128, 32, 0, true), tf32(1, 128, 128),
    tf32(1, 128, 256), tf32(2, 128, 128),
    tf32(2, 128, 128, 0, false, 1, 0, false, false, true), tf32(2, 128, 256),
    tf32(2, 128, 256, 0, false, 1, 0, false, false, true),
    tf32(2, 128, 256, 0, false, 1, 0, false, true), tf32(2, 128, 256, 8),
    tf32(2, 128, 256, 8, false, 1, 0, false, false, true),
    s8(1, 128, 128), s8(1, 128, 256), s8(2, 128, 64, 9, true), s8(2, 128, 128),
    s8(2, 128, 128, 0, false, 1, 0, false, false, true), s8(2, 128, 256),
    s8(2, 128, 256, 0, false, 1, 0, false, false, true),
    s8(2, 128, 256, 0, false, 1, 0, false, true), s8(2, 128, 256, 8),
    s8(2, 128, 256, 8, false, 1, 0, false, false, true),
    u8(1, 128, 128), u8(1, 128, 256), u8(2, 128, 64, 9, true), u8(2, 128, 128),
    u8(2, 128, 128, 0, false, 1, 0, false, false, true), u8(2, 128, 256),
    u8(2, 128, 256, 0, false, 1, 0, false, false, true),
    u8(2, 128, 256, 0, false, 1, 0, false, true), u8(2, 128, 256, 8),
    u8(2, 128, 256, 8, false, 1, 0, false, false, true),
    e4m3(1, 64, 8, 22), e4m3(1, 64, 8, 22, true), e4m3(1, 128, 128),
    e4m3(1, 128, 256), e4m3(2, 128, 64, 9, true), e4m3(2, 128, 128),
    e4m3(2, 128, 128, 0, false, 1, 0, false, false, true),
    e4m3(2, 128, 128, 0, false, 1, 0, false, true),
    e4m3(2, 128, 128, 9, false, 2, 1), e4m3(2, 128, 256),
    e4m3(2, 128, 256, 0, false, 1, 0, false, false, true),
    e4m3(2, 128, 256, 0, false, 1, 0, false, true),
    e4m3(2, 128, 256, 0, false, 1, 1), e4m3(2, 128, 256, 8),
    e4m3(2, 128, 256, 8, false, 1, 0, false, false, true),
    e4m3(2, 128, 256, 8, false, 1, 0, false, true),
    e4m3(2, 128, 256, 9, false, 1, 1), e4m3(2, 128, 256, 10, false, 1, 1),
    e4m3(2, 128, 256, 11, false, 1, 1),
    e5m2(1, 128, 128), e5m2(1, 128, 256), e5m2(2, 128, 64, 9, true),
    e5m2(2, 128, 128), e5m2(2, 128, 128, 0, false, 1, 0, false, false, true),
    e5m2(2, 128, 256), e5m2(2, 128, 256, 0, false, 1, 0, false, false, true),
    e5m2(2, 128, 256, 0, false, 1, 0, false, true), e5m2(2, 128, 256, 8),
    e5m2(2, 128, 256, 8, false, 1, 0, false, false, true),
    e3m2(1, 128, 128), e3m2(1, 128, 256), e3m2(2, 128, 64, 9, true),
    e3m2(2, 128, 128), e3m2(2, 128, 128, 0, false, 1, 0, false, false, true),
    e3m2(2, 128, 256), e3m2(2, 128, 256, 0, false, 1, 0, false, false, true),
    e3m2(2, 128, 256, 0, false, 1, 0, false, true), e3m2(2, 128, 256, 8),
    e3m2(2, 128, 256, 8, false, 1, 0, false, false, true),
    e2m3(1, 128, 128), e2m3(1, 128, 256), e2m3(2, 128, 64, 9, true),
    e2m3(2, 128, 128), e2m3(2, 128, 128, 0, false, 1, 0, false, false, true),
    e2m3(2, 128, 256), e2m3(2, 128, 256, 0, false, 1, 0, false, false, true),
    e2m3(2, 128, 256, 0, false, 1, 0, false, true), e2m3(2, 128, 256, 8),
    e2m3(2, 128, 256, 8, false, 1, 0, false, false, true),
    e2m1(1, 128, 128), e2m1(1, 128, 256), e2m1(2, 128, 64, 9, true),
    e2m1(2, 128, 128), e2m1(2, 128, 128, 0, false, 1, 0, false, false, true),
    e2m1(2, 128, 256), e2m1(2, 128, 256, 0, false, 1, 0, false, false, true),
    e2m1(2, 128, 256, 0, false, 1, 0, false, true), e2m1(2, 128, 256, 8),
    e2m1(2, 128, 256, 8, false, 1, 0, false, false, true),
    e4m3xe5m2(1, 128, 128), e4m3xe5m2(1, 128, 256),
    e4m3xe5m2(2, 128, 64, 9, true), e4m3xe5m2(2, 128, 128),
    e4m3xe5m2(2, 128, 128, 0, false, 1, 0, false, false, true),
    e4m3xe5m2(2, 128, 256),
    e4m3xe5m2(2, 128, 256, 0, false, 1, 0, false, true),
    e4m3xe5m2(2, 128, 256, 8),
    e4m3xe5m2(2, 128, 256, 8, false, 1, 0, false, false, true)>;

inline constexpr auto POLICIES = opengemm::to_array(All{});

inline const Named *named(const char (&dtype)[16]) {
  for (const Named &d : NAMES)
    if (std::strncmp(d.name, dtype, sizeof dtype) == 0)
      return &d;
  return nullptr;
}

inline const Named *named(Elem a, Elem b) {
  for (const Named &d : NAMES)
    if (d.a == a && d.b == b)
      return &d;
  return nullptr;
}

// The split count a request runs: none when the epilogue is doubled, and
// never more than the K tiles. The kernel is specialized on whether it
// accumulates, so the compiled flag follows this count, not the request.
inline int splits_for(const OgGemm &g) {
  if (g.epi_double == 1 || g.split_k <= 1)
    return 1;
  return std::min<int>(g.split_k, (g.k + 127) / 128);
}

// A configs.json entry in the kernel's vocabulary.
inline Policy policy_of(const OgGemm &g, const Named &d) {
  const int group = g.use_2cta ? 2 : 1;
  return Policy{d.a,
                d.b,
                group,
                g.block_m,
                g.output_n,
                g.stages,
                g.swap_ab != 0,
                g.epi_hold,
                g.epi_double,
                g.epi_direct != 0,
                g.use_clc != 0,
                splits_for(g) > 1,
                std::max<int>(g.cluster_m / group, 1),
                std::max<int>(g.cluster_n, 1),
                std::max<int>(g.cluster_k, 1)};
}

// The inverse of policy_of: the configs.json entry that binds to a compiled
// kernel. split_k reads 2 for a kernel that accumulates, the smallest count
// that selects it, and 1 otherwise; the other launch knobs are the caller's.
inline void describe(const Policy &p, OgGemm *g) {
  *g = OgGemm{};
  if (const Named *d = named(p.elem_a, p.elem_b))
    std::strncpy(g->dtype, d->name, sizeof g->dtype - 1);
  g->use_2cta   = p.cta_group == 2;
  g->output_n   = p.mma_n;
  g->use_clc    = p.use_clc;
  g->swap_ab    = p.swap_ab;
  g->cluster_m  = p.cta_group * p.rm;
  g->cluster_n  = p.rn;
  g->cluster_k  = p.rk;
  g->block_m    = p.block_m;
  g->stages     = p.stages;
  g->epi_hold   = p.epi_hold;
  g->epi_double = p.epi_mode;
  g->epi_direct = p.epi_direct;
  g->split_k    = p.split_k ? 2 : 1;
}

}

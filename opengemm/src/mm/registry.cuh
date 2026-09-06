#pragma once

#include "common/registry.cuh"
#include "types.cuh"

namespace mm_registry {

using opengemm::List;

#define OPENGEMM_PAIR(NAME, A, B)                                             \
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
OPENGEMM_PAIR(bf16, bf16, bf16)
OPENGEMM_PAIR(f16, f16, f16)
OPENGEMM_PAIR(tf32, tf32, tf32)
OPENGEMM_PAIR(s8, s8, s8)
OPENGEMM_PAIR(u8, u8, u8)
OPENGEMM_PAIR(e4m3, e4m3, e4m3)
OPENGEMM_PAIR(e5m2, e5m2, e5m2)
OPENGEMM_PAIR(e3m2, e3m2, e3m2)
OPENGEMM_PAIR(e2m3, e2m3, e2m3)
OPENGEMM_PAIR(e2m1, e2m1, e2m1)
OPENGEMM_PAIR(e4m3xe5m2, e4m3, e5m2)
#undef OPENGEMM_PAIR

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

}

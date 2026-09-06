#pragma once

#include "common/registry.cuh"
#include "types.cuh"

namespace smm_registry {

using opengemm::List;

#define OPENGEMM_FORMAT(NAME, ELEM, SF)                                       \
  constexpr Policy NAME(int cta_group, int mma_n = 256, bool swap_ab = false, \
                        int epi_trade = 0, bool deep = false,                 \
                        bool use_clc = false, int rm = 1, int rn = 1,         \
                        int rk = 1) {                                         \
    return Policy{SElem::ELEM, SElem::ELEM, SFElem::SF, cta_group, mma_n,     \
                  swap_ab, epi_trade, deep, use_clc, rm, rn, rk};             \
  }
OPENGEMM_FORMAT(nvfp4, e2m1, ue4m3)
OPENGEMM_FORMAT(mxfp8, e4m3, ue8m0)
OPENGEMM_FORMAT(mxfp4, e2m1, ue8m0)
#undef OPENGEMM_FORMAT

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

}

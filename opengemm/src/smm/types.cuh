#pragma once

#include <cstdint>
#include <cuda.h>

#include "common/ptx.cuh"

enum class SElem { e4m3, e5m2, e3m2, e2m3, e2m1_c8, e2m1 };
enum class SFElem { ue4m3, ue8m0 };
enum class SKind { mxf8f6f4, mxf4, mxf4nvf4 };

constexpr bool packed4(SElem e) { return e == SElem::e2m1; }
constexpr int sbits_of(SElem e) { return packed4(e) ? 4 : 8; }
constexpr int sformat_of(SElem e) {
    switch (e) {
    case SElem::e4m3:    return 0;
    case SElem::e5m2:    return 1;
    case SElem::e3m2:    return 4;
    case SElem::e2m3:    return 3;
    case SElem::e2m1_c8: return 5;
    case SElem::e2m1:    return 1;
    }
    return -1;
}

constexpr SKind skind_of(SElem a, SFElem sf) {
    if (!packed4(a))
        return SKind::mxf8f6f4;
    return sf == SFElem::ue4m3 ? SKind::mxf4nvf4 : SKind::mxf4;
}

constexpr int sf_block_of(SFElem sf) { return sf == SFElem::ue4m3 ? 16 : 32; }
constexpr int sf_format_of(SFElem sf) { return sf == SFElem::ue4m3 ? 0 : 1; }
constexpr int block_k_for(int bits) { return 1024 / bits; }
constexpr int mma_k_for(int bits) { return 256 / bits; }

constexpr int nsf_for(int bits, SFElem sf) {
    return mma_k_for(bits) / sf_block_of(sf);
}

constexpr int EPI_BUF_BYTES  = 128 * 128 * sizeof(uint16_t);
constexpr int SWIZZLE_ATOM   = 8 * 128;
constexpr int SMEM_GRANT_CAP = 232448 - 2048;
constexpr int EPI_STORE_N = 64;

constexpr int epi_buf_count(int mainloop_bytes) {
    for (int bufs = 3; bufs > 1; --bufs)
        if (mainloop_bytes + bufs * EPI_BUF_BYTES + 64 <= SMEM_GRANT_CAP)
            return bufs;
    return 1;
}

constexpr int stages_full_for(int cta_group) {
    return (cta_group == 1) ? 3 : 5;
}

constexpr int traded_stages(int full, int stage_bytes) {
    int stages = full;
    while (stages > 2 && epi_buf_count((stages - 1) * stage_bytes)
                       > epi_buf_count(stages * stage_bytes))
        --stages;
    return stages;
}

constexpr int traded_stages_one(int full, int stage_bytes) {
    return (full > 2 && epi_buf_count((full - 1) * stage_bytes) > epi_buf_count(full * stage_bytes))
        ? full - 1 : full;
}

constexpr int traded_stage_count(int level, int full, int stage_bytes) {
    return level == 1 ? traded_stages(full, stage_bytes)
         : level == 2 ? traded_stages_one(full, stage_bytes)
                      : full;
}

constexpr int deep_stage_count(int stage_bytes) {
    int fit = (SMEM_GRANT_CAP - 64) / stage_bytes;
    return fit < 8 ? fit : 8;
}

struct Policy {
    SElem  elem_a    = SElem::e2m1;
    SElem  elem_b    = SElem::e2m1;
    SFElem elem_sf   = SFElem::ue4m3;
    int    cta_group = 1;
    int    mma_n     = 256;
    bool   swap_ab   = false;
    int    epi_trade = 0;
    bool   deep      = false;
    bool   use_clc   = false;
    int    rm        = 1;
    int    rn        = 1;
    int    rk        = 1;

    constexpr bool operator==(const Policy &) const = default;
};

template <Policy P>
struct Geom {
    static constexpr SElem  elem_a  = P.elem_a;
    static constexpr SElem  elem_b  = P.elem_b;
    static constexpr SFElem elem_sf = P.elem_sf;
    static constexpr SKind  kind    = skind_of(elem_a, elem_sf);
    static constexpr int  elem_bits = sbits_of(elem_a);
    static constexpr int  sf_block  = sf_block_of(elem_sf);
    static constexpr int  nsf       = nsf_for(elem_bits, elem_sf);

    static constexpr int  cta_group = P.cta_group;
    static constexpr int  mma_n     = P.mma_n;
    static constexpr bool swap_ab   = P.swap_ab;
    static constexpr int  epi_trade = P.epi_trade;
    static constexpr bool deep      = P.deep;
    static constexpr bool use_clc   = P.use_clc;
    static constexpr int  rm        = P.rm;
    static constexpr int  rn        = P.rn;
    static constexpr int  rk        = P.rk;

    static constexpr int cluster_x    = cta_group * rm;
    static constexpr int group_ctas   = cta_group * rm * rn;
    static constexpr int cluster_ctas = group_ctas * rk;
    static constexpr int mc_m = rm;
    static constexpr int mc_n = rn;

    static constexpr int mma_m   = cta_group * 128;
    static constexpr int mma_k   = mma_k_for(elem_bits);
    static constexpr int block_m = 128;
    static constexpr int block_n = mma_n / cta_group;
    static constexpr int block_k = block_k_for(elem_bits);
    static constexpr int k_iters = block_k / mma_k;

    static constexpr int a_bytes      = block_m * block_k * elem_bits / 8;
    static constexpr int b_bytes      = block_n * block_k * elem_bits / 8;
    static constexpr int sfa_bytes    = block_m * block_k / sf_block;
    static constexpr int sf_n_blocks  = (mma_n + 127) / 128;
    static constexpr int sfb_bytes    = sfa_bytes * sf_n_blocks;
    static constexpr int stage_bytes = a_bytes + b_bytes + sfa_bytes + sfb_bytes;
    static constexpr int tile_tx      = cta_group * (a_bytes + b_bytes);
    static constexpr int scale_tx     = cta_group * (sfa_bytes + sfb_bytes);

    static constexpr int stages_full = stages_full_for(cta_group);
    static constexpr int num_stages  = deep
        ? deep_stage_count(stage_bytes)
        : traded_stage_count(epi_trade, stages_full, stage_bytes);

    static constexpr int sfa_k_stride     = 4;
    static constexpr int sf_cps           = k_iters * nsf / 4;
    static constexpr int sf_y_stride      = 2 * block_k / (4 * sf_block);

    static constexpr int sfa_stage_stride = sf_cps * 4;
    static constexpr int sfb_stage_stride = sf_cps * 4 * sf_n_blocks;
    static constexpr int scale_stage_cols = sfa_stage_stride + sfb_stage_stride;
    static constexpr int acc_cols         = ((mma_n + 127) / 128) * 128;
    static constexpr int tmem_cols        = 512;

    static constexpr int  epi_tile_n = 128;
    static constexpr int  epi_iters  = (mma_n + epi_tile_n - 1) / epi_tile_n;
    static constexpr int  last_epi_n = mma_n - (epi_iters - 1) * epi_tile_n;

    static constexpr int ring_bytes = (num_stages * stage_bytes + SWIZZLE_ATOM - 1)
            / SWIZZLE_ATOM * SWIZZLE_ATOM;
    static constexpr int  epi_bufs   = epi_buf_count(ring_bytes);
    static constexpr int c_tma_n = (mma_n < epi_tile_n) ? mma_n : epi_tile_n;

    static constexpr bool vec_stage  = !swap_ab && mma_n >= epi_tile_n;
    static constexpr int  store_n    = EPI_STORE_N;
    static constexpr int store_buf_bytes = block_m * store_n * (int)sizeof(uint16_t);

    static constexpr int mma_warp       = 4;
    static constexpr int scale_tma_warp = 5;
    static constexpr int tile_tma_warp  = 6;
    static constexpr int threads        = 128 + 3 * WARP_SIZE;

    static constexpr int smem_bytes = deep
        ? ring_bytes + 64
        : ring_bytes + epi_bufs * EPI_BUF_BYTES + 64;

    static constexpr uint32_t instr_desc = (static_cast<uint32_t>(sformat_of(elem_a)) << 7)
        | (static_cast<uint32_t>(sformat_of(elem_b)) << 10)
        | (static_cast<uint32_t>(mma_n) >> 3 << 17)
        | (static_cast<uint32_t>(sf_format_of(elem_sf)) << 23)
        | (static_cast<uint32_t>(mma_m) >> 4 << 24);
};

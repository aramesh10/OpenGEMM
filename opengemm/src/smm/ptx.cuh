#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include "common/ptx.cuh"
#include "types.cuh"

__device__ __forceinline__ uint64_t make_sf_desc(int addr) {
    constexpr uint64_t SBO = 8ULL * 16;
    return desc_enc(addr) | (desc_enc(SBO) << 32) | (1ULL << 46);
}

__device__ __forceinline__ uint32_t cluster_ctarank() {
    uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}

__device__ __forceinline__ void cluster_sync() {
    cluster_arrive_relaxed();
    cluster_wait();
}

constexpr uint32_t PAIR_PEER_BIT = 0x01000000u;

__device__ __forceinline__ uint32_t pair_leader_bar(int addr) {
    return static_cast<uint32_t>(addr) & ~PAIR_PEER_BIT;
}

__device__ __forceinline__ void mbar_arrive_pair_leader(int mbar) {
    const uint32_t m = pair_leader_bar(mbar);
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(m) : "memory");
}

__device__ __forceinline__ void tile_coords(int tile_id, int num_m, int num_n,
                                            int supergroup, int& m_idx, int& n_idx) {
    const int tpg = num_m * supergroup;
    const int grp = tile_id / tpg;
    const int first_n = grp * supergroup;
    const int ns = (num_n - first_n) < supergroup ? (num_n - first_n) : supergroup;
    const int idx = tile_id - grp * tpg;
    m_idx = idx / ns;
    n_idx = first_n + idx % ns;
}

template <int NUM_SM>
__device__ __forceinline__ void tma_load_2d(int dst, const void* tmap, int x, int y, int mbar) {
    if constexpr (NUM_SM == 1) {
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3}], [%4];"
                     :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
    } else {
        const uint32_t lead = pair_leader_bar(mbar);
        asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                     :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(lead) : "memory");
    }
}

template <int NUM_SM>
__device__ __forceinline__ void tma_load_2d_multicast(int dst, const void* tmap,
                                                      int x, int y, int mbar,
                                                      uint16_t mcast_mask) {
    if constexpr (NUM_SM == 1) {
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                     :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar), "h"(mcast_mask) : "memory");
    } else {
        const uint32_t lead = pair_leader_bar(mbar);
        asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                     :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(lead), "h"(mcast_mask) : "memory");
    }
}

template <int NUM_SM>
__device__ __forceinline__ void tma_load_3d(int dst, const void* tmap, int x, int y, int z, int mbar) {
    if constexpr (NUM_SM == 1) {
        asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3, %4}], [%5];"
                     :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(mbar) : "memory");
    } else {
        const uint32_t lead = pair_leader_bar(mbar);
        asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                     :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(lead) : "memory");
    }
}

template <int NUM_SM>
__device__ __forceinline__ void tma_load_3d_multicast(int dst, const void* tmap, int x, int y, int z, int mbar, uint16_t mcast_mask) {
    if constexpr (NUM_SM == 1) {
        asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3, %4}], [%5], %6;"
                     :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(mbar), "h"(mcast_mask) : "memory");
    } else {
        const uint32_t lead = pair_leader_bar(mbar);
        asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3, %4}], [%5], %6;"
                     :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(lead), "h"(mcast_mask) : "memory");
    }
}

__device__ __forceinline__ uint16_t cvt_bf16(float x) {
    uint16_t v;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(v) : "f"(x));
    return v;
}

__device__ __forceinline__ uint32_t cvt_bf16x2(float hi, float lo) {
    uint32_t v;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(v) : "f"(hi), "f"(lo));
    return v;
}

__device__ __forceinline__ void st_shared_u16(int addr, uint16_t val) {
    asm volatile("st.shared.b16 [%0], %1;" :: "r"(addr), "h"(val) : "memory");
}

__device__ __forceinline__ void st_shared_v4(int addr, uint32_t a, uint32_t b,
                                             uint32_t c, uint32_t d) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

template <int SM>
__device__ __forceinline__ void tmem_relinquish() {
    if constexpr (SM == 1)
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    else
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}

template <int SM, int COLS>
__device__ __forceinline__ void tmem_alloc_all(uint32_t *slot) {
    tmem_alloc<SM, COLS>(static_cast<int>(__cvta_generic_to_shared(slot)));
    __syncwarp();
    tmem_relinquish<SM>();
}

template <int SM>
__device__ __forceinline__ void tcgen05_cp(int taddr, uint64_t s_desc) {
    if constexpr (SM == 1)
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
    else
        asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

template <int SM, SKind K>
__device__ __forceinline__ void tcgen05_mma(int d, uint64_t a, uint64_t b,
                                            uint32_t i, int sfa, int sfb,
                                            int ena_d) {
    if constexpr (SM == 1 && K == SKind::mxf8f6f4)
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::1.kind::mxf8f6f4"
                     ".block_scale.scale_vec::1X"
                     " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                     :: "r"(d), "l"(a), "l"(b), "r"(i), "r"(sfa),
                        "r"(sfb), "r"(ena_d));
    else if constexpr (SM == 1 && K == SKind::mxf4)
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::1.kind::mxf4"
                     ".block_scale.scale_vec::2X"
                     " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                     :: "r"(d), "l"(a), "l"(b), "r"(i), "r"(sfa),
                        "r"(sfb), "r"(ena_d));
    else if constexpr (SM == 1 && K == SKind::mxf4nvf4)
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::1.kind::mxf4nvf4"
                     ".block_scale.scale_vec::4X"
                     " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                     :: "r"(d), "l"(a), "l"(b), "r"(i), "r"(sfa),
                        "r"(sfb), "r"(ena_d));
    else if constexpr (SM == 2 && K == SKind::mxf8f6f4)
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::2.kind::mxf8f6f4"
                     ".block_scale.scale_vec::1X"
                     " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                     :: "r"(d), "l"(a), "l"(b), "r"(i), "r"(sfa),
                        "r"(sfb), "r"(ena_d));
    else if constexpr (SM == 2 && K == SKind::mxf4)
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::2.kind::mxf4"
                     ".block_scale.scale_vec::2X"
                     " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                     :: "r"(d), "l"(a), "l"(b), "r"(i), "r"(sfa),
                        "r"(sfb), "r"(ena_d));
    else if constexpr (SM == 2 && K == SKind::mxf4nvf4)
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
                     "tcgen05.mma.cta_group::2.kind::mxf4nvf4"
                     ".block_scale.scale_vec::4X"
                     " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                     :: "r"(d), "l"(a), "l"(b), "r"(i), "r"(sfa),
                        "r"(sfb), "r"(ena_d));
}

template <int SM>
__device__ __forceinline__ void tcgen05_commit(int mbar, uint16_t mask = 0) {
    if constexpr (SM == 1) {
        if (mask == 0)
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory");
        else
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                         :: "r"(mbar), "h"(mask) : "memory");
    } else {
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                     :: "r"(mbar), "h"(mask) : "memory");
    }
}

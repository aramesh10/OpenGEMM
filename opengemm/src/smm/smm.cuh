#pragma once

#include "device_utils.cuh"
#include "types.cuh"

struct CLC {
    int arrived;
    int finished;
    int ready;
    uint4 *handles;
};

template <int CTA_GROUP>
__device__ __forceinline__ void clc_issue(const CLC &clc, int tile_idx, bool is_leader) {
    const int slot  = tile_idx % CLC_DEPTH;
    const int phase = (tile_idx / CLC_DEPTH) & 1;

    const int answer_landed = clc.arrived  + slot * MBAR;
    const int slot_free     = clc.finished + slot * MBAR;
    const int peer_here     = clc.ready    + slot * MBAR;

    mbar_arrive_tx(answer_landed, sizeof(uint4));
    if (!is_leader) {
        mbar_arrive_cluster_to(peer_here, 0);
        return;
    }
    if (tile_idx >= CLC_DEPTH)      mbar_wait_cluster(slot_free, phase ^ 1);
    if constexpr (CTA_GROUP == 2)   mbar_wait_cluster(peer_here, phase);
    clc_schedule(&clc.handles[slot], answer_landed);
}

template <int CTA_GROUP, bool USE_CLC>
__device__ __forceinline__ int next_tile(const CLC &clc, int tile_id, int tile_idx, int num_clusters) {
    if constexpr (!USE_CLC)
        return tile_id + num_clusters;
    const int slot  = tile_idx % CLC_DEPTH;
    const int phase = (tile_idx / CLC_DEPTH) & 1;

    const int answer_landed = clc.arrived  + slot * MBAR;
    const int slot_free     = clc.finished + slot * MBAR;

    mbar_wait_cluster(answer_landed, phase);
    const uint4 next = clc_query(&clc.handles[slot]);
    mbar_arrive_cluster_to(slot_free, 0);
    return next.x ? next.y / CTA_GROUP : -1;
}

template <Policy P>
__global__ __launch_bounds__(Geom<P>::threads, 1)
void smm_gemm_kernel(const __grid_constant__ CUtensorMap a_tmap,
                        const __grid_constant__ CUtensorMap b_tmap,
                        const __grid_constant__ CUtensorMap c_tmap,
                        const __grid_constant__ CUtensorMap sfa_tmap,
                        const __grid_constant__ CUtensorMap sfb_tmap,
                        uint16_t* c,
                        int m, int n, int k,
                        int supergroup,
                        int epi_direct) {
    using G = Geom<P>;

    constexpr bool SWAP_AB          = G::swap_ab;
    constexpr bool DEEP             = G::deep;
    constexpr bool USE_CLC          = G::use_clc;

    constexpr SKind    KIND         = G::kind;
    constexpr int      NSF          = G::nsf;
    constexpr uint32_t I_DESC       = G::instr_desc;

    constexpr int  CTA_COUNT        = G::cta_group;
    constexpr int  RM               = G::rm;
    constexpr int  RN               = G::rn;
    constexpr int  GROUP_CTAS       = G::group_ctas;
    constexpr int  CLUSTER_CTAS     = G::cluster_ctas;
    constexpr int  MC_M             = G::mc_m;
    constexpr int  MC_N             = G::mc_n;
    constexpr int  CLUSTER_X        = CTA_COUNT * RM;
    constexpr uint16_t PAIR_MASK_0  = (1u << CTA_COUNT) - 1u;
    constexpr uint16_t GROUP_MASK   = (1u << GROUP_CTAS) - 1u;

    constexpr int  MMA_M            = G::mma_m;
    constexpr int  MMA_N            = G::mma_n;
    constexpr int  BLOCK_M          = G::block_m;
    constexpr int  BLOCK_N          = G::block_n;
    constexpr int  BLOCK_K          = G::block_k;
    constexpr int  K_ITERS          = G::k_iters;

    constexpr int  NUM_STAGES       = G::num_stages;
    constexpr int  A_SIZE           = G::a_bytes;
    constexpr int  B_SIZE           = G::b_bytes;
    constexpr int  SFA_SIZE         = G::sfa_bytes;
    constexpr int  SF_N_BLOCKS      = G::sf_n_blocks;
    constexpr int  STAGE_SIZE       = G::stage_bytes;
    constexpr int  TILE_TX          = G::tile_tx;
    constexpr int  SCALE_TX         = G::scale_tx;

    constexpr int  NUM_MBAR         = NUM_STAGES * 3 + 5
                                    + (USE_CLC ? 3 * CLC_DEPTH : 0);

    constexpr int  SFA_K_STRIDE     = G::sfa_k_stride;
    constexpr int  SF_Y_STRIDE      = G::sf_y_stride;
    constexpr int  SFA_STAGE_STRIDE = G::sfa_stage_stride;
    constexpr int  SCALE_STAGE_COLS = G::scale_stage_cols;
    constexpr int  SF_CPS           = G::sf_cps;
    constexpr uint64_t SF_CP_STRIDE = 512 >> 4;
    constexpr int  SIDE_COLS        = G::acc_cols;
    constexpr int  TMEM_COLS        = G::tmem_cols;

    constexpr int      SFB_HALF_DESC_SIZE = SFA_SIZE >> 4;
    constexpr uint64_t KI_DESC_STRIDE     = 32 >> 4;

    constexpr int  EPI_TILE_N       = G::epi_tile_n;
    constexpr int  EPI_ITERS        = G::epi_iters;
    constexpr int  LAST_EPI_N       = G::last_epi_n;
    constexpr int  EPI_BUFS         = G::epi_bufs;
    constexpr bool VEC_STAGE        = G::vec_stage;
    constexpr int  STORE_N          = G::store_n;
    constexpr int  STORE_BUF_BYTES  = G::store_buf_bytes;
    constexpr bool HAS_SHORT_TAIL   = EPI_ITERS > 1 && LAST_EPI_N < EPI_TILE_N;
    constexpr int  FRAG_N           = (EPI_ITERS == 1) ? LAST_EPI_N : EPI_TILE_N;

    constexpr bool RING_STAGE       = DEEP && VEC_STAGE && !USE_CLC;

    constexpr int  MMA_WARP         = G::mma_warp;
    constexpr int  SCALE_TMA_WARP   = G::scale_tma_warp;
    constexpr int  TILE_TMA_WARP    = G::tile_tma_warp;

    const int NUM_ITERS = (k + BLOCK_K - 1) / BLOCK_K;

    const int  tid       = threadIdx.x;
    const int  warp      = tid / WARP_SIZE;

    const int  cta_rank  = cluster_ctarank();
    const int  peer_x    = cta_rank % CTA_COUNT;
    const int  m_peer    = (CLUSTER_CTAS == CTA_COUNT) ? 0 : (cta_rank / CTA_COUNT) % RM;
    const int  n_peer    = (CLUSTER_CTAS == CTA_COUNT) ? 0 : (cta_rank / (CTA_COUNT * RM)) % RN;
    const bool is_leader = (peer_x == 0);

    const int  pair_base = (CLUSTER_CTAS == CTA_COUNT) ? 0 : (cta_rank - peer_x);

    const uint16_t PAIR_MASK = PAIR_MASK_0 << pair_base;
    uint16_t a_mask = 0;
    uint16_t b_mask = 0;

    #pragma unroll
    for (int j = 0; j < RN; ++j)
        a_mask |= 1u << (peer_x + CTA_COUNT * (m_peer + RM * j));

    #pragma unroll
    for (int j = 0; j < RM; ++j)
        b_mask |= 1u << (peer_x + CTA_COUNT * (j + RM * n_peer));

    const uint16_t sfb_mask = ((1u << (CTA_COUNT * RM)) - 1u) << (CTA_COUNT * RM * n_peer);

    const int cluster_idx  = blockIdx.x / CLUSTER_X;
    const int num_clusters = gridDim.x / CLUSTER_X;
    const int num_m_all    = (m + MMA_M - 1) / MMA_M;
    const int num_n_all    = (n + MMA_N - 1) / MMA_N;

    const int num_m        = (num_m_all + MC_M - 1) / MC_M;
    const int num_n        = (num_n_all + MC_N - 1) / MC_N;
    const int num_tiles    = num_m * num_n;

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

    __shared__ int64_t mbar[NUM_STAGES * 3 + 5 + (USE_CLC ? 3 * CLC_DEPTH : 0)];
    const int tile_mbar     = static_cast<int>(__cvta_generic_to_shared(mbar));
    const int scale_mbar    = tile_mbar + NUM_STAGES * MBAR;
    const int mma_mbar      = scale_mbar + NUM_STAGES * MBAR;
    const int tmem_done     = mma_mbar + NUM_STAGES * MBAR;
    const int tmem_empty    = tmem_done + 2 * MBAR;

    const int alloc_done = tmem_empty + 2 * MBAR;
    const int clc_base = alloc_done + 8;

    __shared__ __align__(16) uint4 clc_handles[CLC_DEPTH];
    __shared__ int clc_next;

    const CLC clc = {clc_base,
                     clc_base + CLC_DEPTH * MBAR,
                     clc_base + 2 * CLC_DEPTH * MBAR,
                     clc_handles};

    if (warp == 0) {
        const int lane = tid & (WARP_SIZE - 1);
        if (lane == 0) {
            prefetch_tensormap(&a_tmap);
            prefetch_tensormap(&b_tmap);
            prefetch_tensormap(&c_tmap);
            prefetch_tensormap(&sfa_tmap);
            prefetch_tensormap(&sfb_tmap);
        }
        for (int i = lane; i < NUM_MBAR; i += WARP_SIZE) {
            int count;
            if      (i < NUM_STAGES * 2)     count = 1;
            else if (i < NUM_STAGES * 3)     count = RM * RN;
            else if (i < NUM_STAGES * 3 + 2) count = 1;
            else if (i < NUM_STAGES * 3 + 5) count = CTA_COUNT;
            else count = ((i - (NUM_STAGES * 3 + 5)) / CLC_DEPTH == 1)
                             ? 3 * CTA_COUNT + 1 : 1;
            mbar_init(tile_mbar + i * MBAR, count);
        }
        mbar_fence_init();
    }

    __shared__ uint32_t tmem_base;
    if constexpr (CLUSTER_CTAS == 1) {
        if (warp == MMA_WARP) tmem_alloc_all<CTA_COUNT, TMEM_COLS>(&tmem_base);
        __syncthreads();
    } else {
        cluster_sync();
        if (warp == MMA_WARP) {
            tmem_alloc_all<CTA_COUNT, TMEM_COLS>(&tmem_base);
            if constexpr (CTA_COUNT == 2)
                if (elect_sync()) mbar_arrive_pair_leader(alloc_done);
        }
    }
    if (tid == 0) pdl_arrive();

    if (warp == TILE_TMA_WARP && elect_sync()) {
        int tile_id = cluster_idx;
        int tile_idx = 0;

        int g = 0;
        while (tile_id < num_tiles) {
            if constexpr (USE_CLC)
                clc_issue<CTA_COUNT>(clc, tile_idx, is_leader);

            int m_idx, n_idx;
            tile_coords(tile_id,
                        num_m, num_n, supergroup, m_idx, n_idx);
            const int m_tile = m_idx * MC_M + m_peer;
            const int n_tile = n_idx * MC_N + n_peer;
            const int tile_m = m_tile * MMA_M + peer_x * BLOCK_M;
            const int tile_n = n_tile * MMA_N + peer_x * BLOCK_N;
            for (int iter_k = 0; iter_k < NUM_ITERS; ++iter_k, ++g) {
                const int stage = g % NUM_STAGES;
                const int a_smem = smem + stage * STAGE_SIZE;
                const int b_smem = a_smem + A_SIZE;
                if (g >= NUM_STAGES) mbar_wait(mma_mbar + stage * MBAR, (g / NUM_STAGES - 1) & 1);
                if (is_leader) mbar_arrive_tx(tile_mbar + stage * MBAR, TILE_TX);

                if constexpr (RN > 1) {
                    if (n_peer == 0)
                        tma_load_2d_multicast<CTA_COUNT>(
                            a_smem, &a_tmap, iter_k * BLOCK_K, tile_m,
                            tile_mbar + stage * MBAR, a_mask);
                } else {
                    tma_load_2d<CTA_COUNT>(a_smem, &a_tmap, iter_k * BLOCK_K,
                                          tile_m, tile_mbar + stage * MBAR);
                }
                if constexpr (RM > 1) {
                    if (m_peer == 0)
                        tma_load_2d_multicast<CTA_COUNT>(
                            b_smem, &b_tmap, iter_k * BLOCK_K, tile_n,
                            tile_mbar + stage * MBAR, b_mask);
                } else {
                    tma_load_2d<CTA_COUNT>(b_smem, &b_tmap, iter_k * BLOCK_K,
                                          tile_n, tile_mbar + stage * MBAR);
                }
            }

            tile_id = next_tile<CTA_COUNT, USE_CLC>(clc, tile_id, tile_idx, num_clusters);
            if (tile_id < 0) break;
            ++tile_idx;
        }
    }
    else if (warp == SCALE_TMA_WARP && elect_sync()) {
        int tile_id = cluster_idx;
        int tile_idx = 0;
        int g = 0;
        while (tile_id < num_tiles) {
            int m_idx, n_idx;
            tile_coords(tile_id,
                        num_m, num_n, supergroup, m_idx, n_idx);

            const int m_tile  = m_idx * MC_M + m_peer;
            const int n_tile  = n_idx * MC_N + n_peer;
            const int sfa_row = m_tile * CTA_COUNT + peer_x;
            const int sfb_row = n_tile * SF_N_BLOCKS;

            for (int iter_k = 0; iter_k < NUM_ITERS; ++iter_k, ++g) {
                const int stage = g % NUM_STAGES;
                const int sfa_smem = smem + stage * STAGE_SIZE + A_SIZE + B_SIZE;
                if (g >= NUM_STAGES) mbar_wait(mma_mbar + stage * MBAR, (g / NUM_STAGES - 1) & 1);
                if (is_leader) mbar_arrive_tx(scale_mbar + stage * MBAR, SCALE_TX);

                if constexpr (RN > 1) {
                    if (n_peer == 0)
                        tma_load_3d_multicast<CTA_COUNT>(
                            sfa_smem, &sfa_tmap, 0, SF_Y_STRIDE * iter_k,
                            sfa_row, scale_mbar + stage * MBAR, a_mask);
                } else {
                    tma_load_3d<CTA_COUNT>(sfa_smem, &sfa_tmap,
                                          0, SF_Y_STRIDE * iter_k, sfa_row,
                                          scale_mbar + stage * MBAR);
                }

                if constexpr (CTA_COUNT == 2 && SF_N_BLOCKS == 2) {
                    if (m_peer == 0)
                        tma_load_3d_multicast<2>(
                            sfa_smem + SFA_SIZE + peer_x * SFA_SIZE, &sfb_tmap,
                            0, SF_Y_STRIDE * iter_k, sfb_row + peer_x,
                            scale_mbar + stage * MBAR, sfb_mask);
                } else if constexpr (RM > 1) {
                    if (m_peer == 0)
                        tma_load_3d_multicast<CTA_COUNT>(
                            sfa_smem + SFA_SIZE, &sfb_tmap,
                            0, SF_Y_STRIDE * iter_k, sfb_row,
                            scale_mbar + stage * MBAR, b_mask);
                } else {
                    tma_load_3d<CTA_COUNT>(sfa_smem + SFA_SIZE, &sfb_tmap,
                                          0, SF_Y_STRIDE * iter_k, sfb_row,
                                          scale_mbar + stage * MBAR);
                }
            }

            tile_id = next_tile<CTA_COUNT, USE_CLC>(clc, tile_id, tile_idx, num_clusters);
            if (tile_id < 0) break;
            ++tile_idx;
        }
    }
    else if (warp == MMA_WARP && is_leader && elect_sync()) {
        if constexpr (CTA_COUNT == 2) mbar_wait_cluster(alloc_done, 0);

        int tile_id = cluster_idx;
        int tile_idx = 0;
        int g = 0;
        while (tile_id < num_tiles) {
            const int acc_base = (tile_idx & 1) * SIDE_COLS;
            const int sf_base  = (MMA_N > 128) ? (acc_base ^ SIDE_COLS)
                                                  : 2 * SIDE_COLS;
            if constexpr (MMA_N <= 128) {
                if (tile_idx >= 2) {
                    mbar_wait_cluster(tmem_empty + (tile_idx & 1) * MBAR,
                                      ((tile_idx >> 1) + 1) & 1);
                    tmem_fence();
                }
            }
            if constexpr (MMA_N > 128) {
                if (tile_idx > 0) {
                    #pragma unroll
                    for (int sf_half = 0; sf_half < 2; ++sf_half)
                        mbar_wait_cluster(tmem_empty + sf_half * MBAR,
                                          (tile_idx + 1) & 1);
                    tmem_fence();
                }
            }
            for (int iter_k = 0; iter_k < NUM_ITERS; ++iter_k, ++g) {
                const int stage   = g % NUM_STAGES;
                const int phase   = (g / NUM_STAGES) & 1;
                const int sf_slot = iter_k % NUM_STAGES;

                const int a_smem = smem + stage * STAGE_SIZE;
                const int b_smem = smem + stage * STAGE_SIZE + A_SIZE;
                const int stage_sf_base = sf_base + sf_slot * SCALE_STAGE_COLS;
                const int sfa_tmem   = stage_sf_base;
                const int sfb_tmem_1 = stage_sf_base + SFA_STAGE_STRIDE;
                const int sfb_tmem_2 = sfb_tmem_1 + SFA_K_STRIDE;

                const uint64_t sf_desc = make_sf_desc(b_smem + B_SIZE);

                mbar_wait_cluster(scale_mbar + stage * MBAR, phase);

                #pragma unroll
                for (int w = 0; w < SF_CPS; ++w) {
                    const uint64_t base = sf_desc + w * SF_CP_STRIDE;
                    tcgen05_cp<CTA_COUNT>(sfa_tmem + w * 4, base);
                    tcgen05_cp<CTA_COUNT>(sfb_tmem_1 + w * 4 * SF_N_BLOCKS,
                                          base + SFB_HALF_DESC_SIZE);
                    if constexpr (MMA_N > 128)
                        tcgen05_cp<CTA_COUNT>(
                            sfb_tmem_2 + w * 4 * SF_N_BLOCKS,
                            base + 2 * SFB_HALF_DESC_SIZE);
                }

                mbar_wait_cluster(tile_mbar + stage * MBAR, phase);

                const uint64_t a_desc0 = make_ab_desc(a_smem);
                const uint64_t b_desc0 = make_ab_desc(b_smem);

                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ++ki) {
                    const int sf_byte = ki * NSF;
                    const int sf_word = (sf_byte >> 2) * 4;
                    const uint32_t sf_id = sf_byte & 3;
                    tcgen05_mma<CTA_COUNT, KIND>(acc_base,
                                   a_desc0 + ki * KI_DESC_STRIDE,
                                   b_desc0 + ki * KI_DESC_STRIDE,
                                   I_DESC | (sf_id << 29) | (sf_id << 4),
                                   sfa_tmem   + sf_word,
                                   sfb_tmem_1 + sf_word * SF_N_BLOCKS,
                                   (iter_k == 0 && ki == 0) ? 0 : 1);
                }

                tcgen05_commit<CTA_COUNT>(
                    mma_mbar + stage * MBAR,
                    (CTA_COUNT == 1 && RM * RN == 1) ? 0
                        : (GROUP_CTAS == CTA_COUNT ? PAIR_MASK : GROUP_MASK));
            }

            tcgen05_commit<CTA_COUNT>(
                tmem_done + ((MMA_N <= 128) ? (tile_idx & 1) : 0) * MBAR,
                CTA_COUNT == 1 ? 0 : PAIR_MASK);

            tile_id = next_tile<CTA_COUNT, USE_CLC>(clc, tile_id, tile_idx, num_clusters);
            if (tile_id < 0) break;
            ++tile_idx;
        }
    }
    else if (warp < 4) {
        const int out_smem = DEEP ? smem : smem + G::ring_bytes;
        const int tmem_row = peer_x * BLOCK_M + warp * WARP_SIZE;
        const bool store_leader = (warp == 0) && elect_sync();

        int tile_id = cluster_idx;
        int tile_idx = 0;

        int stores_issued = 0;

        bool cluster_left = false;
        while (tile_id < num_tiles) {
            int m_idx, n_idx;
            tile_coords(tile_id,
                        num_m, num_n, supergroup, m_idx, n_idx);
            const int off_m = (m_idx * MC_M + m_peer) * MMA_M
                            + peer_x * BLOCK_M;
            const int off_n = (n_idx * MC_N + n_peer) * MMA_N;

            const bool ring_free = RING_STAGE && (tile_id + num_clusters >= num_tiles);
            const bool direct_store = (DEEP && !ring_free) || epi_direct ||
                (SWAP_AB
                ? (m % 8 != 0)
                : (n % 8 != 0) || (off_m + BLOCK_M > m) ||
                  (off_n + MMA_N > n));

            if constexpr (MMA_N <= 128)
                mbar_wait_cluster(tmem_done + (tile_idx & 1) * MBAR,
                                  (tile_idx >> 1) & 1);
            else
                mbar_wait_cluster(tmem_done, tile_idx & 1);
            tmem_fence();
            const int acc_base = (tile_idx & 1) * SIDE_COLS;

            float tmp[EPI_ITERS][EPI_TILE_N];

            uint32_t packed[EPI_ITERS][EPI_TILE_N / 2];
            #pragma unroll
            for (int f = 0; f < EPI_ITERS; ++f) {
                const int tmem_col = acc_base + f * EPI_TILE_N;
                tcgen05_ld_32x32bx64(&tmp[f][0],  tmem_row, tmem_col);
                tcgen05_ld_32x32bx64(&tmp[f][64], tmem_row, tmem_col + 64);
                tmem_wait_ld();

                tmem_fence_before();
                bar_sync(1, BLOCK_M);

                if (store_leader) {
                    const int slot = tmem_empty + (EPI_ITERS == 1 ? (tile_idx & 1) : f) * MBAR;
                    if constexpr (CTA_COUNT == 1)
                        mbar_arrive_cluster_to(slot, cta_rank);
                    else
                        mbar_arrive_pair_leader(slot);
                }

                if constexpr (!DEEP) {
                    #pragma unroll
                    for (int j = 0; j < EPI_TILE_N / 2; ++j)
                        packed[f][j] = cvt_bf16x2(tmp[f][2 * j + 1], tmp[f][2 * j]);
                }
            }

            if constexpr (SWAP_AB) {
                if (tile_idx == 0) {
                    if (store_leader) pdl_wait();
                    bar_sync(1, BLOCK_M);
                }
            }

            #pragma unroll
            for (int f = 0; f < EPI_ITERS; ++f) {
                const bool direct_fragment = direct_store || (HAS_SHORT_TAIL && f + 1 == EPI_ITERS);

                if (!direct_fragment && stores_issued >= EPI_BUFS) {
                    if (store_leader) tma_store_wait<EPI_BUFS - 1>();
                    bar_sync(1, BLOCK_M);
                }

                if (direct_fragment) {
                    const int global_m = off_m + tid;
                    const int base_n = off_n + f * EPI_TILE_N;
                    const int frag_n = (f + 1 == EPI_ITERS)
                        ? LAST_EPI_N : EPI_TILE_N;

                    const bool vector_row = !SWAP_AB && (n % 8 == 0) &&
                        (frag_n % 8 == 0) && (base_n + frag_n <= n) &&
                        (global_m < m);
                    if (vector_row) {
                        uint16_t* row = c + (int64_t)global_m * n + base_n;
                        #pragma unroll
                        for (int i = 0; i < EPI_TILE_N; i += 8) {
                            if (i < frag_n) {
                                uint4 v;
                                if constexpr (DEEP) {
                                    v.x = cvt_bf16x2(tmp[f][i + 1], tmp[f][i]);
                                    v.y = cvt_bf16x2(tmp[f][i + 3], tmp[f][i + 2]);
                                    v.z = cvt_bf16x2(tmp[f][i + 5], tmp[f][i + 4]);
                                    v.w = cvt_bf16x2(tmp[f][i + 7], tmp[f][i + 6]);
                                } else {
                                    const int j = i >> 1;
                                    v.x = packed[f][j];
                                    v.y = packed[f][j + 1];
                                    v.z = packed[f][j + 2];
                                    v.w = packed[f][j + 3];
                                }
                                *reinterpret_cast<uint4*>(row + i) = v;
                            }
                        }
                    } else if (global_m < m) {
                        #pragma unroll
                        for (int i = 0; i < EPI_TILE_N; ++i) {
                            const int global_n = base_n + i;
                            if (i < frag_n && global_n < n) {
                                const uint16_t v = DEEP
                                    ? cvt_bf16(tmp[f][i])
                                    : (uint16_t)(packed[f][i >> 1] >> (16 * (i & 1)));
                                if constexpr (SWAP_AB)
                                    c[global_n * m + global_m] = v;
                                else
                                    c[global_m * n + global_n] = v;
                            }
                        }
                    }
                    bar_sync(1, BLOCK_M);
                } else {
                    const int buf = out_smem + (stores_issued % EPI_BUFS) * EPI_BUF_BYTES;
                    if constexpr (SWAP_AB) {
                        #pragma unroll
                        for (int i = 0; i < EPI_TILE_N; ++i)
                            st_shared_u16(
                                buf + (i * BLOCK_M + tid) *
                                          (int)sizeof(uint16_t),
                                (uint16_t)(packed[f][i >> 1]
                                           >> (16 * (i & 1))));
                    } else if constexpr (VEC_STAGE) {
                        #pragma unroll
                        for (int h = 0; h < EPI_TILE_N / STORE_N; ++h) {
                            const int half = buf + h * STORE_BUF_BYTES;
                            #pragma unroll
                            for (int c = 0; c < STORE_N / 8; ++c) {
                                const int j = h * (STORE_N / 2) + c * 4;

                                uint32_t v0, v1, v2, v3;
                                if constexpr (DEEP) {
                                    const int i = 2 * j;
                                    v0 = cvt_bf16x2(tmp[f][i + 1], tmp[f][i + 0]);
                                    v1 = cvt_bf16x2(tmp[f][i + 3], tmp[f][i + 2]);
                                    v2 = cvt_bf16x2(tmp[f][i + 5], tmp[f][i + 4]);
                                    v3 = cvt_bf16x2(tmp[f][i + 7], tmp[f][i + 6]);
                                } else {
                                    v0 = packed[f][j + 0];
                                    v1 = packed[f][j + 1];
                                    v2 = packed[f][j + 2];
                                    v3 = packed[f][j + 3];
                                }
                                st_shared_v4(half + tid * (STORE_N * 2)
                                                  + (c ^ (tid & 7)) * 16,
                                             v0, v1, v2, v3);
                            }
                        }
                    } else {
                        const int frag_n = (f + 1 == EPI_ITERS)
                            ? LAST_EPI_N : EPI_TILE_N;
                        #pragma unroll
                        for (int i = 0; i < FRAG_N; ++i)
                            if (i < frag_n)
                                st_shared_u16(
                                    buf + (tid * frag_n + i) *
                                              (int)sizeof(uint16_t),
                                    (uint16_t)(packed[f][i >> 1]
                                               >> (16 * (i & 1))));
                    }

                    tma_store_fence();
                    bar_sync(1, BLOCK_M);

                    if (store_leader) {
                        if constexpr (!SWAP_AB) {
                            if (tile_idx == 0 && f == 0) pdl_wait();
                        }
                        if constexpr (SWAP_AB) {
                            tma_store_2d(buf, &c_tmap, off_m, off_n + f * EPI_TILE_N);
                        } else if constexpr (VEC_STAGE) {
                            #pragma unroll
                            for (int h = 0; h < EPI_TILE_N / STORE_N; ++h)
                                tma_store_2d(buf + h * STORE_BUF_BYTES, &c_tmap,
                                             off_n + f * EPI_TILE_N
                                                   + h * STORE_N, off_m);
                        } else {
                            tma_store_2d(buf, &c_tmap, off_n + f * EPI_TILE_N, off_m);
                        }
                        tma_store_commit();
                    }
                    ++stores_issued;
                }
            }

            bar_sync(1, BLOCK_M);

            if constexpr (CLUSTER_CTAS > 1 && !USE_CLC) {
                if (tile_id + num_clusters >= num_tiles) {
                    cluster_arrive();
                    cluster_left = true;
                }
            }

            if constexpr (USE_CLC) {
                if (store_leader)
                    clc_next = next_tile<CTA_COUNT, USE_CLC>(clc, tile_id, tile_idx, num_clusters);
                bar_sync(2, BLOCK_M);
                tile_id = clc_next;
                bar_sync(2, BLOCK_M);
                if (tile_id < 0) break;
            } else {
                tile_id += num_clusters;
            }
            ++tile_idx;
        }

        if constexpr (CLUSTER_CTAS > 1) {
            if (!cluster_left) cluster_arrive();
        }

        if (store_leader) tma_store_wait<0>();
    }

    if constexpr (CLUSTER_CTAS > 1) {
        if (warp >= 4) cluster_arrive();
    }
    __syncthreads();
    if constexpr (CLUSTER_CTAS > 1) cluster_wait();
    else                                 __syncthreads();
    if (warp == MMA_WARP) tmem_dealloc<CTA_COUNT, TMEM_COLS>(0);
}


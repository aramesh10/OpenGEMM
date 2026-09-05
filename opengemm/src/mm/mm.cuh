#pragma once

#include "device_utils.cuh"
constexpr int MBAR        = sizeof(int64_t);
constexpr int CLC_DEPTH   = 3;

__device__ __forceinline__ void k_partition(int count, int parts, int idx,
                                            int &begin, int &len) {
    const int per = count / parts;
    const int rem = count % parts;
    begin = idx * per + (idx < rem ? idx : rem);
    len   = per + (idx < rem);
}

constexpr int hold_divisor(int count, int want) {
    int hold = (want < count) ? want : count;
    while (count % hold)
        --hold;
    return hold;
}

constexpr int ld_width(int cols) {
    return cols <= 8 ? 8 : cols <= 16 ? 16 : cols <= 32 ? 32 : 64;
}

template <int W, class T>
__device__ __forceinline__ void tcgen05_ld(T *dst, int row, int col) {
    float *words = reinterpret_cast<float *>(dst);
    if      constexpr (W ==  8) tcgen05_ld_32x32bx8(words, row, col);
    else if constexpr (W == 16) tcgen05_ld_32x32bx16(words, row, col);
    else if constexpr (W == 32) tcgen05_ld_32x32bx32(words, row, col);
    else                        tcgen05_ld_32x32bx64(words, row, col);
}

template <bool SWAP_AB, bool ACC_STORE, class T>
__device__ __forceinline__ void store_c(T *c, int m, int n, int row, int col,
                                        T value) {
    T *p = c + (SWAP_AB ? static_cast<size_t>(row) * n + col
                        : static_cast<size_t>(col) * m + row);
    if constexpr (ACC_STORE) atomicAdd(p, value);
    else                     *p = value;
}

template <Policy P>
struct Geom {
    static constexpr Elem mma_elem_a = P.swap_ab ? P.elem_b : P.elem_a;
    static constexpr Elem mma_elem_b = P.swap_ab ? P.elem_a : P.elem_b;
    static constexpr Kind kind       = KindOf<mma_elem_a, mma_elem_b>::kind;
    static constexpr Acc  acc        = KindOf<mma_elem_a, mma_elem_b>::acc;
    static constexpr int  elem_bits  = KindOf<mma_elem_a, mma_elem_b>::container_bits;
    static constexpr int  input_bytes = elem_bits / 8;
    static constexpr int  global_bits_a = ElemTraits<mma_elem_a>::global_bits;
    static constexpr int  global_bits_b = ElemTraits<mma_elem_b>::global_bits;
    static constexpr CUtensorMapDataType tmap_a = ElemTraits<mma_elem_a>::tmap;
    static constexpr CUtensorMapDataType tmap_b = ElemTraits<mma_elem_b>::tmap;
    using acc_t = typename std::conditional<acc == Acc::s32, int, float>::type;
    static constexpr CUtensorMapDataType tmap_c =
        (acc == Acc::s32) ? CU_TENSOR_MAP_DATA_TYPE_INT32
                          : CU_TENSOR_MAP_DATA_TYPE_FLOAT32;

    static constexpr int  cta_group  = P.cta_group;
    static constexpr int  block_m    = P.block_m;
    static constexpr int  mma_n      = P.mma_n;
    static constexpr bool swap_ab    = P.swap_ab;
    static constexpr bool epi_direct = P.epi_direct;
    static constexpr bool use_clc    = P.use_clc;
    static constexpr bool split_k    = P.split_k;
    static constexpr int  rm         = P.rm;
    static constexpr int  rn         = P.rn;
    static constexpr int  rk         = P.rk;

    static constexpr bool dual_mma  = P.epi_mode == 1;
    static constexpr int  consumers = dual_mma ? 2 : 1;
    static constexpr int  mma_m     = cta_group * block_m;
    static constexpr int  tile_m    = consumers * mma_m;
    static constexpr int  block_n   = mma_n / cta_group;
    static constexpr int  block_k   = block_k_for(elem_bits);
    static constexpr int  mma_k     = mma_k_for(elem_bits);
    static constexpr int  k_iters   = block_k / mma_k;
    static_assert(k_iters == 4, "one stage is 1024 bits of K and one MMA is "
                                "256, for every element width");

    static constexpr int a_bytes     = block_m * block_k * input_bytes;
    static constexpr int b_bytes     = block_n * block_k * input_bytes;
    static constexpr int stage_bytes = consumers * a_bytes + b_bytes;
    static constexpr int a_tx_bytes  = block_m * block_k * global_bits_a / 8;
    static constexpr int b_tx_bytes  = block_n * block_k * global_bits_b / 8;
    static constexpr int stage_tx    =
        cta_group * (consumers * a_tx_bytes + b_tx_bytes);
    static constexpr int stages_want =
        (P.stages > 0) ? P.stages : ((cta_group == 1) ? 3 : 6);

    static constexpr int epi_warps   = 4;
    static constexpr int epi_threads = epi_warps * WARP_SIZE;
    static constexpr int epi_tile_n  =
        (block_m == 64) ? (mma_n / 2) : ((cta_group == 1) ? 128 : 64);
    static constexpr int epi_iters   = (mma_n + epi_tile_n - 1) / epi_tile_n;
    static constexpr int last_epi_n  = mma_n - (epi_iters - 1) * epi_tile_n;
    static constexpr int hold        = hold_divisor(epi_iters, P.epi_hold);

    static constexpr bool acc_store     = split_k || (rk > 1);
    static constexpr bool always_direct = epi_direct || acc_store;
    static constexpr int  epi_bufs      = dual_mma ? 2 : 1;
    static constexpr int  epi_buf_bytes =
        (block_m == 64 || always_direct)
            ? 0
            : epi_tile_n * block_m * static_cast<int>(sizeof(float));
    static constexpr int  c_tma_n =
        (mma_n < epi_tile_n) ? mma_n : epi_tile_n;

    static constexpr int  stages_with_epi =
        fit_stages(stages_want, stage_bytes, epi_bufs * epi_buf_bytes);
    static constexpr int  stages_sans_epi =
        fit_stages(stages_want, stage_bytes, 0);
    static constexpr bool drop_epi_buf = stages_sans_epi > stages_with_epi;
    static constexpr int  num_stages   =
        drop_epi_buf ? stages_sans_epi : stages_with_epi;

    static constexpr int  smem_bytes =
        num_stages * stage_bytes
        + (drop_epi_buf ? 0 : epi_bufs * epi_buf_bytes)
        + 64;

    static constexpr int acc_cols  = (block_m == 64)
        ? ((cta_group == 2) ? 64 : 128)
        : ((mma_n + 127) / 128) * 128;
    static constexpr int tmem_cols = 2 * acc_cols;

    static constexpr int mma_warp = 4 * consumers;
    static constexpr int tma_warp = mma_warp + (dual_mma ? 3 : 1);
    static constexpr int threads  = (dual_mma ? 12 : 6) * WARP_SIZE;

    static constexpr int cluster_m    = cta_group * rm;
    static constexpr int cluster_ctas = cluster_m * rn * rk;

    static constexpr uint32_t instr_desc =
        (static_cast<uint32_t>(acc_format(acc)) << 4)
                  | (static_cast<uint32_t>(ElemTraits<mma_elem_a>::format) << 7)
                  | (static_cast<uint32_t>(ElemTraits<mma_elem_b>::format) << 10)
                  | (static_cast<uint32_t>(mma_n) >> 3 << 17)
                  | (static_cast<uint32_t>(mma_m) >> 4 << 24);

    static_assert(block_m == 128 || block_m == 64);
    static_assert(P.epi_mode == 0 || P.epi_mode == 1);
    static_assert(block_m == 128 || P.epi_mode == 0,
                  "the 64-row block runs the single-consumer epilogue only");
    static_assert(block_m == 128 || !use_clc,
                  "the 64-row epilogue advances by a fixed stride and cannot "
                  "follow a cluster launch control schedule");
    static_assert(!split_k || block_m == 128);
    static_assert(!epi_direct || block_m == 128);
    static_assert(cta_group != 2 ||
                  (mma_n >= 16 && mma_n <= 256 && mma_n % 16 == 0));
    static_assert(block_m != 64 ||
                  (cta_group == 2
                       ? (mma_n >= 16 && mma_n <= 128 && mma_n % 16 == 0)
                       : (mma_n >= 8 && mma_n <= 128 && mma_n % 8 == 0)));
    static_assert(!dual_mma ||
                      !(use_clc || split_k || rm > 1 || rn > 1 || swap_ab),
                  "two MMA issuers share the TMEM ping-pong regions with "
                  "split-K and run the plain persistent schedule only");
    static_assert(cta_group == 2 || (rm == 1 && rn == 1),
                  "a widened cluster needs the 2-CTA MMA's multicast commit");
    static_assert(rm == 1 ||
                  (cta_group == 2 && !dual_mma && !split_k && !use_clc));
    static_assert(rk == 1 || (!split_k && !use_clc && !dual_mma));
    static_assert(cluster_ctas <= 16);
};

struct Clc {
    int arrived;
    int finished;
    int ready;
    uint4 *handles;
};

template <int CTA_GROUP>
__device__ __forceinline__ void clc_issue(const Clc &clc, int tile_idx,
                                          bool is_leader) {
    const int slot  = tile_idx % CLC_DEPTH;
    const int phase = (tile_idx / CLC_DEPTH) & 1;
    mbar_arrive_tx(clc.arrived + slot * MBAR, sizeof(uint4));
    if (!is_leader) {
        mbar_arrive_cluster_to(clc.ready + slot * MBAR, 0);
        return;
    }
    if (tile_idx >= CLC_DEPTH)
        mbar_wait_cluster(clc.finished + slot * MBAR, phase ^ 1);
    if constexpr (CTA_GROUP == 2)
        mbar_wait_cluster(clc.ready + slot * MBAR, phase);
    clc_schedule(&clc.handles[slot], clc.arrived + slot * MBAR);
}

template <int CTA_GROUP, bool USE_CLC>
__device__ __forceinline__ int next_tile(const Clc &clc, int tile_id,
                                         int tile_idx, int num_clusters) {
    if constexpr (!USE_CLC)
        return tile_id + num_clusters;
    const int slot  = tile_idx % CLC_DEPTH;
    const int phase = (tile_idx / CLC_DEPTH) & 1;
    mbar_wait_cluster(clc.arrived + slot * MBAR, phase);
    const uint4 next = clc_query(&clc.handles[slot]);
    mbar_arrive_cluster_to(clc.finished + slot * MBAR, 0);
    return next.x ? static_cast<int>(next.y) / CTA_GROUP : -1;
}

template <Policy P>
__global__ __launch_bounds__(Geom<P>::threads)
void mm_gemm_kernel(const __grid_constant__ CUtensorMap a_tmap,
                     const __grid_constant__ CUtensorMap b_tmap,
                     const __grid_constant__ CUtensorMap c_tmap,
                     typename Geom<P>::acc_t *c,
                     int m, int n, int k,
                     int supergroup, int walk, int splits) {
    using G = Geom<P>;
    constexpr int  CTA_GROUP     = G::cta_group;
    constexpr int  BLOCK_M       = G::block_m;
    constexpr int  BLOCK_N       = G::block_n;
    constexpr int  BLOCK_K       = G::block_k;
    constexpr int  MMA_M         = G::mma_m;
    constexpr int  MMA_N         = G::mma_n;
    constexpr int  MMA_K         = G::mma_k;
    using ACC = typename G::acc_t;
    constexpr int  INPUT_BYTES   = G::input_bytes;
    constexpr Kind KIND          = G::kind;
    constexpr int  TILE_M        = G::tile_m;
    constexpr int  K_ITERS       = G::k_iters;
    constexpr int  A_BYTES       = G::a_bytes;
    constexpr int  STAGE_BYTES   = G::stage_bytes;
    constexpr int  STAGE_TX      = G::stage_tx;
    constexpr int  NUM_STAGES    = G::num_stages;
    constexpr int  CONSUMERS     = G::consumers;
    constexpr bool DUAL_MMA      = G::dual_mma;
    constexpr int  EPI_WARPS     = G::epi_warps;
    constexpr int  EPI_THREADS   = G::epi_threads;
    constexpr int  EPI_TILE_N    = G::epi_tile_n;
    constexpr int  EPI_ITERS     = G::epi_iters;
    constexpr int  LAST_EPI_N    = G::last_epi_n;
    constexpr int  EPI_BUF_BYTES = G::epi_buf_bytes;
    constexpr int  HOLD          = G::hold;
    constexpr bool ALWAYS_DIRECT = G::always_direct;
    constexpr bool DROP_EPI_BUF  = G::drop_epi_buf;
    constexpr bool ACC_STORE     = G::acc_store;
    constexpr int  ACC_COLS      = G::acc_cols;
    constexpr int  TMEM_COLS     = G::tmem_cols;
    constexpr int  MMA_WARP      = G::mma_warp;
    constexpr int  TMA_WARP      = G::tma_warp;
    constexpr int  CM            = G::cluster_m;
    constexpr int  CLUSTER_CTAS  = G::cluster_ctas;
    constexpr bool SWAP_AB       = G::swap_ab;
    constexpr bool USE_CLC       = G::use_clc;
    constexpr bool SPLIT_K       = G::split_k;
    constexpr int  RM = G::rm, RN = G::rn, RK = G::rk;

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    const int  cta_x        = static_cast<int>(cluster_ctaid_x());
    const int  cta_in_group = (RM == 1) ? cta_x : cta_x % CTA_GROUP;
    const int  rm_id        = (RM == 1) ? 0     : cta_x / CTA_GROUP;
    const int  rn_id        = (RN == 1) ? 0     : static_cast<int>(blockIdx.y);
    const int  rk_id        = (RK == 1) ? 0     : static_cast<int>(blockIdx.z);
    const bool is_leader    = (cta_in_group == 0);

    const int plane_base   = rk_id * CM * RN;
    const int group_leader = rm_id * CTA_GROUP + rn_id * CM + plane_base;

    uint16_t a_mcast = 0;
    #pragma unroll
    for (int j = 0; j < RN; ++j)
        a_mcast |= (uint16_t)1u << (cta_in_group + rm_id * CTA_GROUP
                                    + j * CM + plane_base);
    uint16_t b_mcast = 0;
    if constexpr (RM > 1) {
        #pragma unroll
        for (int i = 0; i < RM; ++i)
            b_mcast |= (uint16_t)1u << (cta_in_group + i * CTA_GROUP
                                        + rn_id * CM + plane_base);
    }
    const uint16_t pair_mask =
        (uint16_t)(((1u << CTA_GROUP) - 1u) << group_leader);
    const uint16_t plane_mask =
        (uint16_t)(((1u << (CM * RN)) - 1u) << plane_base);

    const int cluster_idx   = static_cast<int>(blockIdx.x) / CM;
    const int num_clusters  = static_cast<int>(gridDim.x) / CM;
    const int num_m         = (m + TILE_M - 1) / TILE_M;
    const int num_n         = (n + MMA_N - 1) / MMA_N;
    const int num_m_groups  = (num_m + RM - 1) / RM;
    const int num_n_groups  = (num_n + RN - 1) / RN;
    const int spatial_tiles = num_m_groups * num_n_groups;
    const int num_tiles     = SPLIT_K ? spatial_tiles * splits : spatial_tiles;

    const int k_tiles = (k + BLOCK_K - 1) / BLOCK_K;
    const int k_base  = SPLIT_K ? k_tiles / splits : k_tiles;
    const int k_rem   = SPLIT_K ? k_tiles % splits : 0;

    auto tile_origin = [&](int tile_id, int &off_m, int &off_n) {
        const int spatial = SPLIT_K ? tile_id % spatial_tiles : tile_id;
        int m_idx, n_idx;
        tile_coords(spatial, num_m_groups, num_n_groups, supergroup, walk,
                    m_idx, n_idx);
        off_m = (m_idx * RM + rm_id) * TILE_M;
        off_n = (n_idx * RN + rn_id) * MMA_N;
    };

    auto k_range = [&](int tile_id, int &k_begin, int &k_count) {
        const int split_idx = SPLIT_K ? tile_id / spatial_tiles : 0;

        const int lead = (split_idx < k_rem) ? split_idx : k_rem;
        k_begin = SPLIT_K ? split_idx * k_base + lead : 0;
        k_count = SPLIT_K ? k_base + (split_idx < k_rem) : k_base;
        if constexpr (RK > 1) {
            int plane_begin, plane_count;
            k_partition(k_count, RK, rk_id, plane_begin, plane_count);
            k_begin += plane_begin;
            k_count  = plane_count;
        }
    };

    auto sync_cluster = [] {
        if constexpr (CLUSTER_CTAS > 1) cluster_sync();
        else                            __syncthreads();
    };

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

    __shared__ int64_t mbar[NUM_STAGES * 2 + 4 + (USE_CLC ? 3 * CLC_DEPTH : 0)];
    const int stage_full = static_cast<int>(__cvta_generic_to_shared(mbar));
    const int stage_free = stage_full + NUM_STAGES * MBAR;
    const int acc_full   = stage_free + NUM_STAGES * MBAR;
    const int acc_free   = acc_full   + 2 * MBAR;
    const int clc_base   = acc_free   + 2 * MBAR;

    __shared__ __align__(16) uint4 clc_handles[CLC_DEPTH];
    __shared__ int clc_next;

    const Clc clc = {clc_base,
                     clc_base + CLC_DEPTH * MBAR,
                     clc_base + 2 * CLC_DEPTH * MBAR,
                     clc_handles};

    if (tid == 0) {
        prefetch_tensormap(&a_tmap);
        prefetch_tensormap(&b_tmap);
        prefetch_tensormap(&c_tmap);
        #pragma unroll
        for (int s = 0; s < NUM_STAGES; ++s) {
            mbar_init(stage_full + s * MBAR, 1);
            mbar_init(stage_free + s * MBAR, CONSUMERS * RM * RN);
        }
        #pragma unroll
        for (int a = 0; a < 2; ++a) {
            mbar_init(acc_full + a * MBAR, 1);
            mbar_init(acc_free + a * MBAR, CTA_GROUP);
        }
        if constexpr (USE_CLC) {
            #pragma unroll
            for (int s = 0; s < CLC_DEPTH; ++s) {
                mbar_init(clc.arrived  + s * MBAR, 1);
                mbar_init(clc.finished + s * MBAR, 2 * CTA_GROUP + 1);
                mbar_init(clc.ready    + s * MBAR, 1);
            }
        }
        mbar_fence_init();
    }

    if (warp == MMA_WARP) tmem_alloc<CTA_GROUP, TMEM_COLS>(smem);
    sync_cluster();
    if (tid == 0) pdl_arrive();

    if (warp == TMA_WARP && elect_sync()) {
        int tile_id  = cluster_idx;
        int tile_idx = 0;
        int stage    = 0;
        uint32_t parity = 0;
        bool wrapped = false;

        while (tile_id < num_tiles) {
            if constexpr (USE_CLC)
                clc_issue<CTA_GROUP>(clc, tile_idx, is_leader);

            int k_begin, k_count, off_m, off_n;
            k_range(tile_id, k_begin, k_count);
            tile_origin(tile_id, off_m, off_n);
            const int tile_m = off_m + cta_in_group * BLOCK_M;
            const int tile_n = off_n + cta_in_group * BLOCK_N;

            for (int iter_k = 0; iter_k < k_count; ++iter_k) {
                const int a_smem = smem + stage * STAGE_BYTES;
                const int b_smem = a_smem + CONSUMERS * A_BYTES;
                const int mbar   = stage_full + stage * MBAR;
                const int k_off  = (k_begin + iter_k) * BLOCK_K;

                if (wrapped) {
                    if constexpr (BLOCK_M == 64)
                        mbar_wait_ns(stage_free + stage * MBAR, parity ^ 1u);
                    else
                        mbar_wait(stage_free + stage * MBAR, parity ^ 1u);
                }
                if (is_leader)
                    mbar_arrive_tx(mbar, STAGE_TX);

                if constexpr (RN == 1) {
                    #pragma unroll
                    for (int consumer = 0; consumer < CONSUMERS; ++consumer)
                        tma_load_2d<CTA_GROUP>(a_smem + consumer * A_BYTES,
                                               &a_tmap, k_off,
                                               tile_m + consumer * MMA_M,
                                               mbar, group_leader);
                } else if (rn_id == 0) {
                    tma_load_2d_multicast<CTA_GROUP>(a_smem, &a_tmap, k_off,
                                                     tile_m, mbar, a_mcast,
                                                     group_leader);
                }

                if constexpr (RM > 1) {
                    if (rm_id == iter_k % RM)
                        tma_load_2d_multicast<CTA_GROUP>(b_smem, &b_tmap, k_off,
                                                         tile_n, mbar, b_mcast,
                                                         group_leader);
                } else {
                    tma_load_2d<CTA_GROUP>(b_smem, &b_tmap, k_off, tile_n,
                                           mbar, group_leader);
                }

                if (++stage == NUM_STAGES) {
                    stage   = 0;
                    parity ^= 1u;
                    wrapped = true;
                }
            }

            tile_id = next_tile<CTA_GROUP, USE_CLC>(clc, tile_id, tile_idx,
                                                    num_clusters);
            if (tile_id < 0) break;
            ++tile_idx;
        }
    }
    else if (warp >= MMA_WARP && warp < MMA_WARP + CONSUMERS &&
             is_leader && elect_sync()) {
        const int consumer = warp - MMA_WARP;
        int tile_id  = cluster_idx;
        int tile_idx = 0;
        int stage    = 0;
        int parity   = 0;

        while (tile_id < num_tiles) {
            const int acc_slot = DUAL_MMA ? consumer : (tile_idx & 1);
            const int reuse_distance = DUAL_MMA ? 1 : 2;
            if (tile_idx >= reuse_distance) {
                const int phase = DUAL_MMA ? ((tile_idx - 1) & 1)
                                           : ((tile_idx / 2 - 1) & 1);
                if constexpr (BLOCK_M == 64)
                    mbar_wait_cluster_ns(acc_free + acc_slot * MBAR, phase);
                else
                    mbar_wait_cluster(acc_free + acc_slot * MBAR, phase);
                tmem_fence();
            }

            int k_begin, k_count;
            k_range(tile_id, k_begin, k_count);

            for (int iter_k = 0; iter_k < k_count; ++iter_k) {
                const int stage_smem = smem + stage * STAGE_BYTES;
                const int a_smem = stage_smem + consumer * A_BYTES;
                const int b_smem = stage_smem + CONSUMERS * A_BYTES;

                mbar_wait_cluster(stage_full + stage * MBAR, parity);

                constexpr uint64_t DESC_STEP = MMA_K * INPUT_BYTES / 16;
                uint64_t a_desc = make_ab_desc(a_smem);
                uint64_t b_desc = make_ab_desc(b_smem);

                #pragma unroll
                for (int ki = 0; ki < K_ITERS;
                     ++ki, a_desc += DESC_STEP, b_desc += DESC_STEP)
                    tcgen05_mma<CTA_GROUP, KIND>(acc_slot * ACC_COLS, a_desc, b_desc,
                                           G::instr_desc,
                                           (iter_k == 0 && ki == 0) ? 0 : 1);

                tcgen05_commit<CTA_GROUP>(stage_free + stage * MBAR,
                                          (RM == 1 && RN == 1) ? pair_mask
                                                               : plane_mask);
                if (++stage == NUM_STAGES) {
                    stage   = 0;
                    parity ^= 1;
                }
            }
            tcgen05_commit<CTA_GROUP>(acc_full + acc_slot * MBAR, pair_mask);

            tile_id = next_tile<CTA_GROUP, USE_CLC>(clc, tile_id, tile_idx,
                                                    num_clusters);
            if (tile_id < 0) break;
            ++tile_idx;
        }
    }
    else if (BLOCK_M == 64 && warp < EPI_WARPS) {
        constexpr int WARP_N   = (CTA_GROUP == 2) ? MMA_N / 2 : MMA_N;
        constexpr int LD_W     = ld_width(WARP_N);
        constexpr int N_CHUNKS = (WARP_N + LD_W - 1) / LD_W;

        const int  rows_each = (CTA_GROUP == 2) ? WARP_SIZE : 16;
        const int  row_warp  = (CTA_GROUP == 2) ? (warp & 1) : warp;
        const int  n_half    = (CTA_GROUP == 2) ? (warp >> 1) : 0;

        const int  row_in_block = row_warp * rows_each + lane;
        const bool row_stores   = (CTA_GROUP == 2) || (lane < 16);
        const int  lane_base    = warp * WARP_SIZE;
        const bool epi_leader   = (warp == 0) && elect_sync();

        int tile_id  = cluster_idx;
        int tile_idx = 0;
        while (tile_id < num_tiles) {
            int off_m, off_n;
            tile_origin(tile_id, off_m, off_n);
            off_m += cta_in_group * BLOCK_M;
            off_n += n_half * WARP_N;

            const int acc_slot = tile_idx & 1;
            const int phase = (tile_idx / 2) & 1;
            mbar_wait_cluster_ns(acc_full + acc_slot * MBAR, phase);
            tmem_fence();

            ACC acc[N_CHUNKS][LD_W];
            #pragma unroll
            for (int chunk = 0; chunk < N_CHUNKS; ++chunk)
                tcgen05_ld<LD_W>(acc[chunk], lane_base,
                                 acc_slot * ACC_COLS + chunk * LD_W);
            tmem_wait_ld();
            tmem_fence_before();
            bar_sync(1, EPI_THREADS);
            if (epi_leader)
                mbar_arrive_cluster_to(acc_free + acc_slot * MBAR,
                                       group_leader);

            if (tile_idx == 0) {
                if (epi_leader) pdl_wait();
                bar_sync(1, EPI_THREADS);
            }

            const int global_m = off_m + row_in_block;
            if (row_stores && global_m < m) {
                #pragma unroll
                for (int j = 0; j < WARP_N; ++j) {
                    const int global_n = off_n + j;
                    if (global_n < n)
                        store_c<SWAP_AB, ACC_STORE>(c, m, n, global_m, global_n,
                                                    acc[j / LD_W][j % LD_W]);
                }
            }
            tile_id += num_clusters;
            ++tile_idx;
        }
    }
    else if (BLOCK_M == 128 && warp < EPI_WARPS * CONSUMERS) {
        constexpr int FLOAT_BYTES = static_cast<int>(sizeof(float));
        const int  consumer     = warp / EPI_WARPS;
        const int  local_warp   = warp % EPI_WARPS;
        const int  row_in_block = tid - consumer * EPI_THREADS;
        const int  lane_base    = local_warp * WARP_SIZE;
        const int  epi_bar      = 1 + consumer;
        const bool epi_leader   = (local_warp == 0) && elect_sync();
        const int  epi_buf      = smem + NUM_STAGES * STAGE_BYTES
                                       + consumer * EPI_BUF_BYTES;

        int  tile_id     = cluster_idx;
        int  tile_idx    = 0;
        bool pdl_pending = true;

        while (tile_id < num_tiles) {
            int off_m, off_n;
            tile_origin(tile_id, off_m, off_n);
            off_m += (DUAL_MMA ? consumer * MMA_M : 0) + cta_in_group * BLOCK_M;

            const bool direct_store =
                ALWAYS_DIRECT || DROP_EPI_BUF ||
                (SWAP_AB ? (n % 8 != 0)
                         : (m % 8 != 0) || (off_m + BLOCK_M > m)
                                        || (off_n + MMA_N > n));

            const int acc_slot = DUAL_MMA ? consumer : (tile_idx & 1);
            const int phase    = DUAL_MMA ? (tile_idx & 1)
                                          : ((tile_idx / 2) & 1);
            mbar_wait_cluster(acc_full + acc_slot * MBAR, phase);
            tmem_fence();

            constexpr int ACC_N = (EPI_TILE_N < 64) ? 64 : EPI_TILE_N;
            ACC acc[HOLD][ACC_N];

            #pragma unroll
            for (int frag = 0; frag < EPI_ITERS; ++frag) {
                constexpr bool SHORT_TAIL =
                    EPI_ITERS > 1 && LAST_EPI_N < EPI_TILE_N;
                const bool last = (frag + 1 == EPI_ITERS);
                const int  cols = last ? LAST_EPI_N : EPI_TILE_N;
                const int  held = frag % HOLD;

                const bool direct = direct_store || (SHORT_TAIL && last);

                if (held == 0) {
                    #pragma unroll
                    for (int h = 0; h < HOLD; ++h) {
                        const int col = acc_slot * ACC_COLS
                                      + (frag + h) * EPI_TILE_N;
                        tcgen05_ld<64>(&acc[h][0], lane_base, col);
                        if constexpr (EPI_TILE_N == 128)
                            tcgen05_ld<64>(&acc[h][64], lane_base, col + 64);
                    }
                    tmem_wait_ld();
                    if (frag + HOLD == EPI_ITERS) {
                        tmem_fence_before();
                        bar_sync(epi_bar, EPI_THREADS);
                        if (epi_leader)
                            mbar_arrive_cluster_to(acc_free + acc_slot * MBAR,
                                                   group_leader);
                    }
                }

                if (pdl_pending && frag == 0) {
                    if (epi_leader) pdl_wait();
                    bar_sync(epi_bar, EPI_THREADS);
                    pdl_pending = false;
                }

                if (frag > 0) {
                    if (!direct_store && epi_leader) tma_store_wait<0>();
                    bar_sync(epi_bar, EPI_THREADS);
                }

                if (direct) {
                    const int global_m = off_m + row_in_block;
                    if (global_m < m) {
                        #pragma unroll
                        for (int i = 0; i < EPI_TILE_N; ++i) {
                            const int global_n = off_n + frag * EPI_TILE_N + i;
                            if (i < cols && global_n < n)
                                store_c<SWAP_AB, ACC_STORE>(
                                    c, m, n, global_m, global_n, acc[held][i]);
                        }
                    }
                    bar_sync(epi_bar, EPI_THREADS);
                    continue;
                }

                if constexpr (SWAP_AB) {
                    constexpr int FRAG_N = (EPI_ITERS == 1) ? LAST_EPI_N
                                                            : EPI_TILE_N;
                    constexpr int STEP   = (FRAG_N % 4 == 0) ? 4 : 1;
                    const int row = epi_buf + row_in_block * cols * FLOAT_BYTES;
                    #pragma unroll
                    for (int i = 0; i < FRAG_N; i += STEP) {
                        if (STEP == 4 && i + 3 < cols) {
                            st_shared_v4b(row + i * FLOAT_BYTES,
                                         acc[held][i],     acc[held][i + 1],
                                         acc[held][i + 2], acc[held][i + 3]);
                        } else {
                            #pragma unroll
                            for (int j = 0; j < STEP; ++j)
                                if (i + j < cols)
                                    st_shared_b32(row + (i + j) * FLOAT_BYTES,
                                                  acc[held][i + j]);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < EPI_TILE_N; ++i)
                        st_shared_b32(epi_buf + (i * BLOCK_M + row_in_block)
                                                    * FLOAT_BYTES,
                                      acc[held][i]);
                }

                tma_store_fence();
                bar_sync(epi_bar, EPI_THREADS);

                if (epi_leader) {
                    if constexpr (SWAP_AB)
                        tma_store_2d(epi_buf, &c_tmap,
                                     off_n + frag * EPI_TILE_N, off_m);
                    else
                        tma_store_2d(epi_buf, &c_tmap, off_m,
                                     off_n + frag * EPI_TILE_N);
                    tma_store_commit();
                }
            }

            constexpr bool LAST_IS_TMA =
                EPI_ITERS == 1 || LAST_EPI_N == EPI_TILE_N;
            if (!direct_store && LAST_IS_TMA && epi_leader)
                tma_store_wait<0>();
            bar_sync(epi_bar, EPI_THREADS);

            if constexpr (USE_CLC) {
                if (epi_leader)
                    clc_next = next_tile<CTA_GROUP, USE_CLC>(clc, tile_id,
                                                             tile_idx,
                                                             num_clusters);
                bar_sync(2, EPI_THREADS);
                tile_id = clc_next;
                bar_sync(2, EPI_THREADS);
                if (tile_id < 0) break;
            } else {
                tile_id += num_clusters;
            }
            ++tile_idx;
        }
    }

    __syncthreads();
    sync_cluster();
    if (warp == MMA_WARP) tmem_dealloc<CTA_GROUP, TMEM_COLS>(0);
}

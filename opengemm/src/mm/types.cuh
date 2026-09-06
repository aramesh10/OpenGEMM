// The dense kernel's vocabulary: what an element is, the Policy that names
// one kernel, and the Geom that reads a Policy as the geometry the kernel is
// laid out in. Every field of Geom is a compile-time constant, so the kernel
// below it has no configuration to carry.
#pragma once

#include <cuda.h>
#include <cstdint>
#include <type_traits>

#include "common/ptx.cuh"

enum class Elem { bf16, f16, tf32, s8, u8, e4m3, e5m2, e3m2, e2m3, e2m1 };
enum class Kind { f16, tf32, i8, f8f6f4 };
enum class Acc { f32, s32 };

struct ElemSpec {
    int value_bits;
    int container_bits;
    int global_bits;
    int format;
    Kind kind;
    CUtensorMapDataType tmap;
};

constexpr ElemSpec ELEM_SPEC[] = {
    {16, 16, 16, 1, Kind::f16, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16},
    {16, 16, 16, 0, Kind::f16, CU_TENSOR_MAP_DATA_TYPE_FLOAT16},
    {32, 32, 32, 2, Kind::tf32, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32},
    { 8, 8, 8, 1, Kind::i8, CU_TENSOR_MAP_DATA_TYPE_UINT8},
    { 8, 8, 8, 0, Kind::i8, CU_TENSOR_MAP_DATA_TYPE_UINT8},
    { 8, 8, 8, 0, Kind::f8f6f4, CU_TENSOR_MAP_DATA_TYPE_UINT8},
    { 8, 8, 8, 1, Kind::f8f6f4, CU_TENSOR_MAP_DATA_TYPE_UINT8},
    { 6, 8, 6, 4, Kind::f8f6f4, CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B},
    { 6, 8, 6, 3, Kind::f8f6f4, CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B},
    { 4, 8, 4, 5, Kind::f8f6f4, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B},
};

template <Elem E>
struct ElemTraits {
    static constexpr ElemSpec info = ELEM_SPEC[static_cast<int>(E)];
    static constexpr int value_bits = info.value_bits;
    static constexpr int container_bits = info.container_bits;
    static constexpr int global_bits = info.global_bits;
    static constexpr int format = info.format;
    static constexpr Kind kind = info.kind;
    static constexpr CUtensorMapDataType tmap = info.tmap;
};

template <Elem A, Elem B>
struct KindOf {
    static constexpr Kind kind = ElemTraits<A>::kind;
    static constexpr Acc acc = (kind == Kind::i8) ? Acc::s32 : Acc::f32;
    static constexpr int container_bits = ElemTraits<A>::container_bits;
};

constexpr int acc_format(Acc acc) { return acc == Acc::s32 ? 2 : 1; }

constexpr int block_k_for(int bits) { return 1024 / bits; }
constexpr int mma_k_for(int bits) { return 256 / bits; }

struct Policy {
    Elem elem_a;
    Elem elem_b;
    int  cta_group;
    int  block_m    = 128;
    int  mma_n      = 256;
    int  stages     = 0;
    bool swap_ab    = false;
    int  epi_hold   = 4;
    int  epi_mode   = 0;
    bool epi_direct = false;
    bool use_clc    = false;
    bool split_k    = false;
    int  rm         = 1;
    int  rn         = 1;
    int  rk         = 1;

    constexpr bool operator==(const Policy &) const = default;
};

enum : int {
  WALK_GROUP      = 0,
  WALK_SERPENTINE = 1,
  WALK_BLOCK      = 2,
  WALK_GILBERT    = 3,
  WALK_ROW        = 4,
};

// ---- how much of the SM's shared memory a stage may have ----

constexpr int SMEM_BUDGET_BYTES = 227 * 1024;

constexpr int fit_stages(int want, int stage_bytes, int epilogue_bytes) {
  int stages = want;
  while (stages > 1 &&
         stages * stage_bytes + epilogue_bytes + 64 > SMEM_BUDGET_BYTES)
    --stages;
  return stages;
}

constexpr int hold_divisor(int count, int want) {
    int hold = (want < count) ? want : count;
    while (count % hold)
        --hold;
    return hold;
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

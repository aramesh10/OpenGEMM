#pragma once

#include <cuda.h>

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

struct Config {
    Policy policy;
    int supergroup = 1;
    int walk       = 0;
    int l2_promo   = 0;
    int k_pad      = -1;
    int splits     = 1;
};

#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>

#include "common/tmap.cuh"

constexpr int BF16_BYTES = 2;

inline CUresult init_ab_tmap(CUtensorMap *tmap, void *ptr, uint64_t rows,
                             uint64_t k, uint32_t tile_rows, uint32_t tile_k,
                             uint32_t elem_bits) {
  const CUtensorMapDataType dtype =
      (elem_bits == 4) ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B
                       : CU_TENSOR_MAP_DATA_TYPE_UINT8;
  return tmap_2d(tmap, dtype, ptr, rows, k, tile_rows, tile_k,
                 k * elem_bits / 8, CU_TENSOR_MAP_SWIZZLE_128B);
}

inline CUresult init_sf_tmap(CUtensorMap *tmap, void *ptr, uint64_t rows,
                             uint64_t k, uint32_t block_k, uint32_t nblocks,
                             uint32_t sf_block) {
  constexpr uint32_t RANK = 3;
  const uint64_t per_group = 4 * sf_block;
  const uint64_t k_groups = (k + per_group - 1) / per_group;
  uint64_t global_dim[RANK] = {256, 2 * k_groups, (rows + 127) / 128};
  uint64_t global_strides[RANK - 1] = {256, k_groups * 512};
  uint32_t box_dim[RANK] = {256, 2 * (block_k / per_group), nblocks};
  uint32_t elem_strides[RANK] = {1, 1, 1};
  return cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, RANK, ptr, global_dim,
      global_strides, box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline void init_c_tmap(CUtensorMap *tmap, void *ptr, uint64_t rows,
                        uint64_t cols, uint32_t tile_rows, uint32_t tile_cols,
                        CUtensorMapSwizzle swizzle) {
  tmap_2d(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, ptr, rows, cols, tile_rows,
          tile_cols, cols * BF16_BYTES, swizzle);
}

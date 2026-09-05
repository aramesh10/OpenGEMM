#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>

constexpr int OUTPUT_BYTES = 4;

inline CUresult init_ab_tmap(CUtensorMap *tmap, void *ptr, uint64_t rows,
                         uint64_t k, uint32_t tile_rows, uint32_t tile_k,
                         CUtensorMapDataType dtype, uint32_t element_bits,
                         uint64_t pitch = 0,
                         CUtensorMapL2promotion l2 =
                             CU_TENSOR_MAP_L2_PROMOTION_NONE) {
  constexpr uint32_t RANK = 2;
  if (pitch == 0)
    pitch = k;
  uint64_t global_dim[RANK] = {k, rows};
  uint64_t global_strides[RANK - 1] = {pitch * element_bits / 8};
  uint32_t box_dim[RANK] = {tile_k, tile_rows};
  uint32_t elem_strides[RANK] = {1, 1};
  return cuTensorMapEncodeTiled(
      tmap, dtype, RANK, ptr, global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, l2,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline void
init_c_tmap(CUtensorMap *tmap, void *ptr, uint64_t rows, uint64_t cols,
            uint32_t tile_rows, uint32_t tile_cols,
            CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
  constexpr uint32_t RANK = 2;
  uint64_t global_dim[RANK] = {cols, rows};
  uint64_t global_strides[RANK - 1] = {cols * OUTPUT_BYTES};
  uint32_t box_dim[RANK] = {tile_cols, tile_rows};
  uint32_t elem_strides[RANK] = {1, 1};
  cuTensorMapEncodeTiled(
      tmap, dtype, RANK, ptr, global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

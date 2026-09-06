#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>

inline CUresult tmap_2d(CUtensorMap *tmap, CUtensorMapDataType dtype,
                        void *ptr, uint64_t rows, uint64_t cols,
                        uint32_t tile_rows, uint32_t tile_cols,
                        uint64_t row_bytes, CUtensorMapSwizzle swizzle,
                        CUtensorMapL2promotion l2 =
                            CU_TENSOR_MAP_L2_PROMOTION_NONE) {
  constexpr uint32_t RANK = 2;
  uint64_t global_dim[RANK] = {cols, rows};
  uint64_t global_strides[RANK - 1] = {row_bytes};
  uint32_t box_dim[RANK] = {tile_cols, tile_rows};
  uint32_t elem_strides[RANK] = {1, 1};
  return cuTensorMapEncodeTiled(
      tmap, dtype, RANK, ptr, global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

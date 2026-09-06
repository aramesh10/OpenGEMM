// How the dense build describes an operand to the TMA unit. A and B are read
// through a 128B-swizzled tile over rows that may be pitched wider than K; C
// is written through an unswizzled one whose element is the accumulator's.
#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>

#include "common/tmap.cuh"

constexpr int OUTPUT_BYTES = 4;

// `pitch` is the row stride in values, which is K unless the caller padded an
// unaligned K by copying into a wider buffer.
inline CUresult init_ab_tmap(CUtensorMap *tmap, void *ptr, uint64_t rows,
                             uint64_t k, uint32_t tile_rows, uint32_t tile_k,
                             CUtensorMapDataType dtype, uint32_t element_bits,
                             uint64_t pitch = 0,
                             CUtensorMapL2promotion l2 =
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE) {
  if (pitch == 0)
    pitch = k;
  return tmap_2d(tmap, dtype, ptr, rows, k, tile_rows, tile_k,
                 pitch * element_bits / 8, CU_TENSOR_MAP_SWIZZLE_128B, l2);
}

inline void
init_c_tmap(CUtensorMap *tmap, void *ptr, uint64_t rows, uint64_t cols,
            uint32_t tile_rows, uint32_t tile_cols,
            CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
  tmap_2d(tmap, dtype, ptr, rows, cols, tile_rows, tile_cols,
          cols * OUTPUT_BYTES, swizzle);
}

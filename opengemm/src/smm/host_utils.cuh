#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int BF16_BYTES = 2;

inline CUresult init_ab_tmap(CUtensorMap *tmap, void *ptr, uint64_t rows,
                             uint64_t k, uint32_t tile_rows, uint32_t tile_k,
                             uint32_t elem_bits) {
  constexpr uint32_t RANK = 2;
  uint64_t global_dim[RANK] = {k, rows};
  uint64_t global_strides[RANK - 1] = {k * elem_bits / 8};
  uint32_t box_dim[RANK] = {tile_k, tile_rows};
  uint32_t elem_strides[RANK] = {1, 1};
  const CUtensorMapDataType dtype =
      (elem_bits == 4) ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B
                       : CU_TENSOR_MAP_DATA_TYPE_UINT8;
  return cuTensorMapEncodeTiled(
      tmap, dtype, RANK, ptr, global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
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
                        uint64_t cols, uint32_t tile_rows,
                        uint32_t tile_cols, CUtensorMapSwizzle swizzle) {
  constexpr uint32_t RANK = 2;
  uint64_t global_dim[RANK] = {cols, rows};
  uint64_t global_strides[RANK - 1] = {cols * BF16_BYTES};
  uint32_t box_dim[RANK] = {tile_cols, tile_rows};
  uint32_t elem_strides[RANK] = {1, 1};
  cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, RANK, ptr,
                         global_dim, global_strides, box_dim, elem_strides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                         CU_TENSOR_MAP_L2_PROMOTION_NONE,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline int sm_count() {
  static const int count = [] {
    int device = 0, n = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device);
    return n;
  }();
  return count;
}

// Clusters resident at once. One CTA fills an SM - the kernel takes 159-222 KB
// of the SM's 228 KB and all 512 TMEM columns - so this is arithmetic rather
// than an occupancy query. Checked against cudaOccupancyMaxActiveClusters for
// every cluster size the registry builds (1 and 2 CTAs) and for the smallest
// and largest shared-memory footprints: identical in every case. A cluster of
// more than 2 CTAs would need that checked again.
inline int wave_clusters(int cluster_ctas) {
  return sm_count() / cluster_ctas;
}


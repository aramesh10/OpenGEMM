#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "common/abi.h"

namespace opengemm {

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline thread_local char error_text[256];

inline void set_error(const char *what, const char *detail) {
  snprintf(error_text, sizeof error_text, "%s: %s", what,
           detail ? detail : "unknown error");
}

inline int begin_launch(int32_t row, int32_t rows, int32_t device,
                        const char *lister) {
  if (row < 0 || row >= rows) {
    set_error("row is not a kernel this build compiled", lister);
    return OG_ERR_BAD_ROW;
  }
  int current = -1;
  if (cudaGetDevice(&current) != cudaSuccess) {
    set_error("cudaGetDevice", "failed");
    return OG_ERR_LAUNCH;
  }
  if (current == device)
    return OG_OK;
  const cudaError_t moved = cudaSetDevice(device);
  if (moved != cudaSuccess) {
    set_error("cudaSetDevice", cudaGetErrorString(moved));
    return OG_ERR_LAUNCH;
  }
  return OG_OK;
}

template <auto KERNEL, class G>
void configure_kernel(int device) {
  static std::atomic<uint64_t> configured{0};
  const uint64_t bit = uint64_t{1} << (device & 63);
  if (configured.fetch_or(bit) & bit)
    return;
  if constexpr (G::cluster_ctas > 8)
    cudaFuncSetAttribute(KERNEL, cudaFuncAttributeNonPortableClusterSizeAllowed,
                         1);
  cudaFuncSetAttribute(KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       G::smem_bytes);
}

}

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/abi.h"

namespace opengemm {

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline thread_local char error_text[256];

// Record why a call failed and return its code.
template <class... Args>
inline int fail(int code, const char *format, Args... args) {
  snprintf(error_text, sizeof error_text, format, args...);
  return code;
}

inline int check_tmap(CUresult status, const char *which) {
  if (status == CUDA_SUCCESS)
    return OG_OK;
  const char *name = nullptr;
  cuGetErrorName(status, &name);
  return fail(OG_ERR_TENSORMAP, "cuTensorMapEncodeTiled failed for %s: %s",
              which, name ? name : "unknown error");
}

// This library carries its own CUDA runtime state, whose current device
// starts at 0, so a caller on any other device has to be followed.
inline int begin_launch(int32_t row, int32_t rows, int32_t device) {
  if (row < 0 || row >= rows)
    return fail(OG_ERR_LAUNCH, "row %d is not a compiled kernel: bind the OgGemm first", row);
  int current = -1;
  if (cudaGetDevice(&current) != cudaSuccess)
    return fail(OG_ERR_LAUNCH, "cudaGetDevice failed");
  if (current == device)
    return OG_OK;
  const cudaError_t moved = cudaSetDevice(device);
  if (moved != cudaSuccess)
    return fail(OG_ERR_LAUNCH, "cudaSetDevice: %s", cudaGetErrorString(moved));
  return OG_OK;
}

template <auto KERNEL, class G>
void configure_kernel(int device) {
  static std::atomic<uint64_t> configured{0};
  const uint64_t bit = uint64_t{1} << (device & 63);
  if (configured.fetch_or(bit) & bit)
    return;
  if constexpr (G::cluster_ctas > 8)
    cudaFuncSetAttribute(KERNEL, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaFuncSetAttribute(KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize, G::smem_bytes);
}

}

// What both host sides do around a launch: report a failure, get onto the
// caller's device, and tell the driver what the kernel needs. Nothing here
// knows about torch or Python.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "common/abi.h"

namespace opengemm {

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// One buffer per thread, read back through og_<impl>_error(). Only a driver or
// runtime failure fills it; a row that names no compiled kernel says so
// through its status, and the caller has the registry to explain it with.
inline thread_local char error_text[256];

inline void set_error(const char *what, const char *detail) {
  snprintf(error_text, sizeof error_text, "%s: %s", what,
           detail ? detail : "unknown error");
}

// The two things every og_<impl>_launch() settles before it dispatches: that
// the row names a kernel this build compiled, and that the runtime is on the
// device the caller's pointers live on. Returns OG_OK, or the status to hand
// back; `lister` names the entry point that lists the compiled rows.
inline int begin_launch(int32_t row, int32_t rows, int32_t device,
                        const char *lister) {
  if (row < 0 || row >= rows) {
    set_error("row is not a kernel this build compiled", lister);
    return OG_ERR_BAD_ROW;
  }
  // A statically linked runtime keeps its own current device, so the caller's
  // ordinal has to be applied rather than assumed to be the one torch set.
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

// Once per instantiation per device: the driver has to be told this kernel
// wants more than the default 48 KB of shared memory, and a cluster of more
// than 8 CTAs is non-portable and has to be opted into. Both attributes are
// per device, so a process that launches on a second one has to set them again
// there; a single flag would configure whichever device happened to be current
// first and leave the rest to fail the launch.
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

}  // namespace opengemm

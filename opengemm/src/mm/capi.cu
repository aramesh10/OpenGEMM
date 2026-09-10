#include "capi.h"
#include "launch.cuh"

using namespace opengemm_mm;

extern "C" {

OG_API int32_t     og_mm_abi(void)     { return OG_ABI_VERSION; }
OG_API const char *og_mm_error(void)   { return error_text; }
OG_API int32_t     og_mm_kernels(void) { return KERNELS; }

OG_API int32_t og_mm_kernel(int32_t i, OgGemm *out) {
  if (i < 0 || i >= KERNELS)
    return opengemm::fail(OG_ERR_BIND, "kernel %d is out of range; og_mm_kernels() is the count", i);
  mm_registry::describe(POLICIES[i], out);
  out->row = i;
  return OG_OK;
}

OG_API int32_t og_mm_bind(OgGemm *g) { return bind(g); }

OG_API int32_t og_mm_launch(const OgGemm *g, const void *a, const void *b,
                             const void *sfa, const void *sfb, void *c,
                             int32_t device, void *stream) {
  if (const int bad = opengemm::begin_launch(g->row, KERNELS, device))
    return bad;
  return LAUNCHERS[g->row](g, const_cast<void *>(a), const_cast<void *>(b),
                           const_cast<void *>(sfa), const_cast<void *>(sfb), c,
                           device, static_cast<cudaStream_t>(stream));
}

}

// The C surface of the dense kernels. This is the translation unit that
// compiles every kernel registry.cuh names; everything above it is Python.
#include "capi.h"
#include "launch.cuh"

namespace {
using namespace opengemm_mm;
}  // namespace

extern "C" {

OG_API int32_t og_mm_abi_version(void) { return OG_MM_ABI_VERSION; }
OG_API int32_t og_mm_config_bytes(void) { return sizeof(OgMmConfig); }
OG_API int32_t og_mm_shape_bytes(void) { return sizeof(OgMmShape); }

OG_API const char *og_mm_elem_names(void) { return ELEM_NAMES; }
OG_API const char *og_mm_registry_fields(void) { return REGISTRY_FIELDS; }
OG_API int32_t og_mm_registry_rows(void) { return REGISTRY_ROWS; }
OG_API int32_t og_mm_registry_cols(void) { return REGISTRY_COLS; }
OG_API const int32_t *og_mm_registry_data(void) { return REGISTRY.data(); }

OG_API const char *og_mm_error(void) { return error_text; }

OG_API int32_t og_mm_launch(const OgMmConfig *config, const OgMmShape *shape,
                     const void *a, const void *b, void *c, int32_t device,
                     void *stream) {
  // A statically linked runtime keeps its own current device, so the caller's
  // ordinal has to be applied rather than assumed to be the one torch set.
  int current = -1;
  if (cudaGetDevice(&current) != cudaSuccess) {
    set_error("cudaGetDevice", "failed");
    return OG_ERR_LAUNCH;
  }
  if (current != device) {
    const cudaError_t moved = cudaSetDevice(device);
    if (moved != cudaSuccess) {
      set_error("cudaSetDevice", cudaGetErrorString(moved));
      return OG_ERR_LAUNCH;
    }
  }

  int status = OG_OK;
  const bool matched = launch_from_registry(
      mm_registry::All{}, policy_of(*config), status, const_cast<void *>(a),
      const_cast<void *>(b), c, shape->m, shape->n, shape->k, shape->a_pitch,
      shape->b_pitch, config->supergroup, config->walk, config->splits,
      config->l2_promo, device, static_cast<cudaStream_t>(stream));
  if (!matched) {
    set_error("no compiled configuration matches this request",
              "og_mm_registry_data() lists what this build compiles");
    return OG_ERR_NO_POLICY;
  }
  return status;
}

}  // extern "C"

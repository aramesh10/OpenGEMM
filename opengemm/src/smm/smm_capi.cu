// The C surface of the block-scaled kernels. This is the translation unit that
// compiles every kernel registry.cuh names; everything above it is Python.
#include "capi.h"
#include "launch.cuh"

namespace {
using namespace opengemm_smm;
}  // namespace

extern "C" {

OG_API int32_t og_smm_abi_version(void) { return OG_SMM_ABI_VERSION; }
OG_API int32_t og_smm_config_bytes(void) { return sizeof(OgSmmConfig); }
OG_API int32_t og_smm_shape_bytes(void) { return sizeof(OgSmmShape); }

OG_API const char *og_smm_elem_names(void) { return ELEM_NAMES; }
OG_API const char *og_smm_sf_names(void) { return SF_NAMES; }
OG_API const char *og_smm_registry_fields(void) { return REGISTRY_FIELDS; }
OG_API int32_t og_smm_registry_rows(void) { return REGISTRY_ROWS; }
OG_API int32_t og_smm_registry_cols(void) { return REGISTRY_COLS; }
OG_API const int32_t *og_smm_registry_data(void) { return REGISTRY.data(); }

OG_API const char *og_smm_error(void) { return error_text; }

OG_API int32_t og_smm_launch(const OgSmmConfig *config,
                             const OgSmmShape *shape, const void *a,
                             const void *b, const void *sfa, const void *sfb,
                             void *c, int32_t device, void *stream) {
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
      smm_registry::All{}, policy_of(*config), status, const_cast<void *>(a),
      const_cast<void *>(b), const_cast<void *>(sfa), const_cast<void *>(sfb),
      c, shape->m, shape->n, shape->k, config->supergroup, config->epi_direct,
      config->persistent, device, static_cast<cudaStream_t>(stream));
  if (!matched) {
    set_error("no compiled configuration matches this request",
              "og_smm_registry_data() lists what this build compiles");
    return OG_ERR_NO_POLICY;
  }
  return status;
}

}  // extern "C"

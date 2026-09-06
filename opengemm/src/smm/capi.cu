// The C surface of the block-scaled kernels: the translation unit that
// compiles every kernel registry.cuh names.
#include "capi.h"
#include "launch.cuh"

namespace {
using namespace opengemm_smm;
constexpr const char *ROW_LISTER = "og_smm_registry_data() lists the ones "
                                  "that are";
}  // namespace

extern "C" {

OG_API int32_t og_smm_abi_version(void) { return OG_SMM_ABI_VERSION; }

OG_API const char    *og_smm_enums(void)           { return ENUMS;           }
OG_API const char    *og_smm_registry_fields(void) { return REGISTRY_FIELDS; }
OG_API       int32_t  og_smm_registry_rows(void)   { return REGISTRY_ROWS;   }
OG_API const int32_t *og_smm_registry_data(void)   { return REGISTRY.data(); }
OG_API const char    *og_smm_error(void)           { return error_text;      }

OG_API int32_t og_smm_launch(int32_t row, const void *a, const void *b,
                             const void *sfa, const void *sfb, void *c,
                             int32_t m, int32_t n, int32_t k,
                             int32_t supergroup, int32_t epi_direct,
                             int32_t persistent, int32_t device,
                             void *stream) {
  if (const int bad =
          opengemm::begin_launch(row, REGISTRY_ROWS, device, ROW_LISTER))
    return bad;
  return LAUNCHERS[row](const_cast<void *>(a), const_cast<void *>(b),
                        const_cast<void *>(sfa), const_cast<void *>(sfb), c, m,
                        n, k, supergroup, epi_direct, persistent, device,
                        static_cast<cudaStream_t>(stream));
}

}  // extern "C"

// The C surface of the dense kernels: the translation unit that compiles every
// kernel registry.cuh names.
#include "capi.h"
#include "launch.cuh"

namespace {
using namespace opengemm_mm;
constexpr const char *ROW_LISTER = "og_mm_registry_data() lists the ones "
                                   "that are";
}  // namespace

extern "C" {

OG_API int32_t og_mm_abi_version(void) { return OG_MM_ABI_VERSION; }

OG_API const char    *og_mm_enums(void)           { return ENUMS;           }
OG_API const char    *og_mm_registry_fields(void) { return REGISTRY_FIELDS; }
OG_API       int32_t  og_mm_registry_rows(void)   { return REGISTRY_ROWS;   }
OG_API const int32_t *og_mm_registry_data(void)   { return REGISTRY.data(); }
OG_API const char    *og_mm_error(void)           { return error_text;      }

OG_API int32_t og_mm_launch(int32_t row, const void *a, const void *b, void *c,
                            int32_t m, int32_t n, int32_t k, int32_t a_pitch,
                            int32_t b_pitch, int32_t supergroup, int32_t walk,
                            int32_t splits, int32_t l2_promo, int32_t device,
                            void *stream) {
  if (const int bad =
          opengemm::begin_launch(row, REGISTRY_ROWS, device, ROW_LISTER))
    return bad;
  return LAUNCHERS[row](const_cast<void *>(a), const_cast<void *>(b), c, m, n,
                        k, a_pitch, b_pitch, supergroup, walk, splits,
                        l2_promo, device, static_cast<cudaStream_t>(stream));
}

}  // extern "C"

#pragma once

#include "common/abi.h"

#define OG_MM_ABI_VERSION 2

#ifdef __cplusplus
extern "C" {
#endif

OG_API       int32_t  og_mm_abi_version(void);

OG_API const char    *og_mm_enums(void);

OG_API const char    *og_mm_registry_fields(void);
OG_API       int32_t  og_mm_registry_rows(void);
OG_API const int32_t *og_mm_registry_data(void);

OG_API const char    *og_mm_error(void);

OG_API       int32_t  og_mm_launch(int32_t row, const void *a, const void *b,
                                   void *c, int32_t m, int32_t n, int32_t k,
                                   int32_t a_pitch, int32_t b_pitch,
                                   int32_t supergroup, int32_t walk,
                                   int32_t splits, int32_t l2_promo,
                                   int32_t device, void *stream);

#ifdef __cplusplus
}
#endif

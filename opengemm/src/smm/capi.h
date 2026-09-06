#pragma once

#include "common/abi.h"

#define OG_SMM_ABI_VERSION 2

#ifdef __cplusplus
extern "C" {
#endif

OG_API       int32_t  og_smm_abi_version(void);

OG_API const char    *og_smm_enums(void);

OG_API const char    *og_smm_registry_fields(void);
OG_API       int32_t  og_smm_registry_rows(void);
OG_API const int32_t *og_smm_registry_data(void);

OG_API const char    *og_smm_error(void);

OG_API       int32_t  og_smm_launch(int32_t row, const void *a, const void *b,
                                    const void *sfa, const void *sfb, void *c,
                                    int32_t m, int32_t n, int32_t k,
                                    int32_t supergroup, int32_t epi_direct,
                                    int32_t persistent, int32_t device,
                                    void *stream);

#ifdef __cplusplus
}
#endif

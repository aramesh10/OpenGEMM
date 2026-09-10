#pragma once

#include "common/abi.h"

#ifdef __cplusplus
extern "C" {
#endif

OG_API       int32_t  og_smm_abi(void);
OG_API const char    *og_smm_error(void);
OG_API       int32_t  og_smm_kernels(void);
OG_API       int32_t  og_smm_kernel(int32_t i, OgGemm *out);
OG_API       int32_t  og_smm_bind(OgGemm *g);
OG_API       int32_t  og_smm_launch(const OgGemm *g,
                                    const void *a, const void *b,
                                    const void *sfa, const void *sfb,
                                    void *c, int32_t device, void *stream);

#ifdef __cplusplus
}
#endif

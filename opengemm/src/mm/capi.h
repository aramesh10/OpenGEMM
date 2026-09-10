#pragma once

#include "common/abi.h"

#ifdef __cplusplus
extern "C" {
#endif

OG_API       int32_t  og_mm_abi(void);
OG_API const char    *og_mm_error(void);
OG_API       int32_t  og_mm_kernels(void);
OG_API       int32_t  og_mm_kernel(int32_t i, OgGemm *out);
OG_API       int32_t  og_mm_bind(OgGemm *g);
OG_API       int32_t  og_mm_launch(const OgGemm *g,
                                    const void *a, const void *b,
                                    const void *sfa, const void *sfb,
                                    void *c, int32_t device, void *stream);

#ifdef __cplusplus
}
#endif

// The C surface of the block-scaled kernels: no torch, no pybind, no CUDA
// headers, so a caller binds it with ctypes and the library outlives any one
// Python or PyTorch build. Which configuration a shape runs is decided in
// Python.
#pragma once

#include "common/abi.h"

#define OG_SMM_ABI_VERSION 2

#ifdef __cplusplus
extern "C" {
#endif

OG_API       int32_t  og_smm_abi_version(void);

// Every enumerated registry column and its spellings, as
// "<column>=<name>,<name>,...;<column>=...", so a caller decodes a row and
// checks its own table against this one.
OG_API const char    *og_smm_enums(void);

// Every configuration this build compiles, as rows of int32 named by
// og_smm_registry_fields(). The data is static and outlives the call. A row's
// position is the `row` og_smm_launch() takes: row i is kernel i.
OG_API const char    *og_smm_registry_fields(void);
OG_API       int32_t  og_smm_registry_rows(void);
OG_API const int32_t *og_smm_registry_data(void);

// The last failure on this thread, for a status other than OG_OK.
OG_API const char    *og_smm_error(void);

// Enqueue C[m, n] = A[m, k] @ B[n, k].T with per-block scales, using the
// kernel at `row`, on `stream`, owned by `device`. K is in values; the
// block-scaled kernel reads tightly packed rows, so it takes no pitch. The
// caller owns every pointer and has allocated C.
OG_API       int32_t  og_smm_launch(int32_t row, const void *a, const void *b,
                                    const void *sfa, const void *sfb, void *c,
                                    int32_t m, int32_t n, int32_t k,
                                    int32_t supergroup, int32_t epi_direct,
                                    int32_t persistent, int32_t device,
                                    void *stream);

#ifdef __cplusplus
}  // extern "C"
#endif

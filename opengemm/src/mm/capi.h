// The C surface of the dense kernels: no torch, no pybind, no CUDA headers, so
// a caller binds it with ctypes and the library outlives any one Python or
// PyTorch build. Which configuration a shape runs is still decided in Python.
#pragma once

#include "common/abi.h"

#define OG_MM_ABI_VERSION 2

#ifdef __cplusplus
extern "C" {
#endif

OG_API       int32_t  og_mm_abi_version(void);

// Every enumerated registry column and its spellings, as
// "<column>=<name>,<name>,...;<column>=...", so a caller decodes a row and
// checks its own table against this one.
OG_API const char    *og_mm_enums(void);

// Every configuration this build compiles, as rows of int32 named by
// og_mm_registry_fields(). The data is static and outlives the call. A row's
// position is the `row` og_mm_launch() takes: row i is kernel i.
OG_API const char    *og_mm_registry_fields(void);
OG_API       int32_t  og_mm_registry_rows(void);
OG_API const int32_t *og_mm_registry_data(void);

// The last failure on this thread, for a status other than OG_OK.
OG_API const char    *og_mm_error(void);

// Enqueue C[m, n] = A[m, k] @ B[n, k].T with the kernel at `row` on `stream`,
// owned by `device`. K and the pitches are in values, not bytes; a packed
// operand's row extent is narrower than its K. The caller owns every pointer
// and has already allocated and, when the kernel accumulates, zeroed C.
OG_API       int32_t  og_mm_launch(int32_t row, const void *a, const void *b,
                                   void *c, int32_t m, int32_t n, int32_t k,
                                   int32_t a_pitch, int32_t b_pitch,
                                   int32_t supergroup, int32_t walk,
                                   int32_t splits, int32_t l2_promo,
                                   int32_t device, void *stream);

#ifdef __cplusplus
}  // extern "C"
#endif

// The C surface of the dense kernels: no torch, no pybind, no CUDA headers, so
// a caller binds it with ctypes and the compiled library outlives any one
// Python or PyTorch build. Which configuration a shape runs is still decided
// in Python; this is the boundary that decision crosses.
#pragma once

#include <stdint.h>

#define OG_MM_ABI_VERSION 1

// The library is built with -fvisibility=hidden so that the inline helpers in
// host_utils.cuh, which the block-scaled build declares with other signatures,
// cannot collide if both libraries are ever loaded into one namespace. That
// hides these too unless they say otherwise.
#if defined(_WIN32)
#define OG_API
#else
#define OG_API __attribute__((visibility("default")))
#endif

enum {
  OG_OK             =  0,
  OG_ERR_NO_POLICY  = -1,  // the config names no kernel this build compiled
  OG_ERR_TENSORMAP  = -2,  // cuTensorMapEncodeTiled rejected an operand
  OG_ERR_LAUNCH     = -3,  // the launch or the device change failed
};

// Policy as src/mm/types.cuh declares it, followed by the launch arguments the
// kernel reads. Every field is int32 so the struct has no padding and Python
// can check its own view against og_mm_config_bytes(). The elements are Elem
// ordinals, which og_mm_elem_names() spells out.
typedef struct OgMmConfig {
  int32_t elem_a, elem_b;
  int32_t cta_group;                        // 1 or 2, not the harness use_2cta
  int32_t block_m, mma_n, stages;
  int32_t swap_ab, epi_hold, epi_mode, epi_direct, use_clc, split_k;
  int32_t rm, rn, rk;                       // the cluster past the CTA group
  int32_t supergroup, walk, l2_promo;
  int32_t splits;                           // already clamped to the K tiles
} OgMmConfig;

// K and the pitches are in values, not bytes; a packed operand's row extent is
// narrower than its K.
typedef struct OgMmShape {
  int32_t m, n, k, a_pitch, b_pitch;
} OgMmShape;

#ifdef __cplusplus
extern "C" {
#endif

OG_API int32_t        og_mm_abi_version(void);
OG_API int32_t        og_mm_config_bytes(void);
OG_API int32_t        og_mm_shape_bytes(void);

// "bf16,f16,tf32,..." in Elem order, so a caller checks its own table rather
// than trusting that the two were edited together.
OG_API const char    *og_mm_elem_names(void);

// Every configuration this build compiles, as rows of int32 in the harness
// vocabulary. The names are og_mm_registry_fields(), comma separated; the data
// is static and outlives the call.
OG_API const char    *og_mm_registry_fields(void);
OG_API int32_t        og_mm_registry_rows(void);
OG_API int32_t        og_mm_registry_cols(void);
OG_API const int32_t *og_mm_registry_data(void);

// The last failure on this thread, for a status other than OG_OK.
OG_API const char    *og_mm_error(void);

// Enqueue C[m, n] = A[m, k] @ B[n, k].T on `stream`, owned by `device`.
// The caller owns every pointer and has already allocated and, when the
// configuration accumulates, zeroed C.
OG_API int32_t        og_mm_launch(const OgMmConfig *config,
                                   const OgMmShape *shape, const void *a,
                                   const void *b, void *c, int32_t device,
                                   void *stream);

#ifdef __cplusplus
}  // extern "C"
#endif

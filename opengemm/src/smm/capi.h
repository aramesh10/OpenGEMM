// The C surface of the block-scaled kernels: no torch, no pybind, no CUDA
// headers, so a caller binds it with ctypes and the compiled library outlives
// any one Python or PyTorch build. Which configuration a shape runs is still
// decided in Python; this is the boundary that decision crosses.
#pragma once

#include <stdint.h>

#define OG_SMM_ABI_VERSION 1

#ifndef OG_OK
enum {
  OG_OK            =  0,
  OG_ERR_NO_POLICY = -1,  // the config names no kernel this build compiled
  OG_ERR_TENSORMAP = -2,  // cuTensorMapEncodeTiled rejected an operand
  OG_ERR_LAUNCH    = -3,  // the launch or the device change failed
};
#endif

// The library is built with -fvisibility=hidden so that the inline helpers in
// host_utils.cuh, which the dense build declares with other signatures, cannot
// collide if both libraries are ever loaded into one namespace. That hides
// these too unless they say otherwise.
#if defined(_WIN32)
#define OG_API
#else
#define OG_API __attribute__((visibility("default")))
#endif

// Policy as src/smm/types.cuh declares it, followed by the launch arguments.
// Every field is int32 so the struct has no padding and Python can check its
// own view against og_smm_config_bytes(). The elements are SElem and SFElem
// ordinals, which og_smm_elem_names() and og_smm_sf_names() spell out; the two
// enumerations do not share an order with the dense Elem.
typedef struct OgSmmConfig {
  int32_t elem_a, elem_b, elem_sf;
  int32_t cta_group;                        // 1 or 2, not the harness use_2cta
  int32_t mma_n;
  int32_t swap_ab, epi_trade, deep, use_clc;
  int32_t rm, rn, rk;                       // the cluster past the CTA group
  int32_t supergroup, epi_direct, persistent;
} OgSmmConfig;

// K is in values. The block-scaled kernel reads tightly packed rows, so it
// takes no pitch.
typedef struct OgSmmShape {
  int32_t m, n, k;
} OgSmmShape;

#ifdef __cplusplus
extern "C" {
#endif

OG_API int32_t        og_smm_abi_version(void);
OG_API int32_t        og_smm_config_bytes(void);
OG_API int32_t        og_smm_shape_bytes(void);

// "e4m3,e5m2,..." in SElem order and "ue4m3,ue8m0" in SFElem order, so a
// caller checks its own table rather than trusting that the two were edited
// together.
OG_API const char    *og_smm_elem_names(void);
OG_API const char    *og_smm_sf_names(void);

// Every configuration this build compiles, as rows of int32 in the harness
// vocabulary. The names are og_smm_registry_fields(), comma separated; the
// data is static and outlives the call.
OG_API const char    *og_smm_registry_fields(void);
OG_API int32_t        og_smm_registry_rows(void);
OG_API int32_t        og_smm_registry_cols(void);
OG_API const int32_t *og_smm_registry_data(void);

// The last failure on this thread, for a status other than OG_OK.
OG_API const char    *og_smm_error(void);

// Enqueue C[m, n] = A[m, k] @ B[n, k].T with per-block scales on `stream`,
// owned by `device`. The caller owns every pointer and has allocated C.
OG_API int32_t        og_smm_launch(const OgSmmConfig *config,
                                    const OgSmmShape *shape, const void *a,
                                    const void *b, const void *sfa,
                                    const void *sfb, void *c, int32_t device,
                                    void *stream);

#ifdef __cplusplus
}  // extern "C"
#endif

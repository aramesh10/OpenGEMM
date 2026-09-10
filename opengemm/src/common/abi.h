#pragma once

#include <stdint.h>

#if defined(_WIN32)
#define OG_API __declspec(dllexport)
#else
#define OG_API __attribute__((visibility("default")))
#endif

#define OG_ABI_VERSION 3

enum {
  OG_OK            =  0,
  OG_ERR_BIND      = -1,   // unknown dtype, or a K the format cannot take
  OG_ERR_NO_KERNEL = -2,   // no compiled kernel matches the configuration
  OG_ERR_TENSORMAP = -3,
  OG_ERR_LAUNCH    = -4,
};

// One GEMM: its shape, the configs.json entry that picks and drives the
// kernel, and the row og_*_bind resolved. The caller fills every field but
// row; every knob is literal, there are no defaults.
typedef struct {
  int32_t m, n, k;            // k in values, not bytes
  char    dtype[16];          // configs.json name: "bf16", "e4m3xe5m2", "nvfp4"

  // which compiled kernel
  int32_t use_2cta, output_n, use_clc, swap_ab;               // both libraries
  int32_t cluster_m, cluster_n, cluster_k;                    // both libraries
  int32_t block_m, stages, epi_hold, epi_double, epi_direct;  // mm; smm reads epi_direct at launch
  int32_t epi_trade, deep_stages;                             // smm

  // launch time
  int32_t supergroup, split_k, walk, l2_promo;   // mm; bind clamps split_k to the K tiles
  int32_t persistent;                            // smm

  int32_t row;   // set by og_*_bind
} OgGemm;

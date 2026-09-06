#pragma once

#include <stdint.h>

#if defined(_WIN32)
#define OG_API
#else
#define OG_API __attribute__((visibility("default")))
#endif

enum {
  OG_OK            =  0,
  OG_ERR_BAD_ROW   = -1,
  OG_ERR_TENSORMAP = -2,
  OG_ERR_LAUNCH    = -3,
};

// What both C surfaces share: how an entry point is exported, and what a
// status means. src/mm/capi.h and src/smm/capi.h include this and add their
// own entry points; between them there is no torch, no pybind and no CUDA
// header, so a caller binds either library with ctypes.
#pragma once

#include <stdint.h>

// The libraries are built with -fvisibility=hidden, which hides the entry
// points too unless they say otherwise.
#if defined(_WIN32)
#define OG_API
#else
#define OG_API __attribute__((visibility("default")))
#endif

enum {
  OG_OK            =  0,
  OG_ERR_BAD_ROW   = -1,  // the row is not a kernel this build compiled
  OG_ERR_TENSORMAP = -2,  // cuTensorMapEncodeTiled rejected an operand
  OG_ERR_LAUNCH    = -3,  // the launch or the device change failed
};

#pragma once

#include "common/ptx.cuh"

struct Clc {
  int    arrived;
  int    finished;
  int    ready;
  uint4 *handles;
};

template <int CTA_GROUP>
__device__ __forceinline__ void clc_issue(const Clc &clc, int tile_idx,
                                          bool is_leader) {
  const int slot  = tile_idx % CLC_DEPTH;
  const int phase = (tile_idx / CLC_DEPTH) & 1;
  mbar_arrive_tx(clc.arrived + slot * MBAR, sizeof(uint4));
  if (!is_leader) {
    mbar_arrive_cluster_to(clc.ready + slot * MBAR, 0);
    return;
  }
  if (tile_idx >= CLC_DEPTH)
    mbar_wait_cluster(clc.finished + slot * MBAR, phase ^ 1);
  if constexpr (CTA_GROUP == 2)
    mbar_wait_cluster(clc.ready + slot * MBAR, phase);
  clc_schedule(&clc.handles[slot], clc.arrived + slot * MBAR);
}

template <int CTA_GROUP, bool USE_CLC>
__device__ __forceinline__ int next_tile(const Clc &clc, int tile_id,
                                         int tile_idx, int num_clusters) {
  if constexpr (!USE_CLC)
    return tile_id + num_clusters;
  const int slot  = tile_idx % CLC_DEPTH;
  const int phase = (tile_idx / CLC_DEPTH) & 1;
  mbar_wait_cluster(clc.arrived + slot * MBAR, phase);
  const uint4 next = clc_query(&clc.handles[slot]);
  mbar_arrive_cluster_to(clc.finished + slot * MBAR, 0);
  return next.x ? static_cast<int>(next.y) / CTA_GROUP : -1;
}

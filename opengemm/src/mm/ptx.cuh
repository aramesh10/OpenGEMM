#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include "common/ptx.cuh"
#include "types.cuh"

__device__ __forceinline__ uint32_t cluster_ctaid_x() {
  uint32_t x;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(x));
  return x;
}

__device__ __forceinline__ void cluster_sync() {
  cluster_arrive();
  cluster_wait();
}

__host__ __device__ __forceinline__ int walk_sgn(int v) {
  return (v > 0) - (v < 0);
}

__host__ __device__ __forceinline__ int walk_abs(int v) {
  return v < 0 ? -v : v;
}

__host__ __device__ __forceinline__ void gilbert_d2xy(int idx, int w, int h,
                                                      int &out_x, int &out_y) {
  int x = 0, y = 0, ax, ay, bx, by;
  if (w >= h) { ax = w; ay = 0; bx = 0; by = h; }
  else        { ax = 0; ay = h; bx = w; by = 0; }

  int cur = 0;
  for (;;) {
    const int aw = walk_abs(ax + ay);
    const int ah = walk_abs(bx + by);
    const int dax = walk_sgn(ax), day = walk_sgn(ay);
    const int dbx = walk_sgn(bx), dby = walk_sgn(by);

    if (ah == 1) { out_x = x + dax * (idx - cur); out_y = y + day * (idx - cur); return; }
    if (aw == 1) { out_x = x + dbx * (idx - cur); out_y = y + dby * (idx - cur); return; }

    int ax2 = ax >> 1, ay2 = ay >> 1;
    int bx2 = bx >> 1, by2 = by >> 1;

    if (2 * aw > 3 * ah) {
      if ((walk_abs(ax2 + ay2) & 1) && aw > 2) { ax2 += dax; ay2 += day; }
      const int nxt = walk_abs(ax2 + ay2) * ah;
      if (idx - cur < nxt) { ax = ax2; ay = ay2; continue; }
      cur += nxt;
      x += ax2; y += ay2; ax -= ax2; ay -= ay2;
    } else {
      if ((walk_abs(bx2 + by2) & 1) && ah > 2) { bx2 += dbx; by2 += dby; }
      int nxt = walk_abs(bx2 + by2) * walk_abs(ax2 + ay2);
      if (idx - cur < nxt) {
        const int nax = bx2, nay = by2;
        bx = ax2; by = ay2; ax = nax; ay = nay;
        continue;
      }
      cur += nxt;
      nxt = aw * walk_abs((bx - bx2) + (by - by2));
      if (idx - cur < nxt) {
        x += bx2; y += by2; bx -= bx2; by -= by2;
        continue;
      }
      cur += nxt;
      const int nx = x + (ax - dax) + (bx2 - dbx);
      const int ny = y + (ay - day) + (by2 - dby);
      const int nbx = -(ax - ax2), nby = -(ay - ay2);
      x = nx; y = ny; ax = -bx2; ay = -by2; bx = nbx; by = nby;
    }
  }
}

__device__ __forceinline__ void tile_coords(int tile_id, int num_m, int num_n,
                                            int supergroup, int walk,
                                            int &m_idx, int &n_idx) {
  if (walk == WALK_GILBERT) {
    gilbert_d2xy(tile_id, num_m, num_n, m_idx, n_idx);
    return;
  }
  if (walk == WALK_ROW) {
    const int tpg = num_n * supergroup;
    const int grp = tile_id / tpg;
    const int first_m = grp * supergroup;
    const int ms =
        (num_m - first_m) < supergroup ? (num_m - first_m) : supergroup;
    const int idx = tile_id - grp * tpg;
    n_idx = idx / ms;
    m_idx = first_m + idx % ms;
    return;
  }

  const int tpg = num_m * supergroup;
  const int grp = tile_id / tpg;
  const int first_n = grp * supergroup;
  const int ns =
      (num_n - first_n) < supergroup ? (num_n - first_n) : supergroup;
  int idx = tile_id - grp * tpg;

  if (walk == WALK_BLOCK) {
    const int blk = idx / (supergroup * ns);
    const int first_m = blk * supergroup;
    const int ms =
        (num_m - first_m) < supergroup ? (num_m - first_m) : supergroup;
    idx -= blk * supergroup * ns;
    m_idx = first_m + idx % ms;
    n_idx = first_n + idx / ms;
    return;
  }

  m_idx = idx / ns;
  n_idx = first_n + idx % ns;
  if (walk == WALK_SERPENTINE && (grp & 1))
    m_idx = num_m - 1 - m_idx;
}

__device__ __forceinline__ void mbar_wait_ns(uint32_t addr, uint32_t phase) {
  asm volatile("{\n\t.reg .pred P;\n\tWAIT:\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
               "P, [%0], %1, 0x989680;\n\t"
               "@P bra.uni DONE;\n\t"
               "nanosleep.u32 128;\n\t"
               "bra.uni WAIT;\n\tDONE:\n\t}" ::"r"(addr), "r"(phase)
               : "memory");
}

__device__ __forceinline__ void mbar_wait_cluster_ns(int addr, int phase) {
  asm volatile("{\n\t.reg .pred P;\n\tWAIT:\n\t"
               "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
               "P, [%0], %1, 0x989680;\n\t"
               "@P bra.uni DONE;\n\t"
               "nanosleep.u32 128;\n\t"
               "bra.uni WAIT;\n\tDONE:\n\t}" ::"r"(addr), "r"(phase)
               : "memory");
}

template <int CTA_GROUP>
__device__ __forceinline__ void tma_load_2d(int dst, const void *tmap, int x,
                                            int y, int mbar, int leader = 0) {
  if constexpr (CTA_GROUP == 1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::"
        "bytes.cta_group::1 [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
        "l"(tmap), "r"(x), "r"(y), "r"(mbar)
        : "memory");
  } else {
    const uint32_t mbar0 = map_shared_to_cta(mbar, leader);
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
        "l"(tmap), "r"(x), "r"(y), "r"(mbar0)
        : "memory");
  }
}

template <int CTA_GROUP>
__device__ __forceinline__ void
tma_load_2d_multicast(int dst, const void *tmap, int x, int y, int mbar,
                      uint16_t mcast_mask, int leader = 0) {
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], "
                 "[%4], %5;" ::"r"(dst),
                 "l"(tmap), "r"(x), "r"(y), "r"(mbar), "h"(mcast_mask)
                 : "memory");
  } else {
    const uint32_t mbar0 = map_shared_to_cta(mbar, leader);
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global."
                 "mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, "
                 "{%2, %3}], [%4], %5;" ::"r"(dst),
                 "l"(tmap), "r"(x), "r"(y), "r"(mbar0), "h"(mcast_mask)
                 : "memory");
  }
}

__device__ __forceinline__ void st_shared_f32(int addr, float val) {
  asm volatile("st.shared.f32 [%0], %1;" ::"r"(addr), "f"(val) : "memory");
}

template <class T>
__device__ __forceinline__ void st_shared_b32(int addr, T val) {
  float word;
  __builtin_memcpy(&word, &val, sizeof(word));
  st_shared_f32(addr, word);
}

__device__ __forceinline__ void st_shared_v4(int addr, float a, float b,
                                             float c, float d) {
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "f"(a),
               "f"(b), "f"(c), "f"(d)
               : "memory");
}

template <class T>
__device__ __forceinline__ void st_shared_v4b(int addr, T a, T b, T c, T d) {
  float w[4];
  const T in[4] = {a, b, c, d};
  __builtin_memcpy(w, in, sizeof(w));
  st_shared_v4(addr, w[0], w[1], w[2], w[3]);
}

template <int CTA_GROUP, Kind K>
__device__ __forceinline__ void tcgen05_mma(int d, uint64_t a, uint64_t b,
                                            uint32_t i, int ena_d) {
  if constexpr (CTA_GROUP == 1 && K == Kind::f16)
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16"
                 " [%0], %1, %2, %3, p;\n\t}" ::"r"(d),
                 "l"(a), "l"(b), "r"(i), "r"(ena_d)
                 : "memory");
  else if constexpr (CTA_GROUP == 1 && K == Kind::tf32)
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::tf32"
                 " [%0], %1, %2, %3, p;\n\t}" ::"r"(d),
                 "l"(a), "l"(b), "r"(i), "r"(ena_d)
                 : "memory");
  else if constexpr (CTA_GROUP == 1 && K == Kind::i8)
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::i8"
                 " [%0], %1, %2, %3, p;\n\t}" ::"r"(d),
                 "l"(a), "l"(b), "r"(i), "r"(ena_d)
                 : "memory");
  else if constexpr (CTA_GROUP == 1 && K == Kind::f8f6f4)
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f8f6f4"
                 " [%0], %1, %2, %3, p;\n\t}" ::"r"(d),
                 "l"(a), "l"(b), "r"(i), "r"(ena_d)
                 : "memory");
  else if constexpr (CTA_GROUP == 2 && K == Kind::f16)
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::2.kind::f16"
                 " [%0], %1, %2, %3, p;\n\t}" ::"r"(d),
                 "l"(a), "l"(b), "r"(i), "r"(ena_d)
                 : "memory");
  else if constexpr (CTA_GROUP == 2 && K == Kind::tf32)
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::2.kind::tf32"
                 " [%0], %1, %2, %3, p;\n\t}" ::"r"(d),
                 "l"(a), "l"(b), "r"(i), "r"(ena_d)
                 : "memory");
  else if constexpr (CTA_GROUP == 2 && K == Kind::i8)
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::2.kind::i8"
                 " [%0], %1, %2, %3, p;\n\t}" ::"r"(d),
                 "l"(a), "l"(b), "r"(i), "r"(ena_d)
                 : "memory");
  else if constexpr (CTA_GROUP == 2 && K == Kind::f8f6f4)
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::2.kind::f8f6f4"
                 " [%0], %1, %2, %3, p;\n\t}" ::"r"(d),
                 "l"(a), "l"(b), "r"(i), "r"(ena_d)
                 : "memory");
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_commit(int mbar,
                                               uint16_t mask = 0x3) {
  if constexpr (CTA_GROUP == 1)
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
                 "cluster.b64 [%0];" ::"r"(mbar)
                 : "memory");
  else
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::"
                 "cluster.multicast::cluster.b64 [%0], %1;" ::"r"(mbar),
                 "h"(mask)
                 : "memory");
}

__device__ __forceinline__ void tcgen05_ld_32x32bx8(float *tmp, int row,
                                                    int col) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
               "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                 "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
               : "r"((row << 16) | col));
}

__device__ __forceinline__ void tcgen05_ld_32x32bx16(float *tmp, int row,
                                                     int col) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0, %1, %2, %3, %4, %5, %6, %7, "
               "%8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
               : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                 "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7]),
                 "=f"(tmp[8]), "=f"(tmp[9]), "=f"(tmp[10]), "=f"(tmp[11]),
                 "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15])
               : "r"((row << 16) | col));
}

__device__ __forceinline__ void tcgen05_ld_32x32bx32(float *tmp, int row,
                                                     int col) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
               "{%0, %1, %2, %3, %4, %5, %6, %7, "
               "%8, %9, %10, %11, %12, %13, %14, %15, "
               "%16, %17, %18, %19, %20, %21, %22, %23, "
               "%24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
               : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                 "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7]),
                 "=f"(tmp[8]), "=f"(tmp[9]), "=f"(tmp[10]), "=f"(tmp[11]),
                 "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                 "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]),
                 "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                 "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]),
                 "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31])
               : "r"((row << 16) | col));
}

#pragma once

#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int MBAR      = sizeof(int64_t);
constexpr int CLC_DEPTH = 3;

__device__ __forceinline__ constexpr uint64_t desc_enc(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4;
}

__device__ __forceinline__ uint64_t make_ab_desc(int addr) {
  constexpr uint64_t SBO = 8ULL * 128;
  return desc_enc(addr) | (desc_enc(SBO) << 32) | (1ULL << 46) | (1ULL << 62);
}

__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ void bar_sync(int bar_id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(bar_id), "r"(count) : "memory");
}

__device__ __forceinline__ uint32_t map_shared_to_cta(int addr, int dst_cta) {
  uint32_t mapped;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(mapped)
               : "r"(addr), "r"(dst_cta));
  return mapped;
}

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "elect.sync _|p, %1;\n\t"
               "@p mov.s32 %0, 1;\n\t"
               "}"
               : "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}

__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void pdl_arrive() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ void clc_schedule(uint4 *handle,
                                             int completion_mbar) {
  const uint32_t handle_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(handle));
  asm volatile(
      "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx:"
      ":bytes.multicast::cluster::all.b128 [%0], [%1];" ::"r"(handle_addr),
      "r"(completion_mbar)
      : "memory");
}

__device__ __forceinline__ uint4 clc_query(uint4 *handle) {
  uint4 result;
  const uint32_t handle_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(handle));
  asm volatile("{\n\t"
               ".reg .pred SUCCESS;\n\t"
               ".reg .b128 HANDLE;\n\t"
               "mov.u32 %1, 0;\n\t"
               "mov.u32 %2, 0;\n\t"
               "mov.u32 %3, 0;\n\t"
               "ld.shared.b128 HANDLE, [%4];\n\t"
               "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
               "SUCCESS, HANDLE;\n\t"
               "selp.u32 %0, 1, 0, SUCCESS;\n\t"
               "@!SUCCESS bra.uni DONE;\n\t"
               "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
               "{%1, %2, %3, _}, HANDLE;\n\t"
               "fence.proxy.async.shared::cta;\n\t"
               "DONE:\n\t"
               "}"
               : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
               : "r"(handle_addr)
               : "memory");
  return result;
}

__device__ __forceinline__ void mbar_init(int addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr),
               "r"(count));
}

__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;");
}

__device__ __forceinline__ void mbar_wait(uint32_t addr, uint32_t phase) {
  asm volatile("{\n"
               "  .reg .pred p;\n"

               "wait:\n"
               "  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
               "    p, [%0], %1, 0x989680;\n"

               "  @p bra done;\n"
               "  bra wait;\n"

               "done:\n"
               "}\n"
               :
               : "r"(addr), "r"(phase));
}

__device__ __forceinline__ void mbar_wait_cluster(int addr, int phase) {
  asm volatile("{\n\t.reg .pred P;\n\t"
               "WAIT: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
               "P, [%0], %1;\n\t"
               "@P bra.uni DONE;\n\tbra.uni WAIT;\n\tDONE:\n\t}" ::"r"(addr),
               "r"(phase)
               : "memory");
}

__device__ __forceinline__ void mbar_arrive_tx(int addr, int bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
          "r"(addr),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void mbar_arrive_cluster_to(int mbar, int dst_cta) {
  const uint32_t m = map_shared_to_cta(mbar, dst_cta);
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(m)
               : "memory");
}

__device__ __forceinline__ void prefetch_tensormap(const void *tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

__device__ __forceinline__ void
tma_store_2d(int smem_int_ptr, const void *tmap_ptr, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, "
               "{%2, %3}], [%1];" ::"l"(tmap_ptr),
               "r"(smem_int_ptr), "r"(x), "r"(y)
               : "memory");
}

__device__ __forceinline__ void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template <int N> __device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

template <int CTA_GROUP, int COLS>
__device__ __forceinline__ void tmem_alloc(int smem_addr) {
  if constexpr (CTA_GROUP == 1)
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(smem_addr),
        "r"(COLS));
  else
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(smem_addr),
        "r"(COLS));
}

template <int CTA_GROUP, int COLS>
__device__ __forceinline__ void tmem_dealloc(int base_col) {
  if constexpr (CTA_GROUP == 1)
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(base_col),
        "r"(COLS));
  else
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(base_col),
        "r"(COLS));
}

__device__ __forceinline__ void tmem_fence() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void tmem_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}
__device__ __forceinline__ void tmem_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ void tcgen05_ld_32x32bx64(float *tmp, int row,
                                                     int col) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 "
               "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,"
               "  %8,  %9, %10, %11, %12, %13, %14, %15,"
               " %16, %17, %18, %19, %20, %21, %22, %23,"
               " %24, %25, %26, %27, %28, %29, %30, %31,"
               " %32, %33, %34, %35, %36, %37, %38, %39,"
               " %40, %41, %42, %43, %44, %45, %46, %47,"
               " %48, %49, %50, %51, %52, %53, %54, %55,"
               " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
               : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                 "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7]),
                 "=f"(tmp[8]), "=f"(tmp[9]), "=f"(tmp[10]), "=f"(tmp[11]),
                 "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                 "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]),
                 "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                 "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]),
                 "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
                 "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]),
                 "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
                 "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]),
                 "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
                 "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]),
                 "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
                 "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]),
                 "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63])
               : "r"((row << 16) | col));
}

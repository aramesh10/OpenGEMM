/* CUTLASS reference GEMMs for the formats torch has no kernel for.

   scripts/benchmark.py compares OpenGEMM against cuBLAS wherever torch exposes
   a kernel for the format. For u8, e5m2, e3m2, e2m3, e2m1 and mxfp4 it does
   not, so this library stands in: a CUTLASS 4.x sm_100a GEMM per format, in a
   handful of tile shapes each, behind the same launch-and-time interface.

   Operand layout matches OpenGEMM's: A is (M, K) and B is (N, K), both with K
   contiguous, and D is (M, N) row major. D = A x B^T, no C and no epilogue.

   The tile shapes are a small fixed menu, not a tuned choice; the Python side
   times each one and reports the best, which is the fairest thing to compare a
   tuned OpenGEMM configuration against without running a CUTLASS sweep.
*/

#include <cstdint>
#include <string>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

// The library is compiled with -fvisibility=hidden, like the kernel libraries.
#define OGREF_API __attribute__((visibility("default")))

using namespace cute;

namespace {

constexpr int kAbiVersion = 1;
std::string g_error;

int fail(const char *what, cutlass::Status status) {
  g_error = std::string(what) + ": " + cutlass::cutlassGetStatusString(status);
  return -1;
}

/* One GEMM type: the element types, an MMA tile and a cluster.

   Both collectives take the same operator class, which is what selects the
   block-scaled mainloop for the mx/nv formats. */
template <class ElementA_, class ElementB_, class ElementD_, class ElementAcc,
          class ElementCompute, class OpClass, class MmaTile, class Cluster, int AlignA,
          int AlignB>
struct Spec {
  using LayoutA = cutlass::layout::RowMajor;     // A is (M, K), K contiguous
  using LayoutB = cutlass::layout::ColumnMajor;  // B is (N, K), K contiguous
  using LayoutD = cutlass::layout::RowMajor;     // D is (M, N), N contiguous
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD_>::value;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, OpClass, MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementCompute, void,
      LayoutD, AlignD, ElementD_, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, OpClass, ElementA_, LayoutA, AlignA, ElementB_, LayoutB, AlignB,
      ElementAcc, MmaTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using Compute = ElementCompute;
};

/* A Spec's kernel plus the state one launch needs.

   The shape is initialized once and only the operand pointers change after
   that, so rotating through input buffers costs a host-side `update` and not a
   full re-initialization. */
template <class S, bool Scaled>
struct Runner {
  using Gemm = typename S::Gemm;
  using Kernel = typename Gemm::GemmKernel;

  Gemm gemm;
  void *workspace = nullptr;
  size_t workspace_bytes = 0;
  int m = 0, n = 0, k = 0;
  bool ready = false;

  static typename Gemm::Arguments arguments(int m, int n, int k, const void *a, const void *b,
                                            const void *sfa, const void *sfb, void *d) {
    using ElementA = typename Kernel::ElementA;
    using ElementB = typename Kernel::ElementB;
    using ElementD = typename Kernel::ElementD;
    using Compute = typename S::Compute;

    auto stride_a = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, 1});
    auto stride_b = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, 1});
    auto stride_c = cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, {m, n, 1});
    auto stride_d = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m, n, 1});

    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1}, {}, {}};
    if constexpr (Scaled) {
      using Config = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
      using ElementSF = typename Kernel::CollectiveMainloop::ElementSF;
      args.mainloop = {reinterpret_cast<const ElementA *>(a),
                       stride_a,
                       reinterpret_cast<const ElementB *>(b),
                       stride_b,
                       reinterpret_cast<const ElementSF *>(sfa),
                       Config::tile_atom_to_shape_SFA(make_shape(m, n, k, 1)),
                       reinterpret_cast<const ElementSF *>(sfb),
                       Config::tile_atom_to_shape_SFB(make_shape(m, n, k, 1))};
    } else {
      args.mainloop = {reinterpret_cast<const ElementA *>(a), stride_a,
                       reinterpret_cast<const ElementB *>(b), stride_b};
    }
    args.epilogue = {{Compute(1), Compute(0)},
                     nullptr,
                     stride_c,
                     reinterpret_cast<ElementD *>(d),
                     stride_d};
    return args;
  }

  int run(int m, int n, int k, const void *a, const void *b, const void *sfa, const void *sfb,
          void *d, cudaStream_t stream) {
    auto args = arguments(m, n, k, a, b, sfa, sfb, d);
    if (!ready || m != this->m || n != this->n || k != this->k) {
      cutlass::Status status = gemm.can_implement(args);
      if (status != cutlass::Status::kSuccess) return fail("can_implement", status);
      size_t bytes = Gemm::get_workspace_size(args);
      if (bytes > workspace_bytes) {
        cudaFree(workspace);
        if (cudaMalloc(&workspace, bytes) != cudaSuccess) {
          workspace = nullptr;
          workspace_bytes = 0;
          g_error = "could not allocate the CUTLASS workspace";
          return -1;
        }
        workspace_bytes = bytes;
      }
      status = gemm.initialize(args, workspace, stream);
      if (status != cutlass::Status::kSuccess) return fail("initialize", status);
      this->m = m, this->n = n, this->k = k;
      ready = true;
    } else {
      cutlass::Status status = gemm.update(args);
      if (status != cutlass::Status::kSuccess) return fail("update", status);
    }
    cutlass::Status status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) return fail("run", status);
    return 0;
  }
};

template <class S, bool Scaled>
int launch(int m, int n, int k, const void *a, const void *b, const void *sfa, const void *sfb,
           void *d, cudaStream_t stream) {
  static Runner<S, Scaled> runner;
  return runner.run(m, n, k, a, b, sfa, sfb, d, stream);
}

using LaunchFn = int (*)(int, int, int, const void *, const void *, const void *, const void *,
                         void *, cudaStream_t);

struct Entry {
  const char *dtype;
  const char *tile;  // how the Python side names this configuration
  LaunchFn launch;
};

// The tile menu: a 2-SM and a 1-SM shape, in the K tile each MMA wants.
using Tile256x256 = Shape<_256, _256, _128>;
using Tile256x128 = Shape<_256, _128, _128>;
using Tile128x256 = Shape<_128, _256, _128>;
using Scaled256x256 = Shape<_256, _256, _256>;
using Scaled256x128 = Shape<_256, _128, _256>;
using Scaled128x256 = Shape<_128, _256, _256>;
using Cluster2 = Shape<_2, _1, _1>;
using Cluster1 = Shape<_1, _1, _1>;

#define OGREF_DENSE(name, ea, eb, ed, acc, compute, align)                                    \
  {name, "256x256x128_2sm",                                                                   \
   launch<Spec<ea, eb, ed, acc, compute, cutlass::arch::OpClassTensorOp, Tile256x256,          \
               Cluster2, align, align>,                                                        \
          false>},                                                                             \
      {name, "256x128x128_2sm",                                                                \
       launch<Spec<ea, eb, ed, acc, compute, cutlass::arch::OpClassTensorOp, Tile256x128,      \
                   Cluster2, align, align>,                                                    \
              false>},                                                                         \
      {name, "128x256x128_1sm",                                                                \
       launch<Spec<ea, eb, ed, acc, compute, cutlass::arch::OpClassTensorOp, Tile128x256,      \
                   Cluster1, align, align>,                                                    \
              false>}

#define OGREF_SCALED(name, ea, eb, ed, align)                                                  \
  {name, "256x256x256_2sm",                                                                    \
   launch<Spec<ea, eb, ed, float, float, cutlass::arch::OpClassBlockScaledTensorOp,            \
               Scaled256x256, Cluster2, align, align>,                                          \
          true>},                                                                               \
      {name, "256x128x256_2sm",                                                                 \
       launch<Spec<ea, eb, ed, float, float, cutlass::arch::OpClassBlockScaledTensorOp,        \
                   Scaled256x128, Cluster2, align, align>,                                       \
              true>},                                                                            \
      {name, "128x256x256_1sm",                                                                  \
       launch<Spec<ea, eb, ed, float, float, cutlass::arch::OpClassBlockScaledTensorOp,         \
                   Scaled128x256, Cluster1, align, align>,                                        \
              true>}

const Entry kEntries[] = {
    OGREF_DENSE("u8", uint8_t, uint8_t, int32_t, int32_t, int32_t, 16),
    OGREF_DENSE("e5m2", cutlass::float_e5m2_t, cutlass::float_e5m2_t, float, float, float, 16),
    OGREF_DENSE("e3m2", cutlass::float_e3m2_t, cutlass::float_e3m2_t, float, float, float, 128),
    OGREF_DENSE("e2m3", cutlass::float_e2m3_t, cutlass::float_e2m3_t, float, float, float, 128),
    OGREF_DENSE("e2m1", cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, float, float, 128),
    OGREF_SCALED("mxfp4", cutlass::mx_float4_t<cutlass::float_e2m1_t>,
                 cutlass::mx_float4_t<cutlass::float_e2m1_t>, cutlass::bfloat16_t, 32),
};

constexpr int kCount = static_cast<int>(sizeof(kEntries) / sizeof(kEntries[0]));

}  // namespace

extern "C" {

OGREF_API int ogref_abi() { return kAbiVersion; }

OGREF_API const char *ogref_error() { return g_error.c_str(); }

OGREF_API int ogref_configs() { return kCount; }

OGREF_API const char *ogref_config_dtype(int i) {
  return (i < 0 || i >= kCount) ? "" : kEntries[i].dtype;
}

OGREF_API const char *ogref_config_tile(int i) {
  return (i < 0 || i >= kCount) ? "" : kEntries[i].tile;
}

OGREF_API int ogref_launch(int i, int m, int n, int k, const void *a, const void *b, const void *sfa,
                 const void *sfb, void *d, void *stream) {
  if (i < 0 || i >= kCount) {
    g_error = "no such configuration";
    return -1;
  }
  return kEntries[i].launch(m, n, k, a, b, sfa, sfb, d, static_cast<cudaStream_t>(stream));
}
}

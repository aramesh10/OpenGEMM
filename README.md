# OpenGEMM

GEMM kernels for NVIDIA B200 (sm_100a) in CUDA.

```python
import opengemm as og

c = og.gemm(a, b)                    # C[M, N] = A[M, K] @ B[N, K].T
c = og.gemm(a, b, sfa, sfb)          # block-scaled: nvfp4, mxfp8, mxfp4

og.emit_kernel(a, b, file="k.cu")    # emits .cu/.cuh for this shape
c = og.run_kernel("k.cu", a, b)      # compiles emitted kernel and runs it
```

## Install

From PyPI
```bash
pip install opengemm
```

From a clone:
```bash
git clone https://github.com/aramesh10/OpenGEMM.git
cd OpenGEMM
pip install -e .
```

Requirements: 
- sm_100a
- PyTorch 2.8+
- CUDA 12.9+ with `nvcc` on the path, for a clone, an edited kernel, or
  `emit_kernel`. A wheel ships the kernels already compiled and needs only the
  driver.

The kernels are compiled into two libraries that link no libtorch and no
libpython, so one build serves every PyTorch and Python version. A wheel ships
them; a clone compiles them on the first `gemm()` call, about 25 s for both.
To pay that at install time instead:

```bash
python -m opengemm
```

Or from Python, `og.prebuild()`. Either is a no-op once the libraries exist,
so it is safe in a Dockerfile, a post-install step or a test fixture. The
result is cached under `OPENGEMM_CACHE` (default `~/.cache/opengemm`), keyed
by a digest of the sources it was built from, so editing a kernel rebuilds and
a reinstall does not. `OPENGEMM_JIT=1` forces the build even when a wheel
shipped one.

## Agent Quickstart

Give your agent this prompt to use OpenGEMM as a tool:

```
OpenGEMM emits standalone CUDA GEMM kernels for B200 (sm_100a), no GPU
needed to emit:

python -c "
import opengemm as og
S = dict(m=1024, n=1024, k=1024)

og.emit_kernel(**S, atype='bf16', file='k')        # writes k.cu and k.cuh
og.emit_kernel(**S, atype='e4m3', btype='e5m2')   # mixed, names itself
og.emit_kernel(**S, atype='e2m1', sftype='ue4m3') # block-scaled (nvfp4)
src, hdr = og.emit_kernel(**S, atype='bf16')       # the text, always returned
print(src, hdr)
"
atype / btype: bf16 f16 tf32 s8 u8 e4m3 e5m2 e3m2 e2m3 e2m1
sftype (block-scaled): ue4m3 (nvfp4) or ue8m0 (mxfp8, mxfp4)
dtype (output): f32, s32 for s8/u8, bf16 when scaled — inferred, optional.
```

## Dense and block-scaled

`C[M, N] = A[M, K] @ B[N, K].T`. Both operands are row-major with K innermost.

| GEMM | `atype` / `btype` | `sftype` | `dtype` | `torch.dtype` (in → out) |
| --- | --- | --- | --- | --- |
| bfloat16 | bf16 | — | f32 | `bfloat16` → `float32` |
| float16 | f16 | — | f32 | `float16` → `float32` |
| tf32 | tf32 | — | f32 | `float32` → `float32` |
| int8 | s8 | — | s32 | `int8` → `int32` |
| uint8 | u8 | — | s32 | `uint8` → `int32` |
| fp8 | e4m3 | — | f32 | `float8_e4m3fn` → `float32` |
| fp8 | e5m2 | — | f32 | `float8_e5m2` → `float32` |
| mixed fp8 | e4m3, e5m2 | — | f32 | `float8_e4m3fn`, `float8_e5m2` → `float32` |
| fp6 | e3m2 | — | f32 | `uint8` → `float32` |
| fp6 | e2m3 | — | f32 | `uint8` → `float32` |
| fp4 | e2m1 | — | f32 | `uint8` → `float32` |
| nvfp4 | e2m1 | ue4m3 (per 16) | bf16 | `float4_e2m1fn_x2`, `float8_e4m3fn` → `bfloat16` |
| mxfp8 | e4m3 | ue8m0 (per 32) | bf16 | `float8_e4m3fn`, `float8_e8m0fnu` → `bfloat16` |
| mxfp4 | e2m1 | ue8m0 (per 32) | bf16 | `float4_e2m1fn_x2`, `float8_e8m0fnu` → `bfloat16` |

Note: fp6 and fp4 have no torch dtype. They arrive densely packed in `uint8` and are named - `gemm(a, b, atype="e2m1")` 
Use `btype=` when the two operands differ.

Input is `[M, K]` and `[N, K]` with column-major strides `(1, M)` and `(1, N)`
Output is `[M, N]` with column-major strides `(1, M)`

## Tuning and performance

There is no heursitic to choose the config. Optimized configs are stored in `configs.json`. 
If a particular shape has not been optimized, the library autotunes and returns and saves the best config locally to `./opengemm-configs/tuned_configs.json` or to `OPENGEMM_CONFIGS` env variable.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tune.py --dtype f16 --shape 4096 4096 4096
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py --dtype bf16 e4m3    # vs cuBLAS
CUDA_VISIBLE_DEVICES=0 python scripts/test.py                           # correctness
```

`tune.py` ablates every compiled configuration for a shape and records the best performing config to `configs.json`

`CUDA_VISIBLE_DEVICES` above pins tuning to one GPU for repeatable timings; it
is not required. `gemm()` launches on whichever device its operands are on.

## Standalone kernels

```bash
python scripts/emit_kernel.py --dtype e4m3 --shape 4096 4096 4096 --file emitted/e4m3_4k.cu
python scripts/run_kernel.py emitted/e4m3_4k.cu       # correctness, then timing vs cuBLAS
```

OpenGEMM can also emit the optimized CUDA files for a kernel given a shape and dtype. It can be ran with `scripts/run_kernel.py` or built with `nvcc`:

```bash
nvcc -O3 -std=c++20 -gencode=arch=compute_100a,code=sm_100a --expt-relaxed-constexpr -shared -Xcompiler -fPIC -lcuda <KERNEL_FILE>.cu -o <KERNEL_FILE>.so
```

The entry point is `extern "C" void mm_<dtype>_<M>_<N>_<K>(a, b, c, stream)`,
or `smm_<dtype>_<M>_<N>_<K>(a, b, sfa, sfb, c, stream)` for a block-scaled
format. It reads which device the operands are on and launches there, so it
does not care what the caller's current device is.

`emit_kernel` reads only shapes and dtypes, so meta tensors work:
`emit_kernel(torch.empty(4096, 4096, dtype=torch.bfloat16, device="meta"), ...)`.


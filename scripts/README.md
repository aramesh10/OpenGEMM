# scripts

DTYPE: `bf16` `f16` `tf32` `s8` `u8` `e4m3` `e5m2` `e3m2` `e2m3` `e2m1` `e4m3xe5m2` `nvfp4` `mxfp8` `mxfp4`

## `test.py`

Checks `gemm()` against an fp32 reference on six shapes per dtype.

```
python scripts/test.py [--dtype DTYPE] [--emit]
python scripts/test.py
python scripts/test.py --dtype nvfp4
python scripts/test.py --dtype e4m3 --emit
```

## `tune.py`

Sweeps every compiled configuration for a shape and stores the winner in configs.json.

```
python scripts/tune.py --dtype DTYPE (--shape M N K | --all)
python scripts/tune.py --dtype bf16 --shape 4096 4096 4096
python scripts/tune.py --dtype nvfp4 --shape 1 8192 8192
python scripts/tune.py --dtype e4m3 --all
```

## `benchmark.py`

Times every dtype and shape against cuBLAS and writes the results as JSON under results/.

`u8`, `e5m2`, `e3m2`, `e2m3`, `e2m1` and `mxfp4` have no torch kernel to compare
against, so they are measured against the CUTLASS GEMMs in `utils/cutlass_ref.cu`
instead - a few tile shapes each, reporting the fastest one that reproduces the
reference. That needs a CUTLASS 4.x checkout in `OPENGEMM_CUTLASS`, compiled
once and cached; without one those dtypes report "no baseline" as before.

```
python scripts/benchmark.py [--dtype DTYPE ...] [--shape M N K] [--tune-missing] [--quick] [--no-cutlass] [--output PATH] [--resume]
python scripts/benchmark.py
python scripts/benchmark.py --dtype bf16 e4m3 nvfp4 --quick
python scripts/benchmark.py --dtype nvfp4 --shape 8192 8192 8192
python scripts/benchmark.py --tune-missing --resume
OPENGEMM_CUTLASS=~/cutlass python scripts/benchmark.py --dtype e2m1 mxfp4
```

## `emit_kernel.py`

Writes a standalone .cu and .cuh pair for one dtype and shape, with the tuned configuration folded in.

```
python scripts/emit_kernel.py --dtype DTYPE --shape M N K [--file FILE]
python scripts/emit_kernel.py --dtype e4m3 --shape 4096 4096 4096
python scripts/emit_kernel.py --dtype e4m3 --shape 4096 4096 4096 --file emitted/e4m3_4k.cu
python scripts/emit_kernel.py --dtype nvfp4 --shape 32 4096 4096 --file emitted/nvfp4_32_4k.cu
```

## `run_kernel.py`

Compiles an emitted kernel, checks it against the reference, and times it against cuBLAS.

```
python scripts/run_kernel.py FILE
python scripts/run_kernel.py emitted/e4m3_4k.cu
python scripts/run_kernel.py emitted/nvfp4_32_4k.cu
CUDA_VISIBLE_DEVICES=1 python scripts/run_kernel.py emitted/e4m3_4k.cu
```

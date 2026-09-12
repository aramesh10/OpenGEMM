# OpenGEMM API

```python
import opengemm as og
```

## `og.gemm(a, b, sfa=None, sfb=None, out=None, atype=None, btype=None)`

Computes `C[M, N] = A[M, K] @ B[N, K].T`.

**Parameters**
- `a`: `(M, K)` contiguous CUDA tensor.
- `b`: `(N, K)` contiguous CUDA tensor, on the same device as `a`.
- `sfa`: scales of `a` in the 128x4 blocked layout (see `to_blocked`). Block-scaled formats only.
- `sfb`: scales of `b`, same layout.
- `out`: `(M, N)` tensor to write into. Allocated if `None`.
- `atype`: element name for a `uint8` operand: `"u8"`, `"e3m2"`, `"e2m3"` or `"e2m1"`.
- `btype`: element name for `b`. Defaults to `atype`.

**Returns** `(M, N)` tensor: `float32` for dense float, `int32` for `s8`/`u8`, `bfloat16` for block-scaled.

**Example**
```python
c = og.gemm(a, b)                                        # dense
c = og.gemm(a, b, og.to_blocked(sa), og.to_blocked(sb))  # block-scaled
```

## `og.emit_kernel(a=None, b=None, sfa=None, sfb=None, file=None, atype=None, btype=None, m=None, n=None, k=None, sftype=None)`

Writes a standalone `.cu`/`.cuh` kernel for one dtype and shape.

**Parameters**
- `a`, `b`: operands, or meta tensors of the same shape and dtype. `None` to use `m`, `n`, `k` instead.
- `sfa`, `sfb`: scales, for a block-scaled kernel. Only the dtype is read.
- `file`: output stem. Defaults to `<dtype>_<M>_<N>_<K>`.
- `atype`: element of `a`, e.g. `"bf16"`, `"e4m3"`, `"e2m1"`.
- `btype`: element of `b`. Defaults to `atype`.
- `m`, `n`, `k`: shape, when no operands are given.
- `sftype`: scale element, `"ue4m3"` or `"ue8m0"`, for a block-scaled kernel.

**Returns** `(source, header)` text.

**Example**
```python
og.emit_kernel(m=1024, n=1024, k=1024, atype="bf16", file="k")  # writes k.cu, k.cuh
```

## `og.run_kernel(file, a, b, sfa=None, sfb=None, out=None)`

Compiles (once, cached beside the source) and runs a kernel written by `emit_kernel`.

**Parameters**
- `file`: the `.cu` path `emit_kernel` wrote.
- `a`, `b`: operands in the dtype and shape the kernel was emitted for.
- `sfa`, `sfb`: blocked scales. Block-scaled kernels only.
- `out`: `(M, N)` tensor to write into.

**Returns** `(M, N)` tensor, same dtype as `gemm`.

**Example**
```python
c = og.run_kernel("k.cu", a, b)
```

## `og.prebuild(impls=("mm", "smm"))`

Builds the kernel libraries ahead of the first `gemm()` call.

**Parameters**
- `impls`: libraries to build: `"mm"` (dense), `"smm"` (block-scaled).

**Returns** list of `(impl, path)`.

**Example**
```python
og.prebuild()   # same as `python -m opengemm`
```

## `og.quantize(x, name)`

Quantizes values to a block-scaled format.

**Parameters**
- `x`: `(rows, K)` values. K a multiple of the scale block (16 for nvfp4, 32 for mx).
- `name`: `"nvfp4"`, `"mxfp8"` or `"mxfp4"`.

**Returns** `(packed, scales)`: the operand in the format's torch dtype, and `(rows, K / block)` unblocked scales.

**Example**
```python
a, sa = og.quantize(x, "nvfp4")
```

## `og.to_blocked(scales)`

Rearranges scales into the flat 128x4 blocked layout that `gemm` takes.

**Parameters**
- `scales`: `(rows, K / block)` scales, as `quantize` returns them.

**Returns** flat tensor of the same dtype.

**Example**
```python
sfa = og.to_blocked(sa)
```

## `og.pack(x, elem)`

Packs values into dense `uint8` fp6 or fp4 codes.

**Parameters**
- `x`: `(rows, K)` values. K a multiple of 4 for fp6, even for fp4.
- `elem`: `"e3m2"`, `"e2m3"` (4 values in 3 bytes) or `"e2m1"` (2 values per byte).

**Returns** `uint8` tensor of `(rows, K * bits / 8)`.

**Example**
```python
a = og.pack(x, "e3m2")
c = og.gemm(a, a, atype="e3m2")
```

## `og.unpack(packed, elem, k)`

Unpacks codes written by `pack` back to values.

**Parameters**
- `packed`: `uint8` tensor from `pack`.
- `elem`: `"e3m2"`, `"e2m3"` or `"e2m1"`.
- `k`: number of values per row.

**Returns** `(rows, k)` `float32` tensor.

**Example**
```python
x = og.unpack(a, "e3m2", 1024)
```

## `og.dtype_name(a=None, b=None, sfa=None, atype=None, btype=None, sftype=None)`

Names the format for given operands or element names.

**Parameters**
- `a`, `b`: operands, or `None` to name the format from element names.
- `sfa`: scales of `a`, for a block-scaled format.
- `atype`, `btype`: element names. `atype` is required for a `uint8` operand.
- `sftype`: `"ue4m3"` or `"ue8m0"`, for a block-scaled format without operands.

**Returns** the `DTYPES` key, a `str`.

**Example**
```python
og.dtype_name(atype="e2m1", sftype="ue4m3")   # "nvfp4"
og.dtype_name(atype="e4m3", btype="e5m2")     # "e4m3xe5m2"
```

## `og.DTYPES`

Dict of supported formats to their torch dtypes.

**Keys** `bf16`, `f16`, `tf32`, `s8`, `u8`, `e4m3`, `e5m2`, `e3m2`, `e2m3`, `e2m1`, `e4m3xe5m2`, `nvfp4`, `mxfp8`, `mxfp4`.

**Values** `Dtype(impl, a, b, bits, out, sf)`.

**Example**
```python
og.DTYPES["nvfp4"].out   # torch.bfloat16
```

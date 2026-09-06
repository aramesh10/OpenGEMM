"""Element types and block-scaled formats: how each arrives from torch, and
how the harness packs, quantizes and dequantizes it.

Dense types ride the `mm` kernel and are named by the tcgen05 element: bf16,
f16, tf32, s8, u8, e4m3, e5m2, e3m2, e2m3, e2m1, and the mixed e4m3xe5m2.
Block-scaled formats ride `smm`: nvfp4, mxfp8, mxfp4.
"""
import torch

ELEMS = ("bf16", "f16", "tf32", "s8", "u8", "e4m3", "e5m2", "e3m2", "e2m3",
         "e2m1")
ELEM_INDEX = {name: i for i, name in enumerate(ELEMS)}

DENSE_OF_TORCH = {
    torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "tf32",
    torch.int8: "s8", torch.float8_e4m3fn: "e4m3", torch.float8_e5m2: "e5m2",
}
SCALED_OF_TORCH = {
    (torch.float4_e2m1fn_x2, torch.float8_e4m3fn): "nvfp4",
    (torch.float8_e4m3fn, torch.float8_e8m0fnu): "mxfp8",
    (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu): "mxfp4",
}

TOLERANCE_PIVOT_K = 4096


def _scaled_tolerance(rtol, atol, k):
    if rtol == 0.0 and atol == 0.0:
        return 0.0, 0.0
    scale = max(1.0, (k / TOLERANCE_PIVOT_K) ** 0.5)
    return rtol * scale, atol * scale


class Dense:
    """A dense element type: its torch dtype, width, tolerance, packing and
    torch baseline.
    """
    impl = "mm"

    def __init__(self, name, torch_dtype, bits, out_dtype=torch.float32,
                 rtol=1.0e-3, atol=2.0e-2, elem_a=None, elem_b=None,
                 torch_dtype_b=None, int_range=(-127, 128), packer=None,
                 unpacker=None, quantize=None, baseline=None,
                 baseline_unavailable=None):
        self.name = name
        self.torch_dtype = torch_dtype
        self.torch_dtype_b = torch_dtype_b or torch_dtype
        self.bits = bits
        self.out_dtype = out_dtype
        self.rtol, self.atol = rtol, atol
        self.elem_a = elem_a or name
        self.elem_b = elem_b or self.elem_a
        self.int_range = int_range
        self.packer, self.unpacker, self.quantize = packer, unpacker, quantize
        self.baseline = baseline
        self.baseline_unavailable = baseline_unavailable

    def tolerance(self, k):
        """Return `(rtol, atol)` for a reduction of length `k`."""
        return _scaled_tolerance(self.rtol, self.atol, k)

    @property
    def k_align(self):
        """Return the K multiple a row must be for TMA to address it.

        16 bytes' worth for the byte-aligned types, and a flat 128 values for
        the sub-byte ones, whose expanding tensor maps require it of
        globalDim[0].
        """
        return 128 if self.bits < 8 else 16 * 8 // self.bits

    def k_extent(self, k):
        """Return the row extent of a `(rows, K)` operand as torch sees it."""
        return k * self.bits // 8 if self.bits < 8 else k

    def k_values(self, extent):
        """Return K for a row extent as torch sees it; the inverse of
        `k_extent`.
        """
        return extent * 8 // self.bits if self.bits < 8 else extent


class Scaled:
    """A block-scaled format: its element and scale dtypes, block size and
    quantizer.
    """
    impl = "smm"
    rtol, atol = 2.0 ** -6, 2.0 ** -5

    def __init__(self, name, elem, sf, block, amax, packer, unpacker,
                 sf_dtype, sf_ceiling=False, baseline=True,
                 baseline_unavailable=None):
        self.name = name
        self.elem, self.sf = elem, sf
        self.block, self.amax = block, amax
        self.packer, self.unpacker = packer, unpacker
        self.sf_dtype = sf_dtype
        self.sf_ceiling = sf_ceiling
        self.baseline = baseline
        self.baseline_unavailable = baseline_unavailable
        self.per_byte = 2 if elem == "e2m1" else 1
        self.elem_dtype = (torch.float4_e2m1fn_x2 if elem == "e2m1"
                           else torch.float8_e4m3fn)
        self.out_dtype = torch.bfloat16

    def tolerance(self, k):
        """Return `(rtol, atol)` for a reduction of length `k`."""
        return _scaled_tolerance(self.rtol, self.atol, k)

    def k_extent(self, k):
        """Return the row extent of a `(rows, K)` operand as torch sees it."""
        return k // self.per_byte

    def k_values(self, extent):
        """Return K for a row extent as torch sees it; the inverse of
        `k_extent`.
        """
        return extent * self.per_byte


def _float_table(exp_bits, mant_bits):
    bias = (1 << (exp_bits - 1)) - 1
    scale = float(1 << mant_bits)
    values = []
    for code in range(1 << (1 + exp_bits + mant_bits)):
        sign = -1.0 if code >> (exp_bits + mant_bits) else 1.0
        exponent = (code >> mant_bits) & ((1 << exp_bits) - 1)
        mantissa = code & ((1 << mant_bits) - 1)
        values.append(sign * (mantissa / scale * 2.0 ** (1 - bias)
                              if exponent == 0
                              else (1.0 + mantissa / scale)
                              * 2.0 ** (exponent - bias)))
    return values


def _pack_bits(codes, bits):
    per_word = {6: 4, 4: 2}[bits]
    out_bytes = per_word * bits // 8
    rows, k = codes.shape
    q = codes.reshape(rows, k // per_word, per_word).to(torch.int32)
    word = q[..., 0]
    for i in range(1, per_word):
        word = word | (q[..., i] << (bits * i))
    packed = torch.stack([(word >> (8 * b)) & 0xFF for b in range(out_bytes)],
                         dim=-1)
    return packed.reshape(rows, k // per_word * out_bytes).to(
        torch.uint8).contiguous()


def _unpack_bits(packed, k, bits):
    per_word = {6: 4, 4: 2}[bits]
    out_bytes = per_word * bits // 8
    rows = packed.shape[0]
    b = packed.reshape(rows, k // per_word, out_bytes).to(torch.int32)
    word = b[..., 0]
    for i in range(1, out_bytes):
        word = word | (b[..., i] << (8 * i))
    mask = (1 << bits) - 1
    return torch.stack([(word >> (bits * i)) & mask for i in range(per_word)],
                       dim=-1).reshape(rows, k)


def _small_float(name, exp_bits, mant_bits, bits):
    table = {}

    def values(device):
        if device not in table:
            table[device] = torch.tensor(_float_table(exp_bits, mant_bits),
                                         device=device)
        return table[device]

    def pack(x):
        v, x = values(x.device), x.float()
        code = torch.zeros(x.shape, dtype=torch.uint8, device=x.device)
        best = (x - v[0]).abs()
        for c in range(1, v.numel()):
            d = (x - v[c]).abs()
            closer = d < best
            best = torch.where(closer, d, best)
            code[closer] = c
        return _pack_bits(code, bits)

    def unpack(packed, k):
        return values(packed.device)[_unpack_bits(packed, k, bits).long()]

    return Dense(name, torch.uint8, bits, packer=pack, unpacker=unpack,
                 baseline_unavailable=("torch has no float6 dtype" if bits == 6
                                       else "cuBLAS exposes no dense fp4 GEMM"))


def _round_tf32(x):
    bits = x.view(torch.int32)
    return ((bits + 0x1000) & ~0x1FFF).view(torch.float32)


E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def pack_e2m1(x):
    """Pack values to e2m1 codes, two per byte.

    Args:
        x: Values with an even last dimension.

    Returns:
        `uint8` tensor with the last dimension halved, low nibble first.
    """
    values = torch.tensor(E2M1_VALUES[:8], device=x.device)
    magnitude = x.float().abs().unsqueeze(-1)
    distance = (magnitude - values).abs()
    code = distance.argmin(dim=-1)
    upper = (code + 1).clamp(max=len(values) - 1)
    tied = (distance.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
            == distance.gather(-1, code.unsqueeze(-1)).squeeze(-1)) \
        & (upper != code)
    code = torch.where(tied & (code % 2 == 1), upper, code).to(torch.uint8)
    code |= (torch.signbit(x.float()) << 3).to(torch.uint8)
    low, high = code[..., 0::2], code[..., 1::2]
    return (low | (high << 4)).contiguous()


def unpack_e2m1(packed, k):
    """Unpack e2m1 codes to float32 values.

    Args:
        packed: Output of `pack_e2m1`, any leading shape.
        k: Values per row.

    Returns:
        `(rows, k)` float32 tensor.
    """
    values = torch.tensor(E2M1_VALUES, device=packed.device)
    data = packed.reshape(-1, k // 2)
    out = torch.empty((data.shape[0], k), device=packed.device)
    out[:, 0::2] = values[(data & 15).long()]
    out[:, 1::2] = values[(data >> 4).long()]
    return out


def _torch_packer(dtype):
    return lambda x: x.float().to(dtype).view(torch.uint8).contiguous()


def _torch_unpacker(dtype):
    return lambda packed, k: packed.reshape(-1, k).view(dtype).float()


def to_blocked(scales):
    """Rearrange scales into the 128x4 blocked layout cuBLAS and
    torch._scaled_mm consume.

    Args:
        scales: `(rows, K / block)` scale factors.

    Returns:
        Flat tensor of the same dtype, rows padded to 128 and columns to 4.
    """
    rows, cols = scales.shape
    row_blocks = -(-rows // 128)
    col_blocks = -(-cols // 4)
    padded = torch.zeros((row_blocks * 128, col_blocks * 4),
                         dtype=scales.dtype, device=scales.device)
    padded[:rows, :cols] = scales
    atoms = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return atoms.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


def quantize(x, dtype):
    """Quantize rows to a block-scaled format.

    Args:
        x: `(rows, K)` values, K a multiple of the format's block.
        dtype: A `Scaled` entry of `DTYPES`.

    Returns:
        `(packed, scales)`: the packed operand and `(rows, K / block)` scales
        in
        the format's scale dtype.
    """
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // dtype.block, dtype.block)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = (amax / dtype.amax).clamp(min=1e-30)
    if dtype.sf_ceiling:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    stored = scale.squeeze(-1).to(dtype.sf_dtype)
    effective = stored.float().unsqueeze(-1).clamp(min=1e-30)
    data = (blocks / effective).reshape(rows, k)
    return dtype.packer(data.to(torch.bfloat16)), stored


DENSE = {
    "bf16": Dense("bf16", torch.bfloat16, 16, baseline="mm"),
    "f16": Dense("f16", torch.float16, 16, baseline="mm"),
    "tf32": Dense("tf32", torch.float32, 32, baseline="mm_tf32",
                  quantize=_round_tf32),
    "s8": Dense("s8", torch.int8, 8, out_dtype=torch.int32, rtol=0.0, atol=0.0,
                baseline="int_mm"),
    "u8": Dense("u8", torch.uint8, 8, out_dtype=torch.int32, rtol=0.0,
                atol=0.0, int_range=(0, 256),
                baseline_unavailable="torch._int_mm takes signed operands only"),
    "e4m3": Dense("e4m3", torch.float8_e4m3fn, 8, baseline="scaled_mm"),
    "e5m2": Dense("e5m2", torch.float8_e5m2, 8,
                  baseline_unavailable="torch._scaled_mm refuses e5m2 x e5m2"),
    "e3m2": _small_float("e3m2", 3, 2, 6),
    "e2m3": _small_float("e2m3", 2, 3, 6),
    "e2m1": _small_float("e2m1", 2, 1, 4),
    "e4m3xe5m2": Dense("e4m3xe5m2", torch.float8_e4m3fn, 8, elem_a="e4m3",
                       elem_b="e5m2", torch_dtype_b=torch.float8_e5m2,
                       baseline="scaled_mm"),
}

SCALED = {
    "nvfp4": Scaled("nvfp4", "e2m1", "ue4m3", 16, 6.0, pack_e2m1, unpack_e2m1,
                    torch.float8_e4m3fn),
    "mxfp8": Scaled("mxfp8", "e4m3", "ue8m0", 32, 448.0,
                    _torch_packer(torch.float8_e4m3fn),
                    _torch_unpacker(torch.float8_e4m3fn),
                    torch.float8_e8m0fnu, sf_ceiling=True),
    "mxfp4": Scaled("mxfp4", "e2m1", "ue8m0", 32, 6.0, pack_e2m1, unpack_e2m1,
                    torch.float8_e8m0fnu, sf_ceiling=True, baseline=False,
                    baseline_unavailable="torch._scaled_mm refuses e2m1 "
                                         "operands with e8m0 block scales"),
}

DTYPES = {**DENSE, **SCALED}

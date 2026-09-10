"""Packing and quantizing values into the formats gemm() takes."""

import torch

from .dtypes import DTYPES, elems, sf_block

# Exponent, mantissa and storage bits of the elements torch has no dtype for.
SMALL_FLOATS = {"e3m2": (3, 2, 6), "e2m3": (2, 3, 6), "e2m1": (2, 1, 4)}

def _values(elem, device):
    """Every code of a small float as the value it stands for, in code order."""
    exp_bits, mant_bits, _ = SMALL_FLOATS[elem]
    bias = (1 << (exp_bits - 1)) - 1
    values = []
    for code in range(1 << (1 + exp_bits + mant_bits)):
        sign = -1.0 if code >> (exp_bits + mant_bits) else 1.0
        exponent = (code >> mant_bits) & ((1 << exp_bits) - 1)
        mantissa = code & ((1 << mant_bits) - 1)
        if exponent == 0:
            magnitude = mantissa / (1 << mant_bits) * 2.0 ** (1 - bias)
        else:
            magnitude = (1 + mantissa / (1 << mant_bits)) * 2.0 ** (exponent - bias)
        values.append(sign * magnitude)
    return torch.tensor(values, device=device)

def _pack_bits(codes, bits):
    per_word = {6: 4, 4: 2}[bits]
    out_bytes = per_word * bits // 8
    rows, k = codes.shape
    q = codes.reshape(rows, k // per_word, per_word).to(torch.int32)
    word = q[..., 0]
    for i in range(1, per_word):
        word = word | (q[..., i] << (bits * i))
    packed = torch.stack([(word >> (8 * b)) & 0xFF for b in range(out_bytes)], dim=-1)
    return packed.reshape(rows, k // per_word * out_bytes).to(torch.uint8).contiguous()

def _unpack_bits(packed, k, bits):
    per_word = {6: 4, 4: 2}[bits]
    out_bytes = per_word * bits // 8
    rows = packed.shape[0]
    b = packed.reshape(rows, k // per_word, out_bytes).to(torch.int32)
    word = b[..., 0]
    for i in range(1, out_bytes):
        word = word | (b[..., i] << (8 * i))
    mask = (1 << bits) - 1
    return torch.stack([(word >> (bits * i)) & mask for i in range(per_word)], dim=-1).reshape(
        rows, k
    )

def pack(x, elem):
    """Pack `(rows, K)` values to `elem` codes, densely along K.

    Args:
        x: Values; K a multiple of 4 for the 6-bit elements, of 2 for e2m1.
        elem: "e3m2", "e2m3" or "e2m1".

    Returns:
        `uint8` tensor of `(rows, K * bits / 8)`: the nearest representable
        value of each, ties to the even code, first value in the low bits.
    """
    values = _values(elem, x.device)
    x = x.float()
    code = torch.zeros(x.shape, dtype=torch.int32, device=x.device)
    best = (x - values[0]).abs()
    for c in range(1, values.numel()):
        distance = (x - values[c]).abs()
        take = (distance < best) | ((distance == best) & (c % 2 == 0) & (code % 2 == 1))
        best = torch.where(take, distance, best)
        code = torch.where(take, c, code)
    return _pack_bits(code, SMALL_FLOATS[elem][2])

def unpack(packed, elem, k):
    """Unpack `elem` codes written by `pack` to `(rows, k)` float32 values."""
    bits = SMALL_FLOATS[elem][2]
    return _values(elem, packed.device)[_unpack_bits(packed, k, bits).long()]

def pack_e2m1(x):
    """Pack values to e2m1 codes, two per byte, low nibble first."""
    return pack(x, "e2m1")

def unpack_e2m1(packed, k):
    """Unpack e2m1 codes to `(rows, k)` float32 values."""
    return unpack(packed, "e2m1", k)

def round_tf32(x):
    """Round float32 values to the 10 mantissa bits tf32 keeps."""
    bits = x.view(torch.int32)
    return ((bits + 0x1000) & ~0x1FFF).view(torch.float32)

def to_blocked(scales):
    """Rearrange `(rows, K / block)` scales into the 128x4 blocked layout
    cuBLAS and torch._scaled_mm consume.

    Returns:
        Flat tensor of the same dtype, rows padded to 128 and columns to 4.
    """
    rows, cols = scales.shape
    row_blocks = -(-rows // 128)
    col_blocks = -(-cols // 4)
    padded = torch.zeros(
        (row_blocks * 128, col_blocks * 4), dtype=scales.dtype, device=scales.device
    )
    padded[:rows, :cols] = scales
    atoms = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return atoms.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()

def quantize(x, name):
    """Quantize `(rows, K)` values to a block-scaled format.

    Args:
        x: Values, K a multiple of the format's scale block.
        name: "nvfp4", "mxfp8" or "mxfp4".

    Returns:
        `(packed, scales)`: the operand in the format's torch dtype, and the
        `(rows, K / block)` scales in the scale dtype, unblocked.
    """
    d = DTYPES[name]
    elem = elems(name)[0]
    block = sf_block(d)
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // block, block)
    amax = 6.0 if d.bits == 4 else 448.0  # the largest e2m1 or e4m3 value
    scale = (blocks.abs().amax(dim=-1, keepdim=True) / amax).clamp(min=1e-30)
    if d.sf is torch.float8_e8m0fnu:  # power-of-two scales only
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    stored = scale.squeeze(-1).to(d.sf)
    data = (blocks / stored.float().unsqueeze(-1).clamp(min=1e-30)).reshape(rows, k)
    data = data.to(torch.bfloat16)
    packed = pack(data, elem).view(d.a) if elem in SMALL_FLOATS else data.to(d.a)
    return packed, stored
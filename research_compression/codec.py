"""Actual, self-describing bitstreams and differentiable learned feature codec.

Wire format is MFCR + uint32(header length) + UTF-8 JSON header +
2-bit packed spatial allocation map + zlib-compressed packed quantized symbols.
The zlib/DEFLATE stage is a reproducible general-purpose entropy coder,
NOT a learned arithmetic coder. Checkpoint weights are pre-shared.
"""
from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MAGIC = b"MFCR"
VERSION = 1
DEPTHS = (2, 4, 6, 8)
MAX_HEADER = 16384
MAX_ELEMENTS = 16_000_000


def _checked_feature(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 3 or any(d < 1 for d in a.shape) or a.size > MAX_ELEMENTS:
        raise ValueError("A single feature must be [C,H,W] with a bounded shape")
    if not np.all(np.isfinite(a)):
        raise ValueError("Feature contains NaN or infinity")
    return a


def _pack_values(values: np.ndarray, depth: int) -> np.ndarray:
    shifts = np.arange(depth - 1, -1, -1, dtype=np.uint8)
    bits = ((values.astype(np.uint16)[:, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1))


def _unpack_values(bitstream: np.ndarray, offset: int, count: int, depth: int):
    length = count * depth
    if offset + length > len(bitstream):
        raise ValueError("Truncated quantized data")
    raw = bitstream[offset:offset + length].reshape(count, depth)
    weights = 1 << np.arange(depth - 1, -1, -1, dtype=np.uint16)
    return (raw * weights).sum(axis=1).astype(np.float32), offset + length


def encode_symbols(feature: np.ndarray, bit_depths: np.ndarray, *, split: str,
                   rate_id: str = "fixed", rate_fraction: Optional[float] = None) -> bytes:
    """Quantize [C,H,W] and return measured transport payload (one sample).

    Quantization uses a shared per-sample symmetric range; all metadata, the
    allocation map and padding are included in the returned byte count.
    Transport/TCP/IP overhead and pre-shared model weights are not included.
    """
    a = _checked_feature(feature)
    c, h, w = a.shape
    levels = np.asarray(bit_depths, dtype=np.uint8)
    if levels.shape != (h, w) or not np.isin(levels, DEPTHS).all():
        raise ValueError("bit_depths must be [H,W] and contain only 2, 4, 6 or 8")
    if not isinstance(split, str) or len(split) > 128 or not split:
        raise ValueError("split must be a nonempty string of at most 128 chars")
    if not isinstance(rate_id, str) or len(rate_id) > 128:
        raise ValueError("rate_id too long")
    if rate_fraction is not None and not (0 < float(rate_fraction) <= 1):
        raise ValueError("Invalid stored rate fraction")

    maximum = max(float(np.max(np.abs(a))), 1e-8)
    per_location = a.transpose(1, 2, 0).reshape(h * w, c)
    depths = levels.reshape(-1)
    # Packed stream is a sequence of variable-width symbols at each position.
    symbols = []
    for loc, depth in zip(per_location, depths):
        bits = int(depth)
        n = (1 << bits) - 1
        quant = np.rint((np.clip(loc / maximum, -1, 1) + 1) * (n / 2))
        symbols.append(((quant.astype(np.uint16)[:, None] >>
                         np.arange(bits - 1, -1, -1, dtype=np.uint16)) & 1).astype(np.uint8).reshape(-1))
    bit_array = np.concatenate(symbols)
    raw = np.packbits(bit_array).tobytes()
    payload = zlib.compress(raw, level=9)
    allocation_symbols = np.searchsorted(np.array(DEPTHS), levels).astype(np.uint8).reshape(-1)
    allocation_bits = ((allocation_symbols[:, None] >> np.array([1, 0], dtype=np.uint8)) & 1).reshape(-1)
    allocation = np.packbits(allocation_bits).tobytes()
    metadata = {
        "v": VERSION, "shape": [int(c), int(h), int(w)], "split": split,
        "rate_id": rate_id, "rate_fraction": rate_fraction,
        "amax": maximum, "alloc_nbytes": len(allocation),
        "packed_nbytes": len(raw), "symbol_bits": int(bit_array.size),
        "coder": "zlib-deflate", "quant": "symmetric-uniform-per-image"
    }
    head = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(head) > MAX_HEADER:
        raise ValueError("Oversized header")
    return MAGIC + struct.pack(">I", len(head)) + head + allocation + payload


def decode_symbols(message: bytes, *, expected_split: Optional[str] = None):
    """Decode one wire message into a float32 feature and its metadata."""
    if not isinstance(message, bytes) or len(message) < 9 or message[:4] != MAGIC:
        raise ValueError("Not an MFCR message")
    header_len = struct.unpack(">I", message[4:8])[0]
    if header_len < 2 or header_len > MAX_HEADER or 8 + header_len > len(message):
        raise ValueError("Invalid MFCR header length")
    meta = json.loads(message[8:8 + header_len].decode("utf-8"))
    if meta.get("v") != VERSION or meta.get("coder") != "zlib-deflate":
        raise ValueError("Unsupported MFCR codec/version")
    if expected_split is not None and meta.get("split") != expected_split:
        raise ValueError("Split layer mismatch")
    shape = meta.get("shape")
    if (not isinstance(shape, list) or len(shape) != 3 or
        any(not isinstance(s, int) or s < 1 for s in shape) or
        math.prod(shape) > MAX_ELEMENTS):
        raise ValueError("Invalid decoded feature shape")
    c, h, w = shape
    alloc_n = meta.get("alloc_nbytes")
    expected_alloc_n = (2 * h * w + 7) // 8
    if alloc_n != expected_alloc_n:
        raise ValueError("Invalid allocation-map length")
    alloc_start = 8 + header_len
    alloc_stop = alloc_start + alloc_n
    if alloc_stop >= len(message):
        raise ValueError("Truncated allocation/payload")
    alloc_bits = np.unpackbits(np.frombuffer(message[alloc_start:alloc_stop], dtype=np.uint8))
    alloc_symbols = alloc_bits[:2*h*w].reshape(h*w, 2).dot(np.array([2, 1]))
    if (alloc_symbols > 3).any():
        raise ValueError("Invalid allocation code")
    levels = np.array(DEPTHS, dtype=np.uint8)[alloc_symbols].reshape(h, w)
    try:
        decompressor = zlib.decompressobj()
        packed_limit = (c*h*w*8+7)//8
        packed = decompressor.decompress(message[alloc_stop:], packed_limit + 1)
        if len(packed) > packed_limit or not decompressor.eof or decompressor.unused_data:
            raise ValueError("Malformed or oversized compressed payload")
        packed += decompressor.flush()
    except zlib.error as exc:
        raise ValueError("Invalid compressed payload") from exc
    expected_bit_count = int(c * levels.astype(np.int64).sum())
    expected_packed_bytes = (expected_bit_count + 7) // 8
    if (len(packed) != expected_packed_bytes or
        meta.get("packed_nbytes") != expected_packed_bytes or
        meta.get("symbol_bits") != expected_bit_count):
        raise ValueError("Bitstream length does not match the allocation map")
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
    values = np.empty((h*w, c), dtype=np.float32)
    offset = 0
    amax = float(meta["amax"])
    if not math.isfinite(amax) or amax <= 0:
        raise ValueError("Invalid quantization range")
    for i, depth in enumerate(levels.reshape(-1)):
        q, offset = _unpack_values(bits, offset, c, int(depth))
        values[i] = ((2.0 * q / ((1 << int(depth)) - 1)) - 1.0) * amax
    return values.reshape(h, w, c).transpose(2, 0, 1).copy(), {**meta, "allocation": levels}


def fake_quantize(x: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    """Straight-through fake quantization mirroring the serializer's binning."""
    if x.ndim != 4 or depths.shape != (x.shape[0], x.shape[2], x.shape[3]):
        raise ValueError("Expected x=[B,C,H,W] and depths=[B,H,W]")
    maxabs = x.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    bins = (2.0 ** depths.unsqueeze(1).to(x.dtype)) - 1.0
    scaled = ((x/maxabs).clamp(-1, 1) + 1.0) * bins / 2.0
    rounded = torch.round(scaled)
    quantized = (2.0 * rounded / bins - 1.0) * maxabs
    return x + (quantized - x).detach()


@dataclass
class CodecOutput:
    reconstructed: torch.Tensor
    latent: torch.Tensor
    latent_fake_quantized: torch.Tensor
    estimated_bits: torch.Tensor


class ConditionalFeatureCodec(nn.Module):
    """1x1 learned bottleneck, small FiLM rate network and STE quantization.

    forward() provides estimated differentiable rate. encode()/decode() use
    ACTUAL bitstreams; do not equate the two rate measurements.
    """
    def __init__(self, in_channels: int, latent_channels: int = 32, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, latent_channels, 1)
        self.decoder = nn.Conv2d(latent_channels, in_channels, 1)
        self.rate_condition = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, latent_channels*2))
        self.decoder_rate_condition = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, in_channels*2))
        self.log_scale = nn.Parameter(torch.zeros(latent_channels))
        self.latent_channels = latent_channels

    def _latent(self, x, rate_fraction: float):
        if not 0 < rate_fraction <= 1:
            raise ValueError("rate_fraction must be in (0, 1]")
        z = self.encoder(x)
        inp = torch.full((len(x), 1), math.log(rate_fraction), device=x.device, dtype=x.dtype)
        gains, shifts = self.rate_condition(inp).chunk(2, dim=-1)
        return z * (1.0 + 0.1 * torch.tanh(gains[:, :, None, None])) + 0.1 * shifts[:, :, None, None]

    def _decode_latent(self, z, rate_fraction: float):
        fraction = float(rate_fraction)
        if not 0 < fraction <= 1:
            raise ValueError("rate_fraction must be in (0,1]")
        decoded = self.decoder(z)
        inp = torch.full((len(z), 1), math.log(fraction), device=z.device, dtype=z.dtype)
        gain, shift = self.decoder_rate_condition(inp).chunk(2, dim=-1)
        return decoded * (1.0 + 0.1*torch.tanh(gain[:, :, None, None])) + 0.1*shift[:, :, None, None]

    def forward(self, x, depths, rate_fraction: float):
        z = self._latent(x, rate_fraction)
        z_hat = fake_quantize(z, depths)
        reconstructed = self._decode_latent(z_hat, rate_fraction)
        # Factorized Laplace probability mass proxy over unit-spaced symbols.
        scale = self.log_scale.exp().clamp_min(1e-4).view(1,-1,1,1)
        step = (2*z.detach().abs().amax(dim=(1,2,3),keepdim=True).clamp_min(1e-8))/((2.0**depths[:,None].float())-1.0)
        mass = (torch.exp(-torch.relu(z.abs()-step/2)/scale) -
                torch.exp(-(z.abs()+step/2)/scale)).clamp_min(1e-9) / 2.0
        estimated_bits = -torch.log2(mass).sum(dim=(1,2,3))
        return CodecOutput(reconstructed, z, z_hat, estimated_bits)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, depths: torch.Tensor, rate_fraction: float,
               *, split: str, rate_id: str = "fixed") -> list[bytes]:
        z = self._latent(x, rate_fraction).detach().float().cpu().numpy()
        lev = depths.detach().cpu().numpy()
        return [encode_symbols(z[i], lev[i], split=split, rate_id=rate_id,
                               rate_fraction=rate_fraction) for i in range(len(z))]

    @torch.no_grad()
    def decode(self, messages: list[bytes], *, expected_split: str,
               device: torch.device | str = "cpu") -> torch.Tensor:
        decoded = [decode_symbols(m, expected_split=expected_split) for m in messages]
        fractions = [meta.get("rate_fraction") for _,meta in decoded]
        if any(v is None or not isinstance(v,(float,int)) or not 0 < v <= 1 for v in fractions):
            raise ValueError("Conditional decoder requires a valid rate_fraction in each header")
        if len(set(fractions)) > 1:
            raise ValueError("Cannot batch messages with different rate fractions")
        z = torch.from_numpy(np.stack([a for a,_ in decoded])).to(
            device=device, dtype=self.decoder.weight.dtype)
        return self._decode_latent(z, fractions[0])


def fp16_reference_bits(feature: torch.Tensor) -> int:
    """Raw FP16 feature payload: no model/checkpoint/transport metadata."""
    return int(feature.numel() * 16)

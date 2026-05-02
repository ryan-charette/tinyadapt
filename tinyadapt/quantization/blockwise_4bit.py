from __future__ import annotations

import torch


def block_quantize_4bit(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a 1D tensor into unsigned 4-bit values with one FP16 scale per block."""

    if x.dim() != 1:
        raise ValueError("block_quantize_4bit expects a 1D tensor")
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError("group_size must be a positive even integer")
    if x.numel() % group_size != 0:
        raise ValueError("tensor length must be divisible by group_size")

    blocks = x.to(torch.float32).view(-1, group_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    normalized = ((blocks / scale) + 1.0) * 0.5
    values = (normalized * 15.0).round().clamp_(0, 15).to(torch.uint8)
    packed = (values[:, 0::2] & 0x0F) | ((values[:, 1::2] & 0x0F) << 4)
    return packed, scale.to(torch.float16)


def block_dequantize_4bit(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Unpack blockwise 4-bit values back into a flattened FP32 tensor."""

    if packed.dim() != 2:
        raise ValueError("packed tensor must be 2D")
    unpacked = packed.new_empty((packed.shape[0], packed.shape[1] * 2))
    unpacked[:, 0::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    normalized = unpacked.to(torch.float32) / 15.0
    return ((normalized * 2.0) - 1.0).mul(scale.to(torch.float32)).reshape(-1)

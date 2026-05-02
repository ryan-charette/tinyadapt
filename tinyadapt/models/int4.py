from __future__ import annotations

import math

import torch

from tinyadapt.models.bignet import BIGNET_DIM, BigNetBlock, LayerNorm
from tinyadapt.quantization.blockwise_4bit import block_dequantize_4bit, block_quantize_4bit


class Linear4Bit(torch.nn.Module):
    """Weight-only 4-bit linear layer with blockwise scales."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 16,
    ) -> None:
        super().__init__()
        if (in_features * out_features) % group_size != 0:
            raise ValueError("weight element count must be divisible by group_size")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self._shape = (out_features, in_features)

        self.register_buffer(
            "weight_q4",
            torch.zeros(out_features * in_features // group_size, group_size // 2, dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32)) if bias else None
        self._register_load_state_dict_pre_hook(Linear4Bit._load_state_dict_pre_hook, with_module=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        weight = torch.empty(self.out_features, self.in_features)
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        weight_q4, weight_scale = block_quantize_4bit(weight.reshape(-1), self.group_size)
        self.weight_q4.copy_(weight_q4)
        self.weight_scale.copy_(weight_scale)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def _load_state_dict_pre_hook(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        weight_key = f"{prefix}weight"
        if weight_key in state_dict:
            weight = state_dict.pop(weight_key).detach().reshape(-1).to(torch.float32)
            weight_q4, weight_scale = block_quantize_4bit(weight, self.group_size)
            self.weight_q4.copy_(weight_q4)
            self.weight_scale.copy_(weight_scale)

    def quantized_weight(self) -> torch.Tensor:
        return block_dequantize_4bit(self.weight_q4, self.weight_scale).view(self._shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.quantized_weight().to(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return torch.nn.functional.linear(x, weight, bias)


class BigNet4Bit(torch.nn.Module):
    """BigNet with all linear weights stored as blockwise 4-bit buffers."""

    def __init__(self, dim: int = BIGNET_DIM, depth: int = 6, group_size: int = 16) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        for block_index in range(depth):
            layers.append(BigNetBlock(dim, linear_cls=Linear4Bit, group_size=group_size))
            if block_index != depth - 1:
                layers.append(LayerNorm(dim))
        self.model = torch.nn.Sequential(*layers)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

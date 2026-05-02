from __future__ import annotations

import math

import torch

from tinyadapt.models.bignet import BIGNET_DIM, BigNetBlock, LayerNorm
from tinyadapt.models.int4 import Linear4Bit


class QLoRALinear(Linear4Bit):
    """Frozen 4-bit base linear layer plus trainable additive LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int = 8,
        alpha: float | None = None,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, group_size=group_size)
        self.requires_grad_(False)
        if lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        self.lora_rank = lora_rank
        self.alpha = float(alpha if alpha is not None else lora_rank)
        self.scaling = self.alpha / lora_rank
        self.lora_a = torch.nn.Linear(in_features, lora_rank, bias=False)
        self.lora_b = torch.nn.Linear(lora_rank, out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward(x)
        update = self.lora_b(self.lora_a(x.to(self.lora_a.weight.dtype))) * self.scaling
        return base + update.to(base.dtype)


class QLoRABigNet(torch.nn.Module):
    """BigNet with frozen 4-bit base layers and trainable LoRA adapters."""

    def __init__(
        self,
        dim: int = BIGNET_DIM,
        depth: int = 6,
        lora_rank: int = 8,
        group_size: int = 16,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        for block_index in range(depth):
            layers.append(
                BigNetBlock(
                    dim,
                    linear_cls=QLoRALinear,
                    lora_rank=lora_rank,
                    group_size=group_size,
                )
            )
            if block_index != depth - 1:
                layers.append(LayerNorm(dim))
        self.model = torch.nn.Sequential(*layers)
        for name, parameter in self.named_parameters():
            if "lora_" not in name:
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

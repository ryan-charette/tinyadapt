from __future__ import annotations

import math

import torch

from tinyadapt.models.bignet import BIGNET_DIM, BigNetBlock, LayerNorm
from tinyadapt.models.fp16 import HalfLinear


class LoRALinear(HalfLinear):
    """Frozen FP16 base linear layer plus a trainable low-rank additive update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int = 8,
        alpha: float | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
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


class LoraBigNet(torch.nn.Module):
    """BigNet with frozen FP16 base layers and trainable LoRA adapters."""

    def __init__(self, dim: int = BIGNET_DIM, depth: int = 6, lora_rank: int = 8) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        for block_index in range(depth):
            layers.append(BigNetBlock(dim, linear_cls=LoRALinear, lora_rank=lora_rank))
            if block_index != depth - 1:
                layers.append(LayerNorm(dim))
        self.model = torch.nn.Sequential(*layers)
        for name, parameter in self.named_parameters():
            if "lora_" not in name:
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

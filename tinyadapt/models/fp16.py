from __future__ import annotations

import torch

from tinyadapt.models.bignet import BIGNET_DIM, BigNetBlock, LayerNorm


class HalfLinear(torch.nn.Linear):
    """Linear layer that stores weights in FP16 and returns FP32 outputs."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.weight.data = self.weight.data.to(torch.float16)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(torch.float16)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.nn.functional.linear(x.to(torch.float16), self.weight, self.bias)
        return output.to(x.dtype)


class HalfBigNet(torch.nn.Module):
    """BigNet with FP16 linear weights and FP32 normalization."""

    def __init__(self, dim: int = BIGNET_DIM, depth: int = 6) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        for block_index in range(depth):
            layers.append(BigNetBlock(dim, linear_cls=HalfLinear))
            if block_index != depth - 1:
                layers.append(LayerNorm(dim))
        self.model = torch.nn.Sequential(*layers)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

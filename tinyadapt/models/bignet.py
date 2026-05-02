from __future__ import annotations

import torch

BIGNET_DIM = 1024


class LayerNorm(torch.nn.Module):
    """Layer normalization implemented with one-group GroupNorm."""

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = torch.nn.Parameter(torch.empty(num_channels, device=device, dtype=dtype))
            self.bias = torch.nn.Parameter(torch.empty(num_channels, device=device, dtype=dtype))
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.group_norm(x, 1, self.weight, self.bias, self.eps)


class BigNetBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int = BIGNET_DIM,
        linear_cls: type[torch.nn.Module] = torch.nn.Linear,
        **linear_kwargs: object,
    ) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            linear_cls(channels, channels, **linear_kwargs),
            torch.nn.ReLU(),
            linear_cls(channels, channels, **linear_kwargs),
            torch.nn.ReLU(),
            linear_cls(channels, channels, **linear_kwargs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) + x


class BigNet(torch.nn.Module):
    """Reference FP32 residual MLP backbone."""

    def __init__(self, dim: int = BIGNET_DIM, depth: int = 6) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        for block_index in range(depth):
            layers.append(BigNetBlock(dim))
            if block_index != depth - 1:
                layers.append(LayerNorm(dim))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

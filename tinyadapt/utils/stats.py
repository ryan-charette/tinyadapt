from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import chain

import torch

from tinyadapt.models.int4 import Linear4Bit


@dataclass(frozen=True)
class ModelProfile:
    name: str
    total_parameters: int
    trainable_parameters: int
    storage_bytes: int
    reduction_vs_fp32: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def count_parameters(model: torch.nn.Module, *, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    total = 0
    seen: set[int] = set()
    for module in model.modules():
        if isinstance(module, Linear4Bit):
            total += module.in_features * module.out_features
            if module.bias is not None:
                total += module.bias.numel()
                seen.add(id(module.bias))
        for parameter in module.parameters(recurse=False):
            if id(parameter) not in seen:
                total += parameter.numel()
                seen.add(id(parameter))
    return total


def storage_bytes(model: torch.nn.Module) -> int:
    tensors = chain(model.parameters(), model.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def profile_model(
    name: str,
    model: torch.nn.Module,
    *,
    fp32_storage_bytes: int | None = None,
) -> ModelProfile:
    current_storage = storage_bytes(model)
    baseline = current_storage if fp32_storage_bytes is None else fp32_storage_bytes
    reduction = 0.0 if baseline == 0 else 100.0 * (1.0 - current_storage / baseline)
    return ModelProfile(
        name=name,
        total_parameters=count_parameters(model),
        trainable_parameters=count_parameters(model, trainable_only=True),
        storage_bytes=current_storage,
        reduction_vs_fp32=reduction,
    )


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GB"

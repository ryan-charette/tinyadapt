from __future__ import annotations

import torch

from tinyadapt.models import MODEL_NAMES, create_model
from tinyadapt.utils.stats import ModelProfile, profile_model, storage_bytes


def benchmark_memory(
    *,
    checkpoint: str | None = None,
    model_names: tuple[str, ...] = MODEL_NAMES,
    lora_rank: int = 8,
    group_size: int = 16,
) -> list[ModelProfile]:
    models: dict[str, torch.nn.Module] = {
        name: create_model(name, checkpoint, lora_rank=lora_rank, group_size=group_size)
        for name in model_names
    }
    baseline_bytes = storage_bytes(models["fp32"] if "fp32" in models else next(iter(models.values())))
    return [
        profile_model(name, model, fp32_storage_bytes=baseline_bytes)
        for name, model in models.items()
    ]

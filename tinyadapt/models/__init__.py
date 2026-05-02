"""Model variants exposed by TinyAdapt."""

from collections.abc import Mapping
from pathlib import Path

import torch

from tinyadapt.models.bignet import BIGNET_DIM, BigNet, LayerNorm
from tinyadapt.models.fp16 import HalfBigNet, HalfLinear
from tinyadapt.models.int4 import BigNet4Bit, Linear4Bit
from tinyadapt.models.lora import LoRALinear, LoraBigNet
from tinyadapt.models.qlora import QLoRALinear, QLoRABigNet

MODEL_NAMES = ("fp32", "fp16", "int4", "lora", "qlora")


def create_model(
    name: str,
    checkpoint: str | Path | None = None,
    *,
    reference_state: Mapping[str, torch.Tensor] | None = None,
    lora_rank: int = 8,
    group_size: int = 16,
) -> torch.nn.Module:
    """Create a model variant and optionally load an FP32 checkpoint."""

    normalized = name.lower().replace("-", "_")
    if normalized in {"fp32", "bignet", "baseline"}:
        model: torch.nn.Module = BigNet()
        strict = True
    elif normalized in {"fp16", "half", "half_precision"}:
        model = HalfBigNet()
        strict = True
    elif normalized in {"int4", "4bit", "4_bit", "low_precision"}:
        model = BigNet4Bit(group_size=group_size)
        strict = False
    elif normalized in {"lora", "lora_bignet"}:
        model = LoraBigNet(lora_rank=lora_rank)
        strict = False
    elif normalized in {"qlora", "qlora_bignet"}:
        model = QLoRABigNet(lora_rank=lora_rank, group_size=group_size)
        strict = False
    else:
        raise ValueError(f"Unknown model variant: {name}")

    if checkpoint is not None and reference_state is not None:
        raise ValueError("Pass either checkpoint or reference_state, not both")
    if checkpoint is not None:
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=strict)
    elif reference_state is not None:
        model.load_state_dict(dict(reference_state), strict=strict)
    return model


__all__ = [
    "BIGNET_DIM",
    "BigNet",
    "BigNet4Bit",
    "HalfBigNet",
    "HalfLinear",
    "LayerNorm",
    "Linear4Bit",
    "LoRALinear",
    "LoraBigNet",
    "MODEL_NAMES",
    "QLoRALinear",
    "QLoRABigNet",
    "create_model",
]

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from tinyadapt.models import MODEL_NAMES, BigNet, create_model
from tinyadapt.models.bignet import BIGNET_DIM


@dataclass(frozen=True)
class DriftResult:
    name: str
    mean_absolute_difference: float
    max_absolute_difference: float
    relative_error: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@torch.no_grad()
def benchmark_drift(
    *,
    checkpoint: str | None = None,
    model_names: tuple[str, ...] = MODEL_NAMES,
    batch_size: int = 16,
    seed: int = 1337,
    lora_rank: int = 8,
    group_size: int = 16,
    device: torch.device | str = "cpu",
) -> list[DriftResult]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch_size, BIGNET_DIM, generator=generator, device="cpu").to(device)
    reference_state = None if checkpoint is not None else BigNet().state_dict()
    baseline = create_model(
        "fp32",
        checkpoint,
        reference_state=reference_state,
        lora_rank=lora_rank,
        group_size=group_size,
    ).to(device)
    baseline.eval()
    reference = baseline(x)
    results: list[DriftResult] = []
    for name in model_names:
        if name == "fp32":
            continue
        model = create_model(
            name,
            checkpoint,
            reference_state=reference_state,
            lora_rank=lora_rank,
            group_size=group_size,
        ).to(device)
        model.eval()
        output = model(x)
        difference = (output - reference).abs()
        relative = difference.norm() / reference.norm().clamp_min(1e-8)
        results.append(
            DriftResult(
                name=name,
                mean_absolute_difference=float(difference.mean().item()),
                max_absolute_difference=float(difference.max().item()),
                relative_error=float(relative.item()),
            )
        )
    return results

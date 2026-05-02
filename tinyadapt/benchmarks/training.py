from __future__ import annotations

from tinyadapt.models import BigNet, create_model
from tinyadapt.training.datasets import SyntheticClusterConfig
from tinyadapt.training.train_adapter import TrainingResult, train_classifier


def benchmark_adapter_training(
    *,
    checkpoint: str | None = None,
    model_names: tuple[str, ...] = ("fp32", "lora", "qlora"),
    epochs: int = 3,
    lora_rank: int = 8,
    group_size: int = 16,
    device: str = "cpu",
) -> list[TrainingResult]:
    config = SyntheticClusterConfig()
    reference_state = None if checkpoint is not None else BigNet().state_dict()
    results: list[TrainingResult] = []
    for name in model_names:
        model = create_model(
            name,
            checkpoint,
            reference_state=reference_state,
            lora_rank=lora_rank,
            group_size=group_size,
        )
        results.append(
            train_classifier(
                model,
                model_name=name,
                config=config,
                epochs=epochs,
                device=device,
            )
        )
    return results

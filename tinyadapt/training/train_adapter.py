from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from tinyadapt.models.bignet import BIGNET_DIM
from tinyadapt.training.datasets import SyntheticClusterConfig, make_synthetic_loaders
from tinyadapt.training.evaluate import evaluate_classifier
from tinyadapt.utils.seed import seed_everything
from tinyadapt.utils.stats import count_parameters


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    trainable_parameters: int
    validation_accuracy: float
    final_loss: float
    seconds_per_epoch: float
    loss_curve: list[float]

    def to_dict(self) -> dict[str, float | int | str | list[float]]:
        return asdict(self)


def train_classifier(
    backbone: torch.nn.Module,
    *,
    model_name: str,
    num_classes: int = 4,
    config: SyntheticClusterConfig | None = None,
    epochs: int = 3,
    lr: float = 3e-3,
    device: torch.device | str = "cpu",
) -> TrainingResult:
    """Train trainable backbone parameters plus a small classification head."""

    seed_everything((config.seed if config is not None else 1337) + 17)
    config = config or SyntheticClusterConfig(num_classes=num_classes)
    train_loader, val_loader = make_synthetic_loaders(config)
    backbone.to(device)
    head = torch.nn.Linear(BIGNET_DIM, config.num_classes).to(device)
    parameters = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    parameters.extend(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    loss_curve: list[float] = []

    start = time.perf_counter()
    for _epoch in range(epochs):
        backbone.train()
        head.train()
        running_loss = 0.0
        examples = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(backbone(x))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * x.shape[0]
            examples += int(x.shape[0])
        loss_curve.append(running_loss / max(examples, 1))

    elapsed = time.perf_counter() - start
    metrics = evaluate_classifier(backbone, head, val_loader, device=device)
    return TrainingResult(
        model_name=model_name,
        trainable_parameters=count_parameters(backbone, trainable_only=True) + count_parameters(head),
        validation_accuracy=metrics["accuracy"],
        final_loss=metrics["loss"],
        seconds_per_epoch=elapsed / max(epochs, 1),
        loss_curve=loss_curve,
    )

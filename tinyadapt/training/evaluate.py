from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == target).float().mean().item())


@torch.no_grad()
def evaluate_classifier(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    backbone.eval()
    head.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    criterion = torch.nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = head(backbone(x))
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.shape[0]
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_examples += int(x.shape[0])
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }

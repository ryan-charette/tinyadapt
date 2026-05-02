from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from tinyadapt.models.bignet import BIGNET_DIM


@dataclass(frozen=True)
class SyntheticClusterConfig:
    input_dim: int = BIGNET_DIM
    num_classes: int = 4
    samples_per_class: int = 96
    val_fraction: float = 0.25
    noise_std: float = 0.45
    batch_size: int = 64
    seed: int = 1337


def make_synthetic_classification(config: SyntheticClusterConfig) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(config.seed)
    centers = torch.randn(config.num_classes, config.input_dim, generator=generator)
    centers = torch.nn.functional.normalize(centers, dim=-1) * 3.0
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for class_index in range(config.num_classes):
        noise = torch.randn(config.samples_per_class, config.input_dim, generator=generator) * config.noise_std
        features.append(centers[class_index].unsqueeze(0) + noise)
        labels.append(torch.full((config.samples_per_class,), class_index, dtype=torch.long))
    x = torch.cat(features, dim=0)
    y = torch.cat(labels, dim=0)
    order = torch.randperm(x.shape[0], generator=generator)
    return x[order], y[order]


def make_synthetic_loaders(
    config: SyntheticClusterConfig = SyntheticClusterConfig(),
) -> tuple[DataLoader, DataLoader]:
    x, y = make_synthetic_classification(config)
    val_size = int(round(x.shape[0] * config.val_fraction))
    val_size = min(max(val_size, 1), x.shape[0] - 1)
    train_x, val_x = x[:-val_size], x[-val_size:]
    train_y, val_y = y[:-val_size], y[-val_size:]
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed + 1),
    )
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=config.batch_size)
    return train_loader, val_loader

from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics
import time

import torch

from tinyadapt.models import MODEL_NAMES, BigNet, create_model
from tinyadapt.models.bignet import BIGNET_DIM


@dataclass(frozen=True)
class LatencyResult:
    name: str
    batch_size: int
    mean_latency_ms: float
    std_latency_ms: float
    throughput_examples_per_second: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@torch.no_grad()
def benchmark_latency(
    *,
    checkpoint: str | None = None,
    model_names: tuple[str, ...] = MODEL_NAMES,
    batch_size: int = 16,
    warmup_runs: int = 2,
    measured_runs: int = 5,
    lora_rank: int = 8,
    group_size: int = 16,
    device: torch.device | str = "cpu",
) -> list[LatencyResult]:
    x = torch.randn(batch_size, BIGNET_DIM, device=device)
    reference_state = None if checkpoint is not None else BigNet().state_dict()
    results: list[LatencyResult] = []
    for name in model_names:
        model = create_model(
            name,
            checkpoint,
            reference_state=reference_state,
            lora_rank=lora_rank,
            group_size=group_size,
        ).to(device)
        model.eval()
        for _ in range(warmup_runs):
            model(x)
        latencies = []
        for _ in range(measured_runs):
            start = time.perf_counter()
            model(x)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)
        mean_ms = statistics.fmean(latencies)
        std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        throughput = batch_size / max(mean_ms / 1000.0, 1e-8)
        results.append(
            LatencyResult(
                name=name,
                batch_size=batch_size,
                mean_latency_ms=mean_ms,
                std_latency_ms=std_ms,
                throughput_examples_per_second=throughput,
            )
        )
    return results

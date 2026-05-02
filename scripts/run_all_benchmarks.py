from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tinyadapt.benchmarks import (
    benchmark_adapter_training,
    benchmark_drift,
    benchmark_latency,
    benchmark_memory,
)
from tinyadapt.utils.checkpoints import find_default_checkpoint
from tinyadapt.utils.seed import seed_everything


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TinyAdapt memory, drift, latency, and training benchmarks.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--training-epochs", type=int, default=2)
    args = parser.parse_args()

    seed_everything()
    checkpoint = args.checkpoint or find_default_checkpoint()
    checkpoint_str = str(checkpoint) if checkpoint is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    memory = [row.to_dict() for row in benchmark_memory(checkpoint=checkpoint_str, lora_rank=args.rank, group_size=args.group_size)]
    drift = [
        row.to_dict()
        for row in benchmark_drift(
            checkpoint=checkpoint_str,
            batch_size=args.batch_size,
            lora_rank=args.rank,
            group_size=args.group_size,
        )
    ]
    latency = [
        row.to_dict()
        for row in benchmark_latency(
            checkpoint=checkpoint_str,
            batch_size=args.batch_size,
            lora_rank=args.rank,
            group_size=args.group_size,
        )
    ]
    training = [
        row.to_dict()
        for row in benchmark_adapter_training(
            checkpoint=checkpoint_str,
            epochs=args.training_epochs,
            lora_rank=args.rank,
            group_size=args.group_size,
        )
    ]

    payload = {
        "checkpoint": checkpoint_str,
        "memory": memory,
        "drift": drift,
        "latency": latency,
        "training": training,
    }
    (args.output_dir / "benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "memory.csv", memory)
    _write_csv(args.output_dir / "drift.csv", drift)
    _write_csv(args.output_dir / "latency.csv", latency)
    _write_csv(args.output_dir / "training.csv", training)
    print(f"Wrote benchmark results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

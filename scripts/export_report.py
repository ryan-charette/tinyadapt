from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyadapt.utils.stats import format_bytes


def _maybe_write_figures(payload: dict[str, object], output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    memory = {row["name"]: row for row in payload.get("memory", [])}
    training = payload.get("training", [])
    drift = payload.get("drift", [])
    latency = payload.get("latency", [])

    if training:
        names = [row["model_name"] for row in training]
        accuracy = [row["validation_accuracy"] for row in training]
        storage_mb = [memory.get(name, {}).get("storage_bytes", 0) / 2**20 for name in names]
        trainable = [row["trainable_parameters"] for row in training]

        path = output_dir / "memory_vs_accuracy.png"
        plt.figure(figsize=(7, 4))
        plt.scatter(storage_mb, accuracy)
        for name, x_value, y_value in zip(names, storage_mb, accuracy):
            plt.annotate(name, (x_value, y_value), textcoords="offset points", xytext=(4, 4))
        plt.xlabel("Estimated storage (MB)")
        plt.ylabel("Validation accuracy")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        figure_paths.append(path)

        path = output_dir / "trainable_params_vs_accuracy.png"
        plt.figure(figsize=(7, 4))
        plt.scatter(trainable, accuracy)
        for name, x_value, y_value in zip(names, trainable, accuracy):
            plt.annotate(name, (x_value, y_value), textcoords="offset points", xytext=(4, 4))
        plt.xlabel("Trainable parameters")
        plt.ylabel("Validation accuracy")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        figure_paths.append(path)

    if drift:
        path = output_dir / "output_drift_by_model.png"
        plt.figure(figsize=(7, 4))
        plt.bar([row["name"] for row in drift], [row["mean_absolute_difference"] for row in drift])
        plt.ylabel("Mean absolute difference")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        figure_paths.append(path)

    if latency and training:
        accuracy_by_name = {row["model_name"]: row["validation_accuracy"] for row in training}
        rows = [row for row in latency if row["name"] in accuracy_by_name]
        if rows:
            path = output_dir / "latency_vs_accuracy.png"
            plt.figure(figsize=(7, 4))
            x_values = [row["mean_latency_ms"] for row in rows]
            y_values = [accuracy_by_name[row["name"]] for row in rows]
            plt.scatter(x_values, y_values)
            for row, x_value, y_value in zip(rows, x_values, y_values):
                plt.annotate(row["name"], (x_value, y_value), textcoords="offset points", xytext=(4, 4))
            plt.xlabel("Mean latency (ms)")
            plt.ylabel("Validation accuracy")
            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()
            figure_paths.append(path)

    return figure_paths


def _markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    headers = [label for label, _key in columns]
    keys = [key for _label, key in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a TinyAdapt markdown report from benchmark JSON.")
    parser.add_argument("--input", type=Path, default=Path("reports/benchmark_results.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/compression_report.md"))
    args = parser.parse_args()

    if args.input.exists():
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    else:
        payload = {"checkpoint": None, "memory": [], "drift": [], "latency": [], "training": []}

    memory_rows = []
    for row in payload.get("memory", []):
        memory_rows.append(
            {
                "Model": row["name"],
                "Total Parameters": f"{row['total_parameters']:,}",
                "Trainable Parameters": f"{row['trainable_parameters']:,}",
                "Storage": format_bytes(int(row["storage_bytes"])),
                "Reduction vs FP32": f"{row['reduction_vs_fp32']:.1f}%",
            }
        )

    drift_rows = []
    for row in payload.get("drift", []):
        drift_rows.append(
            {
                "Model": row["name"],
                "Mean Abs Diff": f"{row['mean_absolute_difference']:.6f}",
                "Max Abs Diff": f"{row['max_absolute_difference']:.6f}",
                "Relative Error": f"{row['relative_error']:.6f}",
            }
        )

    training_rows = []
    for row in payload.get("training", []):
        training_rows.append(
            {
                "Model": row["model_name"],
                "Trainable Params": f"{row['trainable_parameters']:,}",
                "Validation Accuracy": f"{row['validation_accuracy']:.3f}",
                "Final Loss": f"{row['final_loss']:.4f}",
                "Seconds/Epoch": f"{row['seconds_per_epoch']:.3f}",
            }
        )

    figures = _maybe_write_figures(payload, args.output.parent / "figures")
    figures_markdown = "\n".join(f"![{path.stem}](figures/{path.name})" for path in figures)

    report = f"""# TinyAdapt Compression Report

## Introduction

TinyAdapt compares a full-precision residual MLP against FP16 inference, blockwise 4-bit quantization, LoRA, and QLoRA-style adaptation.

## Motivation

The goal is to measure the practical tradeoff between model storage, output fidelity, inference speed, and downstream adaptation cost.

## Baseline Model

The shared backbone is `BigNet`, a residual MLP over 1024-dimensional vectors. All variants expose the same `model(x)` interface.

## Results

### Memory

{_markdown_table(memory_rows, [("Model", "Model"), ("Total Parameters", "Total Parameters"), ("Trainable Parameters", "Trainable Parameters"), ("Storage", "Storage"), ("Reduction vs FP32", "Reduction vs FP32")]) if memory_rows else "Run `python scripts/run_all_benchmarks.py` to populate this table."}

### Output Drift

{_markdown_table(drift_rows, [("Model", "Model"), ("Mean Abs Diff", "Mean Abs Diff"), ("Max Abs Diff", "Max Abs Diff"), ("Relative Error", "Relative Error")]) if drift_rows else "Run `python scripts/run_all_benchmarks.py` to populate this table."}

### Downstream Adaptation

{_markdown_table(training_rows, [("Model", "Model"), ("Trainable Params", "Trainable Params"), ("Validation Accuracy", "Validation Accuracy"), ("Final Loss", "Final Loss"), ("Seconds/Epoch", "Seconds/Epoch")]) if training_rows else "Run `python scripts/run_all_benchmarks.py` to populate this table."}

### Figures

{figures_markdown if figures_markdown else "Install `matplotlib` and rerun this script after benchmarking to generate charts."}

## Tradeoff Analysis

FP16 is the simplest storage reduction method. It keeps the architecture unchanged and normally has the smallest output drift. The 4-bit model stores far fewer bytes, but reconstructs weights during the forward pass. LoRA reduces fine-tuning cost by training only low-rank adapters. QLoRA combines a compressed frozen base with trainable adapters.

## Limitations

This project uses a compact residual MLP and an offline synthetic classification task. Larger pretrained transformer backbones, calibration data, and GPU-specific kernels would be useful next steps.

## Future Work

- Add GPU latency and memory benchmarks.
- Compare multiple LoRA ranks and quantization block sizes.
- Add ONNX export and a pretrained transformer variant.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()

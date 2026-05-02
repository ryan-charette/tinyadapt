from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyadapt.models import create_model
from tinyadapt.training import train_classifier
from tinyadapt.utils.checkpoints import find_default_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QLoRA adapters on the TinyAdapt synthetic task.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_default_checkpoint()
    model = create_model("qlora", checkpoint, lora_rank=args.rank, group_size=args.group_size)
    result = train_classifier(model, model_name="qlora", epochs=args.epochs, lr=args.lr)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

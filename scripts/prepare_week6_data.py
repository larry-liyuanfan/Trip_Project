"""Create a deterministic, immutable Week 6 train/validation data lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.week6_data import lock_week6_data
from src.training.week6_qlora import Week6TrainingError, load_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/week6/qwen3_vl_8b_qlora.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        payload = lock_week6_data(root, load_training_config(root / args.config))
    except Week6TrainingError as exc:
        raise SystemExit(f"Week 6 data lock error: {exc}") from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

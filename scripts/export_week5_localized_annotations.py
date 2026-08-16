"""Export a Chinese review mirror without changing canonical Week 5 labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.week5_dataset import load_week5_config
from src.data.week5_localization import export_localized_annotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/week5_dataset_qwen3_vl_4b_single_operator.json",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_week5_config(root, args.config)
    print(
        json.dumps(
            export_localized_annotations(root, config),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

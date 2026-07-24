"""Prepare curated Week 3 v2 manifests and human annotation packets."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.config import load_evaluation_config
from src.evaluation.v2_dataset import prepare_curated_v2_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", default="configs/evaluation_week3.yaml")
    parser.add_argument("--target-config", default="configs/evaluation_week3_v2.yaml")
    parser.add_argument(
        "--replacement-after-sales-manifest",
        default=(
            "data/eval/backups/frozen_restore_20260721/"
            "pending_after_sales_v3_before_frozen_restore.jsonl"
        ),
    )
    args = parser.parse_args()
    root = Path.cwd()
    result = prepare_curated_v2_dataset(
        root=root,
        source_config=load_evaluation_config(args.source_config),
        target_config=load_evaluation_config(args.target_config),
        replacement_after_sales_manifest=root / args.replacement_after_sales_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

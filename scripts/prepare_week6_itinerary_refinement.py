"""Create a versioned derived-silver itinerary refinement dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.week6_qlora import Week6TrainingError
from src.training.week6_refinement import build_itinerary_refinement_lock


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--train-input", type=Path, required=True)
    result.add_argument("--validation-input", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--dataset-version", required=True)
    result.add_argument("--silver-weight", type=float, default=0.5)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        payload = build_itinerary_refinement_lock(
            Path.cwd(),
            train_path=args.train_input,
            validation_path=args.validation_input,
            output_dir=args.output_dir,
            dataset_version=args.dataset_version,
            silver_weight=args.silver_weight,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    except Week6TrainingError as exc:
        raise SystemExit(f"Week 6 refinement error: {exc}") from exc


if __name__ == "__main__":
    main()

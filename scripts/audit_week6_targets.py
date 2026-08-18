"""Audit locked Week 6 targets without modifying datasets or model outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.week6_qlora import Week6TrainingError, iter_training_rows
from src.training.week6_quality import summarize_itinerary_targets


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument(
        "--scenario", choices=("itinerary_planning",), required=True
    )
    result.add_argument("--max-examples-per-issue", type=int, default=5)
    result.add_argument("--output", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        payload = summarize_itinerary_targets(
            Path.cwd(),
            iter_training_rows(args.input, scenario=args.scenario),
            max_examples_per_issue=args.max_examples_per_issue,
        )
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8", newline="\n")
        print(text, end="")
    except Week6TrainingError as exc:
        raise SystemExit(f"Week 6 target audit error: {exc}") from exc


if __name__ == "__main__":
    main()

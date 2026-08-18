"""Evaluate a Week 6 itinerary adapter on a locked validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.week6_evaluation import (
    compare_itinerary_evaluations,
    run_itinerary_adapter_evaluation,
)
from src.training.week6_qlora import Week6TrainingError, load_training_config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--eval-input", type=Path, required=True)
    run.add_argument("--adapter-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--max-samples", type=int)
    run.add_argument("--max-new-tokens", type=int, default=2048)
    run.add_argument("--adapter-role", choices=("initial", "candidate"), default="initial")
    run.add_argument("--resume", action="store_true")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline-summary", type=Path, required=True)
    compare.add_argument("--candidate-summary", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "compare":
            baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
            candidate = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
            payload = compare_itinerary_evaluations(baseline, candidate)
            if args.output.exists():
                raise Week6TrainingError("refusing to overwrite a comparison output")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            if payload["status"] != "passed":
                raise SystemExit(2)
            return
        payload = run_itinerary_adapter_evaluation(
            Path.cwd(),
            load_training_config(args.config),
            eval_path=args.eval_input,
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
            adapter_role=args.adapter_role,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    except Week6TrainingError as exc:
        raise SystemExit(f"Week 6 adapter evaluation error: {exc}") from exc


if __name__ == "__main__":
    main()

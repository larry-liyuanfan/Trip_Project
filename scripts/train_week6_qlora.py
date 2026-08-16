"""Explicit Week 6 QLoRA environment, data, and small-pilot entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.week6_qlora import (
    SCENARIOS,
    Week6TrainingError,
    environment_report,
    iter_training_rows,
    load_training_config,
    run_small_sample_training,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Week 6 Qwen3-VL-8B QLoRA pilot")
    result.add_argument("--config", type=Path, default=Path("configs/week6/qwen3_vl_8b_qlora.json"))
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("check-environment")
    validate = sub.add_parser("validate-data")
    validate.add_argument("--scenario", choices=SCENARIOS, required=True)
    validate.add_argument("--input", type=Path, required=True)
    train = sub.add_parser("train-pilot")
    train.add_argument("--scenario", choices=SCENARIOS, required=True)
    train.add_argument("--train-input", type=Path, required=True)
    train.add_argument("--eval-input", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--confirm-dataset-lock", action="store_true")
    train.add_argument("--resume-from-checkpoint", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        config = load_training_config(args.config)
        if args.command == "check-environment":
            payload = environment_report(require_cuda=True)
        elif args.command == "validate-data":
            count = sum(1 for _ in iter_training_rows(args.input, scenario=args.scenario))
            payload = {"status": "ok", "scenario": args.scenario, "records": count}
        else:
            payload = run_small_sample_training(
                config,
                scenario=args.scenario,
                train_path=args.train_input,
                eval_path=args.eval_input,
                output_dir=args.output_dir,
                dataset_lock_confirmed=args.confirm_dataset_lock,
                resume_from_checkpoint=args.resume_from_checkpoint,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    except Week6TrainingError as exc:
        raise SystemExit(f"Week 6 training error: {exc}") from exc


if __name__ == "__main__":
    main()

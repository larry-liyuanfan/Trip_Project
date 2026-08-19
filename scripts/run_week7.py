#!/usr/bin/env python3
"""Run locked Week 7 development experiments and multitask training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.week6_qlora import environment_report
from src.training.week7_inference import run_schema_experiment, run_transformers_development
from src.training.week7_qlora import Week7TrainingError, run_multitask_training


def build_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v1.json")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("check-environment")
    train = commands.add_parser("train-multitask")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--confirm-dataset-lock", action="store_true")
    train.add_argument("--resume-from-checkpoint", type=Path)
    infer = commands.add_parser("evaluate-development")
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--run-id", required=True)
    infer.add_argument("--adapter-dir", type=Path)
    infer.add_argument("--model-role", default="zero_shot")
    infer.add_argument("--max-new-tokens", type=int, default=2048)
    schema = commands.add_parser("schema-experiment")
    schema.add_argument("--output-dir", type=Path, required=True)
    schema.add_argument("--endpoint", required=True)
    schema.add_argument("--served-model", required=True)
    schema.add_argument("--timeout", type=int, default=300)
    return result


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "check-environment":
            payload = environment_report(require_cuda=True)
        elif args.command == "train-multitask":
            payload = run_multitask_training(
                ROOT, args.config, args.output_dir,
                confirm_dataset_lock=args.confirm_dataset_lock,
                resume_from_checkpoint=args.resume_from_checkpoint,
            )
        elif args.command == "evaluate-development":
            payload = run_transformers_development(
                ROOT, args.config, args.output_dir, run_id=args.run_id,
                adapter_dir=args.adapter_dir, model_role=args.model_role,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            payload = run_schema_experiment(
                ROOT, args.config, args.output_dir, endpoint=args.endpoint,
                served_model=args.served_model, timeout=args.timeout,
            )
    except (OSError, ValueError, Week7TrainingError) as exc:
        raise SystemExit(f"Week 7 execution error: {exc}") from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

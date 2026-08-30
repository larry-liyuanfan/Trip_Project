#!/usr/bin/env python3
"""Manage the Week 8 two-stage product candidate and silver continuation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.week8_product_two_stage import (  # noqa: E402
    build_hard_slice_silver_lock,
    load_two_stage_config,
    run_two_stage_continuation_sft,
    run_two_stage_development,
    validate_hard_slice_lock,
)


DEFAULT_CONFIG = ROOT / "configs/week8/product_two_stage_v1.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-config")
    commands.add_parser("build-hard-slice")
    commands.add_parser("validate-hard-slice")
    development = commands.add_parser("development")
    development.add_argument("--output-dir", type=Path, required=True)
    training = commands.add_parser("train")
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--resume-from-checkpoint", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "validate-config":
        config = load_two_stage_config(args.config)
        result = {
            "status": "PASS",
            "schema_version": config["schema_version"],
            "label_provenance": config["policy"]["new_label_provenance"],
            "test_access": config["policy"]["final_test_access"],
        }
    elif args.command == "build-hard-slice":
        result = build_hard_slice_silver_lock(ROOT, args.config)
    elif args.command == "validate-hard-slice":
        _, lock = validate_hard_slice_lock(ROOT, args.config)
        result = {
            "status": "PASS",
            "dataset_version": lock["dataset_version"],
            "lock_sha256": lock["lock_sha256"],
            "final_test_accessed": lock["final_test_accessed"],
        }
    elif args.command == "development":
        result = run_two_stage_development(ROOT, args.config, args.output_dir)
    else:
        result = run_two_stage_continuation_sft(
            ROOT,
            args.config,
            args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

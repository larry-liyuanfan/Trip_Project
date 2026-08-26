#!/usr/bin/env python3
"""Manage the optional Week 8 product-only continuation SFT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.week8_product_sft import (  # noqa: E402
    load_week8_product_sft_config,
    run_week8_product_continuation_sft,
    select_week8_product_sft_candidate,
    validate_week8_product_sft_eligibility,
)


DEFAULT_CONFIG = ROOT / "configs/week8/product_continuation_sft_v1.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = value.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-config")
    eligibility = subparsers.add_parser("validate-eligibility")
    eligibility.add_argument("--prompt-development-dir", type=Path, required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--prompt-development-dir", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--resume-from-checkpoint", type=Path)
    selection = subparsers.add_parser("select-candidate")
    selection.add_argument("--prompt-development-dir", type=Path, required=True)
    selection.add_argument("--training-dir", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "validate-config":
        config = load_week8_product_sft_config(args.config)
        result = {
            "status": "PASS",
            "schema_version": config["schema_version"],
            "run_id": config["experiment_identity"]["run_id"],
        }
    elif args.command == "validate-eligibility":
        result = validate_week8_product_sft_eligibility(
            ROOT, args.config, args.prompt_development_dir
        )
    elif args.command == "train":
        result = run_week8_product_continuation_sft(
            ROOT,
            args.config,
            args.prompt_development_dir,
            args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    else:
        result = select_week8_product_sft_candidate(
            ROOT,
            args.config,
            args.prompt_development_dir,
            args.training_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"FAIL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

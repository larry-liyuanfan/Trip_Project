#!/usr/bin/env python3
"""Manage the scoped Week 8 product-understanding optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.week8_product import (  # noqa: E402
    build_week8_product_lock,
    run_final_test_once,
    run_prompt_development,
    validate_week8_product_lock,
)


DEFAULT_CONFIG = ROOT / "configs/week8/product_understanding_v1.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = value.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-lock")
    build.add_argument("--source-project-root", type=Path)
    sub.add_parser("validate-lock")
    development = sub.add_parser("prompt-development")
    development.add_argument("--output-dir", type=Path, required=True)
    final_test = sub.add_parser("final-test-once")
    final_test.add_argument("--development-dir", type=Path, required=True)
    final_test.add_argument("--output-dir", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "build-lock":
        result = build_week8_product_lock(
            ROOT, args.config, source_root=args.source_project_root
        )
    elif args.command == "validate-lock":
        result = validate_week8_product_lock(ROOT, args.config)
    elif args.command == "prompt-development":
        result = run_prompt_development(ROOT, args.config, args.output_dir)
    else:
        result = run_final_test_once(
            ROOT, args.config, args.development_dir, args.output_dir
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"FAIL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

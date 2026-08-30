#!/usr/bin/env python3
"""Build and validate the immutable Week 7 data lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.week7_data import build_week7_lock, validate_week7_lock


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v4_fix2.json",
    )
    subcommands = command.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build-lock")
    build.add_argument("--source-project-root", type=Path, required=True)
    subcommands.add_parser("validate-lock")
    return command


def main() -> int:
    args = parser().parse_args()
    config = args.config.resolve()
    if args.command == "build-lock":
        output = build_week7_lock(ROOT, args.source_project_root, config)
        print(json.dumps({"status": "PASS", "lock_root": str(output)}, ensure_ascii=False))
    else:
        result = validate_week7_lock(ROOT, config)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

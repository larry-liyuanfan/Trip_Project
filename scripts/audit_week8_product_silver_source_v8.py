#!/usr/bin/env python3
"""Audit untouched Yelp sources for automatic Week 8 product silver evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.week8_product_silver_source_v8 import (  # noqa: E402
    audit_silver_sources,
    load_silver_source_audit_config,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/week8/product_silver_source_audit_v8.json",
    )
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-config")
    audit = commands.add_parser("audit")
    audit.add_argument("--output-dir", type=Path)
    audit.add_argument("--photos-zip", type=Path)
    audit.add_argument("--run-ocr", action="store_true")
    audit.add_argument("--skip-zip-sha256", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "validate-config":
        config = load_silver_source_audit_config(args.config)
        result = {
            "status": "PASS",
            "schema_version": config["schema_version"],
            "label_provenance": config["policy"]["label_provenance"],
            "final_test_access": False,
        }
    else:
        if args.run_ocr and args.photos_zip is None:
            raise SystemExit("--run-ocr requires --photos-zip")
        result = audit_silver_sources(
            ROOT,
            args.config,
            output_dir=args.output_dir,
            photos_zip=args.photos_zip,
            run_ocr=args.run_ocr,
            verify_zip_sha256=not args.skip_zip_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

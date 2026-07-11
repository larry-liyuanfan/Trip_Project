"""Validate the complete Week 2 output contract from the shared config."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_validation import validate_week2_outputs
from src.data.yelp_paths import load_config


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for output validation."""
    parser = argparse.ArgumentParser(description="Validate generated Week 2 Yelp data pipeline outputs.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_processing.yaml"))
    return parser


def main() -> None:
    """Print validation evidence and return a failing exit code on errors."""
    args = build_arg_parser().parse_args()
    result = validate_week2_outputs(load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

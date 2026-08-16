"""Analyze Week 4 prompt pilots and optionally the winner-only full run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.week4_analysis import analyze_full_run, analyze_pilot_runs
from src.evaluation.week4_runner import load_week4_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week4.yaml")
    parser.add_argument("--pilot-run-id", action="append", default=[])
    parser.add_argument("--full-run-id")
    args = parser.parse_args()
    if len(args.pilot_run_id) not in {0, 3}:
        parser.error("--pilot-run-id must be supplied exactly three times")
    root = Path.cwd()
    config = load_week4_config(root, Path(args.config))
    payload = {}
    if args.pilot_run_id:
        payload["pilot"] = analyze_pilot_runs(
            root=root,
            week4_config=config,
            pilot_run_ids=args.pilot_run_id,
        )
    if args.full_run_id:
        payload["full"] = analyze_full_run(
            root=root,
            week4_config=config,
            full_run_id=args.full_run_id,
        )
    if not payload:
        parser.error("provide pilot run IDs and/or a full run ID")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

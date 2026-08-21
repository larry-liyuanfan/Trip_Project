"""Run the local-only Week 7 dialogue human-scoring station."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.week7_dialogue_review import (
    EXPECTED_DATASET_VERSION,
    EXPECTED_MODEL_NAME,
    EXPECTED_RAW_SHA256,
    EXPECTED_RUN_ID,
    create_week7_dialogue_review_app,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 7 dialogue human-scoring station")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument(
        "--dataset-dir",
        default="outputs/week7/locked_data/week7_fresh_multitask_context_20260820_v3",
    )
    parser.add_argument(
        "--raw-outputs",
        default="outputs/week7/human_review/source/multitask_step_000151_raw_outputs.jsonl",
    )
    parser.add_argument("--output-dir", default="outputs/week7/human_review")
    parser.add_argument("--raw-sha256", default=EXPECTED_RAW_SHA256)
    parser.add_argument("--dataset-version", default=EXPECTED_DATASET_VERSION)
    parser.add_argument("--run-id", default=EXPECTED_RUN_ID)
    parser.add_argument("--model-name", default=EXPECTED_MODEL_NAME)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Week 7 human scoring must remain local-only")
    root = Path(__file__).resolve().parents[1]
    app = create_week7_dialogue_review_app(
        root,
        dataset_dir=Path(args.dataset_dir),
        raw_outputs_path=Path(args.raw_outputs),
        output_dir=Path(args.output_dir),
        expected_raw_sha256=args.raw_sha256,
        expected_dataset_version=args.dataset_version,
        expected_run_id=args.run_id,
        expected_model_name=args.model_name,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

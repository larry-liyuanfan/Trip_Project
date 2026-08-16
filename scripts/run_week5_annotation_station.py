"""Run the local-only Week 5 annotation station."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.week5_annotation_station import create_annotation_station


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 5 human annotation station")
    parser.add_argument(
        "--config",
        default="configs/week5_dataset_qwen3_vl_4b_single_operator.json",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--dialogue-run-id")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("annotation station must remain local-only")
    root = Path(__file__).resolve().parents[1]
    uvicorn.run(
        create_annotation_station(root, args.config, dialogue_run_id=args.dialogue_run_id),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()

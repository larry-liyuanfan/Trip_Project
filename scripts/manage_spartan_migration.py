"""Prepare, inspect, and merge a versioned Spartan Week 5 migration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.spartan_migration import (
    merge_spartan_migration,
    prepare_spartan_migration,
    spartan_migration_status,
)
from src.data.week5_dataset import Week5DataError, load_week5_config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Manage Week 5 Spartan migration artifacts")
    sub = result.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", default="configs/week5_dataset_qwen3_vl_4b_gpu.json")
    prepare.add_argument("--source-run-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--migration-id", required=True)
    prepare.add_argument("--shard-count", type=int, default=4)
    prepare.add_argument("--benchmark-count", type=int, default=100)
    status = sub.add_parser("status")
    status.add_argument("--migration-dir", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--migration-dir", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    return result


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main() -> None:
    args = parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        if args.command == "prepare":
            config = load_week5_config(root, args.config)
            payload = prepare_spartan_migration(
                root,
                config,
                source_run_dir=_resolve(root, args.source_run_dir),
                output_dir=_resolve(root, args.output_dir),
                migration_id=args.migration_id,
                shard_count=args.shard_count,
                benchmark_count=args.benchmark_count,
            )
        elif args.command == "status":
            payload = spartan_migration_status(root, _resolve(root, args.migration_dir))
        else:
            payload = merge_spartan_migration(
                root,
                _resolve(root, args.migration_dir),
                _resolve(root, args.output_dir),
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    except Week5DataError as exc:
        raise SystemExit(f"Spartan migration error: {exc}") from exc


if __name__ == "__main__":
    main()

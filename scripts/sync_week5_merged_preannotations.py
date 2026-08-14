"""Create canonical Week 5 preannotations from the final Spartan merge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.week5_dataset import Week5DataError
from src.data.week5_merge_sync import sync_merged_preannotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-dir", type=Path, required=True)
    parser.add_argument("--preserved-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = sync_merged_preannotations(
            merge_dir=args.merge_dir,
            preserved_dir=args.preserved_dir,
            output_dir=args.output_dir,
        )
    except Week5DataError as exc:
        raise SystemExit(f"Week 5 merge sync error: {exc}") from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

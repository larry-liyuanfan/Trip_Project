"""Export and transactionally apply human Week 3 annotation packets."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.annotation_workflow import (
    ANNOTATION_FIELDS,
    apply_annotations,
    export_packet,
)
from src.evaluation.config import load_evaluation_config
from src.evaluation.manifests import (
    ManifestValidationError,
    load_manifest,
    read_jsonl_objects,
    write_jsonl,
)


def _strip_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != "context"} for row in rows]


def _write_manifest_atomically(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".apply.tmp")
    if temporary.exists():
        raise ManifestValidationError(f"temporary apply file already exists: {temporary}")
    write_jsonl(temporary, records)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week3.yaml")
    parser.add_argument("--scenario", choices=sorted(ANNOTATION_FIELDS), required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--include-suggestions", action="store_true")
    apply = subparsers.add_parser("apply")
    apply.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    root = Path.cwd()
    config = load_evaluation_config(args.config)
    manifest_path = root / config["scenarios"][args.scenario]["manifest_path"]
    records = load_manifest(manifest_path, root=root)
    if args.command == "export":
        if args.output.exists():
            raise ManifestValidationError(f"refusing to overwrite packet: {args.output}")
        rows = export_packet(
            records,
            scenario=args.scenario,
            stage="annotation",
            include_suggestions=args.include_suggestions,
        )
        write_jsonl(args.output, rows)
        result = {"status": "exported", "stage": "annotation", "record_count": len(rows)}
    else:
        rows = _strip_context(read_jsonl_objects(args.input))
        updated = apply_annotations(records, rows)
        _write_manifest_atomically(manifest_path, updated)
        result = {"status": "applied", "stage": "annotation", "record_count": len(rows)}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

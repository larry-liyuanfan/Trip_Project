"""Validate a custodian identity-only export without authorizing model execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.protected_identity_inventory import validate_identity_only_import


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--export-manifest-sha256", required=True)
    parser.add_argument("--approved-sources", type=Path, required=True)
    parser.add_argument("--approved-sources-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite identity import validation")
    try:
        report = validate_identity_only_import(
            bundle_root=args.bundle_root,
            export_manifest_sha256=args.export_manifest_sha256,
            approved_sources_path=args.approved_sources,
            approved_sources_sha256=args.approved_sources_sha256,
            expected_scope_count=12,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({
            "status": "IDENTITY_IMPORT_VALIDATION_FAILED",
            "error_type": type(error).__name__,
            "reason": str(error),
            "model_execution_authorized": False,
        }, sort_keys=True), file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in (
        "status", "scope_support", "scope_denominator", "row_support",
        "model_execution_authorized", "coordinator_release_required",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

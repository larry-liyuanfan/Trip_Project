"""Inspect only hash-bound identity metadata; exit 2 when coverage is incomplete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.protected_identity_inventory import audit_from_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite inventory evidence")
    report = audit_from_paths(args.config, args.identity_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in (
        "status", "registry_support", "blocking_registry_ids", "model_execution_authorized",
    )}, indent=2))
    return 2 if report["blocking_registry_ids"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

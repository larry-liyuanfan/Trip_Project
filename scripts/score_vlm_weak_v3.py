"""Score the locked zero/old/current VLM weak v3 one-factor comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evidence_v2 import apply_vlm_v3_gates, score_vlm_v3_comparison
from src.evaluation.relevance_evidence import canonical_json_sha256, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = [row for path in args.result for row in load_jsonl(path)]
    actual_roles = sorted({row.get("variant") for row in rows})
    expected_roles = sorted(config["vlm"]["roles"])
    if actual_roles != expected_roles:
        raise ValueError(f"VLM role support mismatch: {actual_roles} != {expected_roles}")
    report = score_vlm_v3_comparison(rows)
    report["fixed_gates"] = apply_vlm_v3_gates(report, config["vlm"]["fixed_gates"])
    report["result_support"] = len(rows)
    report["result_canonical_sha256"] = canonical_json_sha256(rows)
    report["human_annotation_support"] = 0
    report["fresh_test_used"] = False
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

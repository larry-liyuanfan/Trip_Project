"""Score v5 context-focus development comparison or candidate-only final evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.context_focus_v5 import (
    apply_context_focus_v5_development_gates,
    apply_context_focus_v5_final_gates,
    score_context_focus_v5,
)
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "final"), required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = [row for path in args.result for row in load_jsonl(path)]
    report = score_context_focus_v5(rows)
    candidate = config["vlm"]["candidate_variant"]
    baseline = config["vlm"]["baseline_variant"]
    roles = set(report["variants"])
    if args.split == "development":
        expected = set(config["vlm"]["development_roles"])
        if roles != expected:
            raise ValueError(f"development roles differ: expected={expected} actual={roles}")
        gates = apply_context_focus_v5_development_gates(
            report,
            config["vlm"]["exploration_gates"],
            candidate=candidate,
            baseline=baseline,
        )
    else:
        if roles != {candidate}:
            raise ValueError("final scoring accepts only the development-selected candidate")
        gates = apply_context_focus_v5_final_gates(
            report,
            config["vlm"]["exploration_gates"],
            candidate=candidate,
        )
    report.update({
        "status": "COMPLETED",
        "split": args.split,
        "candidate_variant": candidate,
        "fixed_gates": gates,
        "gate_class": config["gate_class"],
        "human_annotation_support": 0,
        "fresh_test_used": False,
        "raw_result_support": len(rows),
        "raw_result_canonical_sha256": canonical_json_sha256(rows),
        "raw_result_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in args.result
        ],
        "promotion_eligible_as_human_ground_truth": False,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

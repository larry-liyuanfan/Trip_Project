"""Score v4 VLM development comparison or the one-time candidate-only final."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.exploration_v4 import (
    apply_vlm_absolute_v4_gates,
    apply_vlm_exploration_v4_gates,
    score_vlm_exploration_v4,
)
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("development", "final"))
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = [row for path in args.result for row in load_jsonl(path)]
    report = score_vlm_exploration_v4(rows)
    roles = set(report["variants"])
    candidate = config["vlm"]["candidate_variant"]
    if args.split == "development":
        expected_roles = set(config["vlm"]["development_roles"])
        if roles != expected_roles:
            raise ValueError(f"development roles differ: expected={expected_roles} actual={roles}")
        gates = apply_vlm_exploration_v4_gates(
            report,
            config["vlm"]["exploration_gates"],
            config["vlm"]["selection_objective"],
            candidate_variant=candidate,
            baseline_variant=config["vlm"]["baseline_variant"],
        )
    else:
        if roles != {candidate}:
            raise ValueError("final scoring accepts only the development-selected candidate")
        gates = apply_vlm_absolute_v4_gates(
            report, config["vlm"]["exploration_gates"], candidate_variant=candidate
        )
    report.update({
        "schema_version": "vlm_semantic_evidence_v4",
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
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

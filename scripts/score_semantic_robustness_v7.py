"""Score a locked development-only semantic-robustness comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl
from src.evaluation.semantic_robustness_v7 import (
    apply_semantic_robustness_v7_gates,
    score_semantic_robustness_v7,
)


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
    cycle_id = str(config.get("cycle_id", "v7"))
    report = score_semantic_robustness_v7(
        rows,
        cycle_id=cycle_id,
        primary_factor=str(config.get("primary_factor", "robustness_training_data_only")),
    )
    expected_roles = set(config["vlm"]["development_roles"])
    if set(report["variants"]) != expected_roles:
        raise ValueError(f"{cycle_id} development roles differ from the fixed comparison")
    gates = apply_semantic_robustness_v7_gates(
        report,
        config["vlm"]["exploration_gates"],
        config["vlm"]["selection_objective"],
        candidate=config["vlm"]["candidate_variant"],
        baseline=config["vlm"]["baseline_variant"],
    )
    report.update({
        "status": "COMPLETED",
        "split": "development",
        "candidate_variant": config["vlm"]["candidate_variant"],
        "fixed_gates": gates,
        "gate_class": config["gate_class"],
        "human_annotation_support": 0,
        "final_defined_or_consumed": False,
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

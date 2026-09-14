"""Recompute Search v2 gates offline from the frozen, once-consumed holdout metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evidence_v2 import apply_search_v2_gates
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-summary", type=Path, required=True)
    parser.add_argument("--frozen-holdout-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = json.loads(args.frozen_summary.read_text(encoding="utf-8"))
    metrics = json.loads(args.frozen_holdout_metrics.read_text(encoding="utf-8"))
    original_gate = summary["holdout"]["fixed_gates"]
    recomputed = apply_search_v2_gates(
        metrics,
        summary["holdout"]["ann_fidelity"],
        config["search"]["holdout_gates"],
    )
    report = {
        "schema_version": "search_relevance_v2_gate_recompute",
        "status": "completed",
        "operation": "offline_score_only_no_model_no_search_no_holdout_rerun",
        "selection_changed": False,
        "selected_configuration": summary["selected_configuration"],
        "holdout_consumed_again": False,
        "frozen_inputs": {
            "summary_file_sha256": file_sha256(args.frozen_summary),
            "summary_canonical_sha256": canonical_json_sha256(summary),
            "holdout_metrics_file_sha256": file_sha256(args.frozen_holdout_metrics),
            "holdout_metrics_canonical_sha256": canonical_json_sha256(metrics),
            "holdout_result_canonical_sha256": summary["holdout_result_sha256"],
            "original_slurm_job_id": summary["runtime"]["slurm_job_id"],
        },
        "superseded_gate": {
            "status": "INVALID_SUPERSEDED_AGGREGATE_DENOMINATOR",
            "recorded_value": original_gate,
        },
        "recomputed_fixed_gate": recomputed,
        "fresh_test_used": False,
        "human_annotation_support": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

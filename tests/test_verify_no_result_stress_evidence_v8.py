"""Tests for the independent v8 no-result evidence verifier."""

from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_no_result_stress_v8 import METHODS, _method_summaries, _objective
from scripts.verify_no_result_stress_evidence_v8 import verify_evidence_bundle
from src.evaluation.no_result_stress_v8 import apply_no_result_stress_v8_gate
from src.evaluation.relevance_evidence import _aggregate_search_method, canonical_json_sha256, file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "evaluation" / "automated_evidence_v8_no_result.json"
LOCK_PATH = ROOT / "configs" / "evaluation" / "evidence_enhancement" / "no_result_stress_pool_lock_v8.json"
V4_LOCK_PATH = ROOT / "configs" / "evaluation" / "evidence_enhancement" / "exploration_pool_lock_v4.json"
JOB_ID = "30005527"
SOURCE_SHA = "a" * 64
COMMIT = "b" * 40


class VerifyNoResultStressEvidenceV8Tests(unittest.TestCase):
    def test_valid_bundle_passes_and_selection_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            _write_valid_bundle(output)
            report = _verify(output)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["candidate_support"], 25)
            self.assertEqual(report["fixed_gate_status"], "PASS")

            selection_path = output / "selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["selected"]["configuration"]["margin_threshold"] = 0.1
            _write_json(selection_path, selection)
            with self.assertRaisesRegex(ValueError, "selection mismatch"):
                _verify(output)


def _verify(output: Path) -> dict[str, object]:
    return verify_evidence_bundle(
        config_path=CONFIG,
        output_dir=output,
        expected_job_id=JOB_ID,
        expected_source_snapshot_sha256=SOURCE_SHA,
        expected_implementation_commit=COMMIT,
    )


def _write_valid_bundle(output: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    prior_lock = json.loads(V4_LOCK_PATH.read_text(encoding="utf-8"))
    calibration_rows, calibration_metrics = _scored_split(lock["search"]["calibration"])
    validation_rows, validation_metrics = _scored_split(lock["search"]["validation"])
    guard = config["search"]["dual_centroid_guard"]
    candidate_metrics = _method_summaries(calibration_metrics)["dual_centroid_guard"]
    candidates = []
    for margin, similarity in itertools.product(
        guard["margin_threshold_grid"], guard["business_similarity_threshold_grid"]
    ):
        candidates.append({
            "configuration": {
                "margin_threshold": float(margin),
                "business_similarity_threshold": float(similarity),
            },
            "objective": _objective(candidate_metrics, guard["selection_objective"]),
            "candidate_metrics": candidate_metrics,
        })
    selected = sorted(candidates, key=lambda row: (
        -float(row["objective"]),
        float(row["configuration"]["margin_threshold"]),
        float(row["configuration"]["business_similarity_threshold"]),
    ))[0]
    _write_json(output / "calibration_candidates.json", candidates)
    _write_jsonl(output / "calibration_results.jsonl", calibration_rows)
    _write_json(output / "calibration_metrics.json", calibration_metrics)
    selection = {
        "schema_version": "no_result_stress_v8_calibration_selection",
        "selection_data": "calibration_only",
        "validation_read_for_selection": False,
        "prior_v4_training_use": "business_and_non_business_centroids_only",
        "candidate_support": len(candidates),
        "objective_definition": guard["selection_objective"],
        "selected": selected,
        "calibration_query_lock": lock["search"]["calibration"]["query_manifest_canonical_sha256"],
        "prior_v4_training_query_lock": prior_lock["search"]["training"]["query_manifest_canonical_sha256"],
        "centroid_sha256": "e" * 64,
        "source_snapshot_sha256": SOURCE_SHA,
        "implementation_commit_sha": COMMIT,
        "fresh_test_used": False,
        "final_defined_or_consumed": False,
    }
    _write_json(output / "selection.json", selection)
    _write_json(output / "validation_consumption_marker.json", {
        "schema_version": "no_result_stress_v8_validation_consumption",
        "selection_file_sha256": file_sha256(output / "selection.json"),
        "committed_validation_lock": lock["search"]["validation"],
        "single_consumption_policy": "exclusive_marker_written_before_first_validation_manifest_or_annotation_open",
        "slurm_job_id": JOB_ID,
    })
    _write_jsonl(output / "validation_results.jsonl", validation_rows)
    _write_json(output / "validation_metrics.json", validation_metrics)
    gate = apply_no_result_stress_v8_gate(
        validation_metrics,
        config["search"]["validation_gates"],
        candidate="dual_centroid_guard",
        baseline="v4_margin_guard",
    )
    def query_evidence(split_lock: dict[str, object]) -> dict[str, object]:
        return {
            "status": "PASS",
            "query_count": split_lock["query_support"],
            "manifest_sha256": split_lock["query_manifest_canonical_sha256"],
        }

    def annotation_evidence(split_lock: dict[str, object]) -> dict[str, object]:
        return {
            "status": "PASS",
            "support": split_lock["query_support"],
            "annotation_sha256": split_lock["annotation_canonical_sha256"],
            "human_review_support": 0,
            "promotion_eligible_as_human_ground_truth": False,
        }
    summary = {
        "schema_version": "no_result_stress_evidence_v8",
        "status": "COMPLETED",
        "evidence_class": config["evidence_class"],
        "gate_class": config["gate_class"],
        "human_annotation_support": 0,
        "fresh_test_used": False,
        "final_defined_or_consumed": False,
        "ann_fidelity_scope": "NOT_MEASURED_NO_RESULT_CLASSIFIER_STRESS",
        "selected_configuration": selected["configuration"],
        "selection_file_sha256": file_sha256(output / "selection.json"),
        "configuration": {
            "config_sha256": file_sha256(CONFIG),
            "pool_lock_sha256": canonical_json_sha256(lock),
            "retrieval_archive_sha256": config["formal_release_read_only"]["retrieval_archive_sha256"],
            "source_snapshot_sha256": SOURCE_SHA,
            "implementation_commit_sha": COMMIT,
            "embedding_model": config["search"]["embedding_model"],
            "index_support": config["formal_release_read_only"]["expected_index_support"],
        },
        "denominators": {
            "prior_v4_training": lock["prior_v4_training_isolation"]["compared_prior_query_support"],
            "calibration": lock["search"]["calibration"]["query_support"],
            "validation": lock["search"]["validation"]["query_support"],
            "validation_ranking": lock["search"]["validation"]["ranking_support"],
            "validation_no_result": lock["search"]["validation"]["no_result_support"],
            "validation_business_positive": lock["search"]["validation"]["business_positive_support"],
        },
        "validation": {
            "prior_v4_training_queries": query_evidence(prior_lock["search"]["training"]),
            "prior_v4_training_annotations": annotation_evidence(prior_lock["search"]["training"]),
            "calibration_queries": query_evidence(lock["search"]["calibration"]),
            "calibration_annotations": annotation_evidence(lock["search"]["calibration"]),
            "pool_isolation": lock["isolation"],
            "prior_v4_training_isolation": lock["prior_v4_training_isolation"],
            "validation_queries": query_evidence(lock["search"]["validation"]),
            "validation_annotations": annotation_evidence(lock["search"]["validation"]),
        },
        "calibration_metrics": _method_summaries(calibration_metrics),
        "validation_metrics": _method_summaries(validation_metrics),
        "fixed_gate": gate,
        "runtime": {
            "device": "cuda",
            "gpu": "NVIDIA L40S",
            "python": "3.11.3",
            "platform": "Linux",
            "slurm_job_id": JOB_ID,
        },
        "validation_result_canonical_sha256": canonical_json_sha256(validation_rows),
        "validation_result_file_sha256": file_sha256(output / "validation_results.jsonl"),
        "validation_metrics_file_sha256": file_sha256(output / "validation_metrics.json"),
        "promotion_eligible_as_human_ground_truth": False,
    }
    summary["artifact_sha256"] = canonical_json_sha256(summary)
    _write_json(output / "summary.json", summary)


def _scored_split(split_lock: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    per_query = []
    no_result_support = int(split_lock["no_result_support"])
    support = int(split_lock["query_support"])
    for index in range(support):
        expected_no_result = index < no_result_support
        query_id = f"q-{index:03d}"
        slices = ["no_result"] if expected_no_result else ["business_positive"]
        method_result = {"hits": [], "no_result": expected_no_result}
        rows.append({
            "query_id": query_id,
            "methods": {method: dict(method_result) for method in METHODS},
        })
        per_query.append({
            "query_id": query_id,
            "slices": slices,
            "ranking_evaluable": not expected_no_result,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "mrr_at_10": 1.0,
            "ndcg_at_10": 1.0,
            "no_result_correct": True,
            "filter_correct": True,
            "unsupported_constraints_unapplied": True,
            "failed": False,
        })
    methods = {
        method: _aggregate_search_method([dict(item) for item in per_query], 0)
        for method in METHODS
    }
    return rows, {
        "scope": "business_semantic_relevance",
        "query_support": support,
        "methods": methods,
        "query_manifest_sha256": split_lock["query_manifest_canonical_sha256"],
        "annotation_sha256": split_lock["annotation_canonical_sha256"],
        "result_sha256": canonical_json_sha256(rows),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()

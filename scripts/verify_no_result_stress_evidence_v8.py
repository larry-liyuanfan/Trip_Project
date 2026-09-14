"""Verify completed v8 no-result evidence without rerunning CLIP or selection."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_no_result_stress_v8 import METHODS, _method_summaries, _objective
from src.evaluation.no_result_stress_v8 import apply_no_result_stress_v8_gate
from src.evaluation.relevance_evidence import (
    _aggregate_search_method,
    canonical_json_sha256,
    file_sha256,
    load_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-job-id", required=True)
    parser.add_argument("--expected-source-snapshot-sha256", required=True)
    parser.add_argument("--expected-implementation-commit", required=True)
    args = parser.parse_args()
    report = verify_evidence_bundle(
        config_path=args.config,
        output_dir=args.output_dir,
        expected_job_id=args.expected_job_id,
        expected_source_snapshot_sha256=args.expected_source_snapshot_sha256,
        expected_implementation_commit=args.expected_implementation_commit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def verify_evidence_bundle(
    *,
    config_path: Path,
    output_dir: Path,
    expected_job_id: str,
    expected_source_snapshot_sha256: str,
    expected_implementation_commit: str,
) -> dict[str, Any]:
    _validate_sha(expected_source_snapshot_sha256, "expected source snapshot")
    _validate_commit(expected_implementation_commit)
    config = _load_object(config_path)
    lock_path = Path(config["pool"]["committed_lock"])
    if not lock_path.is_absolute():
        lock_path = config_path.resolve().parents[2] / lock_path
    lock = _load_object(lock_path)
    prior_lock_path = Path(config["pool"]["prior_v4_lock"])
    if not prior_lock_path.is_absolute():
        prior_lock_path = config_path.resolve().parents[2] / prior_lock_path
    prior_lock = _load_object(prior_lock_path)
    if canonical_json_sha256(prior_lock) != lock["prior_v4_training_lock_sha256"]:
        raise ValueError("v8 prior-v4 lock SHA mismatch")
    candidates_path = output_dir / "calibration_candidates.json"
    calibration_rows_path = output_dir / "calibration_results.jsonl"
    calibration_metrics_path = output_dir / "calibration_metrics.json"
    selection_path = output_dir / "selection.json"
    marker_path = output_dir / "validation_consumption_marker.json"
    validation_rows_path = output_dir / "validation_results.jsonl"
    validation_metrics_path = output_dir / "validation_metrics.json"
    summary_path = output_dir / "summary.json"
    candidates = _load_list(candidates_path)
    calibration_rows = load_jsonl(calibration_rows_path)
    calibration_metrics = _load_object(calibration_metrics_path)
    selection = _load_object(selection_path)
    marker = _load_object(marker_path)
    validation_rows = load_jsonl(validation_rows_path)
    validation_metrics = _load_object(validation_metrics_path)
    summary = _load_object(summary_path)

    selected = _validate_candidates(candidates, config, lock["search"]["calibration"])
    _validate_selection(
        selection,
        selected,
        config,
        lock,
        prior_lock,
        expected_source_snapshot_sha256,
        expected_implementation_commit,
    )
    _validate_marker(marker, selection_path, lock, expected_job_id)
    _validate_scored_split(
        calibration_metrics,
        calibration_rows,
        lock["search"]["calibration"],
        "calibration",
    )
    if selected["candidate_metrics"] != _method_summaries(calibration_metrics)["dual_centroid_guard"]:
        raise ValueError("v8 selected calibration metrics differ from the scored calibration split")
    _validate_scored_split(
        validation_metrics,
        validation_rows,
        lock["search"]["validation"],
        "validation",
    )
    gate = apply_no_result_stress_v8_gate(
        validation_metrics,
        config["search"]["validation_gates"],
        candidate="dual_centroid_guard",
        baseline="v4_margin_guard",
    )
    _validate_summary(
        summary,
        config,
        config_path,
        lock,
        prior_lock,
        selection,
        calibration_metrics,
        validation_metrics,
        validation_rows,
        validation_rows_path,
        validation_metrics_path,
        gate,
        expected_job_id,
        expected_source_snapshot_sha256,
        expected_implementation_commit,
    )
    artifact_paths = (
        candidates_path,
        calibration_rows_path,
        calibration_metrics_path,
        selection_path,
        marker_path,
        validation_rows_path,
        validation_metrics_path,
        summary_path,
    )
    return {
        "schema_version": "no_result_stress_evidence_verification_v8",
        "status": "PASS",
        "slurm_job_id": expected_job_id,
        "implementation_commit_sha": expected_implementation_commit,
        "source_snapshot_sha256": expected_source_snapshot_sha256,
        "candidate_support": len(candidates),
        "calibration_query_support": len(calibration_rows),
        "validation_query_support": len(validation_rows),
        "fixed_gate_status": gate["status"],
        "artifact_files": {path.name: file_sha256(path) for path in artifact_paths},
        "ann_fidelity_scope": "NOT_MEASURED_NO_RESULT_CLASSIFIER_STRESS",
        "fresh_test_used": False,
        "final_defined_or_consumed": False,
    }


def _validate_candidates(
    candidates: list[Any], config: dict[str, Any], calibration_lock: dict[str, Any]
) -> dict[str, Any]:
    guard = config["search"]["dual_centroid_guard"]
    expected_grid = {
        (float(margin), float(similarity))
        for margin, similarity in itertools.product(
            guard["margin_threshold_grid"],
            guard["business_similarity_threshold_grid"],
        )
    }
    if len(candidates) != len(expected_grid):
        raise ValueError("v8 calibration candidate support mismatch")
    observed: set[tuple[float, float]] = set()
    for row in candidates:
        if not isinstance(row, dict) or not isinstance(row.get("configuration"), dict):
            raise ValueError("v8 calibration candidate is invalid")
        configuration = row["configuration"]
        key = (
            float(configuration.get("margin_threshold")),
            float(configuration.get("business_similarity_threshold")),
        )
        if key in observed:
            raise ValueError("duplicate v8 calibration configuration")
        observed.add(key)
        metrics = row.get("candidate_metrics", {})
        if (
            metrics.get("support") != calibration_lock["query_support"]
            or metrics.get("ranking_support") != calibration_lock["ranking_support"]
            or metrics.get("slices", {}).get("no_result", {}).get("support")
            != calibration_lock["no_result_support"]
            or metrics.get("slices", {}).get("business_positive", {}).get("support")
            != calibration_lock["business_positive_support"]
            or "per_query" in metrics
        ):
            raise ValueError("v8 calibration candidate metric denominators mismatch")
        expected_objective = _objective(metrics, guard["selection_objective"])
        if not _same_number(row.get("objective"), expected_objective):
            raise ValueError("v8 calibration objective mismatch")
    if observed != expected_grid:
        raise ValueError("v8 calibration grid differs from the fixed configuration")
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["objective"]),
            float(row["configuration"]["margin_threshold"]),
            float(row["configuration"]["business_similarity_threshold"]),
        ),
    )[0]


def _validate_selection(
    selection: dict[str, Any],
    selected: dict[str, Any],
    config: dict[str, Any],
    lock: dict[str, Any],
    prior_lock: dict[str, Any],
    expected_source_sha: str,
    expected_commit: str,
) -> None:
    expected = {
        "schema_version": "no_result_stress_v8_calibration_selection",
        "selection_data": "calibration_only",
        "validation_read_for_selection": False,
        "prior_v4_training_use": "business_and_non_business_centroids_only",
        "candidate_support": len(config["search"]["dual_centroid_guard"]["margin_threshold_grid"])
        * len(config["search"]["dual_centroid_guard"]["business_similarity_threshold_grid"]),
        "objective_definition": config["search"]["dual_centroid_guard"]["selection_objective"],
        "selected": selected,
        "calibration_query_lock": lock["search"]["calibration"]["query_manifest_canonical_sha256"],
        "prior_v4_training_query_lock": prior_lock["search"]["training"]["query_manifest_canonical_sha256"],
        "source_snapshot_sha256": expected_source_sha,
        "implementation_commit_sha": expected_commit,
        "fresh_test_used": False,
        "final_defined_or_consumed": False,
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"v8 selection mismatch: {key}")
    _validate_sha(selection.get("centroid_sha256"), "v8 selection centroid_sha256")


def _validate_marker(
    marker: dict[str, Any], selection_path: Path, lock: dict[str, Any], expected_job_id: str
) -> None:
    expected = {
        "schema_version": "no_result_stress_v8_validation_consumption",
        "selection_file_sha256": file_sha256(selection_path),
        "committed_validation_lock": lock["search"]["validation"],
        "single_consumption_policy": "exclusive_marker_written_before_first_validation_manifest_or_annotation_open",
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"v8 validation marker mismatch: {key}")
    if str(marker.get("slurm_job_id")) != str(expected_job_id):
        raise ValueError("v8 validation marker Slurm job mismatch")


def _validate_scored_split(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    split_lock: dict[str, Any],
    split: str,
) -> None:
    if report.get("scope") != "business_semantic_relevance":
        raise ValueError(f"v8 {split} semantic scope mismatch")
    if report.get("query_support") != split_lock["query_support"] or len(rows) != split_lock["query_support"]:
        raise ValueError(f"v8 {split} query support mismatch")
    if report.get("query_manifest_sha256") != split_lock["query_manifest_canonical_sha256"]:
        raise ValueError(f"v8 {split} query lock mismatch")
    if report.get("annotation_sha256") != split_lock["annotation_canonical_sha256"]:
        raise ValueError(f"v8 {split} annotation lock mismatch")
    if report.get("result_sha256") != canonical_json_sha256(rows):
        raise ValueError(f"v8 {split} result canonical SHA mismatch")
    row_index = {row.get("query_id"): row for row in rows}
    if len(row_index) != len(rows) or None in row_index:
        raise ValueError(f"v8 {split} result query identities are invalid")
    if any(set(row.get("methods", {})) != set(METHODS) for row in rows):
        raise ValueError(f"v8 {split} result methods differ from the fixed comparison")
    methods = report.get("methods", {})
    if set(methods) != set(METHODS):
        raise ValueError(f"v8 {split} metric methods differ from the fixed comparison")
    for method in METHODS:
        metrics = methods[method]
        per_query = metrics.get("per_query")
        if not isinstance(per_query, list) or len(per_query) != len(rows):
            raise ValueError(f"v8 {split} {method} per-query support mismatch")
        metric_index = {item.get("query_id"): item for item in per_query if isinstance(item, dict)}
        if set(metric_index) != set(row_index) or len(metric_index) != len(per_query):
            raise ValueError(f"v8 {split} {method} per-query identities mismatch")
        for query_id, item in metric_index.items():
            result = row_index[query_id]["methods"][method]
            expected_no_result = "no_result" in item.get("slices", [])
            if item.get("no_result_correct") != (expected_no_result == bool(result.get("no_result"))):
                raise ValueError(f"v8 {split} {method} no-result outcome mismatch")
        recomputed = _aggregate_search_method(
            per_query,
            sum(bool(item.get("failed")) for item in per_query),
        )
        if metrics != recomputed:
            raise ValueError(f"v8 {split} {method} aggregate differs from per-query metrics")
    ranking_support = next(iter(methods.values()))["ranking_support"]
    no_result_support = next(iter(methods.values()))["slices"]["no_result"]["support"]
    positive_support = next(iter(methods.values()))["slices"]["business_positive"]["support"]
    if ranking_support != split_lock["ranking_support"]:
        raise ValueError(f"v8 {split} ranking support mismatch")
    if no_result_support != split_lock["no_result_support"]:
        raise ValueError(f"v8 {split} no-result support mismatch")
    if positive_support != split_lock["business_positive_support"]:
        raise ValueError(f"v8 {split} business-positive support mismatch")


def _validate_summary(
    summary: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    lock: dict[str, Any],
    prior_lock: dict[str, Any],
    selection: dict[str, Any],
    calibration_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    validation_rows_path: Path,
    validation_metrics_path: Path,
    gate: dict[str, Any],
    expected_job_id: str,
    expected_source_sha: str,
    expected_commit: str,
) -> None:
    expected_status = "COMPLETED" if gate["status"] == "PASS" else "NEGATIVE_EXPERIMENT_GATE_FAILED"
    expected = {
        "schema_version": "no_result_stress_evidence_v8",
        "status": expected_status,
        "evidence_class": config["evidence_class"],
        "gate_class": config["gate_class"],
        "human_annotation_support": 0,
        "fresh_test_used": False,
        "final_defined_or_consumed": False,
        "ann_fidelity_scope": "NOT_MEASURED_NO_RESULT_CLASSIFIER_STRESS",
        "selected_configuration": selection["selected"]["configuration"],
        "selection_file_sha256": file_sha256(validation_rows_path.parent / "selection.json"),
        "denominators": {
            "prior_v4_training": lock["prior_v4_training_isolation"]["compared_prior_query_support"],
            "calibration": lock["search"]["calibration"]["query_support"],
            "validation": lock["search"]["validation"]["query_support"],
            "validation_ranking": lock["search"]["validation"]["ranking_support"],
            "validation_no_result": lock["search"]["validation"]["no_result_support"],
            "validation_business_positive": lock["search"]["validation"]["business_positive_support"],
        },
        "calibration_metrics": _method_summaries(calibration_metrics),
        "validation_metrics": _method_summaries(validation_metrics),
        "fixed_gate": gate,
        "validation_result_canonical_sha256": canonical_json_sha256(validation_rows),
        "validation_result_file_sha256": file_sha256(validation_rows_path),
        "validation_metrics_file_sha256": file_sha256(validation_metrics_path),
        "promotion_eligible_as_human_ground_truth": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"v8 summary mismatch: {key}")
    configuration = summary.get("configuration", {})
    expected_configuration = {
        "config_sha256": file_sha256(config_path),
        "pool_lock_sha256": canonical_json_sha256(lock),
        "retrieval_archive_sha256": config["formal_release_read_only"]["retrieval_archive_sha256"],
        "source_snapshot_sha256": expected_source_sha,
        "implementation_commit_sha": expected_commit,
        "embedding_model": config["search"]["embedding_model"],
        "index_support": config["formal_release_read_only"]["expected_index_support"],
    }
    if configuration != expected_configuration:
        raise ValueError("v8 summary configuration mismatch")
    validation = summary.get("validation", {})
    if validation.get("pool_isolation") != lock["isolation"]:
        raise ValueError("v8 summary pool isolation mismatch")
    if validation.get("prior_v4_training_isolation") != lock["prior_v4_training_isolation"]:
        raise ValueError("v8 summary prior-training isolation mismatch")
    evidence_locks = {
        "prior_v4_training": prior_lock["search"]["training"],
        "calibration": lock["search"]["calibration"],
        "validation": lock["search"]["validation"],
    }
    for prefix, split_lock in evidence_locks.items():
        query_evidence = validation.get(f"{prefix}_queries", {})
        if (
            query_evidence.get("status") != "PASS"
            or query_evidence.get("query_count") != split_lock["query_support"]
            or query_evidence.get("manifest_sha256") != split_lock["query_manifest_canonical_sha256"]
        ):
            raise ValueError(f"v8 summary query evidence failed: {prefix}")
        annotation_evidence = validation.get(f"{prefix}_annotations", {})
        if (
            annotation_evidence.get("status") != "PASS"
            or annotation_evidence.get("support") != split_lock["query_support"]
            or annotation_evidence.get("annotation_sha256") != split_lock["annotation_canonical_sha256"]
            or annotation_evidence.get("human_review_support") != 0
            or annotation_evidence.get("promotion_eligible_as_human_ground_truth") is not False
        ):
            raise ValueError(f"v8 summary annotation evidence failed: {prefix}")
    runtime = summary.get("runtime", {})
    if str(runtime.get("slurm_job_id")) != str(expected_job_id):
        raise ValueError("v8 runtime Slurm job mismatch")
    for key in ("device", "python", "platform"):
        if runtime.get(key) in (None, ""):
            raise ValueError(f"v8 runtime field is missing: {key}")
    unhashed = dict(summary)
    artifact_sha = unhashed.pop("artifact_sha256", None)
    if artifact_sha != canonical_json_sha256(unhashed):
        raise ValueError("v8 summary artifact canonical SHA mismatch")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_list(path: Path) -> list[Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"JSON array required: {path}")
    return value


def _validate_sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} is not a lowercase SHA-256")


def _validate_commit(value: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("implementation commit is not a full lowercase Git SHA")


def _same_number(left: Any, right: float) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isfinite(float(left))
        and math.isclose(float(left), right, rel_tol=1e-12, abs_tol=1e-12)
    )


if __name__ == "__main__":
    main()

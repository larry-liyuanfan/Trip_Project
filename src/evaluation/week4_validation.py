"""Week 4 运行、哈希、评分与比较产物的统一只读验证。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from src.evaluation.baseline_semantics import BaselineSemanticCoder
from src.evaluation.provenance import (
    SHA256_PATTERN,
    canonical_sha256,
    verify_artifact_hashes,
)
from src.evaluation.results import (
    load_run_metadata,
    validate_result_record,
)


class Week4ValidationError(ValueError):
    """Week 4 持久化证据不满足只读验证契约。"""


def validate_week4_delivery(
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """验证既有 Week 4 产物，不创建或修改任何文件。"""
    project_root = Path(root)
    output_root = project_root / config["paths"]["output_dir"]
    validation = config.get("validation")
    if not isinstance(validation, dict):
        raise Week4ValidationError("config.validation must be an object")
    pilot_run_ids = validation.get("pilot_run_ids")
    full_run_id = validation.get("full_run_id")
    artifact_version = validation.get("artifact_version")
    winners = validation.get("expected_winners")
    expected_full_hash = validation.get("expected_full_sample_sha256")
    fewshot_evidence_status = validation.get("fewshot_evidence_status")
    common_comparison_id = validation.get("common_semantic_comparison_id")
    if (
        not isinstance(pilot_run_ids, list)
        or len(pilot_run_ids) != 3
        or len(set(pilot_run_ids)) != 3
        or not all(isinstance(item, str) and item for item in pilot_run_ids)
    ):
        raise Week4ValidationError("validation.pilot_run_ids must contain three IDs")
    if not isinstance(full_run_id, str) or not full_run_id:
        raise Week4ValidationError("validation.full_run_id is required")
    if (
        not isinstance(artifact_version, str)
        or not artifact_version
        or not artifact_version.replace("_", "").isalnum()
    ):
        raise Week4ValidationError("validation.artifact_version is invalid")
    if not isinstance(winners, dict) or set(winners) != {
        "image_product_search",
        "after_sales",
        "itinerary_planning",
    }:
        raise Week4ValidationError("validation.expected_winners is invalid")
    if (
        not isinstance(expected_full_hash, str)
        or SHA256_PATTERN.fullmatch(expected_full_hash) is None
    ):
        raise Week4ValidationError(
            "validation.expected_full_sample_sha256 is invalid"
        )
    if (
        fewshot_evidence_status
        != "descriptive_only_test_gold_demo_contamination"
    ):
        raise Week4ValidationError(
            "validation.fewshot_evidence_status is invalid"
        )
    if not isinstance(common_comparison_id, str) or not common_comparison_id:
        raise Week4ValidationError(
            "validation.common_semantic_comparison_id is required"
        )

    run_summaries = {}
    for run_id in [*pilot_run_ids, full_run_id]:
        run_summaries[run_id] = _validate_run(output_root, run_id)

    pilot_hashes = {
        run_summaries[run_id]["selected_sample_ids_sha256"]
        for run_id in pilot_run_ids
    }
    if len(pilot_hashes) != 1:
        raise Week4ValidationError("pilot runs do not use the same sample set")
    for run_id in pilot_run_ids:
        summary = run_summaries[run_id]
        if summary["run_scope"] != "pilot" or summary["record_count"] != 15:
            raise Week4ValidationError(f"pilot run shape is invalid: {run_id}")
        if summary["scenario_counts"] != {
            "after_sales": 5,
            "image_product_search": 5,
            "itinerary_planning": 5,
        }:
            raise Week4ValidationError(f"pilot scenario counts are invalid: {run_id}")

    full_summary = run_summaries[full_run_id]
    if full_summary["run_scope"] != "full" or full_summary["record_count"] != 450:
        raise Week4ValidationError("full winner run must contain 450 records")
    if full_summary["selected_sample_ids_sha256"] != expected_full_hash:
        raise Week4ValidationError("full winner sample hash is unexpected")
    if full_summary["scenario_counts"] != {
        "after_sales": 150,
        "image_product_search": 200,
        "itinerary_planning": 100,
    }:
        raise Week4ValidationError("full winner scenario counts are invalid")
    if full_summary["prompt_versions_by_scenario"] != winners:
        raise Week4ValidationError("full winner prompt mapping is invalid")

    comparison_summary = _validate_comparisons(
        output_root=output_root,
        pilot_run_ids=pilot_run_ids,
        full_run_id=full_run_id,
        artifact_version=artifact_version,
        winners=winners,
        full_sample_ids=set(full_summary["sample_ids"]),
        fewshot_evidence_status=fewshot_evidence_status,
        common_comparison_id=common_comparison_id,
    )
    common_summary = _validate_common_semantic_comparison(
        output_root=output_root,
        comparison_id=common_comparison_id,
        baseline_run_id="week3_v2_baseline_full_20260724_001",
        winner_run_id=full_run_id,
        full_sample_ids=set(full_summary["sample_ids"]),
        expected_sample_sha256=expected_full_hash,
    )
    return {
        "status": "ok",
        "message": "Week 4 运行、哈希、记录数、评分和比较产物验证通过",
        "pilot_run_ids": pilot_run_ids,
        "pilot_record_count": 45,
        "model_request_error_count": 0,
        "full_run_id": full_run_id,
        "full_record_count": full_summary["record_count"],
        "full_sample_sha256": full_summary["selected_sample_ids_sha256"],
        "winners": winners,
        "business_comparison_status": "comparable_on_common_semantic_track",
        "fewshot_evidence_status": fewshot_evidence_status,
        **comparison_summary,
        **common_summary,
    }


def _validate_run(output_root: Path, run_id: str) -> dict[str, Any]:
    run_dir = output_root / "runs" / run_id
    metadata = load_run_metadata(run_dir / "metadata.json")
    if metadata["run_id"] != run_id:
        raise Week4ValidationError(f"metadata run_id mismatch: {run_id}")
    if metadata["status"] != "completed":
        raise Week4ValidationError(f"run is not completed: {run_id}")
    verify_artifact_hashes(output_root.parents[1], metadata["artifact_hashes"])
    results = _load_jsonl(run_dir / "results.jsonl")
    if metadata["selected_count"] != len(results) or metadata["record_count"] != len(
        results
    ):
        raise Week4ValidationError(f"run record count mismatch: {run_id}")

    sample_ids = []
    scenario_counts: dict[str, int] = {}
    prompt_versions_by_scenario: dict[str, str] = {}
    model_request_error_count = 0
    for raw_record in results:
        record = validate_result_record(raw_record)
        if record["run_id"] != run_id:
            raise Week4ValidationError(f"result run_id mismatch: {run_id}")
        for field in ("input_sha256", "prompt_artifact_sha256"):
            value = raw_record.get(field)
            if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
                raise Week4ValidationError(f"{field} is invalid: {run_id}")
        if raw_record["input_sha256"] != canonical_sha256(
            raw_record["input_metadata"]
        ):
            raise Week4ValidationError(f"input hash mismatch: {run_id}")
        _validate_token_usage(raw_record.get("token_usage"), run_id)
        error = raw_record.get("error")
        if isinstance(error, str) and error.startswith("model_request_error:"):
            model_request_error_count += 1
        sample_id = record["sample_id"]
        if sample_id in sample_ids:
            raise Week4ValidationError(f"duplicate sample_id: {sample_id}")
        sample_ids.append(sample_id)
        scenario = record["scenario"]
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        existing = prompt_versions_by_scenario.setdefault(
            scenario, record["prompt_version"]
        )
        if existing != record["prompt_version"]:
            raise Week4ValidationError(
                f"scenario mixes prompt versions: {scenario}"
            )

    if canonical_sha256(sample_ids) != metadata["selected_sample_ids_sha256"]:
        raise Week4ValidationError(f"selected sample hash mismatch: {run_id}")
    metadata_prompts = metadata.get("prompt_versions_by_scenario")
    if metadata_prompts != prompt_versions_by_scenario:
        raise Week4ValidationError(f"prompt mapping mismatch: {run_id}")
    if model_request_error_count:
        raise Week4ValidationError(
            f"run contains model request errors: {run_id}, "
            f"count={model_request_error_count}"
        )
    return {
        "run_scope": metadata.get("run_scope"),
        "record_count": len(results),
        "selected_sample_ids_sha256": metadata["selected_sample_ids_sha256"],
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "prompt_versions_by_scenario": prompt_versions_by_scenario,
        "sample_ids": sample_ids,
        "model_request_error_count": model_request_error_count,
    }


def _validate_comparisons(
    *,
    output_root: Path,
    pilot_run_ids: list[str],
    full_run_id: str,
    artifact_version: str,
    winners: dict[str, str],
    full_sample_ids: set[str],
    fewshot_evidence_status: str,
    common_comparison_id: str,
) -> dict[str, Any]:
    comparison_dir = output_root / "comparisons"
    pilot = _load_json(
        comparison_dir / f"pilot_comparison_{artifact_version}.json"
    )
    selected = _load_json(
        comparison_dir / f"selected_prompts_{artifact_version}.json"
    )
    full = _load_json(
        comparison_dir / f"full_baseline_comparison_{artifact_version}.json"
    )
    if pilot.get("selection_scope") != "best_among_tested_candidates":
        raise Week4ValidationError("pilot selection scope is invalid")
    if (
        pilot.get("evidence_status") != fewshot_evidence_status
        or pilot.get("effect_claim_allowed") is not False
    ):
        raise Week4ValidationError(
            "Few-Shot pilot must be marked descriptive-only"
        )
    if pilot.get("pilot_run_ids") != pilot_run_ids:
        raise Week4ValidationError("pilot comparison run IDs are invalid")
    if pilot.get("winners") != winners or selected != winners:
        raise Week4ValidationError("selected Prompt artifacts disagree")
    candidate_summaries = pilot.get("candidate_summaries")
    if not isinstance(candidate_summaries, list) or len(candidate_summaries) != 9:
        raise Week4ValidationError("pilot comparison must contain nine summaries")
    if any(
        not isinstance(row, dict)
        or row.get("model_request_error_count") != 0
        for row in candidate_summaries
    ):
        raise Week4ValidationError(
            "pilot comparison contains model request failures"
        )
    if full.get("full_run_id") != full_run_id:
        raise Week4ValidationError("full comparison run ID is invalid")
    if (
        full.get("common_semantic_comparison_id") != common_comparison_id
        or full.get("fewshot_evidence_status") != fewshot_evidence_status
    ):
        raise Week4ValidationError(
            "full comparison does not bind the review limitations"
        )
    optimized = full.get("optimized_summaries")
    if not isinstance(optimized, list) or {
        (row.get("scenario"), row.get("sample_count"))
        for row in optimized
        if isinstance(row, dict)
    } != {
        ("after_sales", 150),
        ("image_product_search", 200),
        ("itinerary_planning", 100),
    }:
        raise Week4ValidationError("full comparison scenario summaries are invalid")
    comparisons = full.get("baseline_comparison")
    if not isinstance(comparisons, list) or len(comparisons) != 3:
        raise Week4ValidationError("baseline comparison rows are invalid")
    for row in comparisons:
        if (
            not isinstance(row, dict)
            or row.get("business_metrics_comparable") is not False
            or row.get("business_comparison_status")
            != "not_comparable_different_prediction_encodings"
            or row.get("baseline_mean_total_tokens") is not None
            or row.get("baseline_token_status") != "PENDING_not_recorded"
            or "business_quality_delta" in row
        ):
            raise Week4ValidationError(
                "baseline comparison mixes incompatible business tracks"
            )

    score_rows = _load_jsonl(
        output_root / "scores" / full_run_id / "sample_scores.jsonl"
    )
    score_ids = {row.get("sample_id") for row in score_rows}
    if len(score_rows) != 450 or score_ids != full_sample_ids:
        raise Week4ValidationError("full score rows do not match the full run")
    bad_cases = _load_jsonl(
        output_root
        / "bad_cases"
        / f"week4_bad_cases_{artifact_version}.jsonl"
    )
    bad_case_counts: dict[str, int] = {}
    for row in bad_cases:
        if row.get("sample_id") not in full_sample_ids:
            raise Week4ValidationError("bad case references an unknown sample")
        categories = row.get("categories")
        if not isinstance(categories, list) or not categories:
            raise Week4ValidationError("bad case categories are invalid")
        for category in categories:
            bad_case_counts[category] = bad_case_counts.get(category, 0) + 1
    expected_bad_counts = full.get("bad_case_counts")
    if dict(sorted(bad_case_counts.items())) != expected_bad_counts:
        raise Week4ValidationError("bad case counts disagree with full comparison")
    return {
        "score_record_count": len(score_rows),
        "bad_case_record_count": len(bad_cases),
        "bad_case_counts": expected_bad_counts,
    }


def _validate_common_semantic_comparison(
    *,
    output_root: Path,
    comparison_id: str,
    baseline_run_id: str,
    winner_run_id: str,
    full_sample_ids: set[str],
    expected_sample_sha256: str,
) -> dict[str, Any]:
    root = output_root / "common_semantic" / comparison_id
    summary = _load_json(root / "summary.json")
    if (
        summary.get("comparison_id") != comparison_id
        or summary.get("scoring_track")
        != "week4_common_semantic_coding_v1"
        or summary.get("coding_version") != "baseline_semantic_coding_v1"
        or summary.get("baseline_run_id") != baseline_run_id
        or summary.get("winner_run_id") != winner_run_id
        or summary.get("paired_sample_count") != 450
        or summary.get("bootstrap_iterations") != 2000
        or summary.get("gold_joined_after_prediction") is not True
        or summary.get("prediction_input_fields") != [
            "scenario",
            "raw_output",
        ]
        or summary.get("selected_sample_ids_sha256")
        != expected_sample_sha256
    ):
        raise Week4ValidationError(
            "common semantic comparison summary is invalid"
        )
    expected_counts = {
        "after_sales": 150,
        "image_product_search": 200,
        "itinerary_planning": 100,
    }
    if summary.get("scenario_counts") != expected_counts:
        raise Week4ValidationError(
            "common semantic scenario counts are invalid"
        )
    source_paths = {
        "baseline_results_sha256": (
            output_root.parents[1]
            / "data/eval/runs"
            / baseline_run_id
            / "results.jsonl"
        ),
        "winner_results_sha256": (
            output_root / "runs" / winner_run_id / "results.jsonl"
        ),
    }
    coder = BaselineSemanticCoder.from_path(
        output_root.parents[1]
        / "configs/evaluation/baseline_semantic_coding_v1.json"
    )
    if summary.get("codebook_sha256") != coder.codebook_sha256:
        raise Week4ValidationError(
            "common semantic codebook hash is invalid"
        )
    for field, path in source_paths.items():
        if summary.get(field) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise Week4ValidationError(
                f"common semantic source hash mismatch: {field}"
            )
    for name, expected_run_id in (
        ("baseline", baseline_run_id),
        ("winner", winner_run_id),
    ):
        source_rows = _load_jsonl(source_paths[f"{name}_results_sha256"])
        predictions = _load_jsonl(
            root / f"{name}_canonical_predictions.jsonl"
        )
        scores = _load_jsonl(root / f"{name}_score/sample_scores.jsonl")
        prediction_ids = {row.get("sample_id") for row in predictions}
        score_ids = {row.get("sample_id") for row in scores}
        if (
            len(predictions) != 450
            or len(scores) != 450
            or prediction_ids != full_sample_ids
            or score_ids != full_sample_ids
            or any(row.get("run_id") != expected_run_id for row in predictions)
            or any(
                row.get("scoring_track")
                != "week4_common_semantic_coding_v1"
                for row in scores
            )
        ):
            raise Week4ValidationError(
                f"common semantic {name} artifacts are invalid"
            )
        for source, prediction_row, score_row in zip(
            source_rows, predictions, scores
        ):
            expected_prediction = coder.encode(
                scenario=source["scenario"],
                raw_output=source["raw_output"],
            )
            if (
                prediction_row.get("sample_id") != source.get("sample_id")
                or prediction_row.get("prediction") != expected_prediction
                or score_row.get("sample_id") != source.get("sample_id")
                or score_row.get("deterministic_prediction")
                != expected_prediction
            ):
                raise Week4ValidationError(
                    f"common semantic {name} prediction binding is invalid"
                )
    comparison_metadata = _load_json(
        root / "paired_comparison/metadata.json"
    )
    if comparison_metadata != {
        key: summary[key]
        for key in comparison_metadata
    }:
        raise Week4ValidationError(
            "common semantic comparison metadata disagrees with summary"
        )
    with (root / "paired_comparison/aggregate_deltas.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        aggregate_rows = list(csv.DictReader(handle))
    required_metrics = {
        ("image_product_search", "business_category_accuracy"),
        ("image_product_search", "price_range_accuracy"),
        ("after_sales", "issue_type_accuracy"),
        ("after_sales", "severity_accuracy"),
        ("after_sales", "ocr_recall"),
        ("itinerary_planning", "constraint_recognition_accuracy"),
        ("itinerary_planning", "itinerary_element_completeness"),
    }
    actual_metrics = {
        (row.get("scenario"), row.get("metric")) for row in aggregate_rows
    }
    if not required_metrics <= actual_metrics:
        raise Week4ValidationError(
            "common semantic comparison lacks required business metrics"
        )
    return {
        "common_semantic_comparison_id": comparison_id,
        "common_semantic_paired_count": 450,
        "common_semantic_metric_count": len(aggregate_rows),
    }


def _validate_token_usage(value: Any, run_id: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }:
        raise Week4ValidationError(f"token_usage is invalid: {run_id}")
    for count in value.values():
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 0
        ):
            raise Week4ValidationError(f"token_usage count is invalid: {run_id}")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week4ValidationError(f"cannot load JSON artifact {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise Week4ValidationError(f"cannot load JSONL artifact {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise Week4ValidationError(f"JSONL artifact must contain objects: {path}")
    return rows

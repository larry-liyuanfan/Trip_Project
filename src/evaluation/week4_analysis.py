"""Scoring, candidate selection, baseline comparison, and bad-case export."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from src.evaluation.config import load_evaluation_config
from src.evaluation.manifests import load_configured_manifests
from src.evaluation.metrics import (
    build_annotation_index,
    load_metric_aliases,
    score_records,
)


BUSINESS_METRICS = {
    "image_product_search": (
        "business_category_accuracy",
        "price_range_accuracy",
        "style_f1",
        "facility_f1",
        "label_completeness",
    ),
    "after_sales": (
        "issue_type_accuracy",
        "severity_accuracy",
        "key_information_f1",
        "ocr_recall",
        "ocr_exact_match",
    ),
    "itinerary_planning": (
        "constraint_recognition_accuracy",
        "hard_constraint_recall",
        "soft_constraint_recall",
        "itinerary_element_completeness",
    ),
}
BASELINE_RUN_ID = "week3_v2_baseline_full_20260724_001"


def analyze_pilot_runs(
    *,
    root: Path,
    week4_config: dict[str, Any],
    pilot_run_ids: list[str],
) -> dict[str, Any]:
    """Score three fixed pilots and choose one tested candidate per scenario."""
    project_root = Path(root)
    week3 = load_evaluation_config(project_root / week4_config["paths"]["week3_config"])
    annotations = build_annotation_index(
        load_configured_manifests(week3, root=project_root)
    )
    aliases = load_metric_aliases(project_root / week3["metrics"]["aliases_path"])
    output_root = project_root / week4_config["paths"]["output_dir"]
    artifact_version = _artifact_version(week4_config)
    summaries: list[dict[str, Any]] = []
    for run_id in pilot_run_ids:
        results = _load_completed_run(output_root / "runs", run_id)
        sample_scores, _, _ = score_records(results, annotations, aliases)
        summaries.extend(_summarize_run(results, sample_scores))
        _write_or_verify_jsonl(
            output_root / "scores" / run_id / "sample_scores.jsonl",
            sample_scores,
        )
    weights = week4_config["selection_weights"]
    winners: dict[str, str] = {}
    for scenario in BUSINESS_METRICS:
        rows = [row for row in summaries if row["scenario"] == scenario]
        _apply_efficiency(rows, "mean_total_tokens", "token_efficiency")
        _apply_efficiency(rows, "mean_latency_ms", "latency_efficiency")
        for row in rows:
            row["selection_score"] = sum(
                (
                    weights["business_quality"] * row["business_quality"],
                    weights["schema_compliance"] * row["schema_compliance"],
                    weights["json_compliance"] * row["json_compliance"],
                    weights["token_efficiency"] * row["token_efficiency"],
                    weights["latency_efficiency"] * row["latency_efficiency"],
                )
            )
        winner = sorted(
            rows,
            key=lambda row: (
                -row["selection_score"],
                -row["business_quality"],
                -row["schema_compliance"],
                row["mean_total_tokens"] if row["mean_total_tokens"] is not None else float("inf"),
                row["mean_latency_ms"],
                row["prompt_version"],
            ),
        )[0]
        winners[scenario] = winner["prompt_version"]
    comparison = {
        "selection_scope": "best_among_tested_candidates",
        "pilot_run_ids": pilot_run_ids,
        "weights": weights,
        "candidate_summaries": summaries,
        "winners": winners,
    }
    comparison_dir = output_root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        comparison_dir / f"pilot_comparison_{artifact_version}.json",
        comparison,
    )
    _write_csv(
        comparison_dir / f"pilot_comparison_{artifact_version}.csv",
        summaries,
    )
    _write_json(
        comparison_dir / f"selected_prompts_{artifact_version}.json",
        winners,
    )
    return comparison


def analyze_full_run(
    *,
    root: Path,
    week4_config: dict[str, Any],
    full_run_id: str,
) -> dict[str, Any]:
    """Score the winner-only full run and compare it with frozen Week 3 baseline."""
    project_root = Path(root)
    week3 = load_evaluation_config(project_root / week4_config["paths"]["week3_config"])
    annotations = build_annotation_index(
        load_configured_manifests(week3, root=project_root)
    )
    aliases = load_metric_aliases(project_root / week3["metrics"]["aliases_path"])
    output_root = project_root / week4_config["paths"]["output_dir"]
    artifact_version = _artifact_version(week4_config)
    results = _load_completed_run(output_root / "runs", full_run_id)
    sample_scores, aggregates, _ = score_records(results, annotations, aliases)
    score_dir = output_root / "scores" / full_run_id
    _write_or_verify_jsonl(score_dir / "sample_scores.jsonl", sample_scores)
    _write_or_verify_json(score_dir / "aggregate_scores.json", aggregates)
    summaries = _summarize_run(results, sample_scores)
    baseline_results = _load_completed_run(
        project_root / "data/eval/runs",
        BASELINE_RUN_ID,
    )
    comparisons = _build_runtime_comparisons(summaries, baseline_results)
    bad_cases = _build_bad_cases(results, sample_scores)
    payload = {
        "full_run_id": full_run_id,
        "optimized_summaries": summaries,
        "baseline_comparison": comparisons,
        "business_metric_note": (
            "baseline lexical coding and optimized structured JSON scoring "
            "are reported separately and have no business-quality delta"
        ),
        "bad_case_counts": _count_bad_cases(bad_cases),
    }
    comparison_dir = output_root / "comparisons"
    _write_json(
        comparison_dir / f"full_baseline_comparison_{artifact_version}.json",
        payload,
    )
    _write_jsonl(
        output_root / "bad_cases" / f"week4_bad_cases_{artifact_version}.jsonl",
        bad_cases,
    )
    return payload


def _summarize_run(
    results: list[dict[str, Any]],
    sample_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_by_id = {row["sample_id"]: row for row in results}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for score in sample_scores:
        grouped.setdefault(score["scenario"], []).append(score)
    rows = []
    for scenario, scores in sorted(grouped.items()):
        scenario_results = [result_by_id[score["sample_id"]] for score in scores]
        token_values = [
            row.get("token_usage", {}).get("total_tokens")
            for row in scenario_results
            if isinstance(row.get("token_usage"), dict)
            and isinstance(row["token_usage"].get("total_tokens"), int)
        ]
        latencies = [float(row["latency_ms"]) for row in scenario_results]
        rows.append(
            {
                "scenario": scenario,
                "prompt_version": scenario_results[0]["prompt_version"],
                "sample_count": len(scores),
                "business_quality": _business_quality(scenario, scores),
                "json_compliance": _mean(scores, "json_compliance"),
                "schema_compliance": _mean(scores, "schema_pass"),
                "mean_total_tokens": statistics.fmean(token_values) if token_values else None,
                "mean_latency_ms": statistics.fmean(latencies),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "model_request_error_count": sum(
                    isinstance(row.get("error"), str)
                    and row["error"].startswith("model_request_error:")
                    for row in scenario_results
                ),
            }
        )
    return rows


def _summarize_runtime(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["scenario"], []).append(result)
    summaries = []
    for scenario, rows in sorted(grouped.items()):
        tokens = [
            row.get("token_usage", {}).get("total_tokens")
            for row in rows
            if isinstance(row.get("token_usage"), dict)
            and isinstance(row["token_usage"].get("total_tokens"), int)
        ]
        latencies = [float(row["latency_ms"]) for row in rows]
        summaries.append(
            {
                "scenario": scenario,
                "json_compliance": statistics.fmean(
                    float(row["json_valid"]) for row in rows
                ),
                "schema_compliance": statistics.fmean(
                    float(row["schema_valid"]) for row in rows
                ),
                "mean_total_tokens": (
                    statistics.fmean(tokens) if tokens else None
                ),
                "mean_latency_ms": statistics.fmean(latencies),
                "p95_latency_ms": _percentile(latencies, 0.95),
            }
        )
    return summaries


def _build_runtime_comparisons(
    optimized_summaries: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """只比较同口径的格式与运行指标，业务评分轨道明确不可直接比较。"""
    baseline_by_scenario = {
        row["scenario"]: row for row in _summarize_runtime(baseline_results)
    }
    comparisons = []
    for optimized in optimized_summaries:
        baseline = baseline_by_scenario[optimized["scenario"]]
        comparisons.append(
            {
                "scenario": optimized["scenario"],
                "business_metrics_comparable": False,
                "business_comparison_status": (
                    "not_comparable_different_prediction_encodings"
                ),
                "baseline_business_track": "baseline_semantic_coding_v1",
                "optimized_business_track": "structured_json_strict",
                "baseline_run_id": BASELINE_RUN_ID,
                "optimized_prompt_version": optimized["prompt_version"],
                "baseline_json_compliance": baseline["json_compliance"],
                "optimized_json_compliance": optimized["json_compliance"],
                "baseline_schema_compliance": baseline["schema_compliance"],
                "optimized_schema_compliance": optimized["schema_compliance"],
                "baseline_mean_total_tokens": baseline["mean_total_tokens"],
                "baseline_token_status": "PENDING_not_recorded",
                "optimized_mean_total_tokens": optimized["mean_total_tokens"],
                "baseline_mean_latency_ms": baseline["mean_latency_ms"],
                "optimized_mean_latency_ms": optimized["mean_latency_ms"],
                "baseline_p95_latency_ms": baseline["p95_latency_ms"],
                "optimized_p95_latency_ms": optimized["p95_latency_ms"],
            }
        )
    return comparisons


def _business_quality(scenario: str, scores: list[dict[str, Any]]) -> float:
    values = []
    for score in scores:
        row_values = [
            score.get(name)
            for name in BUSINESS_METRICS[scenario]
            if isinstance(score.get(name), (int, float))
            and not isinstance(score.get(name), bool)
        ]
        if row_values:
            values.append(statistics.fmean(row_values))
    return statistics.fmean(values) if values else 0.0


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), (int, float))
        and not isinstance(row.get(field), bool)
    ]
    return statistics.fmean(values) if values else 0.0


def _apply_efficiency(
    rows: list[dict[str, Any]],
    value_field: str,
    output_field: str,
) -> None:
    values = [
        row[value_field]
        for row in rows
        if isinstance(row.get(value_field), (int, float))
    ]
    if not values:
        for row in rows:
            row[output_field] = 0.0
        return
    low, high = min(values), max(values)
    for row in rows:
        value = row.get(value_field)
        if not isinstance(value, (int, float)):
            row[output_field] = 0.0
        elif high == low:
            row[output_field] = 1.0
        else:
            row[output_field] = (high - value) / (high - low)


def _build_bad_cases(
    results: list[dict[str, Any]],
    scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_by_id = {row["sample_id"]: row for row in results}
    cases = []
    for score in scores:
        result = result_by_id[score["sample_id"]]
        categories = []
        if score.get("json_compliance") != 1.0 or score.get("schema_pass") != 1.0:
            categories.append("format_error")
        if score["scenario"] == "image_product_search" and score.get(
            "business_category_accuracy"
        ) == 0.0:
            categories.append("classification_error")
        if score["scenario"] == "after_sales":
            if score.get("issue_type_accuracy") == 0.0:
                categories.append("classification_error")
            if score.get("severity_accuracy") == 0.0:
                categories.append("severity_error")
        if score["scenario"] == "itinerary_planning" and any(
            isinstance(score.get(name), (int, float)) and score[name] < 1.0
            for name in (
                "hard_constraint_recall",
                "soft_constraint_recall",
                "constraint_check_coverage",
            )
        ):
            categories.append("constraint_omission")
        parsed = result.get("parsed_output")
        if isinstance(parsed, dict) and result.get("schema_valid") is not True:
            categories.append("field_omission_or_schema_error")
        if not categories:
            continue
        cases.append(
            {
                "run_id": result["run_id"],
                "sample_id": result["sample_id"],
                "scenario": result["scenario"],
                "prompt_version": result["prompt_version"],
                "categories": sorted(set(categories)),
                "error": result.get("error"),
                "raw_output": result.get("raw_output"),
                "parsed_output": parsed,
                "sample_metrics": score,
            }
        )
    return cases


def _count_bad_cases(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        for category in case["categories"]:
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return ordered[index]


def _load_results(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_completed_run(runs_dir: Path, run_id: str) -> list[dict[str, Any]]:
    run_dir = Path(runs_dir) / run_id
    metadata = json.loads(
        (run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("status") != "completed":
        raise ValueError(f"run is not completed: {run_id}")
    results = _load_results(run_dir / "results.jsonl")
    request_errors = [
        row
        for row in results
        if isinstance(row.get("error"), str)
        and row["error"].startswith("model_request_error:")
    ]
    if request_errors:
        raise ValueError(
            f"run contains {len(request_errors)} model request errors: {run_id}"
        )
    if metadata.get("record_count") != len(results):
        raise ValueError(f"run record count mismatch: {run_id}")
    return results


def _artifact_version(config: dict[str, Any]) -> str:
    validation = config.get("validation")
    version = validation.get("artifact_version") if isinstance(validation, dict) else None
    if (
        not isinstance(version, str)
        or not version
        or not version.replace("_", "").isalnum()
    ):
        raise ValueError("validation.artifact_version must be alphanumeric")
    return version


def _write_or_verify_json(path: Path, payload: Any) -> None:
    if not path.exists():
        _write_json(path, payload)
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != payload:
        raise ValueError(f"existing immutable JSON differs: {path}")


def _write_or_verify_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not path.exists():
        _write_jsonl(path, rows)
        return
    if _load_results(path) != rows:
        raise ValueError(f"existing immutable JSONL differs: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

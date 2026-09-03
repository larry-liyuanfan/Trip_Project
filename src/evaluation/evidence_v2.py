"""Fail-closed contracts for the second automated relevance evidence pass."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from src.evaluation.relevance_evidence import score_vlm_comparison


SEARCH_V2_METHODS = (
    "clip_exact",
    "clip_milvus",
    "structured_filter_clip",
    "hard_filter_light_rerank",
)


def validate_calibration_holdout_isolation(
    calibration: list[dict[str, Any]], holdout: list[dict[str, Any]]
) -> dict[str, Any]:
    """Require source, image, and query identities to be disjoint across splits."""
    if not calibration or not holdout:
        raise ValueError("calibration and holdout must both be non-empty")

    def identities(records: list[dict[str, Any]]) -> dict[str, set[str]]:
        return {
            "query_ids": {str(row.get("query_id")) for row in records},
            "source_ids": {str(row.get("source", {}).get("source_id")) for row in records},
            "image_sha256": {str(row.get("image", {}).get("sha256")) for row in records},
        }

    left = identities(calibration)
    right = identities(holdout)
    overlaps = {key: sorted(left[key] & right[key]) for key in left}
    if any(overlaps.values()):
        raise ValueError(f"calibration/holdout identity overlap: {overlaps}")
    return {
        "status": "PASS",
        "calibration_support": len(calibration),
        "holdout_support": len(holdout),
        "overlaps": overlaps,
    }


def select_calibration_configuration(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select only from calibration metrics using a predeclared deterministic order."""
    if not candidates:
        raise ValueError("calibration candidates must not be empty")
    for candidate in candidates:
        if not isinstance(candidate.get("objective"), (int, float)):
            raise ValueError("candidate objective must be numeric")
        config = candidate.get("configuration")
        if not isinstance(config, dict):
            raise ValueError("candidate configuration must be an object")
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["objective"]),
            float(row["configuration"]["star_rating_weight"]),
            float(row["configuration"]["no_result_similarity_threshold"]),
        ),
    )
    return ranked[0]


def apply_search_v2_gates(
    report: dict[str, Any], ann_fidelity: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    """Apply fixed holdout gates without hiding failed or absent denominators."""
    methods = report.get("methods", {})
    candidate = methods.get("hard_filter_light_rerank", {})
    structured = methods.get("structured_filter_clip", {})
    candidate_slices = candidate.get("slices", {})
    structured_slices = structured.get("slices", {})
    no_result_slice = candidate_slices.get("no_result", {})
    hard_filter_slice = candidate_slices.get("hard_filter_before_rerank", {})
    structured_hard_filter_slice = structured_slices.get("hard_filter_before_rerank", {})
    checks = {
        "minimum_query_support": int(candidate.get("support", 0))
        >= int(gates["min_query_support"]),
        "zero_failures": float(candidate.get("failure_rate", 1.0))
        <= float(gates["max_failure_rate"]),
        "no_result_accuracy": _number(no_result_slice.get("no_result_accuracy"))
        >= float(gates["min_no_result_accuracy"]),
        "filter_correctness": _number(hard_filter_slice.get("filter_correctness"))
        >= float(gates["min_filter_correctness"]),
        "ndcg_at_10": _number(hard_filter_slice.get("ndcg_at_10"))
        >= float(gates["min_ndcg_at_10"]),
        "ann_fidelity": _number(ann_fidelity.get("value"))
        >= float(gates["min_ann_recall_at_10"]),
        "structured_filter_ndcg_non_regression": _number(hard_filter_slice.get("ndcg_at_10"))
        + float(gates["max_ndcg_regression_vs_structured"])
        >= _number(structured_hard_filter_slice.get("ndcg_at_10")),
    }
    return {
        "thresholds": gates,
        "denominators": {
            "all_query_support": candidate.get("support"),
            "no_result_slice_support": no_result_slice.get("support"),
            "hard_filter_slice_support": hard_filter_slice.get("support"),
            "hard_filter_slice_ranking_support": hard_filter_slice.get("ranking_support"),
        },
        "observed": {
            "no_result_slice_accuracy": no_result_slice.get("no_result_accuracy"),
            "hard_filter_slice_filter_correctness": hard_filter_slice.get("filter_correctness"),
            "hard_filter_slice_ndcg_at_10": hard_filter_slice.get("ndcg_at_10"),
            "structured_hard_filter_slice_ndcg_at_10": structured_hard_filter_slice.get("ndcg_at_10"),
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def score_vlm_v3_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Extend the shared VLM scorer with explicit weak/synthetic slice denominators."""
    base = score_vlm_comparison(records)
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_variant[str(row.get("variant"))].append(row)

    for variant, rows in by_variant.items():
        slice_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            for slice_name in row.get("slices", []):
                slice_rows[str(slice_name)].append(row)
        known_price = slice_rows.get("known_visible_price", [])
        multi_subject = slice_rows.get("multi_subject_conflict", [])
        insufficient = slice_rows.get("insufficient_visual_evidence", [])
        known_price_correct = sum(
            _norm(row.get("gold", {}).get("price_range"))
            == _norm(row.get("prediction", {}).get("price_range"))
            for row in known_price
        )
        multi_abstentions = sum(_unknown_fields_abstained(row) for row in multi_subject)
        insufficient_abstentions = sum(_unknown_fields_abstained(row) for row in insufficient)
        base["variants"][variant]["slice_metrics"] = {
            "known_visible_price": {
                "support": len(known_price),
                "exact_accuracy": known_price_correct / len(known_price) if known_price else None,
            },
            "multi_subject_conflict": {
                "support": len(multi_subject),
                "all_declared_unknown_fields_abstained_accuracy": (
                    multi_abstentions / len(multi_subject) if multi_subject else None
                ),
            },
            "insufficient_visual_evidence": {
                "support": len(insufficient),
                "all_declared_unknown_fields_abstained_accuracy": (
                    insufficient_abstentions / len(insufficient) if insufficient else None
                ),
            },
        }
    base["evidence_class"] = "weak_and_synthetic_development_only_not_human_ground_truth"
    base["fresh_test_used"] = False
    return base


def apply_vlm_v3_gates(report: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """Apply fixed gates to the declared current variant."""
    current = report["variants"][gates["candidate_variant"]]
    price = current["field_metrics"]["price_range"]
    slices = current["slice_metrics"]
    dialogue = current["dialogue_metrics"]
    checks = {
        "minimum_product_support": current["product_support"] >= int(gates["min_product_support"]),
        "minimum_dialogue_support": current["dialogue_support"] >= int(gates["min_dialogue_support"]),
        "price_f1": _number(price.get("f1")) >= float(gates["min_price_f1"]),
        "unsupported_hallucination_rate": current["unsupported_hallucination_rate"]
        <= float(gates["max_unsupported_hallucination_rate"]),
        "first_attempt_json_compliance": current["first_attempt_json_compliance"]
        >= float(gates["min_first_attempt_json_compliance"]),
        "multi_subject_abstention": _number(
            slices["multi_subject_conflict"]["all_declared_unknown_fields_abstained_accuracy"]
        )
        >= float(gates["min_multi_subject_abstention_accuracy"]),
        "dialogue_context_recall": _number(dialogue.get("context_recall"))
        >= float(gates["min_dialogue_metric"]),
        "dialogue_state_value": _number(dialogue.get("state_value_correct"))
        >= float(gates["min_dialogue_metric"]),
        "dialogue_task_key": _number(dialogue.get("task_key_correct"))
        >= float(gates["min_dialogue_metric"]),
        "dialogue_value": _number(dialogue.get("value_correct"))
        >= float(gates["min_dialogue_metric"]),
        "dialogue_route": _number(dialogue.get("first_turn_routing_correct"))
        >= float(gates["min_dialogue_metric"]),
    }
    return {"thresholds": gates, "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}


def summarize_performance_matrix(
    records: list[dict[str, Any]], gates: dict[str, Any]
) -> dict[str, Any]:
    """Summarize fixed profile/component cells and explicitly retain non-run service cells."""
    if not records:
        raise ValueError("performance matrix records must not be empty")
    stages = ("clip_encode_ms", "milvus_ms", "rerank_ms", "vlm_ms", "end_to_end_ms")
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    not_run: list[dict[str, Any]] = []
    for row in records:
        if row.get("status") == "NOT_RUN":
            not_run.append(row)
            continue
        key = (
            str(row.get("role")),
            str(row.get("profile_id")),
            int(row.get("concurrency", 0)),
            str(row.get("transport")),
        )
        for stage in stages:
            if not isinstance(row.get(stage), (int, float)):
                raise ValueError(f"performance matrix row misses numeric {stage}")
        groups[key].append(row)
    cells: dict[str, Any] = {}
    measured_checks: list[bool] = []
    for key, rows in sorted(groups.items()):
        role, profile, concurrency, transport = key
        cold = [row for row in rows if row.get("phase") == "cold"]
        steady = [row for row in rows if row.get("phase") == "steady"]
        profile_gate = gates["profiles"][profile]
        failure_rate = sum(row.get("failed") is True for row in rows) / len(rows)
        peak_vram = max(float(row.get("peak_vram_mib", 0)) for row in rows)
        steady_stats = {stage: _stats([float(row[stage]) for row in steady]) for stage in stages}
        cold_stats = {stage: _stats([float(row[stage]) for row in cold]) for stage in stages}
        duration = sum(float(row["end_to_end_ms"]) for row in steady) / 1000
        checks = {
            "cold_support": len(cold) >= int(gates["min_cold_repetitions"]),
            "steady_support": len(steady) >= int(gates["min_steady_repetitions"]),
            "end_to_end_p95": _number(steady_stats["end_to_end_ms"]["p95"])
            <= float(profile_gate["max_steady_end_to_end_p95_ms"]),
            "failure_rate": failure_rate <= float(gates["max_failure_rate"]),
            "peak_vram": peak_vram <= float(gates["max_peak_vram_mib"]),
        }
        measured_checks.extend(checks.values())
        cells["|".join(map(str, key))] = {
            "role": role,
            "profile_id": profile,
            "concurrency": concurrency,
            "transport": transport,
            "support": {"cold": len(cold), "steady": len(steady)},
            "stages": {"cold": cold_stats, "steady": steady_stats},
            "peak_vram_mib": peak_vram,
            "throughput_queries_per_second": len(steady) / max(duration, 1e-12),
            "failure_rate": failure_rate,
            "hardware": rows[0].get("hardware"),
            "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        }
    return {
        "scope": "component_pipeline_milvus_lite_not_distributed_service_sla",
        "measured_cells": cells,
        "not_run_cells": not_run,
        "fixed_gates": gates,
        "measured_gate_status": "PASS" if measured_checks and all(measured_checks) else "FAIL",
        "overall_status": "PARTIAL_COMPONENT_EVIDENCE_NO_PRODUCTION_SLA",
    }


def _unknown_fields_abstained(row: dict[str, Any]) -> bool:
    unknown = row.get("gold", {}).get("unknown_fields", [])
    prediction = row.get("prediction", {})
    if not unknown:
        return False
    for field in unknown:
        value = prediction.get(field)
        if isinstance(value, list):
            if any(_norm(item) for item in value):
                return False
        elif _norm(value) not in {"", "unknown"}:
            return False
    return True


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.999999) - 1))
    return {
        "p50": statistics.median(ordered),
        "p95": ordered[index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else float("-inf")


def _norm(value: Any) -> str:
    return " ".join(value.strip().casefold().split()) if isinstance(value, str) else ""

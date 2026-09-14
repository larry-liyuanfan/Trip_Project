"""Scoring and fixed development gates for the synthetic v7 robustness cycle."""

from __future__ import annotations

from typing import Any

from src.evaluation.exploration_v4 import _mean, _number, score_vlm_exploration_v4
from src.evaluation.relevance_evidence import PRODUCT_FIELDS


def score_semantic_robustness_v7(
    records: list[dict[str, Any]],
    *,
    cycle_id: str = "v7",
    primary_factor: str = "robustness_training_data_only",
) -> dict[str, Any]:
    report = score_vlm_exploration_v4(records)
    report["schema_version"] = f"semantic_robustness_metrics_{cycle_id}"
    report["primary_factor"] = primary_factor
    return report


def apply_semantic_robustness_v7_gates(
    report: dict[str, Any],
    gates: dict[str, Any],
    objective: dict[str, Any],
    *,
    candidate: str,
    baseline: str,
) -> dict[str, Any]:
    candidate_result = report["variants"][candidate]
    baseline_result = report["variants"][baseline]
    candidate_score = selection_objective(candidate_result, objective)
    baseline_score = selection_objective(baseline_result, objective)
    candidate_dialogue = candidate_result["dialogue_metrics"]
    baseline_dialogue = baseline_result["dialogue_metrics"]
    candidate_multi = candidate_result["slice_metrics"]["multi_subject_conflict"]
    candidate_insufficient = candidate_result["slice_metrics"]["insufficient_visual_evidence"]
    max_regression = float(gates["max_per_metric_regression"])
    checks = {
        "minimum_product_support": candidate_result["product_support"] >= int(gates["min_product_support"]),
        "minimum_dialogue_support": candidate_result["dialogue_support"] >= int(gates["min_dialogue_support"]),
        **{
            f"{field}_f1": _number(candidate_result["field_metrics"][field].get("f1"))
            >= float(gates["min_field_f1"])
            for field in PRODUCT_FIELDS
        },
        "unsupported_hallucination_rate": candidate_result["unsupported_hallucination_rate"]
        <= float(gates["max_unsupported_hallucination_rate"]),
        "unknown_abstention_accuracy": candidate_result["unknown_field_abstention_accuracy"]
        >= float(gates["min_unknown_abstention_accuracy"]),
        "first_attempt_json_compliance": candidate_result["first_attempt_json_compliance"]
        >= float(gates["min_first_attempt_json_compliance"]),
        "multi_subject_abstention": _number(candidate_multi.get("all_unknown_fields_abstained_accuracy"))
        >= float(gates["min_multi_subject_abstention_accuracy"]),
        "insufficient_evidence_abstention": _number(
            candidate_insufficient.get("all_unknown_fields_abstained_accuracy")
        ) >= float(gates["min_insufficient_evidence_abstention_accuracy"]),
        **{
            f"dialogue_{name}": _number(value) >= float(gates["min_dialogue_metric"])
            for name, value in candidate_dialogue.items()
        },
        "objective_improvement": candidate_score - baseline_score
        >= float(objective["min_improvement_over_baseline"]),
        **{
            f"no_material_field_regression_{field}": _number(
                candidate_result["field_metrics"][field].get("f1")
            ) >= _number(baseline_result["field_metrics"][field].get("f1")) - max_regression
            for field in PRODUCT_FIELDS
        },
        **{
            f"no_material_dialogue_regression_{name}": _number(value)
            >= _number(baseline_dialogue.get(name)) - max_regression
            for name, value in candidate_dialogue.items()
        },
    }
    return {
        "gate_class": "development_only_exploration_not_release_or_production_gate",
        "candidate_variant": candidate,
        "baseline_variant": baseline,
        "thresholds": gates,
        "objective_definition": objective,
        "candidate_objective": candidate_score,
        "baseline_objective": baseline_score,
        "objective_improvement": candidate_score - baseline_score,
        "denominators": {
            "candidate_support": candidate_result["support"],
            "product_support": candidate_result["product_support"],
            "dialogue_support": candidate_result["dialogue_support"],
            "unknown_field_opportunity_support": candidate_result["unknown_field_opportunity_support"],
            "multi_subject_support": candidate_multi["support"],
            "insufficient_evidence_support": candidate_insufficient["support"],
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def selection_objective(result: dict[str, Any], objective: dict[str, Any]) -> float:
    field_mean = _mean([
        result["field_metrics"][field].get("f1") for field in PRODUCT_FIELDS
    ])
    dialogue_mean = _mean(list(result["dialogue_metrics"].values()))
    return (
        field_mean * float(objective["field_f1_mean_weight"])
        + float(result["unknown_field_abstention_accuracy"]) * float(objective["unknown_abstention_weight"])
        + dialogue_mean * float(objective["dialogue_mean_weight"])
        + float(result["first_attempt_json_compliance"]) * float(objective["first_json_weight"])
        + float(result["supported_field_exact_match"]) * float(objective["supported_exact_match_weight"])
    )

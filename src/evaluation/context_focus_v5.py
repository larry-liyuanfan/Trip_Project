"""Pre-registered quality gates for the synthetic context-focus v5 cycle."""

from __future__ import annotations

from typing import Any

from src.evaluation.exploration_v4 import _number, score_vlm_exploration_v4
from src.evaluation.relevance_evidence import PRODUCT_FIELDS


def score_context_focus_v5(records: list[dict[str, Any]]) -> dict[str, Any]:
    report = score_vlm_exploration_v4(records)
    report["schema_version"] = "context_focus_semantic_metrics_v5"
    report["primary_factor"] = "context_focused_training_data_composition_and_support"
    return report


def apply_context_focus_v5_development_gates(
    report: dict[str, Any], gates: dict[str, Any], *, candidate: str, baseline: str
) -> dict[str, Any]:
    candidate_result = report["variants"][candidate]
    baseline_result = report["variants"][baseline]
    candidate_dialogue = candidate_result["dialogue_metrics"]
    baseline_dialogue = baseline_result["dialogue_metrics"]
    multi = candidate_result["slice_metrics"]["multi_subject_conflict"]
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
        "multi_subject_abstention": _number(multi.get("all_unknown_fields_abstained_accuracy"))
        >= float(gates["min_multi_subject_abstention_accuracy"]),
        **{
            f"dialogue_{name}": _number(value) >= float(gates["min_dialogue_metric"])
            for name, value in candidate_dialogue.items()
        },
        "context_recall_improvement": (
            _number(candidate_dialogue.get("context_recall"))
            - _number(baseline_dialogue.get("context_recall"))
        ) >= float(gates["min_context_recall_improvement"]),
        **{
            f"no_material_regression_{name}": _number(candidate_dialogue.get(name))
            >= _number(baseline_dialogue.get(name)) - float(gates["max_other_dialogue_regression"])
            for name in (
                "state_value_correct",
                "task_key_correct",
                "value_correct",
                "first_turn_routing_correct",
            )
        },
    }
    return _gate_record(gates, candidate_result, baseline_result, candidate, baseline, checks)


def apply_context_focus_v5_final_gates(
    report: dict[str, Any], gates: dict[str, Any], *, candidate: str
) -> dict[str, Any]:
    result = report["variants"][candidate]
    dialogue = result["dialogue_metrics"]
    multi = result["slice_metrics"]["multi_subject_conflict"]
    checks = {
        "minimum_product_support": result["product_support"] >= int(gates["min_product_support"]),
        "minimum_dialogue_support": result["dialogue_support"] >= int(gates["min_dialogue_support"]),
        **{
            f"{field}_f1": _number(result["field_metrics"][field].get("f1"))
            >= float(gates["min_field_f1"])
            for field in PRODUCT_FIELDS
        },
        "unsupported_hallucination_rate": result["unsupported_hallucination_rate"]
        <= float(gates["max_unsupported_hallucination_rate"]),
        "unknown_abstention_accuracy": result["unknown_field_abstention_accuracy"]
        >= float(gates["min_unknown_abstention_accuracy"]),
        "first_attempt_json_compliance": result["first_attempt_json_compliance"]
        >= float(gates["min_first_attempt_json_compliance"]),
        "multi_subject_abstention": _number(multi.get("all_unknown_fields_abstained_accuracy"))
        >= float(gates["min_multi_subject_abstention_accuracy"]),
        **{
            f"dialogue_{name}": _number(value) >= float(gates["min_dialogue_metric"])
            for name, value in dialogue.items()
        },
    }
    return _gate_record(gates, result, None, candidate, None, checks)


def _gate_record(
    thresholds: dict[str, Any],
    candidate_result: dict[str, Any],
    baseline_result: dict[str, Any] | None,
    candidate: str,
    baseline: str | None,
    checks: dict[str, bool],
) -> dict[str, Any]:
    return {
        "gate_class": "exploration_only_not_release_or_production_gate",
        "candidate_variant": candidate,
        "baseline_variant": baseline,
        "thresholds": thresholds,
        "denominators": {
            "candidate_support": candidate_result["support"],
            "product_support": candidate_result["product_support"],
            "dialogue_support": candidate_result["dialogue_support"],
            "unknown_field_opportunity_support": candidate_result["unknown_field_opportunity_support"],
            "multi_subject_support": candidate_result["slice_metrics"]["multi_subject_conflict"]["support"],
        },
        "observed": {
            "candidate_context_recall": candidate_result["dialogue_metrics"]["context_recall"],
            "baseline_context_recall": (
                baseline_result["dialogue_metrics"]["context_recall"] if baseline_result else None
            ),
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }

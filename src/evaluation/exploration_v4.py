"""Exploration-only gates for the leak-free v4 search and VLM evidence cycle."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.evaluation.relevance_evidence import PRODUCT_FIELDS, score_vlm_comparison


SEARCH_V4_METHODS = (
    "clip_exact",
    "clip_milvus",
    "structured_filter_clip",
    "hard_filter_light_rerank",
    "hard_filter_clip_business_guard",
)


def validate_three_way_isolation(
    splits: dict[str, list[dict[str, Any]]], *, record_kind: str
) -> dict[str, Any]:
    """Require training/development/final identities to be pairwise disjoint."""
    if set(splits) != {"training", "development", "final"}:
        raise ValueError("three-way isolation requires training, development, and final")
    identities: dict[str, dict[str, set[str]]] = {}
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"{split} split is empty")
        if record_kind == "search":
            identities[split] = {
                "record_id": {str(row.get("query_id")) for row in rows},
                "source_id": {str(row.get("source", {}).get("source_id")) for row in rows},
                "image_sha256": {str(row.get("image", {}).get("sha256")) for row in rows},
            }
        elif record_kind == "vlm":
            identities[split] = {
                "record_id": {str(row.get("sample_id")) for row in rows},
                "source_id": {str(row.get("source_id")) for row in rows},
                "image_sha256": {
                    str(row.get("image_sha256")) for row in rows if row.get("image_sha256")
                },
            }
        else:
            raise ValueError(f"unsupported record_kind: {record_kind}")
    overlaps: dict[str, Any] = {}
    names = ("training", "development", "final")
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            key = f"{left}_vs_{right}"
            overlaps[key] = {
                identity: sorted(identities[left][identity] & identities[right][identity])
                for identity in ("record_id", "source_id", "image_sha256")
            }
    if any(items for pair in overlaps.values() for items in pair.values()):
        raise ValueError(f"three-way identity leakage: {overlaps}")
    return {
        "status": "PASS",
        "record_kind": record_kind,
        "support": {split: len(rows) for split, rows in splits.items()},
        "overlaps": overlaps,
    }


def apply_search_v4_gates(
    report: dict[str, Any], ann_fidelity: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    candidate = report.get("methods", {}).get("hard_filter_clip_business_guard", {})
    no_result = candidate.get("slices", {}).get("no_result", {})
    hard_filter = candidate.get("slices", {}).get("hard_filter_before_rerank", {})
    checks = {
        "minimum_query_support": int(candidate.get("support", 0)) >= int(gates["min_query_support"]),
        "zero_failures": _number(candidate.get("failure_rate"), 1.0) <= float(gates["max_failure_rate"]),
        "no_result_accuracy": _number(no_result.get("no_result_accuracy")) >= float(gates["min_no_result_accuracy"]),
        "filter_correctness": _number(hard_filter.get("filter_correctness")) >= float(gates["min_filter_correctness"]),
        "ndcg_at_10": _number(hard_filter.get("ndcg_at_10")) >= float(gates["min_ndcg_at_10"]),
        "ann_fidelity": _number(ann_fidelity.get("value")) >= float(gates["min_ann_recall_at_10"]),
    }
    return {
        "gate_class": "exploration_only_not_release_or_production_gate",
        "thresholds": gates,
        "denominators": {
            "query_support": candidate.get("support"),
            "ranking_support": candidate.get("ranking_support"),
            "no_result_support": no_result.get("support"),
            "hard_filter_support": hard_filter.get("support"),
        },
        "observed": {
            "no_result_accuracy": no_result.get("no_result_accuracy"),
            "filter_correctness": hard_filter.get("filter_correctness"),
            "ndcg_at_10": hard_filter.get("ndcg_at_10"),
            "ann_recall_at_10": ann_fidelity.get("value"),
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def score_vlm_exploration_v4(records: list[dict[str, Any]]) -> dict[str, Any]:
    report = score_vlm_comparison(records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get("variant"))].append(row)
    for variant, rows in grouped.items():
        product = [row for row in rows if row.get("scenario") == "product"]
        multi = [row for row in product if "multi_subject_conflict" in row.get("slices", [])]
        insufficient = [row for row in product if "insufficient_visual_evidence" in row.get("slices", [])]
        report["variants"][variant]["slice_metrics"] = {
            "multi_subject_conflict": _abstention_slice(multi),
            "insufficient_visual_evidence": _abstention_slice(insufficient),
        }
        report["variants"][variant]["selection_objective_inputs"] = {
            "field_f1_mean": _mean([
                report["variants"][variant]["field_metrics"][field].get("f1")
                for field in PRODUCT_FIELDS
            ]),
            "dialogue_mean": _mean(list(report["variants"][variant]["dialogue_metrics"].values())),
        }
    report["evidence_class"] = "synthetic_programmatic_exploration_only_not_human_ground_truth"
    report["fresh_test_used"] = False
    return report


def apply_vlm_exploration_v4_gates(
    report: dict[str, Any], gates: dict[str, Any], objective: dict[str, Any],
    *, candidate_variant: str, baseline_variant: str,
) -> dict[str, Any]:
    candidate = report["variants"][candidate_variant]
    baseline = report["variants"][baseline_variant]
    candidate_score = vlm_selection_objective(candidate, objective)
    baseline_score = vlm_selection_objective(baseline, objective)
    field_f1 = {
        field: candidate["field_metrics"][field].get("f1") for field in PRODUCT_FIELDS
    }
    dialogue = candidate["dialogue_metrics"]
    multi = candidate["slice_metrics"]["multi_subject_conflict"]
    checks = {
        "minimum_product_support": candidate["product_support"] >= int(gates["min_product_support"]),
        "minimum_dialogue_support": candidate["dialogue_support"] >= int(gates["min_dialogue_support"]),
        **{
            f"{field}_f1": _number(value) >= float(gates["min_field_f1"])
            for field, value in field_f1.items()
        },
        "unsupported_hallucination_rate": candidate["unsupported_hallucination_rate"]
        <= float(gates["max_unsupported_hallucination_rate"]),
        "unknown_abstention_accuracy": candidate["unknown_field_abstention_accuracy"]
        >= float(gates["min_unknown_abstention_accuracy"]),
        "first_attempt_json_compliance": candidate["first_attempt_json_compliance"]
        >= float(gates["min_first_attempt_json_compliance"]),
        "multi_subject_abstention": _number(multi.get("all_unknown_fields_abstained_accuracy"))
        >= float(gates["min_multi_subject_abstention_accuracy"]),
        **{
            f"dialogue_{name}": _number(value) >= float(gates["min_dialogue_metric"])
            for name, value in dialogue.items()
        },
        "objective_improvement_over_checkpoint_87": candidate_score - baseline_score
        >= float(objective["min_improvement_over_baseline"]),
    }
    return {
        "gate_class": "exploration_only_not_release_or_production_gate",
        "candidate_variant": candidate_variant,
        "baseline_variant": baseline_variant,
        "thresholds": gates,
        "objective_definition": objective,
        "candidate_objective": candidate_score,
        "baseline_objective": baseline_score,
        "objective_improvement": candidate_score - baseline_score,
        "denominators": {
            "candidate_support": candidate["support"],
            "product_support": candidate["product_support"],
            "dialogue_support": candidate["dialogue_support"],
            "unknown_field_opportunity_support": candidate["unknown_field_opportunity_support"],
            "multi_subject_support": multi.get("support"),
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def apply_vlm_absolute_v4_gates(
    report: dict[str, Any], gates: dict[str, Any], *, candidate_variant: str,
) -> dict[str, Any]:
    """Apply the same absolute semantic floors when final contains only the candidate."""
    candidate = report["variants"][candidate_variant]
    multi = candidate["slice_metrics"]["multi_subject_conflict"]
    checks = {
        "minimum_product_support": candidate["product_support"] >= int(gates["min_product_support"]),
        "minimum_dialogue_support": candidate["dialogue_support"] >= int(gates["min_dialogue_support"]),
        **{
            f"{field}_f1": _number(candidate["field_metrics"][field].get("f1"))
            >= float(gates["min_field_f1"])
            for field in PRODUCT_FIELDS
        },
        "unsupported_hallucination_rate": candidate["unsupported_hallucination_rate"]
        <= float(gates["max_unsupported_hallucination_rate"]),
        "unknown_abstention_accuracy": candidate["unknown_field_abstention_accuracy"]
        >= float(gates["min_unknown_abstention_accuracy"]),
        "first_attempt_json_compliance": candidate["first_attempt_json_compliance"]
        >= float(gates["min_first_attempt_json_compliance"]),
        "multi_subject_abstention": _number(multi.get("all_unknown_fields_abstained_accuracy"))
        >= float(gates["min_multi_subject_abstention_accuracy"]),
        **{
            f"dialogue_{name}": _number(value) >= float(gates["min_dialogue_metric"])
            for name, value in candidate["dialogue_metrics"].items()
        },
    }
    return {
        "gate_class": "exploration_only_not_release_or_production_gate",
        "candidate_variant": candidate_variant,
        "thresholds": gates,
        "denominators": {
            "candidate_support": candidate["support"],
            "product_support": candidate["product_support"],
            "dialogue_support": candidate["dialogue_support"],
            "unknown_field_opportunity_support": candidate["unknown_field_opportunity_support"],
            "multi_subject_support": multi.get("support"),
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def vlm_selection_objective(variant: dict[str, Any], objective: dict[str, Any]) -> float:
    field_mean = _mean([
        variant["field_metrics"][field].get("f1") for field in PRODUCT_FIELDS
    ])
    dialogue_mean = _mean(list(variant["dialogue_metrics"].values()))
    return (
        field_mean * float(objective["field_f1_mean_weight"])
        + float(variant["unknown_field_abstention_accuracy"]) * float(objective["unknown_abstention_weight"])
        + dialogue_mean * float(objective["dialogue_mean_weight"])
        + float(variant["first_attempt_json_compliance"]) * float(objective["first_json_weight"])
        + float(variant["supported_field_exact_match"]) * float(objective["supported_exact_match_weight"])
    )


def _abstention_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    for row in rows:
        unknown_fields = row.get("gold", {}).get("unknown_fields", [])
        prediction = row.get("prediction", {})
        if unknown_fields and all(_is_abstained(prediction.get(field)) for field in unknown_fields):
            correct += 1
    return {
        "support": len(rows),
        "all_unknown_fields_abstained_accuracy": correct / len(rows) if rows else None,
    }


def _is_abstained(value: Any) -> bool:
    if isinstance(value, list):
        return not any(_norm(item) for item in value)
    return _norm(value) in {"", "unknown"}


def _mean(values: list[Any]) -> float:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _number(value: Any, default: float = float("-inf")) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _norm(value: Any) -> str:
    return " ".join(value.strip().casefold().split()) if isinstance(value, str) else ""

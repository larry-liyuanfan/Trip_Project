"""Fixed quality gate for the development-only no-result stress validation."""

from __future__ import annotations

from typing import Any


def apply_no_result_stress_v8_gate(
    report: dict[str, Any], gates: dict[str, Any], *, candidate: str, baseline: str
) -> dict[str, Any]:
    candidate_metrics = report["methods"][candidate]
    baseline_metrics = report["methods"][baseline]
    candidate_no_result = candidate_metrics["slices"]["no_result"]
    baseline_no_result = baseline_metrics["slices"]["no_result"]
    candidate_positive = candidate_metrics["slices"]["business_positive"]
    baseline_positive = baseline_metrics["slices"]["business_positive"]
    checks = {
        "query_support": candidate_metrics["support"] >= int(gates["min_query_support"]),
        "ranking_support": candidate_metrics["ranking_support"] >= int(gates["min_ranking_support"]),
        "no_result_support": candidate_no_result["support"] >= int(gates["min_no_result_support"]),
        "no_result_accuracy": candidate_no_result["no_result_accuracy"] >= float(gates["min_no_result_accuracy"]),
        "business_positive_acceptance": candidate_positive["no_result_accuracy"]
        >= float(gates["min_business_positive_acceptance"]),
        "ranking_ndcg_at_10": candidate_metrics["ndcg_at_10"] >= float(gates["min_ndcg_at_10"]),
        "filter_correctness": candidate_metrics["filter_correctness"] >= float(gates["min_filter_correctness"]),
        "failure_rate": candidate_metrics["failure_rate"] <= float(gates["max_failure_rate"]),
        "no_material_no_result_regression": candidate_no_result["no_result_accuracy"]
        >= baseline_no_result["no_result_accuracy"] - float(gates["max_metric_regression"]),
        "no_material_positive_regression": candidate_positive["no_result_accuracy"]
        >= baseline_positive["no_result_accuracy"] - float(gates["max_metric_regression"]),
        "no_material_ndcg_regression": candidate_metrics["ndcg_at_10"]
        >= baseline_metrics["ndcg_at_10"] - float(gates["max_metric_regression"]),
    }
    return {
        "gate_class": "synthetic_one_time_validation_not_release_or_production_gate",
        "candidate": candidate,
        "baseline": baseline,
        "thresholds": gates,
        "denominators": {
            "query_support": candidate_metrics["support"],
            "ranking_support": candidate_metrics["ranking_support"],
            "no_result_support": candidate_no_result["support"],
            "business_positive_support": candidate_positive["support"],
        },
        "observed": {
            "candidate_no_result_accuracy": candidate_no_result["no_result_accuracy"],
            "baseline_no_result_accuracy": baseline_no_result["no_result_accuracy"],
            "candidate_business_positive_acceptance": candidate_positive["no_result_accuracy"],
            "baseline_business_positive_acceptance": baseline_positive["no_result_accuracy"],
            "candidate_ndcg_at_10": candidate_metrics["ndcg_at_10"],
            "baseline_ndcg_at_10": baseline_metrics["ndcg_at_10"],
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }

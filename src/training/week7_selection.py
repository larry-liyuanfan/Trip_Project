"""Deterministic Week 7 checkpoint selection with preregistered non-regression gates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.training.week7_data import CORE_SCENARIOS, load_week7_config, sha256_file
from src.training.week7_qlora import Week7TrainingError


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week7TrainingError(f"invalid checkpoint-selection artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise Week7TrainingError(f"checkpoint-selection artifact must be an object: {path}")
    return payload


def _relative_change(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline


def select_development_checkpoint(
    config_path: Path,
    training_dir: Path,
    week6_baseline_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Select the highest composite checkpoint only after every locked gate passes."""
    config_path = Path(config_path).resolve()
    training_dir = Path(training_dir).resolve()
    week6_baseline_path = Path(week6_baseline_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise Week7TrainingError("refusing to overwrite checkpoint-selection evidence")
    config = load_week7_config(config_path)
    config_hash = sha256_file(config_path)
    baseline = _read_json(week6_baseline_path)
    if (
        baseline.get("status") != "COMPLETED"
        or baseline.get("model_role") != "week6_single_task_adapters"
        or baseline.get("split") != "development"
        or baseline.get("config_sha256") != config_hash
        or not baseline.get("dataset_lock_sha256")
        or set(baseline.get("scenarios", {})) != set(CORE_SCENARIOS)
    ):
        raise Week7TrainingError("Week 6 development baseline identity mismatch")

    non_regression = config["evaluation"]["non_regression"]
    scenario_weights = config["evaluation"]["scenario_weights"]
    expected_samples = (
        int(config["dataset"]["development_per_core_scenario"]) * len(CORE_SCENARIOS)
        + int(config["dataset"]["development_dialogue_count"])
    )
    candidates: list[dict[str, Any]] = []
    evaluation_root = training_dir / "development_evaluations"
    for metrics_path in sorted(evaluation_root.glob("step-*/metrics.json")):
        try:
            step = int(metrics_path.parent.name.removeprefix("step-"))
        except ValueError as exc:
            raise Week7TrainingError(f"invalid development evaluation step: {metrics_path.parent.name}") from exc
        checkpoint = training_dir / f"checkpoint-{step}"
        adapter_model = checkpoint / "adapter_model.safetensors"
        if not adapter_model.is_file():
            raise Week7TrainingError(f"development evaluation has no matching adapter checkpoint: step {step}")
        metrics = _read_json(metrics_path)
        if (
            int(metrics.get("sample_count", -1)) != expected_samples
            or set(metrics.get("scenarios", {})) != set(CORE_SCENARIOS)
            or not isinstance(metrics.get("dialogue"), dict)
            or int(metrics["dialogue"].get("sample_count", -1))
            != int(config["dataset"]["development_dialogue_count"])
        ):
            raise Week7TrainingError(f"development evaluation coverage mismatch: step {step}")

        scenario_gates: dict[str, Any] = {}
        recomputed_composite = 0.0
        for scenario in CORE_SCENARIOS:
            current = metrics["scenarios"][scenario]
            prior = baseline["scenarios"][scenario]
            current_composite = float(current["composite"])
            prior_composite = float(prior["composite"])
            recomputed_composite += float(scenario_weights[scenario]) * current_composite
            relative_change = _relative_change(current_composite, prior_composite)
            task_gate = (
                current_composite >= prior_composite
                if relative_change is None
                else relative_change >= -float(non_regression["max_relative_task_drop"])
            )
            support_ratios = {
                metric: (
                    float(current["metric_support"].get(metric, 0)) / float(support)
                    if support else None
                )
                for metric, support in prior["metric_support"].items()
            }
            support_gate = all(
                ratio is None or ratio >= float(non_regression["minimum_support_ratio"])
                for ratio in support_ratios.values()
            )
            current_aggregate = current["aggregate"]
            prior_aggregate = prior["aggregate"]
            json_delta = float(current_aggregate["json_compliance"]) - float(prior_aggregate["json_compliance"])
            schema_delta = float(current_aggregate["schema_pass"]) - float(prior_aggregate["schema_pass"])
            format_gate = (
                json_delta >= float(non_regression["json_schema_absolute_drop"])
                and schema_delta >= float(non_regression["json_schema_absolute_drop"])
            )
            prior_latency = float(prior_aggregate["latency_mean_ms"])
            latency_ratio = (
                float(current_aggregate["latency_mean_ms"]) / prior_latency
                if prior_latency else None
            )
            latency_gate = (
                latency_ratio is not None
                and latency_ratio <= float(non_regression["max_latency_ratio"])
            )
            scenario_gates[scenario] = {
                "candidate_composite": current_composite,
                "baseline_composite": prior_composite,
                "relative_change": relative_change,
                "json_compliance_absolute_change": json_delta,
                "schema_pass_absolute_change": schema_delta,
                "metric_support_ratios": support_ratios,
                "latency_ratio": latency_ratio,
                "task_gate": task_gate,
                "format_gate": format_gate,
                "support_gate": support_gate,
                "latency_gate": latency_gate,
                "passed": bool(task_gate and format_gate and support_gate),
            }
        recorded_composite = float(metrics["weighted_composite"])
        if not math.isclose(recorded_composite, recomputed_composite, rel_tol=0, abs_tol=1e-12):
            raise Week7TrainingError(f"weighted composite mismatch: step {step}")
        failure_rate = float(metrics["failure_rate"])
        failure_gate = (
            failure_rate <= float(non_regression["max_failure_rate"])
            and failure_rate - float(baseline["failure_rate"])
            <= float(non_regression["max_failure_rate_absolute_increase"])
        )
        development_per_scenario = int(config["dataset"]["development_per_core_scenario"])
        baseline_core_latency = sum(
            float(baseline["scenarios"][scenario]["aggregate"]["latency_mean_ms"])
            * development_per_scenario
            for scenario in CORE_SCENARIOS
        ) / (development_per_scenario * len(CORE_SCENARIOS))
        candidate_core_latency = sum(
            float(metrics["scenarios"][scenario]["aggregate"]["latency_mean_ms"])
            * development_per_scenario
            for scenario in CORE_SCENARIOS
        ) / (development_per_scenario * len(CORE_SCENARIOS))
        core_latency_ratio = (
            candidate_core_latency / baseline_core_latency
            if baseline_core_latency else None
        )
        latency_gate = (
            core_latency_ratio is not None
            and core_latency_ratio <= float(non_regression["max_latency_ratio"])
        )
        eligible = failure_gate and latency_gate and all(
            payload["passed"] for payload in scenario_gates.values()
        )
        candidates.append({
            "step": step,
            "checkpoint": str(checkpoint),
            "checkpoint_adapter_sha256": sha256_file(adapter_model),
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "weighted_composite": recorded_composite,
            "failure_rate": failure_rate,
            "failure_gate": failure_gate,
            "core_latency_ms_mean": candidate_core_latency,
            "baseline_core_latency_ms_mean": baseline_core_latency,
            "core_latency_ratio": core_latency_ratio,
            "latency_gate": latency_gate,
            "scenario_gates": scenario_gates,
            "eligible": bool(eligible),
        })
    if not candidates:
        raise Week7TrainingError("no completed development checkpoint evaluations found")
    eligible_candidates = [candidate for candidate in candidates if candidate["eligible"]]
    selected = max(
        eligible_candidates,
        key=lambda candidate: (candidate["weighted_composite"], -candidate["step"]),
        default=None,
    )
    result = {
        "status": "SELECTED" if selected else "BLOCKED_NO_ELIGIBLE_CHECKPOINT",
        "config_sha256": config_hash,
        "dataset_lock_sha256": baseline["dataset_lock_sha256"],
        "selection_rule": "non_regression_gates_then_max_weighted_composite_then_earliest_step",
        "week6_baseline": {
            "path": str(week6_baseline_path),
            "sha256": sha256_file(week6_baseline_path),
        },
        "candidate_count": len(candidates),
        "eligible_count": len(eligible_candidates),
        "selected": selected,
        "candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result

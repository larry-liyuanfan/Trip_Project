"""Deterministic Week 7 checkpoint selection with preregistered non-regression gates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.training.week7_data import CORE_SCENARIOS, canonical_sha256, load_week7_config, sha256_file
from src.training.week7_latency_protocol import validate_latency_protocol_v4
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


def validate_development_raw_artifact(
    metrics: dict[str, Any], metrics_path: Path, expected_count: int,
) -> dict[str, Any]:
    """Validate the immutable raw-output evidence bound by checkpoint metrics."""
    metrics_path = Path(metrics_path).resolve()
    expected_path = (metrics_path.parent / "raw_outputs.jsonl").resolve()
    artifact = metrics.get("raw_outputs")
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256", "count"}:
        raise Week7TrainingError("development metrics raw-output binding is incomplete")
    try:
        artifact_path = Path(str(artifact["path"])).resolve()
        artifact_count = int(artifact["count"])
    except (TypeError, ValueError, OSError) as exc:
        raise Week7TrainingError("development metrics raw-output binding is invalid") from exc
    if artifact_path != expected_path or not artifact_path.is_file():
        raise Week7TrainingError("development raw-output artifact path mismatch")
    with artifact_path.open("r", encoding="utf-8") as handle:
        actual_count = sum(1 for line in handle if line.strip())
    if (
        artifact_count != expected_count
        or actual_count != expected_count
        or artifact.get("sha256") != sha256_file(artifact_path)
    ):
        raise Week7TrainingError("development raw-output artifact hash or count mismatch")
    return artifact


def _validate_combined_baseline(
    config: dict[str, Any],
    config_hash: str,
    baseline_path: Path,
) -> dict[str, Any]:
    baseline = _read_json(baseline_path)
    identity = config["experiment_identity"]
    core_count = int(config["dataset"]["development_per_core_scenario"])
    dialogue_count = int(config["dataset"]["development_dialogue_count"])
    expected_count = core_count * len(CORE_SCENARIOS) + dialogue_count
    dataset_hash = baseline.get("dataset_lock_sha256")
    if (
        baseline.get("status") != "COMPLETED"
        or baseline.get("run_id") != identity["week6_combined_development_run_id"]
        or baseline.get("model_role") != "week6_single_task_adapters"
        or baseline.get("split") != "development"
        or baseline.get("config_sha256") != config_hash
        or not dataset_hash
        or set(baseline.get("scenarios", {})) != set(CORE_SCENARIOS)
        or int(baseline.get("sample_count", -1)) != expected_count
        or not isinstance(baseline.get("dialogue"), dict)
        or int(baseline["dialogue"].get("sample_count", -1)) != dialogue_count
        or set(baseline.get("inputs", {})) != set(CORE_SCENARIOS) | {"dialogue"}
    ):
        raise Week7TrainingError("Week 6 development baseline identity mismatch")

    sample_count = 0
    failure_count = 0
    latency_sum = 0.0
    scenarios: dict[str, Any] = {}
    for scenario in CORE_SCENARIOS:
        input_spec = baseline["inputs"][scenario]
        path = Path(str(input_spec.get("path", "")))
        if not path.is_file() or sha256_file(path) != input_spec.get("sha256"):
            raise Week7TrainingError(f"Week 6 baseline input hash mismatch: {scenario}")
        payload = _read_json(path)
        count = int(payload.get("sample_count", -1))
        if (
            payload.get("status") != "COMPLETED"
            or payload.get("run_id") != identity["development_baseline_run_ids"][scenario]
            or payload.get("model_role") != "week6_single_task_adapter"
            or payload.get("split") != "development"
            or payload.get("scenario_filter") != scenario
            or payload.get("config_sha256") != config_hash
            or payload.get("dataset_lock_sha256") != dataset_hash
            or count != core_count
            or set(payload.get("scenarios", {})) != {scenario}
            or payload.get("adapter_hashes", {}).get("adapter_model.safetensors")
            != config["evaluation"]["week6_adapter_sha256"][scenario]
        ):
            raise Week7TrainingError(f"Week 6 baseline input identity mismatch: {scenario}")
        scenarios[scenario] = payload["scenarios"][scenario]
        if canonical_sha256(scenarios[scenario]) != canonical_sha256(baseline["scenarios"][scenario]):
            raise Week7TrainingError(f"Week 6 combined scenario differs from input: {scenario}")
        sample_count += count
        failure_count += int(payload["failure_count"])
        latency_sum += float(payload["latency_ms_mean"]) * count

    dialogue_spec = baseline["inputs"]["dialogue"]
    dialogue_path = Path(str(dialogue_spec.get("path", "")))
    if not dialogue_path.is_file() or sha256_file(dialogue_path) != dialogue_spec.get("sha256"):
        raise Week7TrainingError("Week 6 dialogue baseline input hash mismatch")
    dialogue = _read_json(dialogue_path)
    route_counts = dialogue.get("routing", {}).get("sample_counts", {})
    expected_route_counts = config["sampling"].get(
        "dialogue_parent_scenario_counts", {}
    ).get(
        "development",
        {scenario: dialogue_count // len(CORE_SCENARIOS) for scenario in CORE_SCENARIOS},
    )
    dialogue_hashes = dialogue.get("adapter_hashes", {})
    if (
        dialogue.get("status") != "COMPLETED"
        or dialogue.get("run_id") != identity["week6_dialogue_development_run_id"]
        or dialogue.get("model_role") != "week6_single_task_adapters"
        or dialogue.get("split") != "development"
        or dialogue.get("scenario_filter") != "dialogue_routed"
        or dialogue.get("config_sha256") != config_hash
        or dialogue.get("dataset_lock_sha256") != dataset_hash
        or int(dialogue.get("sample_count", -1)) != dialogue_count
        or set(dialogue.get("scenarios", {}))
        or not isinstance(dialogue.get("dialogue"), dict)
        or int(dialogue["dialogue"].get("sample_count", -1)) != dialogue_count
        or set(dialogue_hashes) != set(CORE_SCENARIOS)
        or any(
            dialogue_hashes[scenario].get("adapter_model.safetensors")
            != config["evaluation"]["week6_adapter_sha256"][scenario]
            for scenario in CORE_SCENARIOS
        )
        or dialogue.get("routing", {}).get("method") != "target_task_result_v1"
        or route_counts != expected_route_counts
    ):
        raise Week7TrainingError("Week 6 dialogue baseline input identity mismatch")
    if canonical_sha256(dialogue["dialogue"]) != canonical_sha256(baseline["dialogue"]):
        raise Week7TrainingError("Week 6 combined dialogue differs from input")
    sample_count += dialogue_count
    failure_count += int(dialogue["failure_count"])
    latency_sum += float(dialogue["latency_ms_mean"]) * dialogue_count

    weights = config["evaluation"]["scenario_weights"]
    composite = sum(float(weights[name]) * float(scenarios[name]["composite"]) for name in CORE_SCENARIOS)
    if (
        sample_count != expected_count
        or int(baseline.get("failure_count", -1)) != failure_count
        or not math.isclose(float(baseline.get("failure_rate", -1)), failure_count / sample_count, rel_tol=0, abs_tol=1e-12)
        or not math.isclose(float(baseline.get("latency_ms_mean", -1)), latency_sum / sample_count, rel_tol=0, abs_tol=1e-9)
        or baseline.get("latency_ms_median") is not None
        or not math.isclose(float(baseline.get("weighted_composite", -1)), composite, rel_tol=0, abs_tol=1e-12)
    ):
        raise Week7TrainingError("Week 6 combined development aggregate mismatch")
    return baseline


def evaluate_development_candidate(
    config: dict[str, Any],
    baseline: dict[str, Any],
    metrics: dict[str, Any],
    *,
    step: int,
    checkpoint: Path,
    checkpoint_hash: str,
    metrics_path: Path,
    latency_ms_mean: float | None = None,
    baseline_latency_ms_mean: float | None = None,
) -> dict[str, Any]:
    """Recompute every preregistered selection gate from bound artifacts."""
    non_regression = config["evaluation"]["non_regression"]
    scenario_weights = config["evaluation"]["scenario_weights"]
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
        json_delta = float(current_aggregate["json_compliance"]) - float(
            prior_aggregate["json_compliance"]
        )
        schema_delta = float(current_aggregate["schema_pass"]) - float(
            prior_aggregate["schema_pass"]
        )
        format_gate = (
            json_delta >= float(non_regression["json_schema_absolute_drop"])
            and schema_delta >= float(non_regression["json_schema_absolute_drop"])
        )
        prior_latency = float(prior_aggregate["latency_mean_ms"])
        scenario_latency_ratio = (
            float(current_aggregate["latency_mean_ms"]) / prior_latency
            if prior_latency else None
        )
        scenario_gates[scenario] = {
            "candidate_composite": current_composite,
            "baseline_composite": prior_composite,
            "relative_change": relative_change,
            "json_compliance_absolute_change": json_delta,
            "schema_pass_absolute_change": schema_delta,
            "metric_support_ratios": support_ratios,
            "latency_ratio": scenario_latency_ratio,
            "task_gate": task_gate,
            "format_gate": format_gate,
            "support_gate": support_gate,
            "latency_gate": (
                scenario_latency_ratio is not None
                and scenario_latency_ratio <= float(non_regression["max_latency_ratio"])
            ),
            "passed": bool(task_gate and format_gate and support_gate),
        }
    recorded_composite = float(metrics["weighted_composite"])
    if not math.isclose(
        recorded_composite, recomputed_composite, rel_tol=0, abs_tol=1e-12
    ):
        raise Week7TrainingError(f"weighted composite mismatch: step {step}")
    failure_rate = float(metrics["failure_rate"])
    failure_gate = (
        failure_rate <= float(non_regression["max_failure_rate"])
        and failure_rate - float(baseline["failure_rate"])
        <= float(non_regression["max_failure_rate_absolute_increase"])
    )
    baseline_latency = float(
        baseline["latency_ms_mean"]
        if baseline_latency_ms_mean is None else baseline_latency_ms_mean
    )
    candidate_latency = float(
        metrics["latency_ms_mean"] if latency_ms_mean is None else latency_ms_mean
    )
    latency_ratio = candidate_latency / baseline_latency if baseline_latency else None
    latency_gate = (
        latency_ratio is not None
        and latency_ratio <= float(non_regression["max_latency_ratio"])
    )
    eligible = failure_gate and latency_gate and all(
        payload["passed"] for payload in scenario_gates.values()
    )
    return {
        "step": step,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_adapter_sha256": checkpoint_hash,
        "metrics_path": str(Path(metrics_path).resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "weighted_composite": recorded_composite,
        "failure_rate": failure_rate,
        "failure_gate": failure_gate,
        "latency_ms_mean": candidate_latency,
        "baseline_latency_ms_mean": baseline_latency,
        "latency_ratio": latency_ratio,
        "latency_gate": latency_gate,
        "scenario_gates": scenario_gates,
        "eligible": bool(eligible),
    }


def select_development_checkpoint(
    config_path: Path,
    training_dir: Path,
    training_summary_path: Path,
    week6_baseline_path: Path,
    output_path: Path,
    *,
    latency_protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Select the highest composite checkpoint only after every locked gate passes."""
    config_path = Path(config_path).resolve()
    training_dir = Path(training_dir).resolve()
    training_summary_path = Path(training_summary_path).resolve()
    week6_baseline_path = Path(week6_baseline_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise Week7TrainingError("refusing to overwrite checkpoint-selection evidence")
    config = load_week7_config(config_path)
    config_hash = sha256_file(config_path)
    baseline = _validate_combined_baseline(config, config_hash, week6_baseline_path)
    training_summary = _read_json(training_summary_path)
    if training_summary_path.parent != training_dir:
        raise Week7TrainingError("training summary must belong to the selected training directory")
    if (
        training_summary.get("status") != "COMPLETED"
        or training_summary.get("run_id") != config["experiment_identity"]["multitask_sft_run_id"]
        or training_summary.get("config_sha256") != config_hash
        or training_summary.get("dataset_lock_sha256") != baseline["dataset_lock_sha256"]
        or int(training_summary.get("development_samples", -1))
        != int(baseline["sample_count"])
        or not isinstance(training_summary.get("evaluation_steps"), list)
        or not isinstance(training_summary.get("checkpoint_hashes"), dict)
        or not isinstance(training_summary.get("checkpoints"), list)
        or not isinstance(training_summary.get("development_evaluation_artifacts"), dict)
    ):
        raise Week7TrainingError("completed training summary identity mismatch")
    global_step = int(training_summary.get("global_step", -1))
    planned_steps = [int(step) for step in training_summary["evaluation_steps"]]
    if planned_steps != sorted(set(planned_steps)) or not planned_steps or global_step <= 0:
        raise Week7TrainingError("training summary evaluation steps are invalid")
    expected_evaluated_steps = [step for step in planned_steps if step <= global_step]
    latency_protocol = None
    protocol_metrics_by_step: dict[str, tuple[dict[str, Any], Path]] = {}
    gate_baseline = baseline
    if latency_protocol_path is not None:
        latency_protocol_path = Path(latency_protocol_path).resolve()
        latency_protocol = validate_latency_protocol_v4(
            latency_protocol_path,
            config_path=config_path,
            training_summary_path=training_summary_path,
            week6_baseline_path=week6_baseline_path,
        )
        if set(latency_protocol["latency_comparison"]) != {
            str(step) for step in expected_evaluated_steps
        }:
            raise Week7TrainingError(
                "latency protocol does not cover every completed checkpoint"
            )
        baseline_protocol_path = Path(
            latency_protocol["roles"]["week6_single_task_adapters"]["metrics_path"]
        ).resolve()
        gate_baseline = _read_json(baseline_protocol_path)
        for step in expected_evaluated_steps:
            protocol_metrics_path = Path(
                latency_protocol["roles"][f"multitask_step_{step:06d}"]["metrics_path"]
            ).resolve()
            protocol_metrics_by_step[str(step)] = (
                _read_json(protocol_metrics_path), protocol_metrics_path,
            )

    expected_samples = (
        int(config["dataset"]["development_per_core_scenario"]) * len(CORE_SCENARIOS)
        + int(config["dataset"]["development_dialogue_count"])
    )
    candidates: list[dict[str, Any]] = []
    evaluation_root = training_dir / "development_evaluations"
    metrics_paths = sorted(evaluation_root.glob("step-*/metrics.json"))
    actual_steps = []
    for metrics_path in metrics_paths:
        try:
            step = int(metrics_path.parent.name.removeprefix("step-"))
        except ValueError as exc:
            raise Week7TrainingError(f"invalid development evaluation step: {metrics_path.parent.name}") from exc
        actual_steps.append(step)
        checkpoint = training_dir / f"checkpoint-{step}"
        adapter_model = checkpoint / "adapter_model.safetensors"
        if not adapter_model.is_file():
            raise Week7TrainingError(f"development evaluation has no matching adapter checkpoint: step {step}")
        metrics = _read_json(metrics_path)
        checkpoint_name = checkpoint.name
        checkpoint_hash = sha256_file(adapter_model)
        if (
            metrics.get("status") != "COMPLETED"
            or metrics.get("model_role") != "multitask_checkpoint"
            or metrics.get("split") != "development"
            or metrics.get("run_id")
            != f"{training_summary['run_id']}_development_step_{step:06d}"
            or metrics.get("config_sha256") != config_hash
            or metrics.get("dataset_lock_sha256") != baseline["dataset_lock_sha256"]
            or int(metrics.get("global_step", -1)) != step
            or int(metrics.get("sample_count", -1)) != expected_samples
            or set(metrics.get("scenarios", {})) != set(CORE_SCENARIOS)
            or not isinstance(metrics.get("dialogue"), dict)
            or int(metrics["dialogue"].get("sample_count", -1))
            != int(config["dataset"]["development_dialogue_count"])
        ):
            raise Week7TrainingError(f"development evaluation coverage mismatch: step {step}")
        raw_artifact = validate_development_raw_artifact(
            metrics, metrics_path, expected_samples,
        )
        if (
            checkpoint_name not in training_summary["checkpoints"]
            or training_summary["checkpoint_hashes"].get(checkpoint_name) != checkpoint_hash
        ):
            raise Week7TrainingError(f"checkpoint hash is not bound by training summary: step {step}")
        recorded_artifacts = training_summary.get("development_evaluation_artifacts", {})
        expected_artifacts = {
            "raw_outputs_path": str(Path(raw_artifact["path"]).resolve()),
            "raw_outputs_sha256": raw_artifact["sha256"],
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
        }
        if recorded_artifacts.get(str(step)) != expected_artifacts:
            raise Week7TrainingError(
                f"development artifacts are not bound by training summary: step {step}"
            )

        gate_metrics, gate_metrics_path = protocol_metrics_by_step.get(
            str(step), (metrics, metrics_path),
        )
        candidate = evaluate_development_candidate(
            config, gate_baseline, gate_metrics, step=step, checkpoint=checkpoint,
            checkpoint_hash=checkpoint_hash, metrics_path=gate_metrics_path,
        )
        candidate["source_training_metrics_path"] = str(metrics_path.resolve())
        candidate["source_training_metrics_sha256"] = sha256_file(metrics_path)
        candidate["evaluation_protocol"] = (
            None if latency_protocol is None else latency_protocol["schema_version"]
        )
        candidates.append(candidate)
    if actual_steps != expected_evaluated_steps:
        raise Week7TrainingError("development evaluation steps differ from completed training summary")
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
        "training_run_id": training_summary["run_id"],
        "selection_rule": "non_regression_gates_then_max_weighted_composite_then_earliest_step",
        "week6_baseline": {
            "path": str(week6_baseline_path),
            "sha256": sha256_file(week6_baseline_path),
        },
        "training_summary": {
            "path": str(training_summary_path),
            "sha256": sha256_file(training_summary_path),
        },
        "latency_protocol": None if latency_protocol is None else {
            "path": str(latency_protocol_path),
            "sha256": sha256_file(latency_protocol_path),
            "run_id": latency_protocol["run_id"],
            "schema_version": latency_protocol["schema_version"],
            "week6_metrics_path": latency_protocol["roles"]
            ["week6_single_task_adapters"]["metrics_path"],
            "week6_metrics_sha256": latency_protocol["roles"]
            ["week6_single_task_adapters"]["metrics_sha256"],
        },
        "candidate_count": len(candidates),
        "eligible_count": len(eligible_candidates),
        "selected": selected,
        "selected_evidence": None if selected is None else {
            "checkpoint_path": selected["checkpoint"],
            "checkpoint_adapter_sha256": selected["checkpoint_adapter_sha256"],
            "metrics_path": selected["metrics_path"],
            "metrics_sha256": selected["metrics_sha256"],
        },
        "candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result

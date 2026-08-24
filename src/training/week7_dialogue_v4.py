"""Select a corrected Week 7 v4 checkpoint from immutable development evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.training.week7_data import (
    CORE_SCENARIOS,
    canonical_sha256,
    iter_jsonl,
    load_week7_config,
    sha256_file,
    validate_week7_lock,
)
from src.training.week7_evaluation import (
    Week7EvaluationError,
    evaluate_dialogue_automatic_gate,
    summarize_raw_records,
)
from src.training.week7_qlora import Week7TrainingError


_SUMMARY_FIELDS = (
    "sample_count",
    "weighted_composite",
    "core_weighted_composite",
    "scenarios",
    "dialogue",
    "latency_ms_mean",
    "latency_ms_median",
    "failure_count",
    "failure_rate",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week7TrainingError(f"invalid v4 checkpoint-selection artifact: {path}") from exc
    if not isinstance(value, dict):
        raise Week7TrainingError(f"v4 checkpoint-selection artifact must be an object: {path}")
    return value


def _finite_unit_interval(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Week7TrainingError(f"v4 development metric is invalid: {field}") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise Week7TrainingError(f"v4 development metric is outside [0, 1]: {field}")
    return result


def _validate_metrics_identity(
    metrics: dict[str, Any],
    *,
    step: int,
    run_id: str,
    config_hash: str,
    dataset_lock_hash: str,
    expected_samples: int,
    expected_dialogue_samples: int,
) -> None:
    if (
        metrics.get("status") != "COMPLETED"
        or metrics.get("model_role") != "multitask_checkpoint"
        or metrics.get("split") != "development"
        or metrics.get("run_id") != f"{run_id}_development_step_{step:06d}"
        or metrics.get("config_sha256") != config_hash
        or metrics.get("dataset_lock_sha256") != dataset_lock_hash
        or int(metrics.get("global_step", -1)) != step
        or int(metrics.get("sample_count", -1)) != expected_samples
        or set(metrics.get("scenarios", {})) != set(CORE_SCENARIOS)
        or not isinstance(metrics.get("dialogue"), dict)
        or int(metrics["dialogue"].get("sample_count", -1))
        != expected_dialogue_samples
    ):
        raise Week7TrainingError(f"v4 development evaluation identity mismatch: step {step}")


def _validate_and_recompute_metrics(
    root: Path,
    config: dict[str, Any],
    development_rows: list[dict[str, Any]],
    metrics_path: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    raw_binding = metrics.get("raw_outputs")
    expected_raw_path = (metrics_path.parent / "raw_outputs.jsonl").resolve()
    if not isinstance(raw_binding, dict) or set(raw_binding) != {"path", "sha256", "count"}:
        raise Week7TrainingError("v4 development raw-output binding is incomplete")
    try:
        bound_path = Path(str(raw_binding["path"])).resolve()
        bound_count = int(raw_binding["count"])
    except (OSError, TypeError, ValueError) as exc:
        raise Week7TrainingError("v4 development raw-output binding is invalid") from exc
    if bound_path != expected_raw_path or not bound_path.is_file():
        raise Week7TrainingError("v4 development raw-output path mismatch")
    if (
        raw_binding.get("sha256") != sha256_file(bound_path)
        or bound_count != len(development_rows)
    ):
        raise Week7TrainingError("v4 development raw-output hash or count mismatch")
    records = list(iter_jsonl(bound_path))
    if len(records) != bound_count:
        raise Week7TrainingError("v4 development raw-output row count mismatch")
    recomputed = summarize_raw_records(
        root,
        config,
        development_rows,
        records,
        metric_support_protocol=config["evaluation"].get("metric_support_protocol"),
    )
    recorded_summary = {field: metrics.get(field) for field in _SUMMARY_FIELDS}
    recomputed_summary = {field: recomputed.get(field) for field in _SUMMARY_FIELDS}
    if canonical_sha256(recorded_summary) != canonical_sha256(recomputed_summary):
        raise Week7TrainingError("v4 development metrics do not match bound raw outputs")
    return recomputed


def _automatic_gate(config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    try:
        return evaluate_dialogue_automatic_gate(config, metrics)
    except Week7EvaluationError as exc:
        raise Week7TrainingError(str(exc)) from exc


def select_v4_checkpoint(
    root: Path,
    config_path: Path,
    training_dir: Path,
    training_summary_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Select the best v4 checkpoint after replaying every automatic development gate."""
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    training_dir = Path(training_dir).resolve()
    training_summary_path = Path(training_summary_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise Week7TrainingError("refusing to overwrite v4 checkpoint-selection evidence")

    config = load_week7_config(config_path)
    if config.get("schema_version") != "week7_multitask_context_v4":
        raise Week7TrainingError("v4 checkpoint selection requires the v4 config")
    config_hash = sha256_file(config_path)
    lock_validation = validate_week7_lock(root, config_path)
    dataset_lock_hash = str(lock_validation["lock_sha256"])
    dataset_root = (
        root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    ).resolve()
    development_path = dataset_root / "development.jsonl"
    development_rows = list(iter_jsonl(development_path))
    expected_samples = (
        int(config["dataset"]["development_per_core_scenario"]) * len(CORE_SCENARIOS)
        + int(config["dataset"]["development_dialogue_count"])
    )
    expected_dialogue_samples = int(config["dataset"]["development_dialogue_count"])
    if len(development_rows) != expected_samples:
        raise Week7TrainingError("v4 development lock support count mismatch")

    if training_summary_path != training_dir / "run_summary.json":
        raise Week7TrainingError("v4 training summary must be training-dir/run_summary.json")
    training_summary = _read_json(training_summary_path)
    run_id = config["experiment_identity"]["multitask_sft_run_id"]
    if (
        training_summary.get("status") != "COMPLETED"
        or training_summary.get("run_id") != run_id
        or training_summary.get("config_sha256") != config_hash
        or training_summary.get("dataset_lock_sha256") != dataset_lock_hash
        or int(training_summary.get("development_samples", -1)) != expected_samples
        or not isinstance(training_summary.get("evaluation_steps"), list)
        or not isinstance(training_summary.get("checkpoints"), list)
        or not isinstance(training_summary.get("checkpoint_hashes"), dict)
        or not isinstance(training_summary.get("development_evaluation_artifacts"), dict)
    ):
        raise Week7TrainingError("completed v4 training summary identity mismatch")
    global_step = int(training_summary.get("global_step", -1))
    try:
        planned_steps = [int(step) for step in training_summary["evaluation_steps"]]
    except (TypeError, ValueError) as exc:
        raise Week7TrainingError("v4 evaluation steps are invalid") from exc
    if planned_steps != sorted(set(planned_steps)) or not planned_steps or global_step <= 0:
        raise Week7TrainingError("v4 evaluation steps are invalid")
    completed_steps = [step for step in planned_steps if step <= global_step]
    if not completed_steps:
        raise Week7TrainingError("v4 training has no completed development evaluation")
    expected_checkpoint_names = {f"checkpoint-{step}" for step in completed_steps}
    if (
        set(training_summary["checkpoints"]) != expected_checkpoint_names
        or set(training_summary["checkpoint_hashes"]) != expected_checkpoint_names
        or set(training_summary["development_evaluation_artifacts"])
        != {str(step) for step in completed_steps}
    ):
        raise Week7TrainingError("v4 completed checkpoint identities are incomplete or unexpected")

    evaluation_root = training_dir / "development_evaluations"
    discovered_paths = sorted(evaluation_root.glob("step-*/metrics.json"))
    expected_paths = [
        evaluation_root / f"step-{step:06d}" / "metrics.json" for step in completed_steps
    ]
    if [path.resolve() for path in discovered_paths] != [path.resolve() for path in expected_paths]:
        raise Week7TrainingError("v4 development evaluation steps differ from training summary")

    candidates: list[dict[str, Any]] = []
    bound_artifacts = training_summary["development_evaluation_artifacts"]
    for step, metrics_path in zip(completed_steps, expected_paths, strict=True):
        checkpoint = training_dir / f"checkpoint-{step}"
        adapter_model = checkpoint / "adapter_model.safetensors"
        if not adapter_model.is_file():
            raise Week7TrainingError(f"v4 checkpoint adapter is missing: step {step}")
        checkpoint_hash = sha256_file(adapter_model)
        checkpoint_name = checkpoint.name
        if (
            checkpoint_name not in training_summary["checkpoints"]
            or training_summary["checkpoint_hashes"].get(checkpoint_name) != checkpoint_hash
        ):
            raise Week7TrainingError(f"v4 checkpoint hash is not bound by training summary: step {step}")
        metrics = _read_json(metrics_path)
        _validate_metrics_identity(
            metrics,
            step=step,
            run_id=run_id,
            config_hash=config_hash,
            dataset_lock_hash=dataset_lock_hash,
            expected_samples=expected_samples,
            expected_dialogue_samples=expected_dialogue_samples,
        )
        raw_path = metrics_path.parent / "raw_outputs.jsonl"
        expected_binding = {
            "raw_outputs_path": str(raw_path.resolve()),
            "raw_outputs_sha256": sha256_file(raw_path),
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
        }
        if bound_artifacts.get(str(step)) != expected_binding:
            raise Week7TrainingError(
                f"v4 development artifacts are not bound by training summary: step {step}"
            )
        recomputed = _validate_and_recompute_metrics(
            root, config, development_rows, metrics_path, metrics
        )
        _finite_unit_interval(recomputed["weighted_composite"], "weighted_composite")
        _finite_unit_interval(
            recomputed["core_weighted_composite"], "core_weighted_composite"
        )
        gate = _automatic_gate(config, recomputed)
        candidates.append({
            "step": step,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_adapter_sha256": checkpoint_hash,
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
            "raw_outputs_path": str(raw_path.resolve()),
            "raw_outputs_sha256": sha256_file(raw_path),
            "weighted_composite": float(recomputed["weighted_composite"]),
            "core_weighted_composite": float(recomputed["core_weighted_composite"]),
            "automatic_gate": gate,
            "eligible": bool(gate["passed"]),
        })

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = max(
        eligible,
        key=lambda candidate: (candidate["weighted_composite"], -candidate["step"]),
        default=None,
    )
    result: dict[str, Any] = {
        "schema_version": "week7_v4_development_checkpoint_selection_v1",
        "status": "PASS" if selected is not None else "BLOCKED_NO_ELIGIBLE_CHECKPOINT",
        "eligible": selected is not None,
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "dataset_version": config["dataset"]["dataset_version"],
        "dataset_lock_sha256": dataset_lock_hash,
        "development_path": str(development_path),
        "development_sha256": sha256_file(development_path),
        "development_samples": expected_samples,
        "test_read": False,
        "training_run_id": run_id,
        "training_dir": str(training_dir),
        "training_summary_path": str(training_summary_path),
        "training_summary_sha256": sha256_file(training_summary_path),
        "selection_rule": "automatic_gates_then_max_weighted_composite_then_earliest_step",
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "selected_checkpoint": (
            {
                "path": selected["checkpoint"],
                "adapter_sha256": selected["checkpoint_adapter_sha256"],
                "step": selected["step"],
                "metrics_path": selected["metrics_path"],
                "metrics_sha256": selected["metrics_sha256"],
                "raw_outputs_path": selected["raw_outputs_path"],
                "raw_outputs_sha256": selected["raw_outputs_sha256"],
                "weighted_composite": selected["weighted_composite"],
                "core_weighted_composite": selected["core_weighted_composite"],
                "automatic_gate": selected["automatic_gate"],
            }
            if selected is not None else None
        ),
        "candidates": candidates,
    }
    result["selection_sha256"] = canonical_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    if selected is None:
        raise Week7TrainingError(
            f"no v4 checkpoint passed the automatic development gate; "
            f"blocked evidence: {output_path}"
        )
    return result

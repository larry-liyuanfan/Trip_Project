"""Parameter locking and resumable one-shot Week 7 final-test evaluation."""

from __future__ import annotations

import json
import hashlib
import math
import os
import socket
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from src.training.week7_data import CORE_SCENARIOS, canonical_sha256, iter_jsonl, load_week7_config, sha256_file
from src.training.week7_evaluation import (
    Week7EvaluationError,
    compare_schema_decoding,
    summarize_raw_records,
)
from src.training.week7_qlora import Week7TrainingError, structure_aware_messages, training_messages
from src.evaluation.metrics import WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL
from src.training.week7_latency_protocol import (
    _default_model_loader,
    _protocol_runtime,
    _release_model,
    validate_latency_protocol_v4,
)
from src.training.week7_runtime import (
    LATENCY_PROTOCOL_VERSION,
    generate_record,
    inference_runtime,
)
from src.training.week7_selection import (
    evaluate_development_candidate,
    validate_development_raw_artifact,
)


REQUIRED_DEVELOPMENT_EVIDENCE = {
    "week6_development_baseline",
    "zero_shot_development",
    "multitask_development",
    "schema_decoding",
}
FINAL_ROLES = ("week6_single_task_adapters", "multitask", "zero_shot")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week7EvaluationError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise Week7EvaluationError(f"JSON artifact must be an object: {path}")
    return value


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _adapter_spec(
    path: Path,
    expected_sha256: str | None = None,
    expected_base_model: str | None = None,
) -> dict[str, str]:
    adapter_dir = Path(path).resolve()
    model_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not model_path.is_file() or not config_path.is_file():
        raise Week7EvaluationError(f"adapter directory is incomplete: {adapter_dir}")
    model_hash = sha256_file(model_path)
    if expected_sha256 is not None and model_hash != expected_sha256.lower():
        raise Week7EvaluationError(f"adapter SHA-256 mismatch: {adapter_dir}")
    adapter_config = _read_json(config_path)
    if (
        expected_base_model is not None
        and adapter_config.get("base_model_name_or_path") != expected_base_model
    ):
        raise Week7EvaluationError(f"adapter base-model identity mismatch: {adapter_dir}")
    return {
        "adapter_dir": str(adapter_dir),
        "adapter_model_sha256": model_hash,
        "adapter_config_sha256": sha256_file(config_path),
    }


def _require_evidence_identity(
    name: str,
    evidence: dict[str, Any],
    *,
    config_sha256: str,
    dataset_lock_sha256: str,
    model_role: str,
) -> None:
    expected = {
        "status": "COMPLETED",
        "config_sha256": config_sha256,
        "dataset_lock_sha256": dataset_lock_sha256,
        "model_role": model_role,
        "split": "development",
    }
    for field, value in expected.items():
        if evidence.get(field) != value:
            raise Week7EvaluationError(
                f"development evidence identity mismatch ({name}.{field})"
            )


def _validate_development_evidence(
    name: str,
    evidence: dict[str, Any],
    *,
    config: dict[str, Any],
    config_sha256: str,
    dataset_lock_sha256: str,
    selected_checkpoint_sha256: str,
    selected_step: int | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Validate evidence semantics, not merely its completion flag."""
    identity = config["experiment_identity"]
    development_core = int(config["dataset"]["development_per_core_scenario"])
    development_dialogue = int(config["dataset"]["development_dialogue_count"])
    core_total = development_core * len(CORE_SCENARIOS)
    full_total = core_total + development_dialogue

    if name == "week6_development_baseline":
        _require_evidence_identity(
            name, evidence, config_sha256=config_sha256,
            dataset_lock_sha256=dataset_lock_sha256,
            model_role="week6_single_task_adapters",
        )
        if set(evidence.get("scenarios", {})) != set(CORE_SCENARIOS):
            raise Week7EvaluationError("Week 6 development scenario coverage mismatch")
        expected_combined_run = identity.get(
            "week6_combined_development_run_id",
            "week7_dev_week6_adapters_baseline_20260819_v2",
        )
        if evidence.get("run_id") != expected_combined_run:
            raise Week7EvaluationError("Week 6 combined development run identity mismatch")
        if (
            int(evidence.get("sample_count", -1)) != full_total
            or not isinstance(evidence.get("dialogue"), dict)
            or int(evidence["dialogue"].get("sample_count", -1)) != development_dialogue
        ):
            raise Week7EvaluationError("Week 6 development support count mismatch")
        inputs = evidence.get("inputs", {})
        if set(inputs) != set(CORE_SCENARIOS) | {"dialogue"}:
            raise Week7EvaluationError("Week 6 development input evidence is incomplete")
        input_sample_count = 0
        input_failure_count = 0
        input_latency_sum = 0.0
        scenario_payloads: dict[str, Any] = {}
        for scenario in CORE_SCENARIOS:
            input_spec = inputs[scenario]
            path = Path(str(input_spec.get("path", "")))
            if not path.is_file() or sha256_file(path) != input_spec.get("sha256"):
                raise Week7EvaluationError(f"Week 6 scenario evidence hash mismatch: {scenario}")
            scenario_evidence = _read_json(path)
            _require_evidence_identity(
                f"{name}.{scenario}", scenario_evidence,
                config_sha256=config_sha256,
                dataset_lock_sha256=dataset_lock_sha256,
                model_role="week6_single_task_adapter",
            )
            if (
                scenario_evidence.get("run_id") != identity["development_baseline_run_ids"][scenario]
                or scenario_evidence.get("scenario_filter") != scenario
                or set(scenario_evidence.get("scenarios", {})) != {scenario}
                or int(scenario_evidence.get("sample_count", -1)) != development_core
                or scenario_evidence.get("adapter_hashes", {}).get("adapter_model.safetensors")
                != config["evaluation"]["week6_adapter_sha256"][scenario]
            ):
                raise Week7EvaluationError(f"Week 6 scenario evidence mismatch: {scenario}")
            scenario_payloads[scenario] = scenario_evidence["scenarios"][scenario]
            if canonical_sha256(scenario_payloads[scenario]) != canonical_sha256(
                evidence["scenarios"][scenario]
            ):
                raise Week7EvaluationError(
                    f"Week 6 combined scenario differs from input: {scenario}"
                )
            input_sample_count += int(scenario_evidence["sample_count"])
            input_failure_count += int(scenario_evidence["failure_count"])
            input_latency_sum += (
                float(scenario_evidence["latency_ms_mean"])
                * int(scenario_evidence["sample_count"])
            )
        dialogue_spec = inputs["dialogue"]
        dialogue_path = Path(str(dialogue_spec.get("path", "")))
        if not dialogue_path.is_file() or sha256_file(dialogue_path) != dialogue_spec.get("sha256"):
            raise Week7EvaluationError("Week 6 dialogue evidence hash mismatch")
        dialogue_evidence = _read_json(dialogue_path)
        _require_evidence_identity(
            f"{name}.dialogue", dialogue_evidence,
            config_sha256=config_sha256,
            dataset_lock_sha256=dataset_lock_sha256,
            model_role="week6_single_task_adapters",
        )
        dialogue_hashes = dialogue_evidence.get("adapter_hashes", {})
        route_counts = dialogue_evidence.get("routing", {}).get("sample_counts", {})
        expected_route_counts = config["sampling"].get(
            "dialogue_parent_scenario_counts", {}
        ).get(
            "development",
            {
                scenario: development_dialogue // len(CORE_SCENARIOS)
                for scenario in CORE_SCENARIOS
            },
        )
        if (
            dialogue_evidence.get("run_id") != identity.get(
                "week6_dialogue_development_run_id",
                "week7_dev_week6_dialogue_routed_20260819_v2",
            )
            or dialogue_evidence.get("scenario_filter") != "dialogue_routed"
            or int(dialogue_evidence.get("sample_count", -1)) != development_dialogue
            or set(dialogue_evidence.get("scenarios", {}))
            or not isinstance(dialogue_evidence.get("dialogue"), dict)
            or int(dialogue_evidence["dialogue"].get("sample_count", -1)) != development_dialogue
            or set(dialogue_hashes) != set(CORE_SCENARIOS)
            or any(
                dialogue_hashes[scenario].get("adapter_model.safetensors")
                != config["evaluation"]["week6_adapter_sha256"][scenario]
                for scenario in CORE_SCENARIOS
            )
            or dialogue_evidence.get("routing", {}).get("method")
            != "target_task_result_v1"
            or route_counts != expected_route_counts
        ):
            raise Week7EvaluationError("Week 6 dialogue evidence mismatch")
        if canonical_sha256(dialogue_evidence["dialogue"]) != canonical_sha256(
            evidence["dialogue"]
        ):
            raise Week7EvaluationError("Week 6 combined dialogue differs from input")
        input_sample_count += int(dialogue_evidence["sample_count"])
        input_failure_count += int(dialogue_evidence["failure_count"])
        input_latency_sum += (
            float(dialogue_evidence["latency_ms_mean"])
            * int(dialogue_evidence["sample_count"])
        )
        weights = config["evaluation"]["scenario_weights"]
        weighted_composite = sum(
            float(weights[scenario]) * float(scenario_payloads[scenario]["composite"])
            for scenario in CORE_SCENARIOS
        )
        if (
            input_sample_count != full_total
            or int(evidence.get("failure_count", -1)) != input_failure_count
            or not math.isclose(
                float(evidence.get("failure_rate", -1.0)),
                input_failure_count / input_sample_count,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(evidence.get("latency_ms_mean", -1.0)),
                input_latency_sum / input_sample_count,
                rel_tol=0,
                abs_tol=1e-9,
            )
            or evidence.get("latency_ms_median") is not None
            or not math.isclose(
                float(evidence.get("weighted_composite", -1.0)),
                weighted_composite,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise Week7EvaluationError("Week 6 combined development aggregate mismatch")
        return {
            "run_id": evidence.get("run_id"),
            "model_role": evidence["model_role"],
            "scenario_coverage": sorted(evidence["scenarios"]),
            "dialogue_sample_count": development_dialogue,
            "sample_count": full_total,
        }

    if name in {"zero_shot_development", "multitask_development"}:
        is_zero = name == "zero_shot_development"
        role = "zero_shot" if is_zero else "multitask_checkpoint"
        if not is_zero and (selected_step is None or selected_step <= 0):
            raise Week7EvaluationError("selected checkpoint step is required")
        expected_run = (
            identity["zero_shot_development_run_id"]
            if is_zero
            else f"{identity['multitask_sft_run_id']}_development_step_{selected_step:06d}"
        )
        _require_evidence_identity(
            name, evidence, config_sha256=config_sha256,
            dataset_lock_sha256=dataset_lock_sha256, model_role=role,
        )
        if (
            evidence.get("run_id") != expected_run
            or evidence.get("scenario_filter") is not None
            or set(evidence.get("scenarios", {})) != set(CORE_SCENARIOS)
            or int(evidence.get("sample_count", -1)) != full_total
            or not isinstance(evidence.get("dialogue"), dict)
            or int(evidence["dialogue"].get("sample_count", -1)) != development_dialogue
            or (not is_zero and int(evidence.get("global_step", -1)) != selected_step)
        ):
            raise Week7EvaluationError(f"development evidence coverage mismatch: {name}")
        adapter_hash = (evidence.get("adapter_hashes") or {}).get("adapter_model.safetensors")
        if is_zero and evidence.get("adapter_hashes") is not None:
            raise Week7EvaluationError("zero-shot development evidence unexpectedly used an adapter")
        return {
            "run_id": evidence["run_id"], "model_role": role,
            "scenario_coverage": sorted(evidence["scenarios"]),
            "dialogue_sample_count": development_dialogue,
            "sample_count": full_total,
            **({"global_step": selected_step} if not is_zero else {}),
        }

    if name != "schema_decoding":
        raise Week7EvaluationError(f"unexpected development evidence: {name}")
    _require_evidence_identity(
        name, evidence, config_sha256=config_sha256,
        dataset_lock_sha256=dataset_lock_sha256,
        model_role="schema_format_only_experiment",
    )
    expected_runs = {
        "free": identity["schema_free_run_id"],
        "constrained": identity["schema_constrained_run_id"],
    }
    gate = evidence.get("gate")
    modes = evidence.get("modes")
    if (
        evidence.get("scope") != "format_only"
        or evidence.get("semantic_claims") != "FORBIDDEN"
        or evidence.get("paired_order")
        != config["evaluation"]["schema_decoding"]["paired_order"]
        or not isinstance(evidence.get("endpoint_recorded"), str)
        or not evidence.get("endpoint_recorded")
        or evidence.get("run_ids") != expected_runs
        or int(evidence.get("sample_count", -1)) != core_total
        or evidence.get("served_model") != config["base_model"]
        or not isinstance(gate, dict)
        or set(gate) != {"latency", "free_request", "constrained_request", "fallback"}
        or not all(isinstance(value, bool) for value in gate.values())
        or not isinstance(modes, dict)
        or set(modes) != {"free", "constrained"}
    ):
        raise Week7EvaluationError("Schema development evidence identity/gate mismatch")
    model_identity = evidence.get("model_identity")
    completion = evidence.get("completion_eligibility")
    raw_artifacts = evidence.get("raw_artifacts")
    if (
        model_identity != {
            "base_model": config["base_model"],
            "requested_served_model": config["base_model"],
            "registry_model_ids": [config["base_model"]],
            "successful_response_model_ids": [config["base_model"]],
            "verified": True,
        }
        or not isinstance(completion, dict)
        or completion != {
            "eligible": True,
            "free_has_success": True,
            "constrained_operational_has_success": True,
            "served_model_verified": True,
        }
        or not isinstance(raw_artifacts, dict)
        or set(raw_artifacts) != {"free", "constrained"}
    ):
        raise Week7EvaluationError("Schema served-model/completion identity mismatch")
    required_mode_metrics = {
        "json_compliance", "schema_coverage", "request_count",
        "primary_failure_count", "primary_failure_rate",
        "operational_failure_count", "operational_failure_rate",
        "fallback_request_count", "fallback_failure_count",
        "fallback_failure_rate", "latency_ms_mean",
    }
    if any(not required_mode_metrics <= set(payload) for payload in modes.values()):
        raise Week7EvaluationError("Schema development mode metrics are incomplete")
    for mode, mode_metrics in modes.items():
        for metric in required_mode_metrics:
            value = mode_metrics[metric]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise Week7EvaluationError(f"Schema development metric is invalid: {mode}.{metric}")
        if any(
            not 0.0 <= float(mode_metrics[metric]) <= 1.0
            for metric in (
                "json_compliance", "schema_coverage", "primary_failure_rate",
                "operational_failure_rate", "fallback_failure_rate",
            )
        ) or float(mode_metrics["latency_ms_mean"]) < 0:
            raise Week7EvaluationError(f"Schema development metric is out of range: {mode}")
        request_count = int(mode_metrics["request_count"])
        fallback_count = int(mode_metrics["fallback_request_count"])
        if (
            request_count != core_total
            or int(mode_metrics["primary_failure_count"]) < 0
            or int(mode_metrics["operational_failure_count"]) < 0
            or not 0 <= fallback_count <= request_count
            or not 0 <= int(mode_metrics["fallback_failure_count"]) <= fallback_count
            or not math.isclose(
                float(mode_metrics["primary_failure_rate"]),
                int(mode_metrics["primary_failure_count"]) / request_count,
                rel_tol=0, abs_tol=1e-12,
            )
            or not math.isclose(
                float(mode_metrics["operational_failure_rate"]),
                int(mode_metrics["operational_failure_count"]) / request_count,
                rel_tol=0, abs_tol=1e-12,
            )
            or not math.isclose(
                float(mode_metrics["fallback_failure_rate"]),
                int(mode_metrics["fallback_failure_count"]) / fallback_count
                if fallback_count else 0.0,
                rel_tol=0, abs_tol=1e-12,
            )
        ):
            raise Week7EvaluationError(f"Schema request/fallback counts are inconsistent: {mode}")
    free_latency = float(modes["free"]["latency_ms_mean"])
    constrained_latency = float(modes["constrained"]["latency_ms_mean"])
    computed_gate = {
        "latency": free_latency > 0 and constrained_latency / free_latency
        <= float(config["evaluation"]["schema_decoding"]["max_latency_ratio"]),
        "free_request": float(modes["free"]["primary_failure_rate"])
        <= float(config["evaluation"]["non_regression"]["max_failure_rate"]),
        "constrained_request": float(modes["constrained"]["primary_failure_rate"])
        <= float(config["evaluation"]["non_regression"]["max_failure_rate"]),
        "fallback": float(modes["constrained"]["fallback_failure_rate"])
        <= float(config["evaluation"]["schema_decoding"]["max_fallback_failure_rate"]),
    }
    if gate != computed_gate:
        raise Week7EvaluationError("Schema development gate is inconsistent with measured modes")
    if not gate["free_request"] or not gate["fallback"]:
        raise Week7EvaluationError("Schema production free/fallback request gate failed")
    if root is None:
        raise Week7EvaluationError("Schema raw evidence validation requires the repository root")
    raw_records: dict[str, list[dict[str, Any]]] = {}
    for mode in ("free", "constrained"):
        spec = raw_artifacts[mode]
        path = Path(str(spec.get("path", ""))).resolve()
        if (
            not path.is_file()
            or sha256_file(path) != spec.get("sha256")
            or int(spec.get("count", -1)) != core_total
        ):
            raise Week7EvaluationError(f"Schema raw artifact binding mismatch: {mode}")
        records = list(iter_jsonl(path))
        expected_run = expected_runs[mode]
        if (
            len(records) != core_total
            or any(record.get("run_id") != expected_run for record in records)
            or any(record.get("scenario") not in CORE_SCENARIOS for record in records)
        ):
            raise Week7EvaluationError(f"Schema raw record identity mismatch: {mode}")
        if mode == "free":
            record_semantics_mismatch = any(
                bool(record.get("failed")) != bool(record.get("error"))
                or bool(record.get("fallback_used"))
                or bool(record.get("fallback_failed"))
                for record in records
            )
            response_mismatch = any(
                (record.get("response_model") != config["base_model"])
                if not record.get("failed") else record.get("response_model") is not None
                for record in records
            )
        else:
            record_semantics_mismatch = any(
                bool(record.get("constrained_error"))
                != bool(record.get("fallback_used"))
                or bool(record.get("fallback_error"))
                != bool(record.get("fallback_failed"))
                or bool(record.get("failed"))
                != (bool(record.get("fallback_failed")) if record.get("fallback_used") else False)
                or record.get("raw_output")
                != record.get("primary_constrained_raw_output")
                or record.get("operational_raw_output")
                != (
                    record.get("fallback_raw_output")
                    if record.get("fallback_used") and not record.get("fallback_failed")
                    else record.get("primary_constrained_raw_output")
                )
                for record in records
            )
            response_mismatch = any(
                (
                    record.get("primary_response_model") != config["base_model"]
                    if not record.get("constrained_error")
                    else record.get("primary_response_model") is not None
                )
                or (
                    record.get("fallback_response_model") != config["base_model"]
                    if record.get("fallback_used") and not record.get("fallback_failed")
                    else record.get("fallback_response_model") is not None
                    if not record.get("fallback_used") or record.get("fallback_failed")
                    else False
                )
                for record in records
            )
        if record_semantics_mismatch:
            raise Week7EvaluationError(f"Schema raw request/fallback semantics mismatch: {mode}")
        if response_mismatch:
            raise Week7EvaluationError(f"Schema raw served-model identity mismatch: {mode}")
        raw_records[mode] = records
    rows = [
        {"sample_id": record["sample_id"], "scenario": record["scenario"]}
        for record in raw_records["free"]
    ]
    recomputed = compare_schema_decoding(
        Path(root).resolve(), config, rows,
        raw_records["free"], raw_records["constrained"],
    )
    if int(evidence.get("fallback_used_count", -1)) != int(
        recomputed["modes"]["constrained"]["fallback_request_count"]
    ):
        raise Week7EvaluationError("Schema fallback-used count differs from raw evidence")
    for field in ("scope", "semantic_claims", "sample_count", "modes", "deltas", "gate"):
        if canonical_sha256(recomputed[field]) != canonical_sha256(evidence[field]):
            raise Week7EvaluationError(f"Schema comparison differs from raw evidence: {field}")
    return {
        "run_ids": expected_runs,
        "model_role": evidence["model_role"],
        "sample_count": core_total,
        "gate": gate,
        "served_model": config["base_model"],
        "raw_artifact_sha256": {
            mode: raw_artifacts[mode]["sha256"] for mode in ("free", "constrained")
        },
        "selected_mode": "free",
    }


def _validate_training_summary(
    summary_path: Path,
    *,
    config: dict[str, Any],
    config_sha256: str,
    dataset_lock_sha256: str,
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    summary = _read_json(summary_path)
    if (
        summary.get("status") != "COMPLETED"
        or summary.get("run_id") != config["experiment_identity"]["multitask_sft_run_id"]
        or summary.get("config_sha256") != config_sha256
        or summary.get("dataset_lock_sha256") != dataset_lock_sha256
        or not isinstance(summary.get("evaluation_steps"), list)
        or not isinstance(summary.get("checkpoints"), list)
        or not isinstance(summary.get("checkpoint_hashes"), dict)
        or not isinstance(summary.get("development_evaluation_artifacts"), dict)
    ):
        raise Week7EvaluationError("completed multitask training summary identity mismatch")
    steps = [int(step) for step in summary["evaluation_steps"]]
    global_step = int(summary.get("global_step", -1))
    if not steps or steps != sorted(set(steps)) or global_step <= 0:
        raise Week7EvaluationError("training summary evaluation steps are invalid")
    return summary


def _validate_selection_artifact(
    selection_path: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    dataset_lock_sha256: str,
    training_summary_path: Path,
    training_summary_sha256: str,
    week6_baseline_path: Path,
    week6_baseline_sha256: str,
    selected_checkpoint: Path,
    selected_checkpoint_sha256: str,
    selected_metrics_path: Path,
    selected_metrics_sha256: str,
) -> dict[str, Any]:
    selection_path = Path(selection_path).resolve()
    selection = _read_json(selection_path)
    selected = selection.get("selected")
    selected_evidence = selection.get("selected_evidence")
    if (
        selection.get("status") != "SELECTED"
        or selection.get("config_sha256") != config_sha256
        or selection.get("dataset_lock_sha256") != dataset_lock_sha256
        or selection.get("training_run_id")
        != config["experiment_identity"]["multitask_sft_run_id"]
        or not isinstance(selected, dict)
        or selected.get("eligible") is not True
        or not isinstance(selected_evidence, dict)
    ):
        raise Week7EvaluationError("selected checkpoint artifact identity mismatch")

    expected_summary = {
        "path": str(Path(training_summary_path).resolve()),
        "sha256": training_summary_sha256,
    }
    expected_baseline = {
        "path": str(Path(week6_baseline_path).resolve()),
        "sha256": week6_baseline_sha256,
    }
    if selection.get("training_summary") != expected_summary:
        raise Week7EvaluationError("selection training summary binding mismatch")
    if selection.get("week6_baseline") != expected_baseline:
        raise Week7EvaluationError("selection Week 6 baseline binding mismatch")

    protocol = None
    protocol_spec = selection.get("latency_protocol")
    protocol_metrics_by_step: dict[str, tuple[dict[str, Any], Path]] = {}
    gate_baseline = None
    if protocol_spec is not None:
        if not isinstance(protocol_spec, dict):
            raise Week7EvaluationError("selection evaluation-protocol binding is invalid")
        protocol_path = Path(str(protocol_spec.get("path", ""))).resolve()
        if not protocol_path.is_file() or sha256_file(protocol_path) != protocol_spec.get("sha256"):
            raise Week7EvaluationError("selection evaluation-protocol hash mismatch")
        try:
            protocol = validate_latency_protocol_v4(
                protocol_path,
                config_path=config_path,
                training_summary_path=training_summary_path,
                week6_baseline_path=week6_baseline_path,
            )
        except (OSError, KeyError, TypeError, ValueError, Week7TrainingError) as exc:
            raise Week7EvaluationError("selection evaluation protocol is invalid") from exc
        expected_protocol_spec = {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "run_id": protocol["run_id"],
            "schema_version": protocol["schema_version"],
            "week6_metrics_path": protocol["roles"]
            ["week6_single_task_adapters"]["metrics_path"],
            "week6_metrics_sha256": protocol["roles"]
            ["week6_single_task_adapters"]["metrics_sha256"],
        }
        if protocol_spec != expected_protocol_spec:
            raise Week7EvaluationError("selection evaluation-protocol identity mismatch")
        gate_baseline = _read_json(Path(protocol_spec["week6_metrics_path"]))
        for step in protocol["candidate_steps"]:
            metrics_path = Path(
                protocol["roles"][f"multitask_step_{int(step):06d}"]["metrics_path"]
            ).resolve()
            protocol_metrics_by_step[str(int(step))] = (_read_json(metrics_path), metrics_path)

    checkpoint_path = Path(selected_checkpoint).resolve()
    source_metrics_path = Path(selected_metrics_path).resolve()
    selected_step = int(selected.get("step", -1))
    metrics_path = (
        protocol_metrics_by_step[str(selected_step)][1]
        if protocol is not None else source_metrics_path
    )
    expected_evidence = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_adapter_sha256": selected_checkpoint_sha256,
        "metrics_path": str(metrics_path),
        "metrics_sha256": (
            sha256_file(metrics_path) if protocol is not None
            else selected_metrics_sha256
        ),
    }
    if selected_evidence != expected_evidence:
        raise Week7EvaluationError("selection evidence binding mismatch")
    if (
        Path(str(selected.get("checkpoint", ""))).resolve() != checkpoint_path
        or selected.get("checkpoint_adapter_sha256") != selected_checkpoint_sha256
        or Path(str(selected.get("metrics_path", ""))).resolve() != metrics_path
        or selected.get("metrics_sha256") != selected_metrics_sha256
    ):
        if protocol is None:
            raise Week7EvaluationError("selected candidate binding mismatch")
    if protocol is not None and (
        selected.get("metrics_sha256") != sha256_file(metrics_path)
        or selected.get("source_training_metrics_path") != str(source_metrics_path)
        or selected.get("source_training_metrics_sha256") != selected_metrics_sha256
    ):
        raise Week7EvaluationError("selected protocol/source metrics binding mismatch")
    summary = _validate_training_summary(
        training_summary_path, config=config, config_sha256=config_sha256,
        dataset_lock_sha256=dataset_lock_sha256,
    )
    baseline = _read_json(week6_baseline_path)
    _validate_development_evidence(
        "week6_development_baseline", baseline, config=config,
        config_sha256=config_sha256,
        dataset_lock_sha256=dataset_lock_sha256,
        selected_checkpoint_sha256=selected_checkpoint_sha256,
    )
    expected_steps = [
        int(step) for step in summary["evaluation_steps"]
        if int(step) <= int(summary["global_step"])
    ]
    if set(summary["development_evaluation_artifacts"]) != {
        str(step) for step in expected_steps
    }:
        raise Week7EvaluationError("training summary development artifact coverage mismatch")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(expected_steps):
        raise Week7EvaluationError("selection candidate coverage mismatch")
    recomputed_candidates = []
    training_dir = Path(training_summary_path).resolve().parent
    expected_samples = (
        int(config["dataset"]["development_per_core_scenario"])
        * len(CORE_SCENARIOS)
        + int(config["dataset"]["development_dialogue_count"])
    )
    if gate_baseline is None:
        gate_baseline = baseline
    for recorded, step in zip(candidates, expected_steps):
        checkpoint = training_dir / f"checkpoint-{step}"
        metrics_path = (
            training_dir / "development_evaluations" / f"step-{step:06d}"
            / "metrics.json"
        )
        adapter_path = checkpoint / "adapter_model.safetensors"
        if not adapter_path.is_file() or not metrics_path.is_file():
            raise Week7EvaluationError("selection candidate artifacts are missing")
        checkpoint_hash = sha256_file(adapter_path)
        metrics_hash = sha256_file(metrics_path)
        metrics = _read_json(metrics_path)
        if (
            checkpoint.name not in summary["checkpoints"]
            or summary["checkpoint_hashes"].get(checkpoint.name) != checkpoint_hash
            or metrics.get("status") != "COMPLETED"
            or metrics.get("model_role") != "multitask_checkpoint"
            or metrics.get("split") != "development"
            or metrics.get("run_id")
            != f"{summary['run_id']}_development_step_{step:06d}"
            or metrics.get("config_sha256") != config_sha256
            or metrics.get("dataset_lock_sha256") != dataset_lock_sha256
            or int(metrics.get("global_step", -1)) != step
            or int(metrics.get("sample_count", -1)) != expected_samples
            or set(metrics.get("scenarios", {})) != set(CORE_SCENARIOS)
            or not isinstance(metrics.get("dialogue"), dict)
            or int(metrics["dialogue"].get("sample_count", -1))
            != int(config["dataset"]["development_dialogue_count"])
        ):
            raise Week7EvaluationError("selection candidate metrics identity mismatch")
        try:
            raw_artifact = validate_development_raw_artifact(
                metrics, metrics_path, expected_samples
            )
            expected_artifacts = {
                "raw_outputs_path": str(Path(raw_artifact["path"]).resolve()),
                "raw_outputs_sha256": raw_artifact["sha256"],
                "metrics_path": str(metrics_path.resolve()),
                "metrics_sha256": metrics_hash,
            }
            if summary["development_evaluation_artifacts"].get(str(step)) != expected_artifacts:
                raise Week7TrainingError(
                    f"training summary development artifact mismatch: step {step}"
                )
            gate_metrics, gate_metrics_path = protocol_metrics_by_step.get(
                str(step), (metrics, metrics_path),
            )
            recomputed = evaluate_development_candidate(
                config, gate_baseline, gate_metrics, step=step, checkpoint=checkpoint,
                checkpoint_hash=checkpoint_hash, metrics_path=gate_metrics_path,
            )
            recomputed["source_training_metrics_path"] = str(metrics_path.resolve())
            recomputed["source_training_metrics_sha256"] = metrics_hash
            recomputed["evaluation_protocol"] = (
                None if protocol is None else protocol["schema_version"]
            )
        except (KeyError, TypeError, ValueError, Week7TrainingError) as exc:
            raise Week7EvaluationError(
                f"selection candidate gate recomputation failed: step {step}"
            ) from exc
        if not isinstance(recorded, dict) or canonical_sha256(recorded) != canonical_sha256(
            recomputed
        ):
            raise Week7EvaluationError("selection candidate gates were not reproducible")
        expected_gate_hash = sha256_file(
            protocol_metrics_by_step[str(step)][1]
        ) if protocol is not None else metrics_hash
        if recorded.get("metrics_sha256") != expected_gate_hash:
            raise Week7EvaluationError("selection candidate metrics hash mismatch")
        recomputed_candidates.append(recomputed)
    eligible = [candidate for candidate in recomputed_candidates if candidate["eligible"]]
    recomputed_selected = max(
        eligible,
        key=lambda candidate: (candidate["weighted_composite"], -candidate["step"]),
        default=None,
    )
    if (
        recomputed_selected is None
        or canonical_sha256(selected) != canonical_sha256(recomputed_selected)
        or int(selection.get("candidate_count", -1)) != len(recomputed_candidates)
        or int(selection.get("eligible_count", -1)) != len(eligible)
    ):
        raise Week7EvaluationError("selection winner is not the highest eligible checkpoint")
    return selection


def create_parameter_lock(
    root: Path,
    config_path: Path,
    output_path: Path,
    *,
    training_summary_path: Path,
    selection_path: Path,
    selected_checkpoint: Path,
    week6_adapters: dict[str, tuple[Path, str]],
    development_evidence: dict[str, Path],
    schema_decoding_mode: str,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Create an immutable lock without opening the final-test JSONL."""
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = load_week7_config(config_path)
    if schema_decoding_mode != "free":
        raise Week7EvaluationError(
            "parameter locking currently permits only the production-supported free mode"
        )
    if set(week6_adapters) != set(CORE_SCENARIOS):
        raise Week7EvaluationError("exactly three scenario-specific Week 6 adapters are required")
    if set(development_evidence) != REQUIRED_DEVELOPMENT_EVIDENCE:
        raise Week7EvaluationError("all four locked development evidence artifacts are required")

    lock_root = root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    dataset_lock_path = lock_root / "dataset_lock.json"
    dataset_lock = _read_json(dataset_lock_path)
    dataset_claim = dataset_lock.pop("lock_sha256", None)
    if dataset_claim != canonical_sha256(dataset_lock):
        raise Week7EvaluationError("dataset lock canonical SHA-256 mismatch")
    dataset_lock["lock_sha256"] = dataset_claim
    config_hash = sha256_file(config_path)
    if dataset_lock.get("config_sha256") != config_hash:
        raise Week7EvaluationError("dataset lock is bound to a different Week 7 config")
    if dataset_lock.get("test_policy", {}).get("status") != "LOCKED_UNCONSUMED":
        raise Week7EvaluationError("dataset does not declare an unconsumed one-shot test")

    training_summary_path = Path(training_summary_path).resolve()
    training_summary = _validate_training_summary(
        training_summary_path, config=config, config_sha256=config_hash,
        dataset_lock_sha256=dataset_lock["lock_sha256"],
    )

    selected = Path(selected_checkpoint).resolve()
    selected_spec = _adapter_spec(selected, expected_base_model=config["base_model"])
    recorded_hash = training_summary.get("checkpoint_hashes", {}).get(selected.name)
    if (
        selected.parent != training_summary_path.parent
        or selected.name not in training_summary["checkpoints"]
        or recorded_hash != selected_spec["adapter_model_sha256"]
    ):
        raise Week7EvaluationError("selected checkpoint does not match the training summary")

    resolved_evidence = {
        name: Path(path).resolve() for name, path in development_evidence.items()
    }
    selection_path = Path(selection_path).resolve()
    selection = _validate_selection_artifact(
        selection_path, config=config, config_path=config_path,
        config_sha256=config_hash,
        dataset_lock_sha256=dataset_lock["lock_sha256"],
        training_summary_path=training_summary_path,
        training_summary_sha256=sha256_file(training_summary_path),
        week6_baseline_path=resolved_evidence["week6_development_baseline"],
        week6_baseline_sha256=sha256_file(
            resolved_evidence["week6_development_baseline"]
        ),
        selected_checkpoint=selected,
        selected_checkpoint_sha256=selected_spec["adapter_model_sha256"],
        selected_metrics_path=resolved_evidence["multitask_development"],
        selected_metrics_sha256=sha256_file(
            resolved_evidence["multitask_development"]
        ),
    )
    selected_step = int(selection["selected"].get("step", -1))
    if (
        selected.name != f"checkpoint-{selected_step}"
        or selected_step not in [int(step) for step in training_summary["evaluation_steps"]]
        or selected_step > int(training_summary["global_step"])
    ):
        raise Week7EvaluationError("selected checkpoint step is not a completed evaluation step")

    evidence_payload: dict[str, Any] = {}
    for name, resolved in sorted(resolved_evidence.items()):
        evidence = _read_json(resolved)
        identity_summary = _validate_development_evidence(
            name, evidence, config=config, config_sha256=config_hash,
            dataset_lock_sha256=dataset_lock["lock_sha256"],
            selected_checkpoint_sha256=selected_spec["adapter_model_sha256"],
            selected_step=selected_step,
            root=root,
        )
        evidence_payload[name] = {
            "path": str(resolved), "sha256": sha256_file(resolved),
            "identity": identity_summary,
        }

    week6_specs = {
        scenario: _adapter_spec(path, expected, config["base_model"])
        for scenario, (path, expected) in sorted(week6_adapters.items())
    }
    locked_week6_hashes = config["evaluation"].get("week6_adapter_sha256")
    for scenario, spec in week6_specs.items():
        if locked_week6_hashes and spec["adapter_model_sha256"] != locked_week6_hashes[scenario]:
            raise Week7EvaluationError(f"Week 6 adapter does not match the preregistered hash: {scenario}")
    protocol_spec = selection.get("latency_protocol")
    if protocol_spec is None:
        evaluation_runtime = {
            "schema_version": "week7_final_legacy_nf4_runtime_v1",
            "latency_protocol": LATENCY_PROTOCOL_VERSION,
            "inference_precision": "nf4",
            "generation": {
                "max_new_tokens": int(max_new_tokens),
                "warmup_max_new_tokens": 1,
                "do_sample": False,
                "use_cache": True,
                "max_input_length": int(config["training"]["max_length"]),
                "structure_aware_truncation": True,
            },
            "timing": {"warmup_excluded": True},
            "metric_support_protocol": WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL,
        }
    else:
        if not isinstance(protocol_spec, dict):
            raise Week7EvaluationError("selected evaluation protocol is invalid")
        protocol_summary_path = Path(str(protocol_spec.get("path", ""))).resolve()
        protocol_summary = _read_json(protocol_summary_path)
        protocol_config_path = Path(
            str(protocol_summary.get("protocol_config_path", ""))
        ).resolve()
        protocol_config = _read_json(protocol_config_path)
        try:
            runtime_contract = _protocol_runtime(protocol_config)
        except Week7TrainingError as exc:
            raise Week7EvaluationError("unsupported selected evaluation runtime") from exc
        if int(max_new_tokens) != int(runtime_contract["generation"]["max_new_tokens"]):
            raise Week7EvaluationError(
                "parameter-lock max_new_tokens must match the selected evaluation protocol"
            )
        evaluation_runtime = {
            "protocol_summary_path": str(protocol_summary_path),
            "protocol_summary_sha256": sha256_file(protocol_summary_path),
            "protocol_config_path": str(protocol_config_path),
            "protocol_config_sha256": sha256_file(protocol_config_path),
            "schema_version": runtime_contract["schema_version"],
            "latency_protocol": runtime_contract["latency_protocol"],
            "inference_precision": runtime_contract["inference_precision"],
            "generation": runtime_contract["generation"],
            "timing": protocol_config.get("timing"),
            "metric_support_protocol": protocol_config.get("metric_support_protocol"),
        }
    if evaluation_runtime["metric_support_protocol"] != WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL:
        raise Week7EvaluationError(
            "selected evaluation protocol must use gold-evaluable metric support"
        )

    payload: dict[str, Any] = {
        "schema_version": "week7_parameter_lock_v1",
        "status": "LOCKED",
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "dataset_version": config["dataset"]["dataset_version"],
        "dataset_lock_sha256": dataset_lock["lock_sha256"],
        "base_model": config["base_model"],
        "selected_checkpoint": selected_spec["adapter_dir"],
        "selected_checkpoint_sha256": selected_spec["adapter_model_sha256"],
        "selected_checkpoint_config_sha256": selected_spec["adapter_config_sha256"],
        "week6_adapters": week6_specs,
        "development_evidence": evidence_payload,
        "generation": {
            "do_sample": False,
            "max_input_tokens": int(config["training"]["max_length"]),
            "max_new_tokens": int(max_new_tokens),
            "schema_decoding_mode": schema_decoding_mode,
        },
        "comparison_roles": ["week6_single_task_adapters", "multitask", "zero_shot"],
        "test_run_id": config["experiment_identity"]["test_run_id"],
        "training_summary": {
            "path": str(training_summary_path),
            "sha256": sha256_file(training_summary_path),
        },
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "evaluation_protocol": selection.get("latency_protocol"),
        "evaluation_runtime": evaluation_runtime,
    }
    payload["lock_sha256"] = canonical_sha256(payload)
    _write_json_new(Path(output_path), payload)
    return payload


def _validate_parameter_lock(root: Path, config_path: Path, parameter_lock_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = load_week7_config(config_path)
    payload = _read_json(parameter_lock_path)
    claimed_hash = payload.pop("lock_sha256", None)
    if claimed_hash != canonical_sha256(payload):
        raise Week7EvaluationError("parameter lock canonical SHA-256 mismatch")
    payload["lock_sha256"] = claimed_hash
    required = {
        "status", "config_sha256", "dataset_version", "dataset_lock_sha256",
        "base_model", "selected_checkpoint", "selected_checkpoint_sha256",
        "selected_checkpoint_config_sha256", "week6_adapters", "generation",
        "test_run_id", "training_summary", "selection", "development_evidence",
        "evaluation_protocol", "evaluation_runtime",
    }
    if payload.get("status") != "LOCKED" or not required <= set(payload):
        raise Week7EvaluationError("complete parameter lock is required before test")
    if payload["config_sha256"] != sha256_file(config_path):
        raise Week7EvaluationError("parameter lock config identity mismatch")
    if payload["test_run_id"] != config["experiment_identity"]["test_run_id"]:
        raise Week7EvaluationError("parameter lock test run ID mismatch")
    generation = payload["generation"]
    if (
        generation.get("do_sample") is not False
        or generation.get("schema_decoding_mode") != "free"
        or int(generation.get("max_input_tokens", 0)) != int(config["training"]["max_length"])
        or int(generation.get("max_new_tokens", 0)) <= 0
    ):
        raise Week7EvaluationError("parameter lock generation settings are incomplete")
    lock_root = root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    dataset_lock = _read_json(lock_root / "dataset_lock.json")
    dataset_claim = dataset_lock.pop("lock_sha256", None)
    if dataset_claim != canonical_sha256(dataset_lock):
        raise Week7EvaluationError("dataset lock canonical SHA-256 mismatch")
    dataset_lock["lock_sha256"] = dataset_claim
    if payload["dataset_lock_sha256"] != dataset_lock.get("lock_sha256"):
        raise Week7EvaluationError("parameter lock dataset identity mismatch")
    if payload.get("base_model") != config["base_model"]:
        raise Week7EvaluationError("parameter lock base-model identity mismatch")
    selected = _adapter_spec(
        Path(payload["selected_checkpoint"]),
        payload["selected_checkpoint_sha256"],
        config["base_model"],
    )
    if selected["adapter_config_sha256"] != payload.get("selected_checkpoint_config_sha256"):
        raise Week7EvaluationError("selected checkpoint config hash mismatch")
    if set(payload["week6_adapters"]) != set(CORE_SCENARIOS):
        raise Week7EvaluationError("parameter lock does not contain all Week 6 adapters")
    for spec in payload["week6_adapters"].values():
        actual = _adapter_spec(
            Path(spec["adapter_dir"]),
            spec["adapter_model_sha256"],
            config["base_model"],
        )
        if actual["adapter_config_sha256"] != spec.get("adapter_config_sha256"):
            raise Week7EvaluationError("Week 6 adapter config hash mismatch")
    training_summary = payload.get("training_summary", {})
    training_summary_path = Path(training_summary.get("path", "")).resolve()
    if sha256_file(training_summary_path) != training_summary.get("sha256"):
        raise Week7EvaluationError("training summary changed after parameter locking")
    summary_payload = _validate_training_summary(
        training_summary_path, config=config,
        config_sha256=payload["config_sha256"],
        dataset_lock_sha256=payload["dataset_lock_sha256"],
    )
    if set(payload.get("development_evidence", {})) != REQUIRED_DEVELOPMENT_EVIDENCE:
        raise Week7EvaluationError("parameter lock development evidence is incomplete")
    evidence_paths: dict[str, Path] = {}
    for name, evidence in payload["development_evidence"].items():
        evidence_path = Path(evidence.get("path", "")).resolve()
        if sha256_file(evidence_path) != evidence.get("sha256"):
            raise Week7EvaluationError(f"development evidence changed after locking: {name}")
        evidence_paths[name] = evidence_path

    selection_spec = payload.get("selection", {})
    selection_path = Path(selection_spec.get("path", "")).resolve()
    if sha256_file(selection_path) != selection_spec.get("sha256"):
        raise Week7EvaluationError("selection artifact changed after parameter locking")
    selection = _validate_selection_artifact(
        selection_path, config=config, config_path=Path(config_path).resolve(),
        config_sha256=payload["config_sha256"],
        dataset_lock_sha256=payload["dataset_lock_sha256"],
        training_summary_path=training_summary_path,
        training_summary_sha256=training_summary["sha256"],
        week6_baseline_path=evidence_paths["week6_development_baseline"],
        week6_baseline_sha256=payload["development_evidence"]
        ["week6_development_baseline"]["sha256"],
        selected_checkpoint=Path(payload["selected_checkpoint"]),
        selected_checkpoint_sha256=payload["selected_checkpoint_sha256"],
        selected_metrics_path=evidence_paths["multitask_development"],
        selected_metrics_sha256=payload["development_evidence"]
        ["multitask_development"]["sha256"],
    )
    if payload.get("evaluation_protocol") != selection.get("latency_protocol"):
        raise Week7EvaluationError("parameter lock evaluation-protocol binding mismatch")
    protocol_spec = selection.get("latency_protocol")
    if protocol_spec is None:
        runtime_contract = {
            "generation": {
                "max_new_tokens": int(generation["max_new_tokens"]),
                "warmup_max_new_tokens": 1,
                "do_sample": False,
                "use_cache": True,
                "max_input_length": int(config["training"]["max_length"]),
                "structure_aware_truncation": True,
            },
        }
        expected_runtime = {
            "schema_version": "week7_final_legacy_nf4_runtime_v1",
            "latency_protocol": LATENCY_PROTOCOL_VERSION,
            "inference_precision": "nf4",
            "generation": runtime_contract["generation"],
            "timing": {"warmup_excluded": True},
            "metric_support_protocol": WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL,
        }
    else:
        if not isinstance(protocol_spec, dict):
            raise Week7EvaluationError("parameter lock evaluation protocol is invalid")
        protocol_summary_path = Path(str(protocol_spec.get("path", ""))).resolve()
        protocol_summary = _read_json(protocol_summary_path)
        protocol_config_path = Path(
            str(protocol_summary.get("protocol_config_path", ""))
        ).resolve()
        protocol_config = _read_json(protocol_config_path)
        try:
            runtime_contract = _protocol_runtime(protocol_config)
        except Week7TrainingError as exc:
            raise Week7EvaluationError("unsupported locked evaluation runtime") from exc
        expected_runtime = {
            "protocol_summary_path": str(protocol_summary_path),
            "protocol_summary_sha256": sha256_file(protocol_summary_path),
            "protocol_config_path": str(protocol_config_path),
            "protocol_config_sha256": sha256_file(protocol_config_path),
            "schema_version": runtime_contract["schema_version"],
            "latency_protocol": runtime_contract["latency_protocol"],
            "inference_precision": runtime_contract["inference_precision"],
            "generation": runtime_contract["generation"],
            "timing": protocol_config.get("timing"),
            "metric_support_protocol": protocol_config.get("metric_support_protocol"),
        }
    if (
        payload.get("evaluation_runtime") != expected_runtime
        or expected_runtime["metric_support_protocol"]
        != WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL
        or int(generation["max_new_tokens"])
        != int(runtime_contract["generation"]["max_new_tokens"])
    ):
        raise Week7EvaluationError("parameter lock evaluation runtime mismatch")
    selected_step = int(selection["selected"].get("step", -1))
    selected_path = Path(payload["selected_checkpoint"]).resolve()
    if (
        selected_path.parent != training_summary_path.parent
        or selected_path.name != f"checkpoint-{selected_step}"
        or selected_path.name not in summary_payload["checkpoints"]
        or summary_payload["checkpoint_hashes"].get(selected_path.name)
        != payload["selected_checkpoint_sha256"]
        or selected_step not in [int(step) for step in summary_payload["evaluation_steps"]]
        or selected_step > int(summary_payload["global_step"])
    ):
        raise Week7EvaluationError("selection no longer matches the completed training summary")

    for name, evidence in payload["development_evidence"].items():
        evidence_path = evidence_paths[name]
        identity = _validate_development_evidence(
            name, _read_json(evidence_path), config=config,
            config_sha256=payload["config_sha256"],
            dataset_lock_sha256=payload["dataset_lock_sha256"],
            selected_checkpoint_sha256=payload["selected_checkpoint_sha256"],
            selected_step=selected_step,
            root=root,
        )
        if evidence.get("identity") != identity:
            raise Week7EvaluationError(f"development evidence lock identity mismatch: {name}")
    return payload, dataset_lock, lock_root


def _atomic_json_replace(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    os.replace(temporary, path)


def _new_owner() -> dict[str, Any]:
    return {
        "token": uuid.uuid4().hex,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started_unix": time.time(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def _lease_identity(
    run_id: str,
    parameter_lock_sha256: str,
    output_dir: Path,
    declared_test_sha256: str,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "parameter_lock_sha256": parameter_lock_sha256,
        "output_dir": str(output_dir.resolve()),
        "test_file_sha256": declared_test_sha256,
    }


def _acquire_lease(
    marker: Path,
    owner: dict[str, Any],
    identity: dict[str, str],
) -> Path:
    lease = marker.with_suffix(marker.suffix + ".lease")
    try:
        descriptor = os.open(lease, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        active = _read_json(lease).get("owner", {})
        raise Week7EvaluationError(
            "final-test gate has an active owner lease "
            f"({active.get('host')}:{active.get('pid')})"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "schema_version": "week7_final_test_lease_v1",
                "owner": owner,
                "identity": identity,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return lease


def _release_lease(lease: Path | None, owner: dict[str, Any] | None) -> None:
    if lease is None or owner is None or not lease.exists():
        return
    active = _read_json(lease).get("owner", {})
    if active.get("token") != owner.get("token"):
        raise Week7EvaluationError("refusing to release another final-test owner lease")
    lease.unlink()


def _snapshot_partial_raw_outputs(
    output_dir: Path,
    prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prior = prior or {}
    if not isinstance(prior, dict) or not set(prior) <= set(FINAL_ROLES):
        raise Week7EvaluationError("partial raw-output prefix manifest is invalid")
    result: dict[str, Any] = {}
    for role in FINAL_ROLES:
        path = output_dir / "raw_outputs" / f"{role}.jsonl"
        old = prior.get(role)
        if not path.exists():
            if old is not None:
                raise Week7EvaluationError(f"partial raw-output prefix disappeared: {role}")
            continue
        data = path.read_bytes()
        if old is not None:
            if set(old) != {"path", "sha256", "byte_count", "record_count"}:
                raise Week7EvaluationError(
                    f"partial raw-output prefix manifest is invalid: {role}"
                )
            old_bytes = int(old.get("byte_count", -1))
            old_count = int(old.get("record_count", -1))
            prefix_count = sum(
                1 for line in data[:old_bytes].splitlines() if line.strip()
            )
            if (
                old.get("path") != str(path.resolve())
                or old_bytes < 0
                or old_count < 0
                or len(data) < old_bytes
                or hashlib.sha256(data[:old_bytes]).hexdigest()
                != old.get("sha256")
                or prefix_count != old_count
            ):
                raise Week7EvaluationError(
                    f"partial raw-output append-only prefix mismatch: {role}"
                )
        count = sum(1 for line in data.splitlines() if line.strip())
        result[role] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_count": len(data),
            "record_count": count,
        }
    return result


def _validate_failed_partial_outputs(state: dict[str, Any], output_dir: Path) -> None:
    expected = state.get("partial_raw_outputs")
    if not isinstance(expected, dict):
        raise Week7EvaluationError("FAILED marker lacks partial raw-output evidence")
    actual = _snapshot_partial_raw_outputs(output_dir, expected)
    if actual != expected:
        raise Week7EvaluationError("FAILED partial raw outputs changed before resume")


def _mark_test_failed(
    marker: Path,
    owner: dict[str, Any],
    error: BaseException,
) -> None:
    current = _read_json(marker)
    if (
        current.get("status") != "IN_PROGRESS"
        or current.get("owner", {}).get("token") != owner.get("token")
    ):
        raise Week7EvaluationError("cannot fail a final-test marker owned by another process")
    failure = {
        "type": type(error).__name__,
        "message": str(error),
        "failed_unix": time.time(),
        "owner": owner,
    }
    output_dir = Path(str(current.get("output_dir", ""))).resolve()
    partial_raw_outputs = _snapshot_partial_raw_outputs(
        output_dir, current.get("resume_prefix")
    )
    failed = {
        **current,
        "status": "FAILED",
        "failure": failure,
        "failure_history": list(current.get("failure_history", [])) + [failure],
        "partial_raw_outputs": partial_raw_outputs,
    }
    _atomic_json_replace(marker, failed)


def _claim_test_run(
    marker: Path,
    *,
    run_id: str,
    parameter_lock_path: Path,
    parameter_lock_sha256: str,
    output_dir: Path,
    declared_test_sha256: str,
    resume: bool,
) -> tuple[dict[str, Any], bool, Path | None, dict[str, Any] | None]:
    marker.parent.mkdir(parents=True, exist_ok=True)
    expected_identity = (run_id, parameter_lock_sha256, str(output_dir.resolve()))
    lease_identity = _lease_identity(
        run_id, parameter_lock_sha256, output_dir, declared_test_sha256
    )
    if marker.exists():
        current = _read_json(marker)
        identity = (current.get("run_id"), current.get("parameter_lock_sha256"), current.get("output_dir"))
        if identity != expected_identity:
            raise Week7EvaluationError("Week 7 test allowance belongs to another run identity")
        if current.get("status") == "COMPLETED":
            return current, True, None, None
        if current.get("status") == "IN_PROGRESS":
            active = current.get("owner", {})
            raise Week7EvaluationError(
                "final-test run is IN_PROGRESS under active owner "
                f"{active.get('host')}:{active.get('pid')}; concurrent resume is forbidden"
            )
        if current.get("status") != "FAILED" or not resume:
            raise Week7EvaluationError("same-run recovery is allowed only from explicit FAILED state")
        _validate_failed_partial_outputs(current, output_dir)
        owner = _new_owner()
        lease = _acquire_lease(marker, owner, lease_identity)
        try:
            latest = _read_json(marker)
            latest_identity = (
                latest.get("run_id"), latest.get("parameter_lock_sha256"), latest.get("output_dir"),
            )
            if latest.get("status") != "FAILED" or latest_identity != expected_identity:
                raise Week7EvaluationError("final-test state changed while acquiring recovery lease")
            _validate_failed_partial_outputs(latest, output_dir)
            resumed = {
                **latest,
                "status": "IN_PROGRESS",
                "owner": owner,
                "resume_count": int(latest.get("resume_count", 0)) + 1,
                "resume_prefix": latest["partial_raw_outputs"],
            }
            _atomic_json_replace(marker, resumed)
            return resumed, False, lease, owner
        except BaseException:
            _release_lease(lease, owner)
            raise

    if resume:
        raise Week7EvaluationError("cannot resume before an explicit FAILED final-test run exists")
    owner = _new_owner()
    lease = _acquire_lease(marker, owner, lease_identity)
    initial = {
        "schema_version": "week7_test_consumption_v3",
        "status": "IN_PROGRESS",
        "run_id": run_id,
        "parameter_lock_path": str(parameter_lock_path.resolve()),
        "parameter_lock_sha256": parameter_lock_sha256,
        "output_dir": str(output_dir.resolve()),
        "test_file_sha256": declared_test_sha256,
        "owner": owner,
        "resume_count": 0,
        "failure_history": [],
    }
    try:
        with marker.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(initial, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except BaseException:
        _release_lease(lease, owner)
        raise
    return initial, False, lease, owner


def recover_interrupted_final_test(
    root: Path,
    config_path: Path,
    parameter_lock_path: Path,
    output_dir: Path,
    *,
    slurm_job_id: str,
    slurm_job_state: str,
) -> dict[str, Any]:
    """Explicitly fail an orphaned Slurm owner after audited job-state proof."""
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    parameter_lock_path = Path(parameter_lock_path).resolve()
    output_dir = Path(output_dir).resolve()
    parameter_lock, dataset_lock, lock_root = _validate_parameter_lock(
        root, config_path, parameter_lock_path
    )
    declared = dataset_lock.get("files", {}).get("test.jsonl", {})
    if not declared.get("sha256"):
        raise Week7EvaluationError("dataset lock does not bind the final test file")
    marker = (
        lock_root.parent.parent / "test_consumption"
        / f"{parameter_lock['dataset_version']}.json"
    )
    if not marker.is_file():
        raise Week7EvaluationError("no final-test marker exists to recover")
    current = _read_json(marker)
    expected_identity = _lease_identity(
        parameter_lock["test_run_id"], sha256_file(parameter_lock_path),
        output_dir, declared["sha256"],
    )
    marker_identity = {
        key: current.get(key) for key in expected_identity
    }
    if (
        marker_identity != expected_identity
        or current.get("parameter_lock_path") != str(parameter_lock_path)
    ):
        raise Week7EvaluationError("interrupted final-test marker identity mismatch")
    if current.get("status") not in {"IN_PROGRESS", "FAILED"}:
        raise Week7EvaluationError(
            "only an IN_PROGRESS or already FAILED final-test marker can be recovered"
        )

    owner = current.get("owner")
    if not isinstance(owner, dict) or not owner.get("token"):
        raise Week7EvaluationError("interrupted final-test owner identity is incomplete")
    job_id = str(slurm_job_id).strip()
    if not job_id or str(owner.get("slurm_job_id") or "") != job_id:
        raise Week7EvaluationError("Slurm job does not own the interrupted final-test marker")
    state = str(slurm_job_state).strip().upper().split("+")[0]
    terminal_states = {
        "BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL",
        "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT",
    }
    if state == "SIGNAL_TERM":
        if os.environ.get("SLURM_JOB_ID") != job_id:
            raise Week7EvaluationError("SIGNAL_TERM recovery requires the current Slurm owner")
    elif state not in terminal_states:
        raise Week7EvaluationError("Slurm job state is not terminal; active lease is preserved")

    if current.get("status") == "FAILED":
        _validate_failed_partial_outputs(current, output_dir)
        return current

    lease = marker.with_suffix(marker.suffix + ".lease")
    if not lease.is_file():
        raise Week7EvaluationError("interrupted final-test owner lease is missing")
    lease_payload = _read_json(lease)
    if (
        lease_payload.get("identity") != expected_identity
        or lease_payload.get("owner") != owner
    ):
        raise Week7EvaluationError("interrupted final-test lease identity mismatch")
    partial_raw_outputs = _snapshot_partial_raw_outputs(
        output_dir, current.get("resume_prefix")
    )
    failure = {
        "type": "ExternalJobTermination",
        "message": f"Slurm job {job_id} ended with {state}",
        "failed_unix": time.time(),
        "owner": owner,
        "slurm_job_id": job_id,
        "slurm_job_state": state,
        "lease_sha256": sha256_file(lease),
    }
    failed = {
        **current,
        "status": "FAILED",
        "failure": failure,
        "failure_history": list(current.get("failure_history", [])) + [failure],
        "partial_raw_outputs": partial_raw_outputs,
    }
    _atomic_json_replace(marker, failed)
    _release_lease(lease, owner)
    return failed


def _row_task(row: dict[str, Any]) -> str:
    scenario = row.get("scenario")
    if scenario in CORE_SCENARIOS:
        return str(scenario)
    target = row.get("target", {})
    if isinstance(target, dict) and isinstance(target.get("task_result"), dict):
        target = target["task_result"]
    if "business_category" in target:
        return "image_product_search"
    if "issue_type" in target:
        return "after_sales"
    if "itinerary" in target or "hard_constraints" in target:
        return "itinerary_planning"
    raise Week7EvaluationError(f"cannot route dialogue to a Week 6 adapter: {row.get('sample_id')}")


def _transformers_runner(
    root: Path,
    config: dict[str, Any],
    role: str,
    rows: list[dict[str, Any]],
    adapter_dir: Path | None,
    max_new_tokens: int,
    record_sink: Callable[[dict[str, Any]], None],
    evaluation_runtime: dict[str, Any],
    run_id: str,
) -> list[dict[str, Any]]:
    if not rows:
        raise Week7EvaluationError("final-test model route cannot be empty")
    generation = evaluation_runtime["generation"]
    if int(max_new_tokens) != int(generation["max_new_tokens"]):
        raise Week7EvaluationError("final-test generation does not match locked runtime")
    processor, model_loader = _default_model_loader(
        config,
        {"inference_precision": evaluation_runtime["inference_precision"]},
    )
    model = model_loader(adapter_dir)

    def messages(row: dict[str, Any]) -> list[dict[str, Any]]:
        return structure_aware_messages(
            processor,
            training_messages(row),
            int(generation["max_input_length"]),
        )[:-1]

    runtime_options = {
        "latency_protocol": evaluation_runtime["latency_protocol"],
        "cache_implementation": generation.get("cache_implementation"),
        "compile_config": generation.get("compile_config"),
    }
    records: list[dict[str, Any]] = []
    with inference_runtime(model):
        warmup = generate_record(
            model,
            processor,
            messages(rows[0]),
            sample_id=rows[0]["sample_id"],
            run_id=run_id,
            model_name=role,
            max_new_tokens=int(generation["warmup_max_new_tokens"]),
            warmup=True,
            **runtime_options,
        )
        if warmup.get("failed"):
            raise Week7EvaluationError(f"final-test warmup failed for {role}")
        for row in rows:
            record = generate_record(
                model,
                processor,
                messages(row),
                sample_id=row["sample_id"],
                run_id=run_id,
                model_name=role,
                max_new_tokens=max_new_tokens,
                **runtime_options,
            )
            record_sink(record)
            records.append(record)
    del model
    _release_model()
    return records


Runner = Callable[
    [str, list[dict[str, Any]], Path | None, Callable[[dict[str, Any]], None]],
    list[dict[str, Any]],
]


def _load_partial_records(
    path: Path,
    role: str,
    run_id: str,
    row_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        sample_id = record.get("sample_id")
        if (
            record.get("run_role") != role
            or record.get("run_id") != run_id
            or sample_id not in row_ids
            or sample_id in records
        ):
            raise Week7EvaluationError(f"invalid resumable raw-output identity for {role}")
        records[str(sample_id)] = record
    return records


def _run_role_resumable(
    role: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    run_id: str,
    runner: Runner,
    adapter_groups: Iterable[tuple[Path | None, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    row_ids = {row["sample_id"] for row in rows}
    records = _load_partial_records(output_path, role, run_id, row_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        def persist(record: dict[str, Any]) -> None:
            sample_id = record.get("sample_id")
            if sample_id not in row_ids or sample_id in records:
                raise Week7EvaluationError(f"invalid or duplicate generated identity for {role}")
            stored = {**record, "run_id": run_id, "run_role": role}
            handle.write(json.dumps(stored, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            records[str(sample_id)] = stored

        for adapter_dir, group in adapter_groups:
            missing = [row for row in group if row["sample_id"] not in records]
            if not missing:
                continue
            expected_ids = {row["sample_id"] for row in missing}
            generated = runner(role, missing, adapter_dir, persist)
            if len(generated) != len(missing) or {item.get("sample_id") for item in generated} != {row["sample_id"] for row in missing}:
                raise Week7EvaluationError(f"inference runner did not exactly cover {role} rows")
            for record in generated:
                if record["sample_id"] not in records:
                    persist(record)
            if not expected_ids <= set(records):
                raise Week7EvaluationError(f"inference runner did not persist all {role} rows")
    if set(records) != row_ids:
        raise Week7EvaluationError(f"{role} final output coverage is incomplete")
    return [records[row["sample_id"]] for row in rows]


def _relative_change(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline


def build_final_comparison(config: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = summaries["multitask"]
    baseline = summaries["week6_single_task_adapters"]
    zero = summaries["zero_shot"]
    non_regression = config["evaluation"]["non_regression"]
    comparisons: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for baseline_name, reference in (("week6", baseline), ("zero_shot", zero)):
        scenario_changes: dict[str, Any] = {}
        for scenario in CORE_SCENARIOS:
            current = candidate["scenarios"][scenario]
            prior = reference["scenarios"][scenario]
            absolute = float(current["composite"]) - float(prior["composite"])
            relative = _relative_change(float(current["composite"]), float(prior["composite"]))
            support_ratios = {
                metric: (current["metric_support"][metric] / support if support else None)
                for metric, support in prior["metric_support"].items()
            }
            aggregate_current, aggregate_prior = current["aggregate"], prior["aggregate"]
            json_delta = float(aggregate_current["json_compliance"]) - float(aggregate_prior["json_compliance"])
            schema_delta = float(aggregate_current["schema_pass"]) - float(aggregate_prior["schema_pass"])
            task_gate = relative is None and absolute >= 0 or relative is not None and relative >= -float(non_regression["max_relative_task_drop"])
            support_gate = all(value is None or value >= float(non_regression["minimum_support_ratio"]) for value in support_ratios.values())
            format_gate = json_delta >= float(non_regression["json_schema_absolute_drop"]) and schema_delta >= float(non_regression["json_schema_absolute_drop"])
            if baseline_name == "week6":
                gates[scenario] = bool(task_gate and support_gate and format_gate)
            scenario_changes[scenario] = {
                "composite_absolute_change": absolute,
                "composite_relative_change": relative,
                "metric_support_ratios": support_ratios,
                "json_compliance_absolute_change": json_delta,
                "schema_pass_absolute_change": schema_delta,
                "gate": bool(task_gate and support_gate and format_gate),
            }
        comparisons[f"multitask_vs_{baseline_name}"] = scenario_changes

    latency_gate = candidate["latency_ms_mean"] <= baseline["latency_ms_mean"] * float(non_regression["max_latency_ratio"])
    failure_gate = candidate["failure_rate"] <= float(non_regression["max_failure_rate"]) and candidate["failure_rate"] <= baseline["failure_rate"] + float(non_regression["max_failure_rate_absolute_increase"])
    gates.update({"latency": latency_gate, "failure": failure_gate})
    dialogue = {
        role: {
            "format_compliance": summary["dialogue"]["format_compliance"],
            "context_recall": summary["dialogue"]["context_recall"],
            "sample_count": summary["dialogue"]["sample_count"],
            "human_dimensions_status": summary["dialogue"]["human_dimensions_status"],
        }
        for role, summary in summaries.items()
    }
    dialogue_changes = {
        f"multitask_vs_{name}": {
            metric: dialogue["multitask"][metric] - dialogue[role][metric]
            for metric in ("format_compliance", "context_recall")
        }
        for name, role in (
            ("week6", "week6_single_task_adapters"),
            ("zero_shot", "zero_shot"),
        )
    }
    return {
        "status": "COMPLETED",
        "comparisons": comparisons,
        "dialogue": dialogue,
        "dialogue_absolute_changes": dialogue_changes,
        "operational": {
            role: {
                "latency_ms_mean": summary["latency_ms_mean"],
                "latency_ms_median": summary["latency_ms_median"],
                "failure_count": summary["failure_count"],
                "failure_rate": summary["failure_rate"],
                "sample_count": summary["sample_count"],
            }
            for role, summary in summaries.items()
        },
        "non_regression_gates": {**gates, "all_passed": all(gates.values())},
        "models": summaries,
    }


def _final_artifact_paths(output_dir: Path) -> dict[str, Path]:
    paths = {
        f"raw_outputs/{role}.jsonl": output_dir / "raw_outputs" / f"{role}.jsonl"
        for role in FINAL_ROLES
    }
    paths.update({
        f"metrics/{role}.json": output_dir / "metrics" / f"{role}.json"
        for role in FINAL_ROLES
    })
    paths["final_comparison.json"] = output_dir / "final_comparison.json"
    return paths


def _hash_final_artifacts(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in _final_artifact_paths(output_dir).items():
        if not path.is_file():
            raise Week7EvaluationError(f"completed final-test artifact is missing: {name}")
        hashes[name] = sha256_file(path)
    return hashes


def _verify_completed_artifacts(state: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    expected = state.get("artifact_hashes")
    if not isinstance(expected, dict) or set(expected) != set(_final_artifact_paths(output_dir)):
        raise Week7EvaluationError("completed marker artifact manifest is incomplete")
    actual = _hash_final_artifacts(output_dir)
    if actual != expected:
        raise Week7EvaluationError("completed final-test artifact hash mismatch")
    summary_path = output_dir / "final_comparison.json"
    if state.get("summary_path") != str(summary_path) or state.get("summary_sha256") != actual["final_comparison.json"]:
        raise Week7EvaluationError("completed final-test summary identity mismatch")
    return _read_json(summary_path)


def _execute_final_test_suite(
    root: Path,
    config_path: Path,
    parameter_lock_path: Path,
    parameter_lock: dict[str, Any],
    dataset_lock: dict[str, Any],
    lock_root: Path,
    output_dir: Path,
    inference_runner: Runner | None,
    *,
    resume: bool,
) -> dict[str, Any]:
    declared = dataset_lock["files"]["test.jsonl"]
    output_dir.mkdir(parents=True, exist_ok=resume)
    test_path = lock_root / "test.jsonl"
    if sha256_file(test_path) != declared["sha256"]:
        raise Week7EvaluationError("final test file no longer matches the dataset lock")
    rows = list(iter_jsonl(test_path))
    if len(rows) != int(declared["count"]) or any(row.get("split") != "test" for row in rows):
        raise Week7EvaluationError("final test rows do not match the locked count/split")

    config = load_week7_config(config_path)
    max_new_tokens = int(parameter_lock["generation"]["max_new_tokens"])
    evaluation_runtime = parameter_lock["evaluation_runtime"]
    if inference_runner is None:
        def inference_runner(
            role: str,
            selected_rows: list[dict[str, Any]],
            adapter: Path | None,
            record_sink: Callable[[dict[str, Any]], None],
        ) -> list[dict[str, Any]]:
            return _transformers_runner(
                root,
                config,
                role,
                selected_rows,
                adapter,
                max_new_tokens,
                record_sink,
                evaluation_runtime,
                parameter_lock["test_run_id"],
            )

    roles: dict[str, list[tuple[Path | None, list[dict[str, Any]]]]] = {
        "zero_shot": [(None, rows)],
        "multitask": [(Path(parameter_lock["selected_checkpoint"]), rows)],
        "week6_single_task_adapters": [],
    }
    routed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        routed[_row_task(row)].append(row)
    for scenario in CORE_SCENARIOS:
        roles["week6_single_task_adapters"].append((
            Path(parameter_lock["week6_adapters"][scenario]["adapter_dir"]),
            routed[scenario],
        ))

    summaries: dict[str, dict[str, Any]] = {}
    run_id = parameter_lock["test_run_id"]
    for role in FINAL_ROLES:
        records = _run_role_resumable(
            role, rows, output_dir / "raw_outputs" / f"{role}.jsonl", run_id,
            inference_runner, roles[role],
        )
        summaries[role] = summarize_raw_records(
            root,
            config,
            rows,
            records,
            metric_support_protocol=evaluation_runtime["metric_support_protocol"],
        )
        metrics_path = output_dir / "metrics" / f"{role}.json"
        if not metrics_path.exists():
            _write_json_new(metrics_path, summaries[role])
        elif canonical_sha256(_read_json(metrics_path)) != canonical_sha256(summaries[role]):
            raise Week7EvaluationError(f"resumed metrics mismatch for {role}")

    comparison = build_final_comparison(config, summaries)
    comparison.update({
        "run_id": run_id,
        "split": "test",
        "parameter_lock_sha256": sha256_file(parameter_lock_path),
        "dataset_lock_sha256": parameter_lock["dataset_lock_sha256"],
        "test_file_sha256": declared["sha256"],
        "model_identities": {
            "zero_shot": {"base_model": parameter_lock["base_model"], "adapter": None},
            "multitask": {
                "base_model": parameter_lock["base_model"],
                "adapter_model_sha256": parameter_lock["selected_checkpoint_sha256"],
            },
            "week6_single_task_adapters": parameter_lock["week6_adapters"],
        },
        "evaluation_runtime": evaluation_runtime,
    })
    summary_path = output_dir / "final_comparison.json"
    if not summary_path.exists():
        _write_json_new(summary_path, comparison)
    elif canonical_sha256(_read_json(summary_path)) != canonical_sha256(comparison):
        raise Week7EvaluationError("resumed final comparison mismatch")
    return comparison


def run_final_test_suite(
    root: Path,
    config_path: Path,
    parameter_lock_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    inference_runner: Runner | None = None,
) -> dict[str, Any]:
    """Consume the unseen test once; only the same exact run may resume."""
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    parameter_lock_path = Path(parameter_lock_path).resolve()
    output_dir = Path(output_dir).resolve()
    parameter_lock, dataset_lock, lock_root = _validate_parameter_lock(root, config_path, parameter_lock_path)
    run_id = parameter_lock["test_run_id"]
    if output_dir.exists() and not resume:
        raise Week7EvaluationError("final-test output directory already exists")
    declared = dataset_lock.get("files", {}).get("test.jsonl", {})
    if not declared.get("sha256") or not declared.get("count"):
        raise Week7EvaluationError("dataset lock does not bind the final test file")
    marker = lock_root.parent.parent / "test_consumption" / f"{parameter_lock['dataset_version']}.json"
    state, already_completed, lease, owner = _claim_test_run(
        marker,
        run_id=run_id,
        parameter_lock_path=parameter_lock_path,
        parameter_lock_sha256=sha256_file(parameter_lock_path),
        output_dir=output_dir,
        declared_test_sha256=declared["sha256"],
        resume=resume,
    )
    if already_completed:
        return _verify_completed_artifacts(state, output_dir)

    if owner is None:
        raise Week7EvaluationError("claimed final-test run is missing owner identity")
    try:
        comparison = _execute_final_test_suite(
            root, config_path, parameter_lock_path, parameter_lock, dataset_lock,
            lock_root, output_dir, inference_runner, resume=resume,
        )
        artifact_hashes = _hash_final_artifacts(output_dir)
        summary_path = output_dir / "final_comparison.json"
        completed = {
            **state,
            "status": "COMPLETED",
            "completed_unix": time.time(),
            "summary_path": str(summary_path),
            "summary_sha256": artifact_hashes["final_comparison.json"],
            "artifact_hashes": artifact_hashes,
        }
        _atomic_json_replace(marker, completed)
        return comparison
    except BaseException as exc:
        _mark_test_failed(marker, owner, exc)
        raise
    finally:
        _release_lease(lease, owner)

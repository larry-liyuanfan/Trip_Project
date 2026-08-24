"""Run the corrected Week 7 v4 dialogue test exactly once."""

from __future__ import annotations

import gc
import json
import math
import os
import socket
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.training.week6_qlora import environment_report
from src.training.week7_data import (
    ALIGNED_DIALOGUE_CONSTRUCTION_VERSIONS,
    CORE_SCENARIOS,
    _validate_aligned_dialogue,
    canonical_sha256,
    iter_jsonl,
    load_week7_config,
    sha256_file,
    write_jsonl_new,
)
from src.training.week7_evaluation import (
    Week7EvaluationError,
    evaluate_dialogue_automatic_gate,
    summarize_dialogue_raw_records,
    valid_check_constraints_tool_call,
)
from src.training.week7_latency_protocol import (
    _default_model_loader,
    _dialogue_task,
    _release_model,
)
from src.training.week7_dialogue_v4 import (
    _automatic_gate,
    _validate_and_recompute_metrics,
    _validate_metrics_identity,
)
from src.training.week7_qlora import (
    Week7TrainingError,
    assistant_content_text,
    structure_aware_messages,
    training_messages,
)
from src.training.week7_runtime import (
    LATENCY_PROTOCOL_V5_VERSION,
    generate_record,
    inference_runtime,
)


TEST_SCHEMA_VERSION = "week7_corrected_dialogue_test_v4"
CONSUMPTION_SCHEMA_VERSION = "week7_corrected_dialogue_test_consumption_v4"
ROLE_ORDER = ("multitask", "week6_routed", "zero_shot")
AUTOMATIC_DIMENSIONS = (
    "historical_image_reference",
    "requirement_update",
    "context_carryover",
    "logical_consistency",
)
ModelLoader = Callable[[Path | None], Any]
RecordGenerator = Callable[
    [Any, Any, list[dict[str, Any]], str, str, int],
    tuple[list[dict[str, Any]], dict[str, Any]],
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week7TrainingError(f"invalid v4 dialogue-test JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Week7TrainingError(f"v4 dialogue-test JSON must be an object: {path}")
    return value


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json_replace(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise Week7TrainingError(f"stale marker temporary file exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _probe_output_parent(output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    probe = output_dir.parent / f".{output_dir.name}.v4-test-probe-{os.getpid()}"
    try:
        with probe.open("x", encoding="utf-8") as handle:
            handle.write("week7-v4-dialogue-test\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise Week7TrainingError(
            f"v4 dialogue-test output parent is not writable: {output_dir.parent}"
        ) from exc
    finally:
        probe.unlink(missing_ok=True)


def _selection_checkpoint(selection: dict[str, Any]) -> tuple[dict[str, Any], Path, str]:
    selected = selection.get("selected_checkpoint")
    required = {
        "path",
        "adapter_sha256",
        "step",
        "metrics_path",
        "metrics_sha256",
        "raw_outputs_path",
        "raw_outputs_sha256",
        "automatic_gate",
    }
    if not isinstance(selected, dict) or not required.issubset(selected):
        raise Week7TrainingError("v4 selection has no complete selected checkpoint identity")
    if (
        not isinstance(selected["automatic_gate"], dict)
        or selected["automatic_gate"].get("passed") is not True
    ):
        raise Week7TrainingError("selected v4 checkpoint did not pass its automatic gate")
    return selected, Path(str(selected["path"])).resolve(), str(selected["adapter_sha256"])


def _replay_selection_candidates(
    root: Path,
    config: dict[str, Any],
    config_hash: str,
    dataset_lock_hash: str,
    selection: dict[str, Any],
    training_dir: Path,
    training_summary: dict[str, Any],
    development_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = selection.get("candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(candidates) != int(selection.get("candidate_count", -1))
    ):
        raise Week7TrainingError("v4 selection candidate coverage is incomplete")
    expected_samples = len(development_rows)
    expected_dialogue = int(config["dataset"]["development_dialogue_count"])
    try:
        planned_steps = [int(step) for step in training_summary["evaluation_steps"]]
        global_step = int(training_summary["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Week7TrainingError("v4 training evaluation-step identity is invalid") from exc
    completed_steps = {step for step in planned_steps if step <= global_step}
    expected_checkpoint_names = {f"checkpoint-{step}" for step in completed_steps}
    if (
        planned_steps != sorted(set(planned_steps))
        or not completed_steps
        or set(training_summary.get("checkpoints", [])) != expected_checkpoint_names
        or set(training_summary.get("checkpoint_hashes", {})) != expected_checkpoint_names
        or set(training_summary.get("development_evaluation_artifacts", {}))
        != {str(step) for step in completed_steps}
    ):
        raise Week7TrainingError("v4 completed checkpoint evidence is incomplete")
    replayed: list[dict[str, Any]] = []
    seen_steps = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise Week7TrainingError("v4 selection candidate must be an object")
        try:
            step = int(candidate["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Week7TrainingError("v4 selection candidate step is invalid") from exc
        if step in seen_steps:
            raise Week7TrainingError("v4 selection contains duplicate candidate steps")
        seen_steps.add(step)
        checkpoint_name = f"checkpoint-{step}"
        checkpoint = Path(str(candidate.get("checkpoint") or "")).resolve()
        metrics_path = Path(str(candidate.get("metrics_path") or "")).resolve()
        raw_path = Path(str(candidate.get("raw_outputs_path") or "")).resolve()
        expected_checkpoint = (training_dir / checkpoint_name).resolve()
        expected_metrics = (
            training_dir
            / "development_evaluations"
            / f"step-{step:06d}"
            / "metrics.json"
        ).resolve()
        expected_raw = expected_metrics.parent / "raw_outputs.jsonl"
        adapter_model = checkpoint / "adapter_model.safetensors"
        adapter_hash = str(candidate.get("checkpoint_adapter_sha256") or "")
        if (
            checkpoint != expected_checkpoint
            or metrics_path != expected_metrics
            or raw_path != expected_raw
            or not adapter_model.is_file()
            or not metrics_path.is_file()
            or not raw_path.is_file()
            or sha256_file(adapter_model) != adapter_hash
            or sha256_file(metrics_path) != candidate.get("metrics_sha256")
            or sha256_file(raw_path) != candidate.get("raw_outputs_sha256")
            or checkpoint_name not in training_summary.get("checkpoints", [])
            or training_summary.get("checkpoint_hashes", {}).get(checkpoint_name)
            != adapter_hash
        ):
            raise Week7TrainingError(f"v4 candidate artifact identity mismatch: step {step}")
        expected_binding = {
            "raw_outputs_path": str(raw_path),
            "raw_outputs_sha256": candidate["raw_outputs_sha256"],
            "metrics_path": str(metrics_path),
            "metrics_sha256": candidate["metrics_sha256"],
        }
        if training_summary.get("development_evaluation_artifacts", {}).get(str(step)) != expected_binding:
            raise Week7TrainingError(
                f"v4 candidate is not bound by training summary: step {step}"
            )
        metrics = _read_json(metrics_path)
        _validate_metrics_identity(
            metrics,
            step=step,
            run_id=str(training_summary["run_id"]),
            config_hash=config_hash,
            dataset_lock_hash=dataset_lock_hash,
            expected_samples=expected_samples,
            expected_dialogue_samples=expected_dialogue,
        )
        recomputed = _validate_and_recompute_metrics(
            root, config, development_rows, metrics_path, metrics,
        )
        replayed_gate = _automatic_gate(config, recomputed)
        replayed_candidate = {
            "step": step,
            "checkpoint": str(checkpoint),
            "checkpoint_adapter_sha256": adapter_hash,
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "raw_outputs_path": str(raw_path),
            "raw_outputs_sha256": sha256_file(raw_path),
            "weighted_composite": float(recomputed["weighted_composite"]),
            "core_weighted_composite": float(recomputed["core_weighted_composite"]),
            "automatic_gate": replayed_gate,
            "eligible": bool(replayed_gate["passed"]),
        }
        if canonical_sha256(candidate) != canonical_sha256(replayed_candidate):
            raise Week7TrainingError(
                f"v4 candidate differs from replayed evidence: step {step}"
            )
        replayed.append(replayed_candidate)
    if seen_steps != completed_steps:
        raise Week7TrainingError("v4 selection does not cover every completed checkpoint")
    eligible = [candidate for candidate in replayed if candidate["eligible"]]
    if len(eligible) != int(selection.get("eligible_count", -1)):
        raise Week7TrainingError("v4 selection eligible count changed during replay")
    best = max(
        eligible,
        key=lambda candidate: (candidate["weighted_composite"], -candidate["step"]),
        default=None,
    )
    if best is None:
        raise Week7TrainingError("v4 selection replay has no eligible checkpoint")
    return best


def _validate_inputs(
    root: Path,
    config_path: Path,
    selection_path: Path,
    week6_adapters: dict[str, Path],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any], Path, str]:
    config = load_week7_config(config_path)
    if config.get("schema_version") != "week7_multitask_context_v4":
        raise Week7TrainingError("corrected dialogue test requires the v4 config")
    automatic_gate = config.get("evaluation", {}).get("dialogue_automatic_gate", {})
    if (
        automatic_gate.get("enabled") is not True
        or automatic_gate.get("human_input_required") is not False
        or int(config["evaluation"].get("human_review_queue_size", -1)) != 0
    ):
        raise Week7TrainingError("v4 corrected dialogue test must be automatic-only")
    config_hash = sha256_file(config_path)
    lock_root = (
        root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    ).resolve()
    lock_path = lock_root / "dataset_lock.json"
    lock = _read_json(lock_path)
    recorded_lock_hash = str(lock.get("lock_sha256") or "")
    lock_body = dict(lock)
    lock_body.pop("lock_sha256", None)
    if (
        lock.get("schema_version") != "week7_dataset_lock_v4"
        or lock.get("dataset_version") != config["dataset"]["dataset_version"]
        or lock.get("config_sha256") != config_hash
        or not recorded_lock_hash
        or canonical_sha256(lock_body) != recorded_lock_hash
        or lock.get("test_policy") != {
            "status": "LOCKED_UNCONSUMED",
            "may_read_only_after_parameter_lock": True,
            "maximum_evaluations": 1,
        }
    ):
        raise Week7TrainingError("v4 corrected dialogue dataset lock identity mismatch")
    declared_test = lock.get("files", {}).get("test/dialogue.jsonl")
    if not isinstance(declared_test, dict) or not declared_test.get("sha256"):
        raise Week7TrainingError("v4 lock does not bind test/dialogue.jsonl")
    historical_exclusion = lock.get("historical_v3_test_exclusion")
    if (
        not isinstance(historical_exclusion, dict)
        or historical_exclusion.get("status") != "PASS"
    ):
        raise Week7TrainingError(
            "v4 test is not proven disjoint from historical v3 test identities"
        )

    selection = _read_json(selection_path)
    selection_identity = str(selection.get("selection_sha256") or "")
    selection_body = dict(selection)
    selection_body.pop("selection_sha256", None)
    if (
        selection.get("status") != "PASS"
        or selection.get("eligible") is not True
        or selection.get("config_sha256") != config_hash
        or selection.get("dataset_lock_sha256") != recorded_lock_hash
        or not selection_identity
        or canonical_sha256(selection_body) != selection_identity
    ):
        raise Week7TrainingError("v4 selection did not pass the automatic development gate")
    selected, checkpoint, expected_checkpoint_hash = _selection_checkpoint(selection)
    training_dir = Path(str(selection.get("training_dir") or "")).resolve()
    training_summary_path = Path(
        str(selection.get("training_summary_path") or "")
    ).resolve()
    if training_summary_path != training_dir / "run_summary.json":
        raise Week7TrainingError("selection training summary path is not bound to training_dir")
    if (
        not training_summary_path.is_file()
        or sha256_file(training_summary_path) != selection.get("training_summary_sha256")
    ):
        raise Week7TrainingError("selection training summary hash mismatch")
    training_summary = _read_json(training_summary_path)
    development_path = Path(str(selection.get("development_path") or "")).resolve()
    expected_development_path = lock_root / "development.jsonl"
    if (
        training_summary.get("status") != "COMPLETED"
        or training_summary.get("run_id") != selection.get("training_run_id")
        or training_summary.get("run_id")
        != config["experiment_identity"]["multitask_sft_run_id"]
        or training_summary.get("config_sha256") != config_hash
        or training_summary.get("dataset_lock_sha256") != recorded_lock_hash
        or development_path != expected_development_path
        or not development_path.is_file()
        or sha256_file(development_path) != selection.get("development_sha256")
    ):
        raise Week7TrainingError("v4 selection is not bound to completed training evidence")
    development_rows = list(iter_jsonl(development_path))
    if (
        len(development_rows) != int(selection.get("development_samples", -1))
        or len(development_rows) != int(training_summary.get("development_samples", -1))
    ):
        raise Week7TrainingError("v4 development support differs from selection/training")
    best = _replay_selection_candidates(
        root,
        config,
        config_hash,
        recorded_lock_hash,
        selection,
        training_dir,
        training_summary,
        development_rows,
    )
    selected_identity = {
        "path": best["checkpoint"],
        "adapter_sha256": best["checkpoint_adapter_sha256"],
        "step": best["step"],
        "metrics_path": best["metrics_path"],
        "metrics_sha256": best["metrics_sha256"],
        "raw_outputs_path": best["raw_outputs_path"],
        "raw_outputs_sha256": best["raw_outputs_sha256"],
        "weighted_composite": best["weighted_composite"],
        "core_weighted_composite": best["core_weighted_composite"],
        "automatic_gate": best["automatic_gate"],
    }
    if canonical_sha256(selected) != canonical_sha256(selected_identity):
        raise Week7TrainingError("selected checkpoint is not the replayed best candidate")
    adapter_model = checkpoint / "adapter_model.safetensors"
    if not adapter_model.is_file() or sha256_file(adapter_model) != expected_checkpoint_hash:
        raise Week7TrainingError("selected v4 checkpoint hash mismatch")

    if set(week6_adapters) != set(CORE_SCENARIOS):
        raise Week7TrainingError("corrected dialogue test requires exactly three Week 6 adapters")
    configured_hashes = config["evaluation"]["week6_adapter_sha256"]
    for scenario, adapter_dir in week6_adapters.items():
        adapter_model = adapter_dir / "adapter_model.safetensors"
        if (
            not adapter_model.is_file()
            or sha256_file(adapter_model) != configured_hashes[scenario]
        ):
            raise Week7TrainingError(f"Week 6 adapter hash mismatch: {scenario}")
    if output_dir.exists():
        raise Week7TrainingError("refusing to overwrite corrected dialogue test evidence")
    _probe_output_parent(output_dir)
    return config, lock, lock_root, selection, checkpoint, expected_checkpoint_hash


def _claim_test(
    marker: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    lock: dict[str, Any],
    selection_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    marker.parent.mkdir(parents=True, exist_ok=True)
    claim = {
        "schema_version": CONSUMPTION_SCHEMA_VERSION,
        "status": "CLAIMED",
        "dataset_version": lock["dataset_version"],
        "run_id": config["experiment_identity"]["test_run_id"],
        "config_sha256": sha256_file(config_path),
        "dataset_lock_sha256": lock["lock_sha256"],
        "declared_dialogue_test_sha256": lock["files"]["test/dialogue.jsonl"]["sha256"],
        "selection_sha256": sha256_file(selection_path),
        "output_dir": str(output_dir),
        "owner": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "claimed_at": _utc_now(),
        "completed_at": None,
        "failure": None,
    }
    try:
        _write_json_new(marker, claim)
    except FileExistsError as exc:
        raise Week7TrainingError(
            f"v4 corrected dialogue test was already consumed: {marker}"
        ) from exc
    return claim


def _load_test_dialogues(
    root: Path,
    lock_root: Path,
    lock: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    test_path = lock_root / "test" / "dialogue.jsonl"
    declared = lock["files"]["test/dialogue.jsonl"]
    if sha256_file(test_path) != declared["sha256"]:
        raise Week7TrainingError("v4 dialogue test file hash changed after consumption claim")
    rows = list(iter_jsonl(test_path))
    if len(rows) != int(declared["count"]) or any(
        row.get("split") != "test" or row.get("scenario") != "dialogue" for row in rows
    ):
        raise Week7TrainingError("v4 dialogue test rows do not match the locked count/split")
    expected_construction = config["sampling"].get(
        "dialogue_construction_version", "aligned_concrete_turns_v4"
    )
    if expected_construction not in ALIGNED_DIALOGUE_CONSTRUCTION_VERSIONS:
        raise Week7TrainingError("unsupported corrected dialogue construction")
    for row in rows:
        if row.get("construction_version") != expected_construction:
            raise Week7TrainingError("legacy dialogue construction entered the v4 test")
        _validate_aligned_dialogue(row)
        image_path = root / str(row["image_path"])
        if sha256_file(image_path) != row.get("image_sha256"):
            raise Week7TrainingError(f"v4 test image hash mismatch: {row.get('sample_id')}")
    expected_count = int(lock["counts"]["test"].get("dialogue", -1))
    expected_routes = lock.get("dialogue_parent_scenario_counts", {}).get("test")
    actual_routes = dict(Counter(str(row.get("parent_scenario")) for row in rows))
    if len(rows) != expected_count or actual_routes != expected_routes:
        raise Week7TrainingError("v4 corrected dialogue test coverage or routing changed")
    return rows


def _dimension_means(metrics: dict[str, Any]) -> dict[str, float]:
    scores = metrics.get("dialogue", {}).get("scores", [])
    if not scores:
        raise Week7TrainingError("dialogue metrics contain no sample scores")
    result = {}
    for name in AUTOMATIC_DIMENSIONS:
        values = [item.get(name) for item in scores]
        if any(not isinstance(value, (int, float)) for value in values):
            raise Week7TrainingError(f"automatic dialogue dimension is unavailable: {name}")
        result[name] = statistics.fmean(float(value) for value in values)
    return result


def _sequential_record_generator(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    run_id: str,
    model_name: str,
    max_new_tokens: int,
    *,
    runtime_options: dict[str, Any],
    max_length: int = 8192,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise Week7TrainingError("v4 sequential dialogue route is empty")
    options = dict(runtime_options)
    warmup_max_new_tokens = int(options.pop("warmup_max_new_tokens", 1))
    first_messages = training_messages(rows[0])
    first_assistant = next(
        index for index, message in enumerate(first_messages)
        if message.get("role") == "assistant"
    )
    records = []
    with inference_runtime(model):
        warmup = generate_record(
            model, processor,
            structure_aware_messages(
                processor, first_messages[:first_assistant], max_length,
            ),
            sample_id=f"{rows[0]['sample_id']}#warmup",
            run_id=run_id, model_name=model_name,
            max_new_tokens=warmup_max_new_tokens, warmup=True, **options,
        )
        for row in rows:
            messages = training_messages(row)
            conversation: list[dict[str, Any]] = []
            turn_outputs = []
            total_latency_ms = 0.0
            failed = False
            for message_index, message in enumerate(messages):
                if message.get("role") != "assistant":
                    conversation.append(message)
                    continue
                generation_messages = structure_aware_messages(
                    processor, conversation, max_length,
                )
                generated = generate_record(
                    model, processor, generation_messages,
                    sample_id=(
                        f"{row['sample_id']}#assistant-{len(turn_outputs):02d}"
                    ),
                    run_id=run_id, model_name=model_name,
                    max_new_tokens=max_new_tokens, **options,
                )
                raw_output = str(generated.get("raw_output") or "")
                generation_failed = bool(generated.get("failed"))
                expected_output = assistant_content_text(message.get("content"))
                expects_tool_call = "<tool_call>" in expected_output
                has_tool_marker = (
                    "<tool_call>" in raw_output or "</tool_call>" in raw_output
                )
                protocol_valid = (
                    valid_check_constraints_tool_call(raw_output)
                    if expects_tool_call
                    else not has_tool_marker
                )
                latency_ms = float(generated.get("latency_ms", 0.0))
                turn_outputs.append({
                    "assistant_turn_index": len(turn_outputs),
                    "message_index": message_index,
                    "expected_output": expected_output,
                    "raw_output": raw_output,
                    "failed": generation_failed,
                    "latency_ms": latency_ms,
                    "error": generated.get("error"),
                    "input_token_count": generated.get("input_token_count"),
                    "generated_token_count": generated.get("generated_token_count"),
                    "generation_max_new_tokens": generated.get(
                        "generation_max_new_tokens"
                    ),
                    "limit_reached": (
                        generated.get("generated_token_count") is not None
                        and int(generated["generated_token_count"])
                        >= max_new_tokens
                    ),
                    "protocol_valid": protocol_valid,
                })
                total_latency_ms += latency_ms
                failed = failed or generation_failed or not protocol_valid
                conversation.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": raw_output}],
                })
                if generation_failed or not protocol_valid:
                    # 生成或工具协议失败时不得注入锁定的 tool result 或伪造后续轮。
                    break
            records.append({
                "sample_id": row["sample_id"],
                "run_id": run_id,
                "model_name": model_name,
                "raw_output": turn_outputs[-1]["raw_output"],
                "latency_ms": total_latency_ms,
                "failed": failed,
                "turn_outputs": turn_outputs,
                "generation_mode": "sequential_assistant_turns_v4",
            })
    return records, warmup


def _validate_sequential_records(
    rows: list[dict[str, Any]], records: list[dict[str, Any]],
) -> None:
    rows_by_id = {str(row["sample_id"]): row for row in rows}
    if (
        len(records) != len(rows_by_id)
        or {str(record.get("sample_id")) for record in records} != set(rows_by_id)
    ):
        raise Week7TrainingError("sequential dialogue records do not exactly cover the route")
    for record in records:
        row = rows_by_id[str(record["sample_id"])]
        expected_messages = training_messages(row)
        assistant_positions = [
            index
            for index, message in enumerate(expected_messages)
            if message.get("role") == "assistant"
        ]
        turn_outputs = record.get("turn_outputs")
        if (
            record.get("generation_mode") != "sequential_assistant_turns_v4"
            or not isinstance(turn_outputs, list)
            or not turn_outputs
            or len(turn_outputs) > len(assistant_positions)
        ):
            raise Week7TrainingError(
                f"sequential assistant-turn coverage mismatch: {row['sample_id']}"
            )
        for turn_index, (message_index, turn) in enumerate(
            zip(assistant_positions, turn_outputs)
        ):
            expected = assistant_content_text(
                expected_messages[message_index].get("content")
            )
            if (
                not isinstance(turn, dict)
                or turn.get("assistant_turn_index") != turn_index
                or turn.get("message_index") != message_index
                or turn.get("expected_output") != expected
                or not isinstance(turn.get("raw_output"), str)
                or not isinstance(turn.get("failed"), bool)
                or not isinstance(turn.get("protocol_valid"), bool)
                or turn.get("error") is not None
                and not isinstance(turn.get("error"), str)
                or turn.get("input_token_count") is not None
                and (
                    not isinstance(turn.get("input_token_count"), int)
                    or int(turn["input_token_count"]) < 0
                )
                or turn.get("generated_token_count") is not None
                and (
                    not isinstance(turn.get("generated_token_count"), int)
                    or int(turn["generated_token_count"]) < 0
                )
                or turn.get("generation_max_new_tokens") is not None
                and (
                    not isinstance(turn.get("generation_max_new_tokens"), int)
                    or int(turn["generation_max_new_tokens"]) <= 0
                )
                or not isinstance(turn.get("limit_reached"), bool)
                or not isinstance(turn.get("latency_ms"), (int, float))
                or not math.isfinite(float(turn["latency_ms"]))
                or float(turn["latency_ms"]) < 0.0
            ):
                raise Week7TrainingError(
                    f"sequential assistant-turn identity mismatch: {row['sample_id']}"
                )
        if len(turn_outputs) < len(assistant_positions) and not (
            record.get("failed") is True
            and (
                turn_outputs[-1].get("failed") is True
                or turn_outputs[-1].get("protocol_valid") is False
            )
        ):
            raise Week7TrainingError(
                f"incomplete sequential turns lack a protocol failure: {row['sample_id']}"
            )
        if record.get("raw_output") != turn_outputs[-1]["raw_output"]:
            raise Week7TrainingError(
                f"final raw output differs from final assistant turn: {row['sample_id']}"
            )


def _persist_role(
    output_dir: Path,
    role: str,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    warmups: list[dict[str, Any]],
    *,
    run_id: str,
    model_identity: Any,
    scoring_protocol: str = "gold_exact_v1",
) -> dict[str, Any]:
    role_dir = output_dir / "roles" / role
    role_dir.mkdir(parents=True, exist_ok=False)
    _validate_sequential_records(rows, records)
    raw_path = role_dir / "raw_outputs.jsonl"
    warmup_path = role_dir / "warmups.jsonl"
    write_jsonl_new(raw_path, records)
    write_jsonl_new(warmup_path, warmups)
    metrics = summarize_dialogue_raw_records(
        rows, records, scoring_protocol=scoring_protocol,
    )
    metrics.update({
        "status": "COMPLETED",
        "model_role": role,
        "run_id": run_id,
        "split": "test",
        "human_evaluation": "NOT_PERFORMED_AUTOMATIC_ONLY",
        "automatic_dimensions": _dimension_means(metrics),
        "model_identity": model_identity,
        "raw_outputs": {
            "path": str(raw_path),
            "sha256": sha256_file(raw_path),
            "count": len(records),
        },
        "warmups": {
            "path": str(warmup_path),
            "sha256": sha256_file(warmup_path),
            "count": len(warmups),
        },
    })
    metrics_path = role_dir / "metrics.json"
    _write_json_new(metrics_path, metrics)
    return metrics


def _relative_change(candidate: float, baseline: float) -> float | None:
    return None if math.isclose(baseline, 0.0) else (candidate - baseline) / baseline


def _comparison(config: dict[str, Any], roles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metric_paths = {
        "format_compliance": ("dialogue", "format_compliance"),
        "context_recall": ("dialogue", "context_recall"),
        "context_state_value_accuracy": ("dialogue", "context_state_value_accuracy"),
        "task_result_key_coverage": ("dialogue", "task_result_key_coverage"),
        "task_result_value_accuracy": ("dialogue", "task_result_value_accuracy"),
        "sequential_turn_coverage": ("dialogue", "sequential_turn_coverage"),
        "sequential_turn_failure_rate": (
            "dialogue", "sequential_turn_failure_rate",
        ),
        "automatic_composite": ("dialogue", "automatic_composite"),
        "failure_rate": (None, "failure_rate"),
        "latency_ms_mean": (None, "latency_ms_mean"),
    }
    dialogue_count = int(config["dataset"]["test_dialogue_count"])
    tool_stride = round(1.0 / float(config["sampling"]["tool_call_dialogue_fraction"]))
    expected_tool_support = sum(
        index % tool_stride == 0 for index in range(dialogue_count)
    )
    if (
        "minimum_tool_protocol_compliance"
        in config["evaluation"]["dialogue_automatic_gate"]
        and any(
            int(payload.get("dialogue", {}).get("tool_protocol_support_count", -1))
            != expected_tool_support
            for payload in roles.values()
        )
    ):
        raise Week7TrainingError("final dialogue tool support identity mismatch")
    optional_dialogue_metrics = (
        "initial_task_stable_value_accuracy",
        "anchor_retention",
        "tool_protocol_compliance",
        "sequential_protocol_coverage",
        "sequential_semantic_accuracy",
    )
    available_optional_metrics = {
        metric_name
        for metric_name in optional_dialogue_metrics
        if all(
            isinstance(payload.get("dialogue", {}).get(metric_name), (int, float))
            for payload in roles.values()
        )
    }
    for metric_name in sorted(available_optional_metrics):
        metric_paths[metric_name] = ("dialogue", metric_name)
    for dimension in AUTOMATIC_DIMENSIONS:
        metric_paths[f"automatic_dimension.{dimension}"] = (
            "automatic_dimensions",
            dimension,
        )

    def value(payload: dict[str, Any], path: tuple[str | None, str]) -> float:
        section, key = path
        source = payload if section is None else payload[section]
        return float(source[key])

    changes: dict[str, Any] = {}
    for baseline_role in ("week6_routed", "zero_shot"):
        changes[baseline_role] = {}
        for name, path in metric_paths.items():
            candidate = value(roles["multitask"], path)
            baseline = value(roles[baseline_role], path)
            changes[baseline_role][name] = {
                "multitask": candidate,
                "baseline": baseline,
                "absolute_change": candidate - baseline,
                "relative_change": _relative_change(candidate, baseline),
            }
    try:
        checks = evaluate_dialogue_automatic_gate(
            config, roles["multitask"]
        )["checks"]
    except Week7EvaluationError as exc:
        raise Week7TrainingError(str(exc)) from exc
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "gate_checks": checks,
        "human_scores_used": False,
        "automatic_dimensions": {
            role: payload["automatic_dimensions"] for role, payload in roles.items()
        },
        "role_metrics": {
            role: {
                "sample_count": payload["sample_count"],
                "format_compliance": payload["dialogue"]["format_compliance"],
                "context_recall": payload["dialogue"]["context_recall"],
                "context_state_value_accuracy": payload["dialogue"]["context_state_value_accuracy"],
                "task_result_key_coverage": payload["dialogue"]["task_result_key_coverage"],
                "task_result_value_accuracy": payload["dialogue"]["task_result_value_accuracy"],
                "sequential_turn_coverage": payload["dialogue"]["sequential_turn_coverage"],
                "sequential_turn_failure_rate": payload["dialogue"]["sequential_turn_failure_rate"],
                "automatic_composite": payload["dialogue"]["automatic_composite"],
                "failure_rate": payload["failure_rate"],
                "latency_ms_mean": payload["latency_ms_mean"],
                **{
                    metric_name: payload["dialogue"][metric_name]
                    for metric_name in sorted(available_optional_metrics)
                },
            }
            for role, payload in roles.items()
        },
        "multitask_changes": changes,
    }


def run_corrected_dialogue_test_once(
    root: Path,
    config_path: Path,
    selection_path: Path,
    output_dir: Path,
    week6_adapters: dict[str, Path],
    *,
    processor: Any = None,
    model_loader: ModelLoader | None = None,
    record_generator: RecordGenerator | None = None,
) -> dict[str, Any]:
    """Consume locked v4 test dialogue rows once and compare three model roles."""
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    selection_path = Path(selection_path).resolve()
    output_dir = Path(output_dir).resolve()
    week6_adapters = {
        scenario: Path(path).resolve() for scenario, path in week6_adapters.items()
    }
    config, lock, lock_root, selection, checkpoint, checkpoint_hash = _validate_inputs(
        root, config_path, selection_path, week6_adapters, output_dir,
    )
    scoring_protocol = config["evaluation"].get(
        "dialogue_scoring_protocol", "gold_exact_v1"
    )
    max_new_tokens = int(
        config["evaluation"].get("generation_max_new_tokens", 2048)
    )
    if max_new_tokens <= 0:
        raise Week7TrainingError(
            "corrected dialogue generation_max_new_tokens must be positive"
        )
    max_length = int(config["training"]["max_length"])
    if max_length <= 0:
        raise Week7TrainingError("corrected dialogue max_length must be positive")
    report = (
        environment_report(require_cuda=True)
        if model_loader is None
        else {"status": "injected-test-runtime"}
    )
    if report["status"] not in {"ok", "injected-test-runtime"}:
        raise Week7TrainingError(f"v4 dialogue-test environment is not ready: {report['status']}")
    if report["status"] == "ok" and not os.environ.get("SLURM_JOB_ID"):
        raise Week7TrainingError("v4 corrected dialogue inference must run inside Slurm")
    marker = (
        lock_root.parent.parent
        / "test_consumption"
        / f"{lock['dataset_version']}-corrected-dialogue-v4.json"
    )
    claim = _claim_test(
        marker,
        config=config,
        config_path=config_path,
        lock=lock,
        selection_path=selection_path,
        output_dir=output_dir,
    )
    try:
        # 只有原子占用成功后才允许读取 test 内容或计算其哈希。
        rows = _load_test_dialogues(root, lock_root, lock, config)
        runtime_options = {
            "latency_protocol": LATENCY_PROTOCOL_V5_VERSION,
            "cache_implementation": "static",
            "compile_config": {
                "backend": "inductor",
                "mode": "reduce-overhead",
                "fullgraph": False,
                "dynamic": True,
            },
            "warmup_max_new_tokens": 32,
        }
        if model_loader is None:
            processor, model_loader = _default_model_loader(
                config, {"inference_precision": "bf16"},
            )
        if processor is None or model_loader is None:
            raise Week7TrainingError("v4 dialogue-test model loader requires a processor")
        if record_generator is None:
            def record_generator(
                model: Any,
                proc: Any,
                selected_rows: list[dict[str, Any]],
                run_id: str,
                model_name: str,
                max_new_tokens: int,
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                return _sequential_record_generator(
                    model,
                    proc,
                    selected_rows,
                    run_id,
                    model_name,
                    max_new_tokens,
                    runtime_options=runtime_options,
                    max_length=max_length,
                )

        output_dir.mkdir(parents=True, exist_ok=False)
        run_id = config["experiment_identity"]["test_run_id"]
        model_name = config["base_model"]
        role_metrics: dict[str, dict[str, Any]] = {}

        def generate_route(
            role: str,
            routes: list[tuple[Path | None, list[dict[str, Any]]]],
            identity: Any,
        ) -> None:
            all_records: list[dict[str, Any]] = []
            warmups: list[dict[str, Any]] = []
            for adapter, route_rows in routes:
                model = model_loader(adapter)
                try:
                    records, warmup = record_generator(
                        model,
                        processor,
                        route_rows,
                        f"{run_id}_{role}",
                        model_name,
                        max_new_tokens,
                    )
                    all_records.extend(records)
                    warmups.append(warmup)
                finally:
                    del model
                    gc.collect()
                    _release_model()
            role_metrics[role] = _persist_role(
                output_dir,
                role,
                rows,
                all_records,
                warmups,
                run_id=f"{run_id}_{role}",
                model_identity=identity,
                scoring_protocol=scoring_protocol,
            )

        generate_route(
            "multitask",
            [(checkpoint, rows)],
            {"adapter_dir": str(checkpoint), "adapter_sha256": checkpoint_hash},
        )
        routed: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            scenario = str(row.get("parent_scenario") or _dialogue_task(row))
            if scenario not in CORE_SCENARIOS or _dialogue_task(row) != scenario:
                raise Week7TrainingError(f"v4 dialogue route identity mismatch: {row.get('sample_id')}")
            routed[scenario].append(row)
        generate_route(
            "week6_routed",
            [(week6_adapters[scenario], routed[scenario]) for scenario in CORE_SCENARIOS],
            {
                scenario: {
                    "adapter_dir": str(week6_adapters[scenario]),
                    "adapter_sha256": config["evaluation"]["week6_adapter_sha256"][scenario],
                    "sample_count": len(routed[scenario]),
                }
                for scenario in CORE_SCENARIOS
            },
        )
        generate_route("zero_shot", [(None, rows)], {"base_model": model_name, "adapter": None})
        if tuple(role_metrics) != ROLE_ORDER:
            raise Week7TrainingError("v4 dialogue-test role order changed")

        comparison = _comparison(config, role_metrics)
        result = {
            "schema_version": TEST_SCHEMA_VERSION,
            "status": "COMPLETED",
            "gate_status": comparison["status"],
            "run_id": run_id,
            "split": "test",
            "scope": "corrected_dialogue_only",
            "sample_count": len(rows),
            "human_evaluation": "NOT_PERFORMED_AUTOMATIC_ONLY",
            "config_sha256": sha256_file(config_path),
            "dataset_lock_sha256": lock["lock_sha256"],
            "test_file_sha256": lock["files"]["test/dialogue.jsonl"]["sha256"],
            "selection_sha256": sha256_file(selection_path),
            "selection_status": selection["status"],
            "selected_checkpoint_sha256": checkpoint_hash,
            "runtime": {
                "precision": "bf16",
                "latency_protocol": LATENCY_PROTOCOL_V5_VERSION,
                "max_new_tokens": max_new_tokens,
                "max_length": max_length,
                "dialogue_construction_version": config["sampling"].get(
                    "dialogue_construction_version", "aligned_concrete_turns_v4"
                ),
                "dialogue_scoring_protocol": scoring_protocol,
                "cache_implementation": "static",
                "compile_config": runtime_options["compile_config"],
            },
            "comparison": comparison,
            "role_artifacts": {
                role: {
                    "metrics_path": str(output_dir / "roles" / role / "metrics.json"),
                    "metrics_sha256": sha256_file(output_dir / "roles" / role / "metrics.json"),
                    "raw_outputs_sha256": role_metrics[role]["raw_outputs"]["sha256"],
                    "warmups_sha256": role_metrics[role]["warmups"]["sha256"],
                }
                for role in ROLE_ORDER
            },
            "completed_at": _utc_now(),
        }
        summary_path = output_dir / "final_comparison.json"
        _write_json_new(summary_path, result)
        claim.update({
            "status": "COMPLETED",
            "completed_at": _utc_now(),
            "actual_dialogue_test_sha256": sha256_file(
                lock_root / "test" / "dialogue.jsonl"
            ),
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "failure": None,
        })
        _atomic_json_replace(marker, claim)
        return result
    except BaseException as exc:
        claim.update({
            "status": "FAILED",
            "completed_at": _utc_now(),
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        })
        _atomic_json_replace(marker, claim)
        raise

"""Parameter locking and resumable one-shot Week 7 final-test evaluation."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from src.training.week6_qlora import environment_report
from src.training.week7_data import CORE_SCENARIOS, canonical_sha256, iter_jsonl, load_week7_config, sha256_file
from src.training.week7_evaluation import Week7EvaluationError, summarize_raw_records
from src.training.week7_qlora import Week7TrainingError, structure_aware_messages, training_messages


REQUIRED_DEVELOPMENT_EVIDENCE = {
    "week6_development_baseline",
    "zero_shot_development",
    "multitask_development",
    "schema_decoding",
}


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


def create_parameter_lock(
    root: Path,
    config_path: Path,
    output_path: Path,
    *,
    training_summary_path: Path,
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
    if schema_decoding_mode not in {"free", "constrained"}:
        raise Week7EvaluationError("schema_decoding_mode must be free or constrained")
    if set(week6_adapters) != set(CORE_SCENARIOS):
        raise Week7EvaluationError("exactly three scenario-specific Week 6 adapters are required")
    if set(development_evidence) != REQUIRED_DEVELOPMENT_EVIDENCE:
        raise Week7EvaluationError("all four locked development evidence artifacts are required")

    lock_root = root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    dataset_lock_path = lock_root / "dataset_lock.json"
    dataset_lock = _read_json(dataset_lock_path)
    config_hash = sha256_file(config_path)
    if dataset_lock.get("config_sha256") != config_hash:
        raise Week7EvaluationError("dataset lock is bound to a different Week 7 config")
    if dataset_lock.get("test_policy", {}).get("status") != "LOCKED_UNCONSUMED":
        raise Week7EvaluationError("dataset does not declare an unconsumed one-shot test")

    training_summary_path = Path(training_summary_path).resolve()
    training_summary = _read_json(training_summary_path)
    if training_summary.get("status") != "COMPLETED":
        raise Week7EvaluationError("completed multitask training summary is required")
    if training_summary.get("config_sha256") != config_hash:
        raise Week7EvaluationError("training summary config identity mismatch")
    if training_summary.get("dataset_lock_sha256") != dataset_lock.get("lock_sha256"):
        raise Week7EvaluationError("training summary dataset identity mismatch")

    selected = Path(selected_checkpoint).resolve()
    selected_spec = _adapter_spec(selected, expected_base_model=config["base_model"])
    best_name = Path(str(training_summary.get("best_checkpoint") or "")).name
    if not best_name or selected.name != best_name:
        raise Week7EvaluationError("selected checkpoint must be the development-best checkpoint")
    recorded_hash = training_summary.get("checkpoint_hashes", {}).get(selected.name)
    if recorded_hash != selected_spec["adapter_model_sha256"]:
        raise Week7EvaluationError("selected checkpoint does not match the training summary")

    evidence_payload: dict[str, Any] = {}
    for name, evidence_path in sorted(development_evidence.items()):
        resolved = Path(evidence_path).resolve()
        evidence = _read_json(resolved)
        if evidence.get("status") != "COMPLETED":
            raise Week7EvaluationError(f"development evidence is incomplete: {name}")
        evidence_payload[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}

    week6_specs = {
        scenario: _adapter_spec(path, expected, config["base_model"])
        for scenario, (path, expected) in sorted(week6_adapters.items())
    }
    locked_week6_hashes = config["evaluation"].get("week6_adapter_sha256")
    for scenario, spec in week6_specs.items():
        if locked_week6_hashes and spec["adapter_model_sha256"] != locked_week6_hashes[scenario]:
            raise Week7EvaluationError(f"Week 6 adapter does not match the preregistered hash: {scenario}")
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
        "test_run_id", "training_summary", "development_evidence",
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
        or generation.get("schema_decoding_mode") not in {"free", "constrained"}
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
    if sha256_file(Path(training_summary.get("path", ""))) != training_summary.get("sha256"):
        raise Week7EvaluationError("training summary changed after parameter locking")
    if set(payload.get("development_evidence", {})) != REQUIRED_DEVELOPMENT_EVIDENCE:
        raise Week7EvaluationError("parameter lock development evidence is incomplete")
    for name, evidence in payload["development_evidence"].items():
        if sha256_file(Path(evidence.get("path", ""))) != evidence.get("sha256"):
            raise Week7EvaluationError(f"development evidence changed after locking: {name}")
    return payload, dataset_lock, lock_root


def _atomic_json_replace(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    os.replace(temporary, path)


def _claim_test_run(
    marker: Path,
    *,
    run_id: str,
    parameter_lock_path: Path,
    parameter_lock_sha256: str,
    output_dir: Path,
    declared_test_sha256: str,
    resume: bool,
) -> tuple[dict[str, Any], bool]:
    marker.parent.mkdir(parents=True, exist_ok=True)
    initial = {
        "schema_version": "week7_test_consumption_v1",
        "status": "IN_PROGRESS",
        "run_id": run_id,
        "parameter_lock_path": str(parameter_lock_path.resolve()),
        "parameter_lock_sha256": parameter_lock_sha256,
        "output_dir": str(output_dir.resolve()),
        "test_file_sha256": declared_test_sha256,
    }
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        current = _read_json(marker)
        identity = (current.get("run_id"), current.get("parameter_lock_sha256"), current.get("output_dir"))
        expected = (run_id, parameter_lock_sha256, str(output_dir.resolve()))
        if identity != expected:
            raise Week7EvaluationError("Week 7 test allowance belongs to another run identity")
        if current.get("status") == "COMPLETED":
            return current, True
        if current.get("status") != "IN_PROGRESS" or not resume:
            raise Week7EvaluationError("same-run test recovery requires --resume")
        return current, False
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(initial, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return initial, False


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
) -> list[dict[str, Any]]:
    report = environment_report(require_cuda=True)
    if report.get("status") != "ok":
        raise Week7TrainingError(f"final test environment is not ready: {report.get('status')}")
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = config["quantization"]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["base_model"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        attn_implementation=config["training"]["attn_implementation"],
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    processor = AutoProcessor.from_pretrained(config["base_model"])
    model.eval()
    records: list[dict[str, Any]] = []
    for row in rows:
        messages = structure_aware_messages(
            processor, training_messages(row), int(config["training"]["max_length"])
        )[:-1]
        started = time.perf_counter()
        raw_output, error = "", None
        try:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", truncation=False,
            )
            device = next(model.parameters()).device
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            raw_output = processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )[0].strip()
        except (RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "sample_id": row["sample_id"], "model_name": role, "raw_output": raw_output,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "failed": error is not None, "error": error,
        }
        record_sink(record)
        records.append(record)
    del model
    torch.cuda.empty_cache()
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
    if inference_runner is None and parameter_lock["generation"]["schema_decoding_mode"] != "free":
        raise Week7EvaluationError(
            "the local Transformers final runner supports only a locked free-decoding mode"
        )
    declared = dataset_lock.get("files", {}).get("test.jsonl", {})
    if not declared.get("sha256") or not declared.get("count"):
        raise Week7EvaluationError("dataset lock does not bind the final test file")
    marker = lock_root.parent.parent / "test_consumption" / f"{parameter_lock['dataset_version']}.json"
    if resume and not marker.exists():
        raise Week7EvaluationError("cannot resume before the exact final-test run has started")
    state, already_completed = _claim_test_run(
        marker,
        run_id=run_id,
        parameter_lock_path=parameter_lock_path,
        parameter_lock_sha256=sha256_file(parameter_lock_path),
        output_dir=output_dir,
        declared_test_sha256=declared["sha256"],
        resume=resume,
    )
    if already_completed:
        summary_path = Path(state["summary_path"])
        if sha256_file(summary_path) != state.get("summary_sha256"):
            raise Week7EvaluationError("completed final-test summary hash mismatch")
        return _read_json(summary_path)

    output_dir.mkdir(parents=True, exist_ok=resume)
    test_path = lock_root / "test.jsonl"
    if sha256_file(test_path) != declared["sha256"]:
        raise Week7EvaluationError("final test file no longer matches the dataset lock")
    rows = list(iter_jsonl(test_path))
    if len(rows) != int(declared["count"]) or any(row.get("split") != "test" for row in rows):
        raise Week7EvaluationError("final test rows do not match the locked count/split")

    config = load_week7_config(config_path)
    max_new_tokens = int(parameter_lock["generation"]["max_new_tokens"])
    if inference_runner is None:
        def inference_runner(
            role: str,
            selected_rows: list[dict[str, Any]],
            adapter: Path | None,
            record_sink: Callable[[dict[str, Any]], None],
        ) -> list[dict[str, Any]]:
            return _transformers_runner(
                root, config, role, selected_rows, adapter, max_new_tokens, record_sink,
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
        roles["week6_single_task_adapters"].append((Path(parameter_lock["week6_adapters"][scenario]["adapter_dir"]), routed[scenario]))

    summaries: dict[str, dict[str, Any]] = {}
    for role in ("week6_single_task_adapters", "multitask", "zero_shot"):
        records = _run_role_resumable(
            role, rows, output_dir / "raw_outputs" / f"{role}.jsonl", run_id,
            inference_runner, roles[role],
        )
        summaries[role] = summarize_raw_records(root, config, rows, records)
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
    })
    summary_path = output_dir / "final_comparison.json"
    if not summary_path.exists():
        _write_json_new(summary_path, comparison)
    elif canonical_sha256(_read_json(summary_path)) != canonical_sha256(comparison):
        raise Week7EvaluationError("resumed final comparison mismatch")
    completed = {
        **state,
        "status": "COMPLETED",
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
    }
    _atomic_json_replace(marker, completed)
    return comparison

"""Immutable Week 7 Transformers and OpenAI-compatible inference runners."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.evaluation.schema_validation import load_output_schema
from src.training.week6_qlora import environment_report
from src.training.week7_data import CORE_SCENARIOS, iter_jsonl, load_week7_config, sha256_file
from src.training.week7_evaluation import (
    compare_schema_decoding,
    summarize_dialogue_raw_records,
    summarize_raw_records,
)
from src.training.week7_qlora import (
    Week7TrainingError,
    _generate_record as generate_training_record,
    structure_aware_messages,
    training_messages,
)
from src.training.week7_runtime import generate_record, inference_runtime


def _validate_system_candidate_adapter(
    config: dict[str, Any],
    config_path: Path,
    dataset_lock: dict[str, Any],
    adapter_dir: Path,
    adapter_hashes: dict[str, str],
) -> None:
    """Bind a candidate adapter to its completed continuation-SFT run."""

    summary_path = adapter_dir.parent / "run_summary.json"
    if not summary_path.is_file():
        raise Week7TrainingError("system-repair candidate has no run summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "COMPLETED"
        or summary.get("run_id")
        != config["experiment_identity"]["multitask_sft_run_id"]
        or summary.get("config_sha256") != sha256_file(config_path)
        or summary.get("dataset_lock_sha256") != dataset_lock.get("lock_sha256")
        or summary.get("adapter_only") is not True
        or summary.get("adapter_reload_verified") is not True
        or summary.get("adapter_hashes", {}).get("adapter_model.safetensors")
        != adapter_hashes.get("adapter_model.safetensors")
        or summary.get("continued_from_adapter", {}).get("adapter_model_sha256")
        != config["continuation"]["adapter_model_sha256"]
    ):
        raise Week7TrainingError("system-repair candidate provenance mismatch")


def _write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_transformers_development(
    root: Path, config_path: Path, output_dir: Path, *, run_id: str,
    adapter_dir: Path | None = None, model_role: str = "zero_shot",
    max_new_tokens: int = 2048, scenario: str | None = None,
) -> dict[str, Any]:
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise Week7TrainingError("refusing to overwrite an inference run")
    config = load_week7_config(config_path)
    lock_root = root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    dataset_lock = json.loads((lock_root / "dataset_lock.json").read_text(encoding="utf-8"))
    if dataset_lock.get("config_sha256") != sha256_file(config_path):
        raise Week7TrainingError("development config SHA-256 does not match the dataset lock")
    rows = list(iter_jsonl(lock_root / "development.jsonl"))
    if scenario is not None:
        if scenario not in CORE_SCENARIOS:
            raise Week7TrainingError("development scenario filter is invalid")
        rows = [row for row in rows if row["scenario"] == scenario]
    report = environment_report(require_cuda=True)
    if report["status"] != "ok":
        raise Week7TrainingError(f"inference environment is not ready: {report['status']}")
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = config["quantization"]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["base_model"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16, device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        attn_implementation=config["training"]["attn_implementation"],
    )
    adapter_hashes = None
    if adapter_dir is not None:
        adapter_dir = Path(adapter_dir).resolve()
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            raise Week7TrainingError("adapter directory is incomplete")
        adapter_hashes = {path.name: sha256_file(path) for path in adapter_dir.iterdir() if path.is_file()}
        if scenario is not None:
            expected_run = config["experiment_identity"]["development_baseline_run_ids"][scenario]
            expected_hash = config["evaluation"]["week6_adapter_sha256"][scenario]
            if run_id != expected_run or model_role != "week6_single_task_adapter":
                raise Week7TrainingError("Week 6 development adapter run identity mismatch")
            if adapter_hashes.get("adapter_model.safetensors") != expected_hash:
                raise Week7TrainingError("Week 6 development adapter SHA-256 mismatch")
        elif model_role == "multitask_existing":
            expected_run = (
                f"{config['system_repair']['repair_id']}_existing_adapter_development"
            )
            expected_hash = config["continuation"]["adapter_model_sha256"]
            if (
                run_id != expected_run
                or adapter_hashes.get("adapter_model.safetensors") != expected_hash
            ):
                raise Week7TrainingError(
                    "existing multitask development adapter identity mismatch"
                )
        elif (
            model_role != "multitask"
            or run_id
            != f"{config['experiment_identity']['multitask_sft_run_id']}_development"
        ):
            raise Week7TrainingError("multitask development run identity mismatch")
        else:
            _validate_system_candidate_adapter(
                config,
                config_path,
                dataset_lock,
                adapter_dir,
                adapter_hashes,
            )
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    elif scenario is not None or model_role != "zero_shot" or run_id != config["experiment_identity"]["zero_shot_development_run_id"]:
        raise Week7TrainingError("zero-shot development run identity mismatch")
    processor = AutoProcessor.from_pretrained(config["base_model"])
    records = []
    with inference_runtime(model):
        for row in rows:
            records.append(
                generate_training_record(
                    root,
                    model,
                    processor,
                    row,
                    run_id,
                    max_new_tokens,
                    int(config["training"]["max_length"]),
                )
            )
    summary = summarize_raw_records(root, config, rows, records)
    summary.update({
        "status": "COMPLETED", "run_id": run_id, "model_role": model_role,
        "config_sha256": sha256_file(config_path), "dataset_lock_sha256": dataset_lock["lock_sha256"],
        "adapter_hashes": adapter_hashes, "split": "development", "scenario_filter": scenario,
    })
    _write_jsonl_new(output_dir / "raw_outputs.jsonl", records)
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return summary


WEEK6_DIALOGUE_DEVELOPMENT_RUN_ID = "week7_dev_week6_dialogue_routed_20260819_v2"
WEEK6_COMBINED_DEVELOPMENT_RUN_ID = "week7_dev_week6_adapters_baseline_20260819_v2"


def _week6_dialogue_development_run_id(config: dict[str, Any]) -> str:
    return str(config["experiment_identity"].get(
        "week6_dialogue_development_run_id", WEEK6_DIALOGUE_DEVELOPMENT_RUN_ID,
    ))


def _week6_combined_development_run_id(config: dict[str, Any]) -> str:
    return str(config["experiment_identity"].get(
        "week6_combined_development_run_id", WEEK6_COMBINED_DEVELOPMENT_RUN_ID,
    ))


def _dialogue_task(row: dict[str, Any]) -> str:
    target = row.get("target", {})
    if isinstance(target, dict) and isinstance(target.get("task_result"), dict):
        target = target["task_result"]
    if not isinstance(target, dict):
        raise Week7TrainingError(f"dialogue target is not routable: {row.get('sample_id')}")
    if "business_category" in target:
        return "image_product_search"
    if "issue_type" in target:
        return "after_sales"
    if "itinerary" in target or "hard_constraints" in target:
        return "itinerary_planning"
    raise Week7TrainingError(f"dialogue target is not routable: {row.get('sample_id')}")


def _generate_transformers_records(
    config: dict[str, Any], processor: Any, model: Any, rows: list[dict[str, Any]],
    *, run_id: str, model_role: str, max_new_tokens: int,
) -> list[dict[str, Any]]:
    records = []
    with inference_runtime(model):
        for row in rows:
            messages = structure_aware_messages(
                processor, training_messages(row), int(config["training"]["max_length"])
            )[:-1]
            records.append(generate_record(
                model,
                processor,
                messages,
                sample_id=row["sample_id"],
                run_id=run_id,
                model_name=model_role,
                max_new_tokens=max_new_tokens,
            ))
    return records


def run_week6_dialogue_development(
    root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    adapter_dirs: dict[str, Path],
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Route the 24 locked development dialogues to their underlying Week 6 adapter."""
    import gc

    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise Week7TrainingError("refusing to overwrite an inference run")
    if set(adapter_dirs) != set(CORE_SCENARIOS):
        raise Week7TrainingError("exactly three Week 6 adapters are required")
    config = load_week7_config(config_path)
    lock_root = root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    dataset_lock = json.loads((lock_root / "dataset_lock.json").read_text(encoding="utf-8"))
    if dataset_lock.get("config_sha256") != sha256_file(config_path):
        raise Week7TrainingError("development config SHA-256 does not match the dataset lock")
    rows = [
        row for row in iter_jsonl(lock_root / "development.jsonl")
        if row.get("scenario") == "dialogue"
    ]
    expected_count = int(config["dataset"]["development_dialogue_count"])
    if len(rows) != expected_count:
        raise Week7TrainingError("development dialogue support count mismatch")
    routed = {scenario: [] for scenario in CORE_SCENARIOS}
    for row in rows:
        routed[_dialogue_task(row)].append(row)
    if any(not routed[scenario] for scenario in CORE_SCENARIOS):
        raise Week7TrainingError("development dialogues do not cover all three adapter routes")

    resolved_adapters: dict[str, Path] = {}
    adapter_hashes: dict[str, dict[str, str]] = {}
    for scenario in CORE_SCENARIOS:
        adapter = Path(adapter_dirs[scenario]).resolve()
        model_path = adapter / "adapter_model.safetensors"
        if not model_path.is_file():
            raise Week7TrainingError(f"Week 6 adapter directory is incomplete: {scenario}")
        model_hash = sha256_file(model_path)
        if model_hash != config["evaluation"]["week6_adapter_sha256"][scenario]:
            raise Week7TrainingError(f"Week 6 adapter SHA-256 mismatch: {scenario}")
        resolved_adapters[scenario] = adapter
        adapter_hashes[scenario] = {
            path.name: sha256_file(path) for path in adapter.iterdir() if path.is_file()
        }

    report = environment_report(require_cuda=True)
    if report["status"] != "ok":
        raise Week7TrainingError(f"inference environment is not ready: {report['status']}")
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(config["base_model"])
    run_id = _week6_dialogue_development_run_id(config)
    quant = config["quantization"]
    records = []
    for scenario in CORE_SCENARIOS:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["base_model"],
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
            attn_implementation=config["training"]["attn_implementation"],
        )
        model = PeftModel.from_pretrained(
            model, str(resolved_adapters[scenario]), is_trainable=False,
        )
        model.eval()
        records.extend(_generate_transformers_records(
            config, processor, model, routed[scenario],
            run_id=run_id,
            model_role="week6_single_task_adapters",
            max_new_tokens=max_new_tokens,
        ))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    summary = summarize_dialogue_raw_records(rows, records)
    summary.update({
        "status": "COMPLETED",
        "run_id": run_id,
        "model_role": "week6_single_task_adapters",
        "config_sha256": sha256_file(config_path),
        "dataset_lock_sha256": dataset_lock["lock_sha256"],
        "adapter_hashes": adapter_hashes,
        "split": "development",
        "scenario_filter": "dialogue_routed",
        "routing": {
            "method": "target_task_result_v1",
            "sample_counts": {scenario: len(routed[scenario]) for scenario in CORE_SCENARIOS},
        },
    })
    _write_jsonl_new(output_dir / "raw_outputs.jsonl", records)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return summary


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _openai_messages(root: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = training_messages(row)[:-1]
    converted = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            converted.append({"role": message["role"], "content": content})
            continue
        blocks = []
        for item in content:
            if item.get("type") == "image":
                blocks.append({"type": "image_url", "image_url": {"url": _data_url(root / item["path"])}})
            elif item.get("type") == "text":
                blocks.append({"type": "text", "text": item["text"]})
        converted.append({"role": message["role"], "content": blocks})
    return converted


def _request_completion(
    endpoint: str, payload: dict[str, Any], timeout: int,
) -> tuple[str, float, str | None, str | None]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return (
            str(result["choices"][0]["message"]["content"]),
            (time.perf_counter() - started) * 1000,
            None,
            str(result["model"]),
        )
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        return "", (time.perf_counter() - started) * 1000, f"{type(exc).__name__}: {exc}", None


def _request_model_registry(endpoint: str, timeout: int) -> list[str]:
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/models", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        model_ids = [str(item["id"]) for item in payload["data"]]
    except (urllib.error.URLError, TimeoutError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise Week7TrainingError(f"cannot verify the served model registry: {exc}") from exc
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise Week7TrainingError("served model registry is empty or ambiguous")
    return model_ids


def run_schema_experiment(root: Path, config_path: Path, output_dir: Path, *, endpoint: str, served_model: str, timeout: int = 300) -> dict[str, Any]:
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise Week7TrainingError("refusing to overwrite the schema experiment")
    config = load_week7_config(config_path)
    if served_model != config["base_model"]:
        raise Week7TrainingError("served model must exactly match the locked Qwen3-VL-8B base model")
    registry_model_ids = _request_model_registry(endpoint, timeout)
    if registry_model_ids != [served_model]:
        raise Week7TrainingError("served model registry does not exactly match the locked base model")
    lock_root = root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    dataset_lock = json.loads((lock_root / "dataset_lock.json").read_text(encoding="utf-8"))
    if dataset_lock.get("config_sha256") != sha256_file(config_path):
        raise Week7TrainingError("schema experiment config SHA-256 does not match the dataset lock")
    rows = [row for row in iter_jsonl(lock_root / "development.jsonl") if row["scenario"] in CORE_SCENARIOS]
    output_dir.mkdir(parents=True, exist_ok=False)
    by_mode: dict[str, list[dict[str, Any]]] = {"free": [], "constrained": []}
    handles = {
        mode: (output_dir / f"{mode}_raw_outputs.jsonl").open("x", encoding="utf-8", newline="\n")
        for mode in by_mode
    }
    try:
        for row in rows:
            free_payload: dict[str, Any] = {
                "model": served_model, "messages": _openai_messages(root, row),
                "temperature": 0, "max_tokens": 2048,
            }
            free_raw, free_latency, free_error, free_response_model = _request_completion(
                endpoint, free_payload, timeout,
            )
            free_record = {
                "run_id": config["experiment_identity"]["schema_free_run_id"],
                "sample_id": row["sample_id"], "raw_output": free_raw,
                "latency_ms": free_latency, "failed": free_error is not None,
                "fallback_used": False, "fallback_failed": False, "error": free_error,
                "scenario": row["scenario"], "response_model": free_response_model,
            }
            by_mode["free"].append(free_record)
            handles["free"].write(json.dumps(free_record, ensure_ascii=False, sort_keys=True) + "\n")
            handles["free"].flush()

            payload: dict[str, Any] = {
                "model": served_model, "messages": _openai_messages(root, row),
                "temperature": 0, "max_tokens": 2048,
            }
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": f"week7_{row['scenario']}_v1", "strict": True, "schema": load_output_schema(root, row["scenario"], "v1")},
            }
            raw, latency, error, primary_response_model = _request_completion(
                endpoint, payload, timeout,
            )
            primary_constrained_raw = raw
            constrained_error = error
            fallback_used = error is not None
            fallback_error = None
            fallback_raw = ""
            fallback_response_model = None
            if fallback_used:
                fallback_raw, fallback_latency, fallback_error, fallback_response_model = (
                    _request_completion(endpoint, free_payload, timeout)
                )
                latency += fallback_latency
            constrained_record = {
                "run_id": config["experiment_identity"]["schema_constrained_run_id"],
                "sample_id": row["sample_id"], "raw_output": primary_constrained_raw,
                "primary_constrained_raw_output": primary_constrained_raw,
                "fallback_raw_output": fallback_raw,
                "operational_raw_output": fallback_raw if fallback_used and fallback_error is None else primary_constrained_raw,
                "latency_ms": latency,
                "failed": fallback_error is not None if fallback_used else False,
                "fallback_used": fallback_used, "fallback_failed": fallback_error is not None,
                "constrained_error": constrained_error, "fallback_error": fallback_error,
                "scenario": row["scenario"],
                "primary_response_model": primary_response_model,
                "fallback_response_model": fallback_response_model,
            }
            by_mode["constrained"].append(constrained_record)
            handles["constrained"].write(json.dumps(constrained_record, ensure_ascii=False, sort_keys=True) + "\n")
            handles["constrained"].flush()
    finally:
        for handle in handles.values():
            handle.close()
    comparison = compare_schema_decoding(root, config, rows, by_mode["free"], by_mode["constrained"])
    response_models = {
        str(model)
        for record in by_mode["free"]
        for model in (record.get("response_model"),)
        if model is not None
    } | {
        str(model)
        for record in by_mode["constrained"]
        for model in (
            record.get("primary_response_model"), record.get("fallback_response_model"),
        )
        if model is not None
    }
    raw_artifacts = {
        mode: {
            "path": str((output_dir / f"{mode}_raw_outputs.jsonl").resolve()),
            "sha256": sha256_file(output_dir / f"{mode}_raw_outputs.jsonl"),
            "count": len(records),
        }
        for mode, records in by_mode.items()
    }
    completion_eligible = (
        comparison["modes"]["free"]["operational_failure_count"] < len(rows)
        and comparison["modes"]["constrained"]["operational_failure_count"] < len(rows)
        and response_models == {served_model}
    )
    comparison.update({
        "status": "COMPLETED" if completion_eligible else "FAILED_REQUESTS",
        "served_model": served_model,
        "model_identity": {
            "base_model": config["base_model"],
            "requested_served_model": served_model,
            "registry_model_ids": registry_model_ids,
            "successful_response_model_ids": sorted(response_models),
            "verified": response_models == {served_model},
        },
        "endpoint_recorded": endpoint, "config_sha256": sha256_file(config_path),
        "dataset_lock_sha256": dataset_lock["lock_sha256"],
        "model_role": "schema_format_only_experiment", "split": "development",
        "run_ids": {
            "free": config["experiment_identity"]["schema_free_run_id"],
            "constrained": config["experiment_identity"]["schema_constrained_run_id"],
        },
        "paired_order": "sample_interleaved_free_then_constrained",
        "fallback_used_count": sum(bool(record["fallback_used"]) for record in by_mode["constrained"]),
        "raw_artifacts": raw_artifacts,
        "completion_eligibility": {
            "eligible": completion_eligible,
            "free_has_success": comparison["modes"]["free"]["operational_failure_count"] < len(rows),
            "constrained_operational_has_success": comparison["modes"]["constrained"]["operational_failure_count"] < len(rows),
            "served_model_verified": response_models == {served_model},
        },
    })
    (output_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if not completion_eligible:
        raise Week7TrainingError(
            "schema experiment requests/model identity are incomplete; comparison is not COMPLETED"
        )
    return comparison


def combine_week6_development_baseline(
    config_path: Path,
    scenario_metrics: dict[str, Path],
    dialogue_metrics: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Combine three independently generated, preregistered Week 6 adapter baselines."""
    config = load_week7_config(config_path)
    if set(scenario_metrics) != set(CORE_SCENARIOS):
        raise Week7TrainingError("exactly three Week 6 scenario metrics are required")
    scenarios: dict[str, Any] = {}
    sample_count = 0
    failure_count = 0
    latency_weighted = 0.0
    inputs = {}
    config_hash = sha256_file(config_path)
    dataset_lock_sha256 = None
    for scenario, path in sorted(scenario_metrics.items()):
        resolved = Path(path).resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        expected_run = config["experiment_identity"]["development_baseline_run_ids"][scenario]
        expected_hash = config["evaluation"]["week6_adapter_sha256"][scenario]
        if (
            payload.get("status") != "COMPLETED"
            or payload.get("run_id") != expected_run
            or payload.get("model_role") != "week6_single_task_adapter"
            or payload.get("split") != "development"
            or payload.get("scenario_filter") != scenario
            or payload.get("adapter_hashes", {}).get("adapter_model.safetensors") != expected_hash
            or set(payload.get("scenarios", {})) != {scenario}
            or payload.get("config_sha256") != config_hash
            or not payload.get("dataset_lock_sha256")
            or int(payload.get("sample_count", -1))
            != int(config["dataset"]["development_per_core_scenario"])
        ):
            raise Week7TrainingError(f"Week 6 development evidence mismatch: {scenario}")
        if dataset_lock_sha256 is None:
            dataset_lock_sha256 = payload["dataset_lock_sha256"]
        elif payload["dataset_lock_sha256"] != dataset_lock_sha256:
            raise Week7TrainingError("Week 6 development dataset lock mismatch")
        count = int(payload["sample_count"])
        scenarios[scenario] = payload["scenarios"][scenario]
        sample_count += count
        failure_count += int(payload["failure_count"])
        latency_weighted += float(payload["latency_ms_mean"]) * count
        inputs[scenario] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    dialogue_path = Path(dialogue_metrics).resolve()
    dialogue_payload = json.loads(dialogue_path.read_text(encoding="utf-8"))
    expected_dialogue_count = int(config["dataset"]["development_dialogue_count"])
    dialogue_hashes = dialogue_payload.get("adapter_hashes", {})
    if (
        dialogue_payload.get("status") != "COMPLETED"
        or dialogue_payload.get("run_id") != _week6_dialogue_development_run_id(config)
        or dialogue_payload.get("model_role") != "week6_single_task_adapters"
        or dialogue_payload.get("split") != "development"
        or dialogue_payload.get("scenario_filter") != "dialogue_routed"
        or dialogue_payload.get("config_sha256") != config_hash
        or dialogue_payload.get("dataset_lock_sha256") != dataset_lock_sha256
        or int(dialogue_payload.get("sample_count", -1)) != expected_dialogue_count
        or set(dialogue_payload.get("scenarios", {}))
        or not isinstance(dialogue_payload.get("dialogue"), dict)
        or int(dialogue_payload["dialogue"].get("sample_count", -1)) != expected_dialogue_count
        or set(dialogue_hashes) != set(CORE_SCENARIOS)
        or any(
            dialogue_hashes[scenario].get("adapter_model.safetensors")
            != config["evaluation"]["week6_adapter_sha256"][scenario]
            for scenario in CORE_SCENARIOS
        )
    ):
        raise Week7TrainingError("Week 6 routed dialogue development evidence mismatch")
    route_counts = dialogue_payload.get("routing", {}).get("sample_counts", {})
    expected_route_counts = config["sampling"].get(
        "dialogue_parent_scenario_counts", {}
    ).get(
        "development",
        {
            scenario: expected_dialogue_count // len(CORE_SCENARIOS)
            for scenario in CORE_SCENARIOS
        },
    )
    if (
        dialogue_payload.get("routing", {}).get("method") != "target_task_result_v1"
        or route_counts != expected_route_counts
    ):
        raise Week7TrainingError("Week 6 routed dialogue coverage mismatch")
    dialogue_count = int(dialogue_payload["sample_count"])
    sample_count += dialogue_count
    failure_count += int(dialogue_payload["failure_count"])
    latency_weighted += float(dialogue_payload["latency_ms_mean"]) * dialogue_count
    inputs["dialogue"] = {"path": str(dialogue_path), "sha256": sha256_file(dialogue_path)}
    scenario_weights = config["evaluation"]["scenario_weights"]
    result = {
        "status": "COMPLETED",
        "run_id": _week6_combined_development_run_id(config),
        "model_role": "week6_single_task_adapters",
        "split": "development",
        "sample_count": sample_count,
        "weighted_composite": sum(float(scenario_weights[name]) * float(scenarios[name]["composite"]) for name in CORE_SCENARIOS),
        "scenarios": scenarios,
        "dialogue": dialogue_payload["dialogue"],
        "latency_ms_mean": latency_weighted / sample_count,
        "latency_ms_median": None,
        "failure_count": failure_count,
        "failure_rate": failure_count / sample_count,
        "inputs": inputs,
        "config_sha256": config_hash,
        "dataset_lock_sha256": dataset_lock_sha256,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result

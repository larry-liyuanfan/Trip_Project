"""Fair single-allocation latency re-evaluation for locked Week 7 checkpoints."""

from __future__ import annotations

import gc
import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any, Callable

from src.evaluation.metrics import WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL
from src.training.week6_qlora import environment_report
from src.training.week7_data import (
    CORE_SCENARIOS,
    canonical_sha256,
    iter_jsonl,
    load_week7_config,
    sha256_file,
)
from src.training.week7_evaluation import summarize_raw_records
from src.training.week7_qlora import Week7TrainingError, structure_aware_messages, training_messages
from src.training.week7_runtime import (
    LATENCY_PROTOCOL_VERSION,
    generate_record,
    inference_runtime,
)


PROTOCOL_SCHEMA_VERSION = "week7_development_latency_protocol_v4"
BASELINE_RAW_HASH_KEYS = {
    "image_product_search": "product_raw",
    "after_sales": "after_sales_raw",
    "itinerary_planning": "itinerary_raw",
    "dialogue": "dialogue_raw",
}

ModelLoader = Callable[[Path | None], Any]
RecordGenerator = Callable[
    [Any, Any, list[dict[str, Any]], str, str, int],
    tuple[list[dict[str, Any]], dict[str, Any]],
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week7TrainingError(f"invalid latency-protocol artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise Week7TrainingError(f"latency-protocol artifact must be an object: {path}")
    return payload


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def _default_record_generator(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    run_id: str,
    model_name: str,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise Week7TrainingError("latency protocol cannot evaluate an empty model route")

    def messages(row: dict[str, Any]) -> list[dict[str, Any]]:
        return structure_aware_messages(
            processor, training_messages(row), 8192,
        )[:-1]

    with inference_runtime(model):
        warmup = generate_record(
            model,
            processor,
            messages(rows[0]),
            sample_id=rows[0]["sample_id"],
            run_id=run_id,
            model_name=model_name,
            max_new_tokens=1,
            warmup=True,
        )
        records = [
            generate_record(
                model,
                processor,
                messages(row),
                sample_id=row["sample_id"],
                run_id=run_id,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            )
            for row in rows
        ]
    return records, warmup


def _validate_source_evidence(
    config: dict[str, Any],
    config_hash: str,
    lock: dict[str, Any],
    development_path: Path,
    training_dir: Path,
    training_summary_path: Path,
    week6_baseline_path: Path,
    week6_baseline_evidence_path: Path,
    week6_adapters: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_count = (
        int(config["dataset"]["development_per_core_scenario"]) * len(CORE_SCENARIOS)
        + int(config["dataset"]["development_dialogue_count"])
    )
    declared = lock.get("files", {}).get("development.jsonl", {})
    if (
        lock.get("config_sha256") != config_hash
        or declared.get("sha256") != sha256_file(development_path)
        or int(declared.get("count", -1)) != expected_count
    ):
        raise Week7TrainingError("latency protocol development lock identity mismatch")

    baseline = _read_json(week6_baseline_path)
    baseline_evidence = _read_json(week6_baseline_evidence_path)
    if (
        baseline.get("status") != "COMPLETED"
        or baseline.get("config_sha256") != config_hash
        or baseline.get("dataset_lock_sha256") != lock.get("lock_sha256")
        or int(baseline.get("sample_count", -1)) != expected_count
        or baseline_evidence.get("status") != "COMPLETED"
        or baseline_evidence.get("config_sha256") != config_hash
        or baseline_evidence.get("dataset_lock_sha256") != lock.get("lock_sha256")
        or baseline_evidence.get("test_consumed") is not False
        or baseline_evidence.get("artifact_sha256", {}).get("week6_combined_metrics")
        != sha256_file(week6_baseline_path)
    ):
        raise Week7TrainingError("latency protocol Week 6 baseline identity mismatch")
    inputs = baseline.get("inputs", {})
    if set(inputs) != set(CORE_SCENARIOS) | {"dialogue"}:
        raise Week7TrainingError("latency protocol Week 6 baseline inputs are incomplete")
    baseline_raw: dict[str, Any] = {}
    for role, spec in inputs.items():
        metrics_path = Path(str(spec.get("path", ""))).resolve()
        raw_path = metrics_path.parent / "raw_outputs.jsonl"
        expected_hash = baseline_evidence["artifact_sha256"].get(
            BASELINE_RAW_HASH_KEYS[role]
        )
        if (
            not metrics_path.is_file()
            or sha256_file(metrics_path) != spec.get("sha256")
            or not raw_path.is_file()
            or sha256_file(raw_path) != expected_hash
        ):
            raise Week7TrainingError(f"latency protocol Week 6 raw binding mismatch: {role}")
        baseline_raw[role] = {
            "path": str(raw_path),
            "sha256": expected_hash,
        }
    if set(week6_adapters) != set(CORE_SCENARIOS):
        raise Week7TrainingError("latency protocol requires exactly three Week 6 adapters")
    for scenario, adapter in week6_adapters.items():
        model_path = adapter / "adapter_model.safetensors"
        if (
            not model_path.is_file()
            or sha256_file(model_path)
            != config["evaluation"]["week6_adapter_sha256"][scenario]
        ):
            raise Week7TrainingError(f"latency protocol Week 6 adapter mismatch: {scenario}")

    summary = _read_json(training_summary_path)
    if (
        training_summary_path.parent != training_dir
        or summary.get("status") != "COMPLETED"
        or summary.get("run_id") != config["experiment_identity"]["multitask_sft_run_id"]
        or summary.get("config_sha256") != config_hash
        or summary.get("dataset_lock_sha256") != lock.get("lock_sha256")
    ):
        raise Week7TrainingError("latency protocol training summary identity mismatch")
    completed_steps = [
        int(step) for step in summary.get("evaluation_steps", [])
        if int(step) <= int(summary.get("global_step", -1))
    ]
    if not completed_steps:
        raise Week7TrainingError("latency protocol has no completed checkpoint steps")
    source_candidates: dict[str, Any] = {}
    for step in completed_steps:
        checkpoint = training_dir / f"checkpoint-{step}"
        model_path = checkpoint / "adapter_model.safetensors"
        artifacts = summary.get("development_evaluation_artifacts", {}).get(str(step), {})
        raw_path = Path(str(artifacts.get("raw_outputs_path", ""))).resolve()
        metrics_path = Path(str(artifacts.get("metrics_path", ""))).resolve()
        if (
            not model_path.is_file()
            or sha256_file(model_path) != summary.get("checkpoint_hashes", {}).get(checkpoint.name)
            or not raw_path.is_file()
            or sha256_file(raw_path) != artifacts.get("raw_outputs_sha256")
            or not metrics_path.is_file()
            or sha256_file(metrics_path) != artifacts.get("metrics_sha256")
        ):
            raise Week7TrainingError(f"latency protocol checkpoint evidence mismatch: step {step}")
        source_candidates[str(step)] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_adapter_sha256": sha256_file(model_path),
            "training_raw_outputs_path": str(raw_path),
            "training_raw_outputs_sha256": sha256_file(raw_path),
            "training_metrics_path": str(metrics_path),
            "training_metrics_sha256": sha256_file(metrics_path),
        }
    return baseline_raw, source_candidates, summary


def _validate_protocol_config(
    protocol_config_path: Path,
    *,
    base_config_path: Path,
    config: dict[str, Any],
    lock: dict[str, Any],
    development_path: Path,
    training_summary_path: Path,
    week6_baseline_path: Path,
    week6_baseline_evidence_path: Path,
    baseline_raw: dict[str, Any],
    source_candidates: dict[str, Any],
) -> dict[str, Any]:
    protocol = _read_json(protocol_config_path)
    expected_order = [
        *[f"week6_{scenario}" for scenario in CORE_SCENARIOS],
        *[
            f"multitask_step_{int(step):06d}"
            for step in protocol.get("candidate_steps", [])
        ],
        "zero_shot",
    ]
    generation = protocol.get("generation", {})
    timing = protocol.get("timing", {})
    dataset = protocol.get("dataset", {})
    sources = protocol.get("source_evidence", {})
    if (
        protocol.get("schema_version") != "week7_evaluation_protocol_v4"
        or not isinstance(protocol.get("run_id"), str)
        or not protocol["run_id"]
        or protocol.get("base_config")
        != {"path": str(base_config_path.relative_to(base_config_path.parents[2])).replace("\\", "/"),
            "sha256": sha256_file(base_config_path)}
        or dataset.get("version") != config["dataset"]["dataset_version"]
        or dataset.get("lock_sha256") != lock.get("lock_sha256")
        or dataset.get("development_sha256") != sha256_file(development_path)
        or int(dataset.get("development_count", -1)) != int(
            lock["files"]["development.jsonl"]["count"]
        )
        or dataset.get("test_allowed") is not False
        or generation != {
            "max_new_tokens": 2048,
            "warmup_max_new_tokens": 1,
            "do_sample": False,
            "use_cache": True,
            "max_input_length": 8192,
            "structure_aware_truncation": True,
        }
        or timing.get("protocol") != LATENCY_PROTOCOL_VERSION
        or timing.get("scope") != "apply_chat_template+device_transfer+generate+decode"
        or any(
            timing.get(name) is not True
            for name in (
                "cuda_synchronize_before", "cuda_synchronize_after",
                "model_loading_excluded", "warmup_excluded",
                "single_slurm_allocation",
            )
        )
        or timing.get("sequential_model_order") != expected_order
        or protocol.get("metric_support_protocol")
        != WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL
        or float(protocol.get("max_latency_ratio", -1))
        != float(config["evaluation"]["non_regression"]["max_latency_ratio"])
        or sources.get("training_summary_sha256") != sha256_file(training_summary_path)
        or sources.get("week6_combined_metrics_sha256") != sha256_file(week6_baseline_path)
        or sources.get("week6_baseline_evidence_sha256")
        != sha256_file(week6_baseline_evidence_path)
        or sources.get("week6_raw_sha256")
        != {role: spec["sha256"] for role, spec in baseline_raw.items()}
        or sources.get("candidates")
        != {
            step: {
                "checkpoint_adapter_sha256": spec["checkpoint_adapter_sha256"],
                "training_raw_outputs_sha256": spec["training_raw_outputs_sha256"],
            }
            for step, spec in source_candidates.items()
        }
    ):
        raise Week7TrainingError("evaluation protocol v4 config identity mismatch")
    return protocol


def _default_model_loader(config: dict[str, Any]) -> tuple[Any, ModelLoader]:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(config["base_model"])
    quant = config["quantization"]

    def load(adapter: Path | None) -> Any:
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
        if adapter is not None:
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        return model

    return processor, load


def _release_model() -> None:
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _persist_role(
    root: Path,
    config: dict[str, Any],
    output_dir: Path,
    role: str,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    warmups: list[dict[str, Any]],
    identity: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    role_dir = output_dir / "roles" / role
    raw_path = role_dir / "raw_outputs.jsonl"
    warmup_path = role_dir / "warmups.jsonl"
    _write_jsonl_new(raw_path, records)
    _write_jsonl_new(warmup_path, warmups)
    metrics = summarize_raw_records(
        root,
        config,
        rows,
        records,
        metric_support_protocol=WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL,
    )
    metrics.update({
        "status": "COMPLETED",
        "model_role": role,
        "split": "development",
        **identity,
        "latency_protocol": LATENCY_PROTOCOL_VERSION,
        "runtime": runtime,
        "raw_outputs": {
            "path": str(raw_path.resolve()),
            "sha256": sha256_file(raw_path),
            "count": len(records),
        },
        "warmups": {
            "path": str(warmup_path.resolve()),
            "sha256": sha256_file(warmup_path),
            "count": len(warmups),
        },
    })
    metrics_path = role_dir / "metrics.json"
    _write_json_new(metrics_path, metrics)
    return {
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "raw_outputs_path": str(raw_path.resolve()),
        "raw_outputs_sha256": sha256_file(raw_path),
        "warmups_path": str(warmup_path.resolve()),
        "warmups_sha256": sha256_file(warmup_path),
        "sample_count": len(records),
        "latency_ms_mean": metrics["latency_ms_mean"],
        "failure_rate": metrics["failure_rate"],
    }


def run_latency_protocol_v4(
    root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    protocol_config_path: Path,
    training_dir: Path,
    training_summary_path: Path,
    week6_baseline_path: Path,
    week6_baseline_evidence_path: Path,
    week6_adapters: dict[str, Path],
    processor: Any = None,
    model_loader: ModelLoader | None = None,
    record_generator: RecordGenerator | None = None,
) -> dict[str, Any]:
    """Re-evaluate every latency comparator sequentially in one allocation."""
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    protocol_config_path = Path(protocol_config_path).resolve()
    output_dir = Path(output_dir).resolve()
    training_dir = Path(training_dir).resolve()
    training_summary_path = Path(training_summary_path).resolve()
    week6_baseline_path = Path(week6_baseline_path).resolve()
    week6_baseline_evidence_path = Path(week6_baseline_evidence_path).resolve()
    week6_adapters = {
        scenario: Path(path).resolve() for scenario, path in week6_adapters.items()
    }
    if output_dir.exists():
        raise Week7TrainingError("refusing to overwrite latency protocol evidence")
    config = load_week7_config(config_path)
    config_hash = sha256_file(config_path)
    lock_root = root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    lock = _read_json(lock_root / "dataset_lock.json")
    development_path = lock_root / "development.jsonl"
    baseline_raw, source_candidates, training_summary = _validate_source_evidence(
        config, config_hash, lock, development_path, training_dir,
        training_summary_path, week6_baseline_path, week6_baseline_evidence_path,
        week6_adapters,
    )
    protocol_config = _validate_protocol_config(
        protocol_config_path,
        base_config_path=config_path,
        config=config,
        lock=lock,
        development_path=development_path,
        training_summary_path=training_summary_path,
        week6_baseline_path=week6_baseline_path,
        week6_baseline_evidence_path=week6_baseline_evidence_path,
        baseline_raw=baseline_raw,
        source_candidates=source_candidates,
    )
    run_id = protocol_config["run_id"]
    max_new_tokens = int(protocol_config["generation"]["max_new_tokens"])
    available_steps = [int(step) for step in source_candidates]
    steps = [int(step) for step in protocol_config["candidate_steps"]]
    if steps != available_steps:
        raise Week7TrainingError("latency protocol candidate steps are not completed checkpoints")
    rows = list(iter_jsonl(development_path))
    if any(row.get("split") != "development" for row in rows):
        raise Week7TrainingError("latency protocol received non-development rows")
    expected_ids = {row["sample_id"] for row in rows}
    if len(expected_ids) != len(rows):
        raise Week7TrainingError("latency protocol development sample IDs are not unique")

    report = environment_report(require_cuda=True) if model_loader is None else {
        "status": "injected-test-runtime",
    }
    if report["status"] not in {"ok", "injected-test-runtime"}:
        raise Week7TrainingError(f"latency protocol environment is not ready: {report['status']}")
    if report["status"] == "ok" and not os.environ.get("SLURM_JOB_ID"):
        raise Week7TrainingError("latency protocol v4 must run inside one Slurm allocation")
    if model_loader is None:
        processor, model_loader = _default_model_loader(config)
    if processor is None or model_loader is None:
        raise Week7TrainingError("latency protocol model loader requires a processor")
    record_generator = record_generator or _default_record_generator

    started_unix = time.time()
    output_dir.mkdir(parents=True, exist_ok=False)
    identity = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "run_id": run_id,
        "protocol_config_path": str(protocol_config_path),
        "protocol_config_sha256": sha256_file(protocol_config_path),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "dataset_lock_path": str((lock_root / "dataset_lock.json").resolve()),
        "dataset_lock_sha256": lock["lock_sha256"],
        "development_path": str(development_path.resolve()),
        "development_sha256": sha256_file(development_path),
        "development_count": len(rows),
        "training_summary_path": str(training_summary_path),
        "training_summary_sha256": sha256_file(training_summary_path),
        "week6_baseline_path": str(week6_baseline_path),
        "week6_baseline_sha256": sha256_file(week6_baseline_path),
        "week6_baseline_evidence_path": str(week6_baseline_evidence_path),
        "week6_baseline_evidence_sha256": sha256_file(week6_baseline_evidence_path),
        "candidate_steps": steps,
        "max_new_tokens": max_new_tokens,
        "warmup_max_new_tokens": 1,
        "latency_protocol": LATENCY_PROTOCOL_VERSION,
        "execution_order": [
            *[f"week6_{scenario}" for scenario in CORE_SCENARIOS],
            *[f"multitask_step_{step:06d}" for step in steps],
            "zero_shot",
        ],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "environment": report,
        "source_week6_raw_outputs": baseline_raw,
        "source_candidates": {str(step): source_candidates[str(step)] for step in steps},
    }
    _write_json_new(output_dir / "run_identity.json", identity)
    role_identity = {
        "run_id": run_id,
        "config_sha256": config_hash,
        "dataset_lock_sha256": lock["lock_sha256"],
        "development_sha256": sha256_file(development_path),
    }

    role_evidence: dict[str, Any] = {}
    baseline_records: list[dict[str, Any]] = []
    baseline_warmups: list[dict[str, Any]] = []
    baseline_loads = []
    dialogue_rows = [row for row in rows if row["scenario"] == "dialogue"]
    for scenario in CORE_SCENARIOS:
        selected_rows = [row for row in rows if row["scenario"] == scenario]
        selected_rows += [row for row in dialogue_rows if _dialogue_task(row) == scenario]
        load_started = time.perf_counter()
        model = model_loader(week6_adapters[scenario])
        load_seconds = time.perf_counter() - load_started
        records, warmup = record_generator(
            model, processor, selected_rows, run_id,
            "week6_single_task_adapters", max_new_tokens,
        )
        baseline_records.extend(records)
        baseline_warmups.append({**warmup, "adapter_route": scenario})
        baseline_loads.append({"adapter_route": scenario, "load_seconds": load_seconds})
        del model
        _release_model()
    if {record["sample_id"] for record in baseline_records} != expected_ids:
        raise Week7TrainingError("latency protocol Week 6 routing coverage mismatch")
    baseline_records.sort(key=lambda record: next(
        index for index, row in enumerate(rows) if row["sample_id"] == record["sample_id"]
    ))
    role_evidence["week6_single_task_adapters"] = _persist_role(
        root, config, output_dir, "week6_single_task_adapters", rows,
        baseline_records, baseline_warmups,
        {
            **role_identity,
            "source_baseline_sha256": sha256_file(week6_baseline_path),
        },
        {"model_loads": baseline_loads},
    )

    for step in steps:
        role = f"multitask_step_{step:06d}"
        checkpoint = Path(source_candidates[str(step)]["checkpoint_path"])
        load_started = time.perf_counter()
        model = model_loader(checkpoint)
        load_seconds = time.perf_counter() - load_started
        records, warmup = record_generator(
            model, processor, rows, run_id, role, max_new_tokens,
        )
        del model
        _release_model()
        role_evidence[role] = _persist_role(
            root, config, output_dir, role, rows, records, [warmup],
            {
                **role_identity,
                "global_step": step,
                "checkpoint_adapter_sha256": source_candidates[str(step)][
                    "checkpoint_adapter_sha256"
                ],
                "source_training_raw_outputs_sha256": source_candidates[str(step)][
                    "training_raw_outputs_sha256"
                ],
            },
            {"model_load_seconds": load_seconds},
        )

    load_started = time.perf_counter()
    model = model_loader(None)
    load_seconds = time.perf_counter() - load_started
    records, warmup = record_generator(
        model, processor, rows, run_id, "zero_shot", max_new_tokens,
    )
    del model
    _release_model()
    role_evidence["zero_shot"] = _persist_role(
        root, config, output_dir, "zero_shot", rows, records, [warmup],
        role_identity,
        {"model_load_seconds": load_seconds},
    )

    summary = {
        **identity,
        "status": "COMPLETED",
        "started_unix": started_unix,
        "completed_unix": time.time(),
        "duration_seconds": time.time() - started_unix,
        "roles": role_evidence,
        "latency_comparison": {
            str(step): {
                "candidate_latency_ms_mean": role_evidence[
                    f"multitask_step_{step:06d}"
                ]["latency_ms_mean"],
                "baseline_latency_ms_mean": role_evidence[
                    "week6_single_task_adapters"
                ]["latency_ms_mean"],
                "latency_ratio": role_evidence[f"multitask_step_{step:06d}"][
                    "latency_ms_mean"
                ] / role_evidence["week6_single_task_adapters"]["latency_ms_mean"],
            }
            for step in steps
        },
    }
    _write_json_new(output_dir / "protocol_summary.json", summary)
    return summary


def validate_latency_protocol_v4(
    protocol_path: Path,
    *,
    config_path: Path,
    training_summary_path: Path,
    week6_baseline_path: Path,
) -> dict[str, Any]:
    """Validate immutable v4 files before their latency values reach selection."""
    protocol_path = Path(protocol_path).resolve()
    config_path = Path(config_path).resolve()
    expected_training_summary_path = Path(training_summary_path).resolve()
    week6_baseline_path = Path(week6_baseline_path).resolve()
    payload = _read_json(protocol_path)
    protocol_config_path = Path(str(payload.get("protocol_config_path", ""))).resolve()
    protocol_config = _read_json(protocol_config_path)
    config = load_week7_config(config_path)
    config_hash = sha256_file(config_path)
    root = config_path.parents[2]
    lock_root = (
        root / config["dataset"]["output_root"]
        / config["dataset"]["dataset_version"]
    ).resolve()
    dataset_lock_path = lock_root / "dataset_lock.json"
    dataset_lock = _read_json(dataset_lock_path)
    claimed_lock_hash = dataset_lock.pop("lock_sha256", None)
    if claimed_lock_hash != canonical_sha256(dataset_lock):
        raise Week7TrainingError("latency protocol v4 dataset lock canonical hash mismatch")
    dataset_lock["lock_sha256"] = claimed_lock_hash
    canonical_development_path = (lock_root / "development.jsonl").resolve()
    declared_development = dataset_lock.get("files", {}).get("development.jsonl", {})
    if (
        dataset_lock.get("config_sha256") != config_hash
        or payload.get("dataset_lock_path") != str(dataset_lock_path)
        or payload.get("dataset_lock_sha256") != claimed_lock_hash
        or payload.get("development_path") != str(canonical_development_path)
        or not canonical_development_path.is_file()
        or payload.get("development_sha256") != declared_development.get("sha256")
        or sha256_file(canonical_development_path) != declared_development.get("sha256")
        or int(payload.get("development_count", -1))
        != int(declared_development.get("count", -2))
    ):
        raise Week7TrainingError("latency protocol v4 canonical development binding mismatch")
    generation_config = protocol_config.get("generation", {})
    timing_config = protocol_config.get("timing", {})
    if (
        payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or payload.get("status") != "COMPLETED"
        or payload.get("latency_protocol") != LATENCY_PROTOCOL_VERSION
        or payload.get("config_sha256") != config_hash
        or payload.get("training_summary_sha256") != sha256_file(training_summary_path)
        or payload.get("week6_baseline_sha256") != sha256_file(week6_baseline_path)
        or not protocol_config_path.is_file()
        or payload.get("protocol_config_sha256") != sha256_file(protocol_config_path)
        or protocol_config.get("schema_version") != "week7_evaluation_protocol_v4"
        or protocol_config.get("run_id") != payload.get("run_id")
        or protocol_config.get("candidate_steps") != payload.get("candidate_steps")
        or timing_config.get("sequential_model_order")
        != payload.get("execution_order")
        or generation_config != {
            "max_new_tokens": 2048,
            "warmup_max_new_tokens": 1,
            "do_sample": False,
            "use_cache": True,
            "max_input_length": 8192,
            "structure_aware_truncation": True,
        }
        or timing_config.get("protocol") != LATENCY_PROTOCOL_VERSION
        or timing_config.get("scope")
        != "apply_chat_template+device_transfer+generate+decode"
        or any(
            timing_config.get(name) is not True
            for name in (
                "cuda_synchronize_before", "cuda_synchronize_after",
                "model_loading_excluded", "warmup_excluded",
                "single_slurm_allocation",
            )
        )
        or protocol_config.get("metric_support_protocol")
        != WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL
        or float(protocol_config.get("max_latency_ratio", -1))
        != float(config["evaluation"]["non_regression"]["max_latency_ratio"])
        or int(payload.get("max_new_tokens", -1)) != 2048
        or int(payload.get("warmup_max_new_tokens", -1)) != 1
    ):
        raise Week7TrainingError("latency protocol v4 identity mismatch")
    roles = payload.get("roles")
    steps = payload.get("candidate_steps")
    expected_roles = {
        "week6_single_task_adapters", "zero_shot",
        *{f"multitask_step_{int(step):06d}" for step in steps or []},
    }
    if not isinstance(roles, dict) or set(roles) != expected_roles:
        raise Week7TrainingError("latency protocol v4 role coverage mismatch")
    development_path = canonical_development_path
    development_rows = list(iter_jsonl(development_path))
    development_ids = {row["sample_id"] for row in development_rows}
    if (
        len(development_rows) != int(payload.get("development_count", -1))
        or len(development_ids) != len(development_rows)
        or any(row.get("split") != "development" for row in development_rows)
    ):
        raise Week7TrainingError("latency protocol v4 development coverage mismatch")
    source_candidates = payload.get("source_candidates", {})
    training_summary_path = Path(str(payload.get("training_summary_path", ""))).resolve()
    if (
        training_summary_path != expected_training_summary_path
        or not training_summary_path.is_file()
        or sha256_file(training_summary_path) != payload.get("training_summary_sha256")
    ):
        raise Week7TrainingError("latency protocol v4 training-summary path mismatch")
    training_summary = _read_json(training_summary_path)
    completed_steps = [
        int(step) for step in training_summary.get("evaluation_steps", [])
        if int(step) <= int(training_summary.get("global_step", -1))
    ]
    if completed_steps != [int(step) for step in steps]:
        raise Week7TrainingError("latency protocol v4 training step coverage mismatch")
    for step in completed_steps:
        spec = source_candidates.get(str(step), {})
        checkpoint = Path(str(spec.get("checkpoint_path", ""))).resolve()
        adapter = checkpoint / "adapter_model.safetensors"
        raw_path = Path(str(spec.get("training_raw_outputs_path", ""))).resolve()
        metrics_path = Path(str(spec.get("training_metrics_path", ""))).resolve()
        summary_artifacts = training_summary.get(
            "development_evaluation_artifacts", {}
        ).get(str(step), {})
        if (
            checkpoint.parent != training_summary_path.parent
            or checkpoint.name != f"checkpoint-{step}"
            or not adapter.is_file()
            or sha256_file(adapter) != spec.get("checkpoint_adapter_sha256")
            or training_summary.get("checkpoint_hashes", {}).get(checkpoint.name)
            != spec.get("checkpoint_adapter_sha256")
            or not raw_path.is_file()
            or sha256_file(raw_path) != spec.get("training_raw_outputs_sha256")
            or summary_artifacts.get("raw_outputs_path") != str(raw_path)
            or summary_artifacts.get("raw_outputs_sha256")
            != spec.get("training_raw_outputs_sha256")
            or not metrics_path.is_file()
            or sha256_file(metrics_path) != spec.get("training_metrics_sha256")
            or summary_artifacts.get("metrics_path") != str(metrics_path)
            or summary_artifacts.get("metrics_sha256")
            != spec.get("training_metrics_sha256")
        ):
            raise Week7TrainingError(
                f"latency protocol v4 source checkpoint mismatch: step {step}"
            )
    source_week6 = payload.get("source_week6_raw_outputs", {})
    if set(source_week6) != set(CORE_SCENARIOS) | {"dialogue"}:
        raise Week7TrainingError("latency protocol v4 source baseline coverage mismatch")
    for role, spec in source_week6.items():
        path = Path(str(spec.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != spec.get("sha256"):
            raise Week7TrainingError(
                f"latency protocol v4 source baseline mismatch: {role}"
            )
    week6_baseline_evidence_path = Path(
        str(payload.get("week6_baseline_evidence_path", ""))
    ).resolve()
    if (
        not week6_baseline_evidence_path.is_file()
        or sha256_file(week6_baseline_evidence_path)
        != payload.get("week6_baseline_evidence_sha256")
    ):
        raise Week7TrainingError(
            "latency protocol v4 Week 6 baseline evidence mismatch"
        )
    _validate_protocol_config(
        protocol_config_path,
        base_config_path=config_path,
        config=config,
        lock=dataset_lock,
        development_path=canonical_development_path,
        training_summary_path=training_summary_path,
        week6_baseline_path=week6_baseline_path,
        week6_baseline_evidence_path=week6_baseline_evidence_path,
        baseline_raw=source_week6,
        source_candidates=source_candidates,
    )
    for role, evidence in roles.items():
        metrics_path = Path(str(evidence.get("metrics_path", ""))).resolve()
        raw_path = Path(str(evidence.get("raw_outputs_path", ""))).resolve()
        warmups_path = Path(str(evidence.get("warmups_path", ""))).resolve()
        if (
            not metrics_path.is_file()
            or sha256_file(metrics_path) != evidence.get("metrics_sha256")
            or not raw_path.is_file()
            or sha256_file(raw_path) != evidence.get("raw_outputs_sha256")
            or not warmups_path.is_file()
            or sha256_file(warmups_path) != evidence.get("warmups_sha256")
            or int(evidence.get("sample_count", -1)) != int(payload["development_count"])
        ):
            raise Week7TrainingError(f"latency protocol v4 artifact mismatch: {role}")
        metrics = _read_json(metrics_path)
        records = list(iter_jsonl(raw_path))
        warmups = list(iter_jsonl(warmups_path))
        expected_warmups = 3 if role == "week6_single_task_adapters" else 1
        if (
            metrics.get("status") != "COMPLETED"
            or metrics.get("model_role") != role
            or metrics.get("run_id") != payload.get("run_id")
            or metrics.get("latency_protocol") != LATENCY_PROTOCOL_VERSION
            or metrics.get("raw_outputs", {}).get("sha256") != evidence["raw_outputs_sha256"]
            or metrics.get("warmups", {}).get("sha256") != evidence["warmups_sha256"]
            or float(metrics.get("latency_ms_mean", -1))
            != float(evidence.get("latency_ms_mean", -2))
            or metrics.get("metric_support_protocol")
            != WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL
            or len(records) != int(payload["development_count"])
            or {record.get("sample_id") for record in records} != development_ids
            or any(
                record.get("run_id") != payload.get("run_id")
                or record.get("model_name") != role
                or record.get("latency_protocol") != LATENCY_PROTOCOL_VERSION
                or record.get("warmup") is not False
                or int(record.get("generation_max_new_tokens", -1)) != 2048
                or (
                    not record.get("failed")
                    and (
                        not isinstance(record.get("input_token_count"), int)
                        or not isinstance(record.get("generated_token_count"), int)
                    )
                )
                for record in records
            )
            or len(warmups) != expected_warmups
            or any(
                warmup.get("run_id") != payload.get("run_id")
                or warmup.get("model_name") != role
                or warmup.get("latency_protocol") != LATENCY_PROTOCOL_VERSION
                or warmup.get("warmup") is not True
                or int(warmup.get("generation_max_new_tokens", -1)) != 1
                for warmup in warmups
            )
        ):
            raise Week7TrainingError(f"latency protocol v4 metrics mismatch: {role}")
        recomputed = summarize_raw_records(
            Path(config_path).resolve().parents[2],
            config,
            development_rows,
            records,
            metric_support_protocol=WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL,
        )
        for field in (
            "sample_count", "weighted_composite", "scenarios", "dialogue",
            "latency_ms_mean", "latency_ms_median", "failure_count", "failure_rate",
            "metric_support_protocol",
        ):
            if canonical_sha256(metrics.get(field)) != canonical_sha256(recomputed.get(field)):
                raise Week7TrainingError(
                    f"latency protocol v4 metrics differ from raw: {role}.{field}"
                )
        if role.startswith("multitask_step_"):
            step = str(int(role.removeprefix("multitask_step_")))
            source = source_candidates.get(step, {})
            if (
                int(metrics.get("global_step", -1)) != int(step)
                or metrics.get("checkpoint_adapter_sha256")
                != source.get("checkpoint_adapter_sha256")
                or metrics.get("source_training_raw_outputs_sha256")
                != source.get("training_raw_outputs_sha256")
            ):
                raise Week7TrainingError(
                    f"latency protocol v4 checkpoint provenance mismatch: {role}"
                )
        elif role == "week6_single_task_adapters" and (
            metrics.get("source_baseline_sha256") != payload.get("week6_baseline_sha256")
        ):
            raise Week7TrainingError("latency protocol v4 baseline provenance mismatch")
    baseline_latency = float(roles["week6_single_task_adapters"]["latency_ms_mean"])
    comparisons = payload.get("latency_comparison")
    if not isinstance(comparisons, dict) or set(comparisons) != {str(int(step)) for step in steps}:
        raise Week7TrainingError("latency protocol v4 comparison coverage mismatch")
    for step in steps:
        role = f"multitask_step_{int(step):06d}"
        candidate_latency = float(roles[role]["latency_ms_mean"])
        expected = {
            "candidate_latency_ms_mean": candidate_latency,
            "baseline_latency_ms_mean": baseline_latency,
            "latency_ratio": candidate_latency / baseline_latency,
        }
        if comparisons[str(int(step))] != expected:
            raise Week7TrainingError(f"latency protocol v4 comparison mismatch: step {step}")
    return payload

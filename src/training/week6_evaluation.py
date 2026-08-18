"""Run immutable adapter inference on the locked Week 6 itinerary validation set."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from src.training.week6_qlora import (
    Week6TrainingError,
    _normalize_processor_messages,
    environment_report,
    iter_training_rows,
)
from src.training.week6_quality import audit_itinerary_target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_itinerary_predictions(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate generated prediction audits without treating loss as business quality."""
    counts: Counter[str] = Counter()
    for record in records:
        audit = record.get("audit")
        if not isinstance(audit, dict) or not isinstance(audit.get("checks"), dict):
            raise Week6TrainingError("evaluation result is missing a target audit")
        counts["rows"] += 1
        counts["passed"] += bool(audit.get("passed"))
        for name, passed in audit["checks"].items():
            counts[name] += bool(passed)
    if not counts["rows"]:
        raise Week6TrainingError("evaluation summary requires at least one result")
    return dict(sorted(counts.items()))


_COMPARISON_IDENTITY_FIELDS = (
    "scenario",
    "base_model",
    "evaluation_input_sha256",
    "dataset_lock",
    "selected_sample_ids_sha256",
    "selected_samples",
    "generation",
)
_BUSINESS_COUNT_FIELDS = (
    "passed",
    "json_valid",
    "schema_valid",
    "expected_days_parsed",
    "day_count_match",
    "day_indices_sequential",
    "constraint_text_exact_match",
    "constraint_check_exact_coverage",
    "required_elements_complete",
)


def compare_itinerary_evaluations(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Gate a candidate only on an identity-matched business evaluation."""
    reasons: list[str] = []
    for name, payload in (("baseline", baseline), ("candidate", candidate)):
        if payload.get("status") != "completed":
            reasons.append(f"{name} evaluation is not completed")
        counts = payload.get("counts")
        if not isinstance(counts, dict):
            reasons.append(f"{name} evaluation has no counts")
        elif counts.get("rows") != payload.get("selected_samples"):
            reasons.append(f"{name} evaluation row count is incomplete")
    for field in _COMPARISON_IDENTITY_FIELDS:
        if baseline.get(field) != candidate.get(field):
            reasons.append(f"evaluation identity differs at {field}")

    baseline_counts = baseline.get("counts", {})
    candidate_counts = candidate.get("counts", {})
    deltas: dict[str, int] = {}
    if isinstance(baseline_counts, dict) and isinstance(candidate_counts, dict):
        for field in _BUSINESS_COUNT_FIELDS:
            baseline_value = baseline_counts.get(field)
            candidate_value = candidate_counts.get(field)
            if not isinstance(baseline_value, int) or not isinstance(candidate_value, int):
                reasons.append(f"comparison count is missing: {field}")
                continue
            deltas[field] = candidate_value - baseline_value
            if candidate_value < baseline_value:
                reasons.append(f"candidate regressed at {field}")
        if deltas.get("passed", 0) <= 0:
            reasons.append("candidate did not improve fully-passed rows")

    return {
        "status": "passed" if not reasons else "failed",
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "selected_samples": baseline.get("selected_samples"),
        "count_deltas": deltas,
        "reasons": reasons,
    }


def _validate_candidate_adapter_provenance(
    config: dict[str, Any], adapter_dir: Path, adapter_hashes: dict[str, str]
) -> str:
    summary_path = adapter_dir.parent / "run_summary.json"
    if not summary_path.is_file():
        raise Week6TrainingError("candidate adapter is missing its training summary")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Week6TrainingError("candidate training summary is invalid JSON") from exc
    expected_initial = config.get("refinement", {}).get(
        "expected_initial_adapter_file_sha256", {}
    )
    expected_dataset = config.get("dataset", {}).get("dataset_version")
    checks = (
        summary.get("status") == "completed",
        summary.get("scenario") == "itinerary_planning",
        summary.get("adapter_only") is True,
        summary.get("adapter_reload_verified") is True,
        summary.get("adapter_file_sha256") == adapter_hashes,
        summary.get("initial_adapter_file_sha256") == expected_initial,
        summary.get("dataset_lock", {}).get("dataset_version") == expected_dataset,
    )
    if not all(checks):
        raise Week6TrainingError("candidate adapter provenance does not match the refinement config")
    return _sha256_file(summary_path)


def _read_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Week6TrainingError(
                f"invalid evaluation result at line {line_number}"
            ) from exc
        if not isinstance(record.get("sample_id"), str):
            raise Week6TrainingError("evaluation result requires sample_id")
        records.append(record)
    if len({record["sample_id"] for record in records}) != len(records):
        raise Week6TrainingError("evaluation results contain duplicate sample IDs")
    return records


def run_itinerary_adapter_evaluation(
    root: Path,
    config: dict[str, Any],
    *,
    eval_path: Path,
    adapter_dir: Path,
    output_dir: Path,
    max_samples: int | None = None,
    max_new_tokens: int = 2048,
    resume: bool = False,
    adapter_role: str = "initial",
) -> dict[str, Any]:
    """Generate and audit one immutable adapter evaluation run with safe resume."""
    if max_samples is not None and max_samples <= 0:
        raise Week6TrainingError("max_samples must be positive")
    if max_new_tokens <= 0:
        raise Week6TrainingError("max_new_tokens must be positive")
    if adapter_role not in {"initial", "candidate"}:
        raise Week6TrainingError("adapter_role must be initial or candidate")
    if not eval_path.is_file():
        raise Week6TrainingError("evaluation input does not exist")
    if not adapter_dir.is_dir():
        raise Week6TrainingError("adapter directory does not exist")
    adapter_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(adapter_dir.iterdir())
        if path.is_file()
    }
    if not {"adapter_config.json", "adapter_model.safetensors"} <= set(adapter_hashes):
        raise Week6TrainingError("adapter directory is incomplete")
    training_summary_sha256 = None
    if adapter_role == "initial":
        expected_hashes = config.get("refinement", {}).get(
            "expected_initial_adapter_file_sha256", {}
        )
        if expected_hashes and any(
            adapter_hashes.get(name) != expected for name, expected in expected_hashes.items()
        ):
            raise Week6TrainingError("evaluation adapter hashes do not match the config")
    else:
        training_summary_sha256 = _validate_candidate_adapter_provenance(
            config, adapter_dir, adapter_hashes
        )

    source_rows = list(
        islice(
            iter_training_rows(eval_path, scenario="itinerary_planning"),
            max_samples,
        )
    )
    if not source_rows:
        raise Week6TrainingError("evaluation input is empty")
    source_locks = {
        json.dumps(row["dataset_lock"], sort_keys=True, separators=(",", ":"))
        for row in source_rows
    }
    if len(source_locks) != 1:
        raise Week6TrainingError("evaluation input contains mixed dataset locks")
    dataset_lock = json.loads(next(iter(source_locks)))
    expected_dataset_version = config.get("dataset", {}).get("dataset_version")
    if dataset_lock.get("dataset_version") != expected_dataset_version:
        raise Week6TrainingError("evaluation input does not match the configured dataset")
    manifest = {
        "schema_version": "week6_adapter_evaluation_v1",
        "scenario": "itinerary_planning",
        "base_model": config["base_model"],
        "adapter_dir": adapter_dir.as_posix(),
        "adapter_role": adapter_role,
        "adapter_file_sha256": adapter_hashes,
        "training_summary_sha256": training_summary_sha256,
        "evaluation_input": eval_path.as_posix(),
        "evaluation_input_sha256": _sha256_file(eval_path),
        "dataset_lock": dataset_lock,
        "selected_sample_ids_sha256": hashlib.sha256(
            "\n".join(row["sample_id"] for row in source_rows).encode("utf-8")
        ).hexdigest(),
        "selected_samples": len(source_rows),
        "generation": {"do_sample": False, "max_new_tokens": max_new_tokens},
        "git_commit": os.environ.get("TRIP_GIT_COMMIT"),
        "run_id": os.environ.get("TRIP_RUN_ID"),
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    manifest_path = output_dir / "run_manifest.json"
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    if output_dir.exists():
        if not resume:
            raise Week6TrainingError("refusing to overwrite an evaluation output directory")
        if not manifest_path.is_file() or manifest_path.read_text(encoding="utf-8") != manifest_text:
            raise Week6TrainingError("evaluation resume identity does not match")
        if summary_path.exists():
            raise Week6TrainingError("completed evaluation cannot be resumed")
    else:
        output_dir.mkdir(parents=True)
        manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")

    existing = _read_results(results_path)
    completed_ids = {record["sample_id"] for record in existing}
    selected_ids = {row["sample_id"] for row in source_rows}
    if not completed_ids <= selected_ids:
        raise Week6TrainingError("evaluation results do not belong to the selected input")

    report = environment_report(require_cuda=True)
    if report["status"] != "ok":
        raise Week6TrainingError(f"evaluation environment is not ready: {report['status']}")

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    if peft_config.base_model_name_or_path != config["base_model"]:
        raise Week6TrainingError("evaluation adapter points to an unexpected base model")
    quant = config["quantization"]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["base_model"],
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        attn_implementation=config["training"].get("attn_implementation", "sdpa"),
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    processor = AutoProcessor.from_pretrained(config["base_model"])
    max_length = int(config["scenarios"]["itinerary_planning"]["max_length"])
    torch.cuda.reset_peak_memory_stats()
    started_at = time.time()

    with results_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, source_row in enumerate(source_rows, 1):
            if source_row["sample_id"] in completed_ids:
                continue
            messages = _normalize_processor_messages(source_row["messages"][:-1])
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = inputs.to(model.device)
            sample_started = time.time()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            raw_output = processor.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            prediction_row = json.loads(json.dumps(source_row, ensure_ascii=False))
            prediction_row["messages"][-1]["content"] = raw_output
            audit = audit_itinerary_target(root, prediction_row)
            record = {
                "sample_id": source_row["sample_id"],
                "scenario": "itinerary_planning",
                "raw_output": raw_output,
                "latency_seconds": time.time() - sample_started,
                "generated_tokens": int(new_tokens.shape[1]),
                "audit": audit,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index % 10 == 0:
                os.fsync(handle.fileno())
    if hasattr(os, "sync"):
        os.sync()

    records = _read_results(results_path)
    if len(records) != len(source_rows):
        raise Week6TrainingError("evaluation ended before all selected samples completed")
    payload = {
        "status": "completed",
        **manifest,
        "counts": summarize_itinerary_predictions(records),
        "duration_seconds": time.time() - started_at,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload

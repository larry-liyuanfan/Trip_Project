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
from src.training.week6_qlora import _normalize_processor_messages, environment_report
from src.training.week7_data import CORE_SCENARIOS, iter_jsonl, load_week7_config, sha256_file
from src.training.week7_evaluation import compare_schema_decoding, summarize_raw_records
from src.training.week7_qlora import Week7TrainingError, training_messages


def _write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_transformers_development(
    root: Path, config_path: Path, output_dir: Path, *, run_id: str,
    adapter_dir: Path | None = None, model_role: str = "zero_shot",
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise Week7TrainingError("refusing to overwrite an inference run")
    config = load_week7_config(config_path)
    lock_root = root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    rows = list(iter_jsonl(lock_root / "development.jsonl"))
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
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    processor = AutoProcessor.from_pretrained(config["base_model"])
    model.eval()
    records = []
    for row in rows:
        messages = _normalize_processor_messages(training_messages(row)[:-1])
        started = time.perf_counter()
        raw, error = "", None
        try:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", truncation=True, max_length=int(config["training"]["max_length"]),
            )
            device = next(model.parameters()).device
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        except (RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        records.append({
            "run_id": run_id, "sample_id": row["sample_id"], "model_name": model_role,
            "raw_output": raw, "latency_ms": (time.perf_counter() - started) * 1000,
            "failed": error is not None, "error": error,
        })
    summary = summarize_raw_records(root, config, rows, records)
    summary.update({
        "status": "COMPLETED", "run_id": run_id, "model_role": model_role,
        "config_sha256": sha256_file(config_path), "dataset_lock_sha256": json.loads((lock_root / "dataset_lock.json").read_text(encoding="utf-8"))["lock_sha256"],
        "adapter_hashes": adapter_hashes, "split": "development",
    })
    _write_jsonl_new(output_dir / "raw_outputs.jsonl", records)
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
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


def _request_completion(endpoint: str, payload: dict[str, Any], timeout: int) -> tuple[str, float, str | None]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result["choices"][0]["message"]["content"]), (time.perf_counter() - started) * 1000, None
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        return "", (time.perf_counter() - started) * 1000, f"{type(exc).__name__}: {exc}"


def run_schema_experiment(root: Path, config_path: Path, output_dir: Path, *, endpoint: str, served_model: str, timeout: int = 300) -> dict[str, Any]:
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise Week7TrainingError("refusing to overwrite the schema experiment")
    config = load_week7_config(config_path)
    lock_root = root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    rows = [row for row in iter_jsonl(lock_root / "development.jsonl") if row["scenario"] in CORE_SCENARIOS]
    by_mode: dict[str, list[dict[str, Any]]] = {"free": [], "constrained": []}
    for mode in by_mode:
        for row in rows:
            payload: dict[str, Any] = {
                "model": served_model, "messages": _openai_messages(root, row),
                "temperature": 0, "max_tokens": 2048,
            }
            if mode == "constrained":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": f"week7_{row['scenario']}_v1", "strict": True, "schema": load_output_schema(root, row["scenario"], "v1")},
                }
            raw, latency, error = _request_completion(endpoint, payload, timeout)
            by_mode[mode].append({
                "run_id": config["experiment_identity"][f"schema_{mode}_run_id"],
                "sample_id": row["sample_id"], "raw_output": raw, "latency_ms": latency,
                "failed": error is not None, "fallback_failed": error is not None, "error": error,
            })
    comparison = compare_schema_decoding(root, config, rows, by_mode["free"], by_mode["constrained"])
    comparison.update({"status": "COMPLETED", "served_model": served_model, "endpoint_recorded": endpoint, "config_sha256": sha256_file(config_path)})
    output_dir.mkdir(parents=True, exist_ok=False)
    for mode, records in by_mode.items():
        with (output_dir / f"{mode}_raw_outputs.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return comparison

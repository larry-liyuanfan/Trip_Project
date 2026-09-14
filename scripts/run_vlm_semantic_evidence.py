"""Run one locked VLM role on the weak development semantic pool."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.relevance_evidence import (
    canonical_json_sha256,
    file_sha256,
    load_jsonl,
)


PRODUCT_PROMPT = """Analyze only directly visible evidence in the image. Return exactly one JSON object with keys business_category, style_tags, visible_facilities, price_range. business_category must be restaurant, hotel, attraction, or other. If equally prominent subjects conflict, use other instead of choosing one. style_tags and visible_facilities must be arrays of short lowercase strings and must be empty when the evidence is ambiguous. price_range must be unknown unless a legible visible numeric price proves the fixed development rule: below 20 is budget, 20 through 60 is mid_range, 61 through 150 is premium, and above 150 is luxury. Do not infer Wi-Fi, parking, service quality, or price from appearance."""
DIALOGUE_PROMPT = """Read the dialogue state case. Return exactly one JSON object with keys context_facts, state, task, value, route. context_facts must be an array of retained facts. Keep the newest explicit correction. route must be image_product_search for these cases."""
CORRECTION_PROMPT = """The previous answer was not a strict JSON object with the required keys. Return only the corrected JSON object, without Markdown or explanation."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--role", required=True, choices=(
        "zero_shot", "old_unified_adapter", "current_system_repair_checkpoint_87"
    ))
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.role == "zero_shot" and args.adapter_path:
        raise ValueError("zero_shot must not receive an adapter")
    if args.role != "zero_shot" and not args.adapter_path:
        raise ValueError("adapter role requires --adapter-path")
    if args.adapter_path:
        model_file = args.adapter_path / "adapter_model.safetensors"
        if not model_file.is_file():
            raise FileNotFoundError(f"missing adapter model: {model_file}")
        actual = file_sha256(model_file)
        if args.adapter_sha256 and actual != args.adapter_sha256:
            raise ValueError("adapter model SHA-256 mismatch")

    records = load_jsonl(args.manifest)
    for record in records:
        if record.get("scenario") != "product":
            continue
        image_path = args.asset_dir / record["image_relative_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"missing VLM asset: {image_path}")
        if file_sha256(image_path) != record.get("image_sha256"):
            raise ValueError(f"{record['sample_id']}: VLM image SHA-256 mismatch")
        if not isinstance(record.get("source_id"), str) or not record["source_id"]:
            raise ValueError(f"{record['sample_id']}: source_id is required")
    data_lock = canonical_json_sha256(records)
    prompt_lock = canonical_json_sha256(
        {"product": PRODUCT_PROMPT, "dialogue": DIALOGUE_PROMPT, "correction": CORRECTION_PROMPT}
    )
    generation = {"do_sample": False, "max_new_tokens": args.max_new_tokens, "max_retries": 1}
    generation_lock = canonical_json_sha256(generation)
    model, processor, torch = _load_model(args)
    output_rows: list[dict[str, Any]] = []
    for record in records:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        first_raw = _generate(model, processor, torch, record, args.asset_dir, args.max_new_tokens)
        first_prediction = _strict_json_object(first_raw)
        correction_triggered = first_prediction is None
        correction_raw = None
        prediction = first_prediction
        if correction_triggered:
            correction_raw = _generate(
                model,
                processor,
                torch,
                record,
                args.asset_dir,
                args.max_new_tokens,
                previous=first_raw,
            )
            prediction = _extract_json_object(correction_raw)
        prediction = prediction or {}
        elapsed_ms = (time.perf_counter() - started) * 1000
        row = {
            "variant": args.role,
            "sample_id": record["sample_id"],
            "scenario": record["scenario"],
            "data_lock_sha256": data_lock,
            "base_model": args.base_model,
            "base_revision": args.base_revision,
            "prompt_sha256": prompt_lock,
            "generation_config_sha256": generation_lock,
            "adapter_model_sha256": (
                file_sha256(args.adapter_path / "adapter_model.safetensors")
                if args.adapter_path else None
            ),
            "label_provenance": record["label_provenance"],
            "slices": record["slices"],
            "gold": record["gold"],
            "prediction": prediction,
            "first_attempt_json_valid": first_prediction is not None,
            "correction_triggered": correction_triggered,
            "first_attempt_raw": first_raw,
            "correction_raw": correction_raw,
            "latency_ms": elapsed_ms,
            "peak_vram_mib": (
                torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            ),
        }
        if record["scenario"] == "dialogue":
            row.update(_dialogue_metrics(record["gold"], prediction))
        output_rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "status": "completed",
        "role": args.role,
        "support": len(output_rows),
        "data_lock_sha256": data_lock,
        "prompt_sha256": prompt_lock,
        "generation_config_sha256": generation_lock,
        "result_sha256": canonical_json_sha256(output_rows),
        "first_attempt_json_compliance": sum(row["first_attempt_json_valid"] for row in output_rows) / len(output_rows),
        "correction_trigger_rate": sum(row["correction_triggered"] for row in output_rows) / len(output_rows),
        "mean_latency_ms": sum(row["latency_ms"] for row in output_rows) / len(output_rows),
        "peak_vram_mib": max(row["peak_vram_mib"] for row in output_rows),
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=quantization,
        trust_remote_code=False,
    )
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=False)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        trust_remote_code=False,
    )
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = min(int(image_processor.max_pixels or 1024 * 1024), 1024 * 1024)
    return model, processor, torch


def _generate(model, processor, torch, record, asset_dir, max_new_tokens, previous=None):
    prompt = PRODUCT_PROMPT if record["scenario"] == "product" else DIALOGUE_PROMPT
    content: list[dict[str, Any]] = []
    if record["scenario"] == "product":
        image_path = (asset_dir / record["image_relative_path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"missing VLM asset: {image_path}")
        content.append({"type": "image", "image": str(image_path)})
    else:
        prompt = f"{prompt}\n\nCASE:\n{record['dialogue']}"
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    if previous is not None:
        messages.extend(
            [
                {"role": "assistant", "content": [{"type": "text", "text": previous}]},
                {"role": "user", "content": [{"type": "text", "text": CORRECTION_PROMPT}]},
            ]
        )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs["input_ids"], generated)]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def _strict_json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json_object(value: str) -> dict[str, Any] | None:
    strict = _strict_json_object(value)
    if strict is not None:
        return strict
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        return None
    return _strict_json_object(value[start:end + 1])


def _dialogue_metrics(gold: dict[str, Any], prediction: dict[str, Any]) -> dict[str, bool]:
    expected_facts = {_norm(item) for item in gold.get("context_facts", [])}
    actual_facts = {_norm(item) for item in prediction.get("context_facts", [])}
    return {
        "context_recall": bool(expected_facts) and expected_facts.issubset(actual_facts),
        "state_value_correct": _norm(gold.get("state")) == _norm(prediction.get("state")),
        "task_key_correct": _norm(gold.get("task")) == _norm(prediction.get("task")),
        "value_correct": _norm(gold.get("value")) == _norm(prediction.get("value")),
        "first_turn_routing_correct": _norm(gold.get("route")) == _norm(prediction.get("route")),
    }


def _norm(value: Any) -> str:
    return " ".join(value.strip().casefold().split()) if isinstance(value, str) else ""


if __name__ == "__main__":
    main()

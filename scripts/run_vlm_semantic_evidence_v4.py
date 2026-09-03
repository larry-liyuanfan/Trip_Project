"""Run one adapter role on a locked v4 development or one-time final split."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_vlm_semantic_evidence import (
    CORRECTION_PROMPT,
    _dialogue_metrics,
    _extract_json_object,
    _load_model,
    _strict_json_object,
)
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("development", "final"))
    parser.add_argument("--role", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--selection-record", type=Path)
    parser.add_argument("--final-marker", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_lock = json.loads(Path(config["pool"]["committed_lock"]).read_text(encoding="utf-8"))
    actual_lock = json.loads((args.bundle_dir / "bundle_lock.json").read_text(encoding="utf-8"))
    if actual_lock != expected_lock:
        raise ValueError("generated bundle lock differs from committed lock")
    if args.role == "zero_shot" and args.adapter_path:
        raise ValueError("zero_shot must not receive an adapter")
    if args.role != "zero_shot" and not args.adapter_path:
        raise ValueError("adapter role requires --adapter-path")
    if args.adapter_path:
        adapter_file = args.adapter_path / "adapter_model.safetensors"
        if not adapter_file.is_file() or (
            args.adapter_sha256 and file_sha256(adapter_file) != args.adapter_sha256
        ):
            raise ValueError("adapter model is missing or SHA-256 mismatched")

    manifest_path = args.bundle_dir / f"vlm_{args.split}_manifest.jsonl"
    if args.split == "final":
        if not args.selection_record or not args.final_marker:
            raise ValueError("final requires --selection-record and --final-marker")
        selection = json.loads(args.selection_record.read_text(encoding="utf-8"))
        if selection.get("fixed_gates", {}).get("status") != "PASS":
            raise ValueError("development gate must pass before final consumption")
        if selection.get("candidate_variant") != args.role:
            raise ValueError("final role differs from the development-selected candidate")
        marker = {
            "schema_version": f"vlm_semantic_{config.get('cycle_id', 'v4')}_final_consumption",
            "selection_file_sha256": file_sha256(args.selection_record),
            "committed_final_lock": expected_lock["vlm"]["final"],
            "role": args.role,
            "single_consumption_policy": "exclusive_marker_written_before_first_final_manifest_open",
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        }
        records = load_final_after_marker(manifest_path, args.final_marker, marker)
    else:
        if args.selection_record or args.final_marker:
            raise ValueError("development run must not receive final-consumption arguments")
        records = load_jsonl(manifest_path)
    split_lock = expected_lock["vlm"][args.split]
    if (
        len(records) != split_lock["sample_support"]
        or canonical_json_sha256(records) != split_lock["manifest_canonical_sha256"]
        or file_sha256(manifest_path) != split_lock["manifest_file_sha256"]
    ):
        raise ValueError(f"VLM {args.split} manifest differs from committed lock")
    for record in records:
        if record.get("split") != args.split:
            raise ValueError("VLM manifest split mismatch")
        if record.get("scenario") == "product":
            image_path = args.bundle_dir / record["image_relative_path"]
            if not image_path.is_file() or file_sha256(image_path) != record["image_sha256"]:
                raise ValueError(f"VLM image mismatch: {record.get('sample_id')}")

    data_lock = canonical_json_sha256(records)
    prompt_lock = canonical_json_sha256({
        "product": sorted({row["prompt"] for row in records if row["scenario"] == "product"}),
        "dialogue": sorted({row["prompt"] for row in records if row["scenario"] == "dialogue"}),
        "correction": CORRECTION_PROMPT,
    })
    generation = {"do_sample": False, "max_new_tokens": args.max_new_tokens, "max_retries": 1}
    generation_lock = canonical_json_sha256(generation)
    model, processor, torch = _load_model(args)
    output_rows = []
    for record in records:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        first_raw = _generate(model, processor, torch, record, args.bundle_dir, args.max_new_tokens)
        first_prediction = _strict_json_object(first_raw)
        correction_triggered = first_prediction is None
        correction_raw = None
        prediction = first_prediction
        if correction_triggered:
            correction_raw = _generate(
                model, processor, torch, record, args.bundle_dir, args.max_new_tokens,
                previous=first_raw,
            )
            prediction = _extract_json_object(correction_raw)
        prediction = prediction or {}
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
                file_sha256(args.adapter_path / "adapter_model.safetensors") if args.adapter_path else None
            ),
            "label_provenance": record["label_provenance"],
            "slices": record["slices"],
            "gold": record["gold"],
            "prediction": prediction,
            "first_attempt_json_valid": first_prediction is not None,
            "correction_triggered": correction_triggered,
            "first_attempt_raw": first_raw,
            "correction_raw": correction_raw,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "peak_vram_mib": (
                torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            ),
        }
        if record["scenario"] == "dialogue":
            row.update(_dialogue_metrics(record["gold"], prediction))
        output_rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output, output_rows)
    summary = {
        "schema_version": f"vlm_semantic_role_evidence_{config.get('cycle_id', 'v4')}",
        "status": "COMPLETED",
        "split": args.split,
        "role": args.role,
        "support": len(output_rows),
        "product_support": sum(row["scenario"] == "product" for row in output_rows),
        "dialogue_support": sum(row["scenario"] == "dialogue" for row in output_rows),
        "data_lock_sha256": data_lock,
        "prompt_sha256": prompt_lock,
        "generation_config_sha256": generation_lock,
        "result_sha256": canonical_json_sha256(output_rows),
        "first_attempt_json_compliance": sum(row["first_attempt_json_valid"] for row in output_rows) / len(output_rows),
        "correction_trigger_rate": sum(row["correction_triggered"] for row in output_rows) / len(output_rows),
        "mean_latency_ms": sum(row["latency_ms"] for row in output_rows) / len(output_rows),
        "peak_vram_mib": max(row["peak_vram_mib"] for row in output_rows),
        "final_consumed_once": args.split == "final",
        "fresh_test_used": False,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    _write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def load_final_after_marker(
    manifest_path: Path, marker_path: Path, marker: dict[str, Any], loader=load_jsonl
) -> list[dict[str, Any]]:
    _write_json(marker_path, marker)
    return loader(manifest_path)


def _generate(
    model: Any, processor: Any, torch: Any, record: dict[str, Any], bundle_dir: Path,
    max_new_tokens: int, previous: str | None = None,
) -> str:
    prompt = str(record["prompt"])
    content: list[dict[str, Any]] = []
    if record["scenario"] == "product":
        content.append({
            "type": "image",
            "image": str((bundle_dir / record["image_relative_path"]).resolve()),
        })
    else:
        prompt = f"{prompt}\n\nCASE:\n{record['dialogue']}"
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    if previous is not None:
        messages.extend([
            {"role": "assistant", "content": [{"type": "text", "text": previous}]},
            {"role": "user", "content": [{"type": "text", "text": CORRECTION_PROMPT}]},
        ])
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", truncation=False,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
    trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs["input_ids"], generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

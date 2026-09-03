"""Run one true-process-cold fixed-length Milvus Lite component performance profile."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_relevance_evidence import _build_milvus_store, _read_retrieval_archive
from scripts.run_vlm_semantic_evidence import PRODUCT_PROMPT, _load_model
from src.evaluation.relevance_evidence import file_sha256, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--retrieval-archive", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--role", required=True, choices=(
        "old_unified_adapter", "current_system_repair_checkpoint_87"
    ))
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    adapter_file = args.adapter_path / "adapter_model.safetensors"
    if not adapter_file.is_file() or file_sha256(adapter_file) != args.adapter_sha256:
        raise ValueError("adapter model is missing or SHA-256 mismatched")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_lock = json.loads(Path(config["pool"]["committed_lock"]).read_text(encoding="utf-8"))
    actual_lock = json.loads((args.bundle_dir / "bundle_lock.json").read_text(encoding="utf-8"))
    if actual_lock != expected_lock:
        raise ValueError("generated bundle lock differs from committed lock")
    if file_sha256(args.retrieval_archive) != config["formal_release_read_only"]["retrieval_archive_sha256"]:
        raise ValueError("formal retrieval archive SHA-256 mismatch")
    profiles = {row["profile_id"]: row for row in config["performance"]["profiles"]}
    if args.profile_id not in profiles:
        raise ValueError(f"unknown profile: {args.profile_id}")
    profile = profiles[args.profile_id]
    manifest = load_jsonl(args.bundle_dir / "vlm_manifest_weak_v3.jsonl")
    samples = {row["sample_id"]: row for row in manifest}
    sample = samples[profile["sample_id"]]
    if sample.get("scenario") != "product":
        raise ValueError("performance profile requires a product image sample")
    image_path = args.bundle_dir / sample["image_relative_path"]
    if file_sha256(image_path) != sample["image_sha256"]:
        raise ValueError("performance image SHA-256 mismatch")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, CLIPModel

    startup_started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_name = config["search"]["embedding_model"]
    clip_processor = AutoProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).to(device).eval()
    archive = _read_retrieval_archive(args.retrieval_archive)
    milvus_path = args.output.with_suffix(".milvus.db")
    store = _build_milvus_store(milvus_path, archive["vectors"], archive["metadata"], config)
    vlm_model, vlm_processor, torch = _load_model(args)
    startup_ms = (time.perf_counter() - startup_started) * 1000
    hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    max_new_tokens = int(profile["max_new_tokens"])
    image = Image.open(image_path).convert("RGB")
    rows: list[dict[str, Any]] = []

    cold = _run_once(
        clip_model, clip_processor, vlm_model, vlm_processor, torch,
        store, archive["metadata"], image, str(image_path.resolve()), device, max_new_tokens,
    )
    cold.update(_dimensions(args, profile, hardware, "cold"))
    cold["startup_ms"] = startup_ms
    cold["end_to_end_ms"] += startup_ms
    cold["cold_scope"] = "true_process_cold_including_model_and_index_startup"
    rows.append(cold)
    for _ in range(int(config["performance"]["warmup_repetitions"])):
        _run_once(
            clip_model, clip_processor, vlm_model, vlm_processor, torch,
            store, archive["metadata"], image, str(image_path.resolve()), device, max_new_tokens,
        )
    for _ in range(int(config["performance"]["steady_repetitions"])):
        row = _run_once(
            clip_model, clip_processor, vlm_model, vlm_processor, torch,
            store, archive["metadata"], image, str(image_path.resolve()), device, max_new_tokens,
        )
        row.update(_dimensions(args, profile, hardware, "steady"))
        row["startup_ms"] = 0.0
        row["cold_scope"] = None
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "completed",
        "role": args.role,
        "profile_id": args.profile_id,
        "support": len(rows),
        "cold_support": 1,
        "steady_support": len(rows) - 1,
        "transport": "component_milvus_lite",
        "production_sla_supported": False,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }, indent=2, sort_keys=True))


def _run_once(
    clip_model, clip_processor, vlm_model, vlm_processor, torch,
    store, metadata, image, vlm_image_path: str, device: str, max_new_tokens: int,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total_started = time.perf_counter()
    started = time.perf_counter()
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        embedding = clip_model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    query_vector = embedding[0].detach().float().cpu().numpy()
    clip_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    result = store.search(query_vector.astype(float).tolist(), top_k=10)
    milvus_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    hits = result[0] if result else []
    sorted(
        hits,
        key=lambda hit: -(
            float(hit.get("distance", 0.0))
            + 0.02 * float(hit.get("entity", {}).get("star_rating", 0.0)) / 5.0
        ),
    )
    rerank_ms = (time.perf_counter() - started) * 1000

    messages = [{"role": "user", "content": [
        {"type": "image", "image": vlm_image_path},
        {"type": "text", "text": PRODUCT_PROMPT},
    ]}]
    vlm_inputs = vlm_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
    )
    model_device = next(vlm_model.parameters()).device
    vlm_inputs = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in vlm_inputs.items()
    }
    started = time.perf_counter()
    with torch.inference_mode():
        generated = vlm_model.generate(
            **vlm_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    vlm_ms = (time.perf_counter() - started) * 1000
    input_tokens = int(vlm_inputs["input_ids"].shape[-1])
    output_tokens = int(generated.shape[-1] - input_tokens)
    return {
        "clip_encode_ms": clip_ms,
        "milvus_ms": milvus_ms,
        "rerank_ms": rerank_ms,
        "vlm_ms": vlm_ms,
        "end_to_end_ms": (time.perf_counter() - total_started) * 1000,
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        ),
        "input_token_count": input_tokens,
        "output_token_count": output_tokens,
        "failed": False,
    }


def _dimensions(
    args: argparse.Namespace, profile: dict[str, Any], hardware: str, phase: str
) -> dict[str, Any]:
    return {
        "role": args.role,
        "profile_id": profile["profile_id"],
        "sample_id": profile["sample_id"],
        "max_new_tokens": profile["max_new_tokens"],
        "phase": phase,
        "concurrency": 1,
        "transport": "component_milvus_lite",
        "milvus_deployment": "Milvus_Lite_local_file",
        "http_service": False,
        "hardware": hardware,
        "status": "MEASURED",
    }


if __name__ == "__main__":
    main()

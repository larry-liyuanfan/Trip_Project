"""Benchmark the locked CLIP -> Milvus -> rerank -> VLM evidence path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_relevance_evidence import _build_milvus_store, _read_retrieval_archive
from scripts.run_vlm_semantic_evidence import _load_model
from src.evaluation.relevance_evidence import file_sha256, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--retrieval-archive", type=Path, required=True)
    parser.add_argument("--role", required=True, choices=(
        "old_unified_adapter", "current_system_repair_checkpoint_87"
    ))
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    adapter_model = args.adapter_path / "adapter_model.safetensors"
    if not adapter_model.is_file():
        raise FileNotFoundError(f"missing adapter: {adapter_model}")
    if args.adapter_sha256 and file_sha256(adapter_model) != args.adapter_sha256:
        raise ValueError("adapter model SHA-256 mismatch")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_archive = config["formal_release_read_only"]["retrieval_archive_sha256"]
    if file_sha256(args.retrieval_archive) != expected_archive:
        raise ValueError("formal retrieval archive SHA-256 mismatch")
    query = load_jsonl(args.query_manifest)[1]
    image_path = args.asset_dir / query["image"]["relative_path"]
    archive = _read_retrieval_archive(args.retrieval_archive)

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, CLIPModel

    startup_started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_name = config["search"]["embedding_model"]
    clip_processor = AutoProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).to(device).eval()
    milvus_path = args.output.with_suffix(".milvus.db")
    if milvus_path.exists():
        raise FileExistsError(f"Milvus benchmark database exists: {milvus_path}")
    store = _build_milvus_store(
        milvus_path,
        archive["vectors"],
        archive["metadata"],
        config,
    )
    vlm_model, vlm_processor, torch = _load_model(args)
    startup_ms = (time.perf_counter() - startup_started) * 1000
    hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    repeats = config["performance"]
    rows = []

    cold = _run_once(
        clip_model, clip_processor, vlm_model, vlm_processor, torch,
        store, image_path, archive["metadata"], device, args.max_new_tokens,
    )
    cold["phase"] = "cold"
    cold["startup_ms"] = startup_ms
    cold["end_to_end_ms"] += startup_ms
    cold["hardware"] = hardware
    cold["role"] = args.role
    rows.append(cold)

    for _ in range(int(repeats["warmup_repetitions"])):
        _run_once(
            clip_model, clip_processor, vlm_model, vlm_processor, torch,
            store, image_path, archive["metadata"], device, args.max_new_tokens,
        )
    for _ in range(int(repeats["steady_repetitions"])):
        row = _run_once(
            clip_model, clip_processor, vlm_model, vlm_processor, torch,
            store, image_path, archive["metadata"], device, args.max_new_tokens,
        )
        row["phase"] = "steady"
        row["startup_ms"] = 0.0
        row["hardware"] = hardware
        row["role"] = args.role
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "completed",
        "role": args.role,
        "cold_support": 1,
        "steady_support": int(repeats["steady_repetitions"]),
        "startup_ms": startup_ms,
        "hardware": hardware,
    }, ensure_ascii=False, indent=2, sort_keys=True))


def _run_once(
    clip_model,
    clip_processor,
    vlm_model,
    vlm_processor,
    torch,
    store,
    image_path,
    metadata,
    device,
    max_new_tokens,
):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total_started = time.perf_counter()
    image = Image.open(image_path).convert("RGB")

    started = time.perf_counter()
    clip_inputs = clip_processor(images=image, return_tensors="pt")
    clip_inputs = {key: value.to(device) for key, value in clip_inputs.items()}
    with torch.inference_mode():
        vector = clip_model.get_image_features(**clip_inputs)
        vector = vector / vector.norm(dim=-1, keepdim=True)
    query_vector = vector[0].detach().float().cpu().numpy()
    clip_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    result = store.search(query_vector.astype(float).tolist(), top_k=10)
    milvus_ms = (time.perf_counter() - started) * 1000
    hits = result[0] if result else []

    started = time.perf_counter()
    reranked = sorted(
        hits,
        key=lambda hit: -(
            float(hit.get("distance", 0.0))
            + 0.15 * (hit.get("entity", {}).get("business_category") == "hotel")
        ),
    )
    top_metadata = [
        {
            "image_id": hit.get("entity", {}).get("image_id"),
            "business_category": hit.get("entity", {}).get("business_category"),
            "city": hit.get("entity", {}).get("city"),
            "price_range": hit.get("entity", {}).get("price_range"),
        }
        for hit in reranked[:3]
    ]
    rerank_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    prompt = (
        "Return only JSON with keys business_category, visible_facilities, price_range, "
        "retrieved_image_ids. Use visible image evidence; price_range must be unknown unless "
        "a visible price proves it. Candidate metadata: " + json.dumps(top_metadata)
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path.resolve())},
        {"type": "text", "text": prompt},
    ]}]
    inputs = vlm_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
    )
    vlm_device = next(vlm_model.parameters()).device
    inputs = {key: value.to(vlm_device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    if generated.shape[0] != 1:
        raise RuntimeError("unexpected VLM batch support")
    vlm_ms = (time.perf_counter() - started) * 1000
    total_ms = (time.perf_counter() - total_started) * 1000
    return {
        "query_id": "q02_hotel_visual",
        "clip_encode_ms": clip_ms,
        "milvus_ms": milvus_ms,
        "rerank_ms": rerank_ms,
        "vlm_ms": vlm_ms,
        "end_to_end_ms": total_ms,
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        ),
        "failed": False,
    }


if __name__ == "__main__":
    from PIL import Image
    main()

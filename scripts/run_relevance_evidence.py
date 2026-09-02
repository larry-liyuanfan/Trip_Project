"""Validate, run, and score the development relevance-evidence protocol."""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import statistics
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.relevance_evidence import (
    SEARCH_METHODS,
    canonical_json_sha256,
    compare_performance,
    file_sha256,
    load_jsonl,
    score_ann_fidelity,
    score_search_results,
    score_vlm_comparison,
    summarize_performance,
    validate_annotation_protocol,
    validate_asset_source_registry,
    validate_query_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/evidence_enhancement_v1.json"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-pool")
    validate.add_argument("--asset-dir", type=Path)
    validate.add_argument("--retrieval-archive", type=Path)
    validate.add_argument("--release-manifest", type=Path)
    validate.add_argument("--output", type=Path)

    run_search = subparsers.add_parser("run-search")
    run_search.add_argument("--asset-dir", type=Path, required=True)
    run_search.add_argument("--retrieval-archive", type=Path, required=True)
    run_search.add_argument("--output", type=Path, required=True)

    score_search = subparsers.add_parser("score-search")
    score_search.add_argument("--results", type=Path, required=True)
    score_search.add_argument("--output", type=Path, required=True)

    score_vlm = subparsers.add_parser("score-vlm")
    score_vlm.add_argument("--results", type=Path, required=True)
    score_vlm.add_argument("--output", type=Path, required=True)

    score_perf = subparsers.add_parser("score-performance")
    score_perf.add_argument("--results", type=Path, required=True)
    score_perf.add_argument("--output", type=Path, required=True)

    compare_perf = subparsers.add_parser("compare-performance")
    compare_perf.add_argument("--candidate", type=Path, required=True)
    compare_perf.add_argument("--baseline", type=Path, required=True)
    compare_perf.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    config = _load_json(args.config)
    if args.command == "validate-pool":
        report = validate_pool(config, args)
        if args.output:
            _write_json_exclusive(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "run-search":
        report = run_search_evaluation(config, args.asset_dir, args.retrieval_archive, args.output)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "score-search":
        queries, annotations = _load_protocol(config)
        results = load_jsonl(args.results)
        report = score_search_results(queries, annotations, results)
        _write_json_exclusive(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "score-vlm":
        report = score_vlm_comparison(load_jsonl(args.results))
        _write_json_exclusive(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "score-performance":
        report = summarize_performance(
            load_jsonl(args.results), config["performance"]["fixed_gates"]
        )
        _write_json_exclusive(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        report = compare_performance(
            load_jsonl(args.candidate),
            load_jsonl(args.baseline),
            config["performance"]["fixed_gates"],
        )
        _write_json_exclusive(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def validate_pool(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    queries, annotations = _load_protocol(config)
    query_report = validate_query_manifest(queries, args.asset_dir)
    annotation_report = validate_annotation_protocol(queries, annotations)
    registry = load_jsonl(Path(config["search"]["query_asset_source_registry"]))
    registry_report = validate_asset_source_registry(queries, registry, args.asset_dir)
    release_report: dict[str, Any] = {"status": "NOT_RUN_NOT_PROVIDED"}
    if args.retrieval_archive:
        expected = config["formal_release_read_only"]["retrieval_archive_sha256"]
        actual = file_sha256(args.retrieval_archive)
        if actual != expected:
            raise ValueError("formal retrieval archive SHA-256 mismatch")
        members = _read_retrieval_archive(args.retrieval_archive)
        historical = members["benchmark"]
        release_report = {
            "status": "PASS",
            "archive_sha256": actual,
            "member_sha256": members["member_sha256"],
            "historical_benchmark": {
                "scope": "ann_vs_exact_vector_query_only",
                "query_support": historical["search"]["query_count"],
                "recall_at_10": historical["search"]["recall_at_k"],
                "mean_vector_query_latency_ms": historical["search"]["mean_latency_ms"],
                "p95_vector_query_latency_ms": historical["search"]["p95_latency_ms"],
                "business_relevance_supported": False,
                "end_to_end_latency_supported": False,
            },
        }
    if args.release_manifest:
        expected = config["formal_release_read_only"]["release_manifest_sha256"]
        actual = file_sha256(args.release_manifest)
        if actual != expected:
            raise ValueError("formal release manifest SHA-256 mismatch")
        release_report["release_manifest_sha256"] = actual
    return {
        "schema_version": "evidence_enhancement_validation_v1",
        "status": "PASS",
        "evidence_class": config["evidence_class"],
        "query_pool": query_report,
        "annotation_protocol": annotation_report,
        "asset_source_registry": registry_report,
        "formal_release_read_only": release_report,
        "historical_aggregate_0_780639_audit": {
            "status": "EVIDENCE_GAP_RAW_SAMPLE_OUTPUTS_NOT_IN_LOCAL_HANDOFF",
            "reported_value": 0.780639,
            "recomputed": False,
            "fresh_test_120_reused": False,
        },
    }


def run_search_evaluation(
    config: dict[str, Any],
    asset_dir: Path,
    retrieval_archive: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"search output already exists: {output}")
    expected_archive = config["formal_release_read_only"]["retrieval_archive_sha256"]
    if file_sha256(retrieval_archive) != expected_archive:
        raise ValueError("formal retrieval archive SHA-256 mismatch")
    queries, annotations = _load_protocol(config)
    validate_query_manifest(queries, asset_dir)
    validate_annotation_protocol(queries, annotations)
    annotations_by_id = {item["query_id"]: item for item in annotations}
    archive = _read_retrieval_archive(retrieval_archive)

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config["search"]["embedding_model"]
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    vectors = archive["vectors"]
    metadata = archive["metadata"]
    if len(vectors) != len(metadata):
        raise ValueError("formal vectors and metadata have different supports")

    milvus_path = output.with_suffix(".milvus.db")
    if milvus_path.exists():
        raise FileExistsError(f"Milvus evidence database already exists: {milvus_path}")
    store = _build_milvus_store(milvus_path, vectors, metadata, config)
    top_k = int(config["search"]["top_k"])
    candidate_pool = int(config["search"]["candidate_pool"])
    threshold = float(config["search"]["no_result_similarity_threshold"])
    rows: list[dict[str, Any]] = []
    ann_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for query in queries:
        query_started = time.perf_counter()
        image = Image.open(asset_dir / query["image"]["relative_path"]).convert("RGB")
        stage_started = time.perf_counter()
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            embedding = model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        query_vector = embedding[0].detach().float().cpu().numpy()
        clip_ms = (time.perf_counter() - stage_started) * 1000

        similarities = vectors @ query_vector
        exact_indices = np.argsort(-similarities)
        exact_hits = _hits_from_indices(exact_indices[:top_k], similarities, metadata)

        stage_started = time.perf_counter()
        milvus_result = store.search(query_vector.astype(float).tolist(), top_k=top_k)
        milvus_ms = (time.perf_counter() - stage_started) * 1000
        milvus_hits = _milvus_hits(milvus_result)

        supported_filters = {
            key: value
            for key, value in query["requested_filters"].items()
            if key in config["search"]["supported_filters"]
        }
        eligible = [
            index
            for index, item in enumerate(metadata)
            if all(_norm(item.get(key)) == _norm(value) for key, value in supported_filters.items())
        ]
        filtered_indices = sorted(eligible, key=lambda index: -float(similarities[index]))
        filtered_hits = _hits_from_indices(filtered_indices[:top_k], similarities, metadata)

        stage_started = time.perf_counter()
        candidates = exact_indices[:candidate_pool]
        rerank_scores = {
            int(index): float(similarities[index])
            + sum(
                0.5
                for key, value in supported_filters.items()
                if _norm(metadata[int(index)].get(key)) == _norm(value)
            )
            - sum(
                0.5
                for key, value in supported_filters.items()
                if _norm(metadata[int(index)].get(key)) != _norm(value)
            )
            for index in candidates
        }
        reranked = sorted(candidates, key=lambda index: -rerank_scores[int(index)])
        rerank_hits = _hits_from_indices(reranked[:top_k], similarities, metadata)
        rerank_ms = (time.perf_counter() - stage_started) * 1000

        relevant_total = sum(
            _metadata_grade(item, annotations_by_id[query["query_id"]]) >= 2
            for item in metadata
        )
        unsupported = query["unsupported_constraints"]
        methods = {
            "clip_exact": _method_result(exact_hits, threshold, relevant_total, unsupported),
            "clip_milvus": _method_result(milvus_hits, threshold, relevant_total, unsupported),
            "structured_filter_clip": _method_result(filtered_hits, threshold, relevant_total, unsupported),
            "lightweight_rerank": _method_result(rerank_hits, threshold, relevant_total, unsupported),
        }
        rows.append({"query_id": query["query_id"], "methods": methods})
        ann_rows.append(
            {
                "query_id": query["query_id"],
                "exact_ids": [item["image_id"] for item in exact_hits],
                "ann_ids": [item["image_id"] for item in milvus_hits],
            }
        )
        timing_rows.append(
            {
                "query_id": query["query_id"],
                "clip_encode_ms": clip_ms,
                "milvus_ms": milvus_ms,
                "rerank_ms": rerank_ms,
                "search_path_ms": (time.perf_counter() - query_started) * 1000,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl_exclusive(output, rows)
    semantic = score_search_results(queries, annotations, rows)
    ann = score_ann_fidelity(ann_rows, top_k=top_k)
    summary = {
        "status": "completed",
        "scope": "development_weak_labels_not_human_ground_truth",
        "device": device,
        "embedding_model": model_name,
        "query_support": len(rows),
        "ann_fidelity": ann,
        "business_semantics": {
            method: {
                key: value
                for key, value in semantic["methods"][method].items()
                if key not in {"per_query"}
            }
            for method in SEARCH_METHODS
        },
        "search_stage_timing": {
            key: _stats([row[key] for row in timing_rows])
            for key in ("clip_encode_ms", "milvus_ms", "rerank_ms", "search_path_ms")
        },
        "end_to_end_performance": "NOT_RUN_MISSING_VLM_STAGE",
        "historical_2_41ms_comparability": "NOT_COMPARABLE_VECTOR_QUERY_ONLY",
        "result_sha256": canonical_json_sha256(rows),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        },
    }
    _write_json_exclusive(output.with_suffix(".summary.json"), summary)
    _write_json_exclusive(output.with_suffix(".semantic.json"), semantic)
    return {"summary": summary, "semantic": semantic}


def _build_milvus_store(path: Path, vectors: Any, metadata: list[dict[str, Any]], config: dict[str, Any]):
    from src.retrieval.milvus_vectors import FILTER_FIELDS, OTAMilvusVectorStore

    index = config["search"]["milvus"]
    store_config = {
        "connection": {"uri": str(path), "timeout_seconds": 120},
        "collection": {
            "name": "ota_business_image_vector",
            "vector_dimension": 512,
            "embedding_model": config["search"]["embedding_model"],
            "consistency_level": "Strong",
        },
        "index": {
            **index,
            "scalar_fields": sorted(FILTER_FIELDS),
        },
    }
    store = OTAMilvusVectorStore(store_config)
    store.create_collection()
    rows = []
    for index_value, metadata_row in enumerate(metadata):
        row = {key: metadata_row[key] for key in FILTER_FIELDS}
        row["multimodal_vector"] = vectors[index_value].astype(float).tolist()
        rows.append(row)
    store.batch_insert(rows)
    store.client.flush(collection_name=store.collection)
    store.build_indexes()
    store.client.load_collection(collection_name=store.collection)
    if store.count_visible_entities() != len(rows):
        raise RuntimeError("Milvus visible support does not match the formal vector lock")
    return store


def _read_retrieval_archive(path: Path) -> dict[str, Any]:
    import numpy as np

    names = {
        "metadata": "retrieval/clip_metadata_1000.jsonl",
        "vectors": "retrieval/clip_vectors_1000.npz",
        "benchmark": "retrieval/milvus_benchmark_1000.json",
    }
    with tarfile.open(path, "r:gz") as archive:
        found = set(archive.getnames())
        if found != set(names.values()):
            raise ValueError(f"unexpected retrieval archive members: {sorted(found)}")
        payloads = {
            key: archive.extractfile(member).read()
            for key, member in names.items()
        }
    vector_archive = np.load(io.BytesIO(payloads["vectors"]))
    vector_key = "multimodal_vector"
    if vector_key not in vector_archive.files:
        raise ValueError("formal vector archive misses multimodal_vector")
    return {
        "metadata": [json.loads(line) for line in payloads["metadata"].decode("utf-8").splitlines() if line],
        "vectors": vector_archive[vector_key],
        "benchmark": json.loads(payloads["benchmark"]),
        "member_sha256": {
            names[key]: __import__("hashlib").sha256(payload).hexdigest()
            for key, payload in payloads.items()
        },
    }


def _load_protocol(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        load_jsonl(Path(config["search"]["query_manifest"])),
        load_jsonl(Path(config["search"]["weak_annotations"])),
    )


def _hits_from_indices(indices: Any, similarities: Any, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**metadata[int(index)], "similarity": float(similarities[int(index)])}
        for index in indices
    ]


def _milvus_hits(result: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {**hit.get("entity", {}), "similarity": float(hit.get("distance", 0.0))}
        for hit in (result[0] if result else [])
    ]


def _method_result(hits: list[dict[str, Any]], threshold: float, relevant_total: int, unsupported: list[str]) -> dict[str, Any]:
    no_result = not hits or max(float(item.get("similarity", -1.0)) for item in hits) < threshold
    return {
        "hits": [] if no_result else hits,
        "no_result": no_result,
        "relevant_total": relevant_total,
        "unsupported_constraints_unapplied": unsupported,
    }


def _metadata_grade(hit: dict[str, Any], annotation: dict[str, Any]) -> int:
    rules = annotation["grade_rules"]
    category = _norm(rules.get("target_business_category"))
    hit_category = _norm(hit.get("business_category"))
    excluded = {_norm(item) for item in rules.get("excluded_business_categories", [])}
    if hit_category in excluded:
        return 0
    required_match = all(
        _norm(hit.get(key)) == _norm(value)
        for key, value in rules.get("required_metadata", {}).items()
    )
    if category and hit_category == category and required_match:
        return 3
    if category and hit_category == category:
        return 2
    return 1 if hit_category else 0


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.999999) - 1))
    return {
        "support": len(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_exclusive(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _norm(value: Any) -> str:
    return " ".join(value.strip().casefold().split()) if isinstance(value, str) else ""


if __name__ == "__main__":
    main()

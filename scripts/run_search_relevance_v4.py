"""Train a CLIP business guard, select on development, and consume final once."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_relevance_evidence import (
    _build_milvus_store,
    _hits_from_indices,
    _metadata_grade,
    _method_result,
    _milvus_hits,
    _read_retrieval_archive,
)
from src.evaluation.exploration_v4 import (
    SEARCH_V4_METHODS,
    apply_search_v4_gates,
    validate_three_way_isolation,
)
from src.evaluation.relevance_evidence import (
    canonical_json_sha256,
    file_sha256,
    load_jsonl,
    score_ann_fidelity,
    score_search_results,
    validate_annotation_protocol,
    validate_query_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--retrieval-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_lock = json.loads(Path(config["pool"]["committed_lock"]).read_text(encoding="utf-8"))
    actual_lock = json.loads((args.bundle_dir / "bundle_lock.json").read_text(encoding="utf-8"))
    if actual_lock != expected_lock:
        raise ValueError("generated bundle lock differs from committed lock")
    if file_sha256(args.retrieval_archive) != config["formal_release_read_only"]["retrieval_archive_sha256"]:
        raise ValueError("formal retrieval archive SHA-256 mismatch")

    training, training_annotations = _load_split(args.bundle_dir, expected_lock, "training")
    development, development_annotations = _load_split(args.bundle_dir, expected_lock, "development")
    validation = {
        "training_queries": validate_query_manifest(training, args.bundle_dir),
        "training_annotations": validate_annotation_protocol(training, training_annotations),
        "development_queries": validate_query_manifest(development, args.bundle_dir),
        "development_annotations": validate_annotation_protocol(development, development_annotations),
        "training_development_isolation": _pair_isolation(training, development),
    }
    archive = _read_retrieval_archive(args.retrieval_archive)
    model_bundle = _load_clip_and_store(config, archive, args.output_dir / "formal_index.milvus.db")
    prepared_training = _prepare(
        training, training_annotations, args.bundle_dir, archive, model_bundle, config
    )
    centroids = _fit_business_guard(prepared_training, training_annotations)
    prepared_development = _prepare(
        development, development_annotations, args.bundle_dir, archive, model_bundle, config
    )

    grid = config["search"]["business_guard"]
    objective_definition = grid["selection_objective"]
    candidates: list[dict[str, Any]] = []
    for threshold, star_weight in itertools.product(
        grid["development_threshold_grid"], grid["star_rating_weight_grid"]
    ):
        candidate = {"business_guard_threshold": float(threshold), "star_rating_weight": float(star_weight)}
        rows, ann_rows = _evaluate(prepared_development, candidate, centroids, config)
        report = score_search_results(
            development, development_annotations, rows, methods=SEARCH_V4_METHODS
        )
        metrics = report["methods"]["hard_filter_clip_business_guard"]
        candidates.append({
            "configuration": candidate,
            "objective": _objective(metrics, objective_definition),
            "candidate_metrics": _without_per_query(metrics),
            "ann_fidelity": score_ann_fidelity(ann_rows, top_k=int(config["search"]["top_k"])),
        })
    selected = sorted(
        candidates,
        key=lambda row: (
            -float(row["objective"]),
            float(row["configuration"]["star_rating_weight"]),
            float(row["configuration"]["business_guard_threshold"]),
        ),
    )[0]
    development_rows, development_ann_rows = _evaluate(
        prepared_development, selected["configuration"], centroids, config
    )
    development_report = score_search_results(
        development, development_annotations, development_rows, methods=SEARCH_V4_METHODS
    )
    development_ann = score_ann_fidelity(
        development_ann_rows, top_k=int(config["search"]["top_k"])
    )
    development_gate = apply_search_v4_gates(
        development_report, development_ann, config["search"]["development_gates"]
    )
    selection = {
        "schema_version": "search_relevance_v4_development_selection",
        "selection_data": "development_only",
        "training_use": "business_guard_centroids_only",
        "final_read_for_selection": False,
        "selected": selected,
        "candidate_support": len(candidates),
        "objective_definition": objective_definition,
        "development_gate": development_gate,
        "training_query_lock": canonical_json_sha256(training),
        "development_query_lock": canonical_json_sha256(development),
        "centroid_sha256": canonical_json_sha256({
            key: [float(value) for value in vector.tolist()] for key, vector in centroids.items()
        }),
    }
    _write_json(args.output_dir / "development_candidates.json", candidates)
    _write_jsonl(args.output_dir / "development_results.jsonl", development_rows)
    _write_json(args.output_dir / "development_metrics.json", development_report)
    _write_json(args.output_dir / "selection.json", selection)
    if development_gate["status"] != "PASS":
        summary = {
            "schema_version": "search_relevance_evidence_v4",
            "status": "NEGATIVE_EXPERIMENT_DEVELOPMENT_GATE_FAILED",
            "final_consumed": False,
            "fresh_test_used": False,
            "selected_configuration": selected["configuration"],
            "development": {"metrics": _method_summaries(development_report), "fixed_gates": development_gate},
            "runtime": _runtime(model_bundle),
        }
        _write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    marker_path = args.output_dir / "final_consumption_marker.json"
    marker = {
        "schema_version": "search_relevance_v4_final_consumption",
        "selection_sha256": canonical_json_sha256(selection),
        "committed_final_lock": expected_lock["search"]["final"],
        "single_consumption_policy": "exclusive_marker_written_before_first_final_manifest_or_annotation_open",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    final, final_annotations = load_final_after_marker(args.bundle_dir, marker_path, marker)
    _verify_split_lock(final, final_annotations, args.bundle_dir, expected_lock, "final")
    validation.update({
        "final_queries": validate_query_manifest(final, args.bundle_dir),
        "final_annotations": validate_annotation_protocol(final, final_annotations),
        "three_way_isolation": validate_three_way_isolation(
            {"training": training, "development": development, "final": final}, record_kind="search"
        ),
    })
    prepared_final = _prepare(final, final_annotations, args.bundle_dir, archive, model_bundle, config)
    final_rows, final_ann_rows = _evaluate(
        prepared_final, selected["configuration"], centroids, config
    )
    final_report = score_search_results(final, final_annotations, final_rows, methods=SEARCH_V4_METHODS)
    final_ann = score_ann_fidelity(final_ann_rows, top_k=int(config["search"]["top_k"]))
    final_gate = apply_search_v4_gates(final_report, final_ann, config["search"]["final_gates"])
    _write_jsonl(args.output_dir / "final_results.jsonl", final_rows)
    _write_json(args.output_dir / "final_metrics.json", final_report)
    summary = {
        "schema_version": "search_relevance_evidence_v4",
        "status": "COMPLETED",
        "evidence_class": config["evidence_class"],
        "gate_class": config["gate_class"],
        "human_annotation_support": 0,
        "fresh_test_used": False,
        "selected_configuration": selected["configuration"],
        "validation": validation,
        "development": {
            "query_support": len(development),
            "metrics": _method_summaries(development_report),
            "ann_fidelity": development_ann,
            "fixed_gates": development_gate,
        },
        "final": {
            "query_support": len(final),
            "consumed_once": True,
            "metrics": _method_summaries(final_report),
            "ann_fidelity": final_ann,
            "fixed_gates": final_gate,
        },
        "runtime": _runtime(model_bundle),
        "final_result_sha256": canonical_json_sha256(final_rows),
        "promotion_eligible_as_human_ground_truth": False,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def load_final_after_marker(
    bundle_dir: Path, marker_path: Path, marker: dict[str, Any], loader=load_jsonl
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _write_json(marker_path, marker)
    return (
        loader(bundle_dir / "search_final_manifest.jsonl"),
        loader(bundle_dir / "search_final_annotations.jsonl"),
    )


def _load_split(
    bundle_dir: Path, expected_lock: dict[str, Any], split: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries = load_jsonl(bundle_dir / f"search_{split}_manifest.jsonl")
    annotations = load_jsonl(bundle_dir / f"search_{split}_annotations.jsonl")
    _verify_split_lock(queries, annotations, bundle_dir, expected_lock, split)
    return queries, annotations


def _verify_split_lock(
    queries: list[dict[str, Any]], annotations: list[dict[str, Any]], bundle_dir: Path,
    expected_lock: dict[str, Any], split: str,
) -> None:
    query_path = bundle_dir / f"search_{split}_manifest.jsonl"
    annotation_path = bundle_dir / f"search_{split}_annotations.jsonl"
    actual = {
        "query_support": len(queries),
        "query_manifest_canonical_sha256": canonical_json_sha256(queries),
        "query_manifest_file_sha256": file_sha256(query_path),
        "annotation_canonical_sha256": canonical_json_sha256(annotations),
        "annotation_file_sha256": file_sha256(annotation_path),
    }
    if actual != expected_lock["search"][split]:
        raise ValueError(f"search {split} split differs from the committed lock")


def _pair_isolation(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    def values(rows: list[dict[str, Any]], path: tuple[str, ...]) -> set[str]:
        output = set()
        for row in rows:
            value: Any = row
            for part in path:
                value = value.get(part, {}) if isinstance(value, dict) else None
            output.add(str(value))
        return output
    overlaps = {
        "query_id": sorted(values(left, ("query_id",)) & values(right, ("query_id",))),
        "source_id": sorted(values(left, ("source", "source_id")) & values(right, ("source", "source_id"))),
        "image_sha256": sorted(values(left, ("image", "sha256")) & values(right, ("image", "sha256"))),
    }
    if any(overlaps.values()):
        raise ValueError(f"training/development identity overlap: {overlaps}")
    return {"status": "PASS", "overlaps": overlaps}


def _load_clip_and_store(config: dict[str, Any], archive: dict[str, Any], path: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = config["search"]["embedding_model"]
    processor = AutoProcessor.from_pretrained(name)
    model = CLIPModel.from_pretrained(name).to(device).eval()
    return {
        "torch": torch,
        "processor": processor,
        "model": model,
        "store": _build_milvus_store(path, archive["vectors"], archive["metadata"], config),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _prepare(
    queries: list[dict[str, Any]], annotations: list[dict[str, Any]], bundle_dir: Path,
    archive: dict[str, Any], model_bundle: dict[str, Any], config: dict[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np
    from PIL import Image

    annotation_index = {row["query_id"]: row for row in annotations}
    vectors, metadata = archive["vectors"], archive["metadata"]
    top_k = int(config["search"]["top_k"])
    prepared = []
    for query in queries:
        image = Image.open(bundle_dir / query["image"]["relative_path"]).convert("RGB")
        inputs = model_bundle["processor"](images=image, return_tensors="pt")
        inputs = {key: value.to(model_bundle["device"]) for key, value in inputs.items()}
        with model_bundle["torch"].inference_mode():
            embedding = model_bundle["model"].get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        query_vector = embedding[0].detach().float().cpu().numpy()
        similarities = vectors @ query_vector
        exact_indices = np.argsort(-similarities)
        filters = {
            key: value for key, value in query["requested_filters"].items()
            if key in config["search"]["supported_filters"]
        }
        eligible = [
            index for index, row in enumerate(metadata)
            if all(_norm(row.get(key)) == _norm(value) for key, value in filters.items())
        ]
        prepared.append({
            "query": query,
            "query_vector": query_vector,
            "similarities": similarities,
            "metadata": metadata,
            "eligible": eligible,
            "exact_hits": _hits_from_indices(exact_indices[:top_k], similarities, metadata),
            "milvus_hits": _milvus_hits(model_bundle["store"].search(query_vector.astype(float).tolist(), top_k=top_k)),
            "relevant_total": sum(
                _metadata_grade(row, annotation_index[query["query_id"]]) >= 2 for row in metadata
            ),
        })
    return prepared


def _fit_business_guard(
    prepared: list[dict[str, Any]], annotations: list[dict[str, Any]]
) -> dict[str, Any]:
    import numpy as np

    labels = {row["query_id"]: row.get("business_guard_label") for row in annotations}
    groups = {
        name: [item["query_vector"] for item in prepared if labels[item["query"]["query_id"]] == name]
        for name in ("business", "non_business")
    }
    if any(not values for values in groups.values()):
        raise ValueError("business guard training requires both classes")
    centroids = {}
    for name, values in groups.items():
        centroid = np.mean(np.stack(values), axis=0)
        centroids[name] = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    return centroids


def _evaluate(
    prepared: list[dict[str, Any]], candidate: dict[str, float], centroids: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    top_k = int(config["search"]["top_k"])
    threshold = float(candidate["business_guard_threshold"])
    star_weight = float(candidate["star_rating_weight"])
    rows, ann_rows = [], []
    for item in prepared:
        similarities, metadata = item["similarities"], item["metadata"]
        structured = sorted(item["eligible"], key=lambda index: -float(similarities[index]))
        reranked = sorted(
            item["eligible"],
            key=lambda index: -(
                float(similarities[index])
                + star_weight * float(metadata[index].get("star_rating", 0.0)) / 5.0
            ),
        )
        structured_hits = _hits_from_indices(structured[:top_k], similarities, metadata)
        reranked_hits = _hits_from_indices(reranked[:top_k], similarities, metadata)
        query = item["query"]
        guard_score = float(item["query_vector"] @ centroids["business"] - item["query_vector"] @ centroids["non_business"])
        guard_reject = not query["requested_filters"] and guard_score < threshold
        candidate_hits = [] if guard_reject else reranked_hits
        relevant_total, unsupported = item["relevant_total"], query["unsupported_constraints"]
        methods = {
            "clip_exact": _method_result(item["exact_hits"], -2.0, relevant_total, unsupported),
            "clip_milvus": _method_result(item["milvus_hits"], -2.0, relevant_total, unsupported),
            "structured_filter_clip": _method_result(structured_hits, -2.0, relevant_total, unsupported),
            "hard_filter_light_rerank": _method_result(reranked_hits, -2.0, relevant_total, unsupported),
            "hard_filter_clip_business_guard": _method_result(candidate_hits, -2.0, relevant_total, unsupported),
        }
        methods["hard_filter_clip_business_guard"]["business_guard_score"] = guard_score
        methods["hard_filter_clip_business_guard"]["business_guard_rejected"] = guard_reject
        rows.append({"query_id": query["query_id"], "methods": methods})
        ann_rows.append({
            "query_id": query["query_id"],
            "exact_ids": [hit["image_id"] for hit in item["exact_hits"]],
            "ann_ids": [hit["image_id"] for hit in item["milvus_hits"]],
        })
    return rows, ann_rows


def _objective(metrics: dict[str, Any], definition: dict[str, Any]) -> float:
    no_result = metrics.get("slices", {}).get("no_result", {}).get("no_result_accuracy")
    hard_filter = metrics.get("slices", {}).get("hard_filter_before_rerank", {})
    return (
        _number(hard_filter.get("ndcg_at_10")) * float(definition["ndcg_at_10_weight"])
        + _number(no_result) * float(definition["no_result_accuracy_weight"])
        + _number(hard_filter.get("filter_correctness")) * float(definition["filter_correctness_weight"])
    )


def _method_summaries(report: dict[str, Any]) -> dict[str, Any]:
    return {name: _without_per_query(value) for name, value in report["methods"].items()}


def _without_per_query(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "per_query"}


def _runtime(model_bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "device": model_bundle["device"],
        "gpu": model_bundle["gpu"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "milvus_scope": "Milvus_Lite_for_ANN_fidelity_only_not_service_benchmark",
    }


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _norm(value: Any) -> str:
    return " ".join(value.strip().casefold().split()) if isinstance(value, str) else ""


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

"""Calibrate on one locked synthetic split, then consume the holdout exactly once."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time
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
from src.evaluation.evidence_v2 import (
    SEARCH_V2_METHODS,
    apply_search_v2_gates,
    select_calibration_configuration,
    validate_calibration_holdout_isolation,
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
    _verify_bundle_lock(config, args.bundle_dir)
    if file_sha256(args.retrieval_archive) != config["formal_release_read_only"]["retrieval_archive_sha256"]:
        raise ValueError("formal retrieval archive SHA-256 mismatch")

    calibration = load_jsonl(args.bundle_dir / "search_calibration_manifest.jsonl")
    calibration_annotations = load_jsonl(args.bundle_dir / "search_calibration_annotations.jsonl")
    holdout = load_jsonl(args.bundle_dir / "search_holdout_manifest.jsonl")
    holdout_annotations = load_jsonl(args.bundle_dir / "search_holdout_annotations.jsonl")
    validation = {
        "calibration_queries": validate_query_manifest(calibration, args.bundle_dir),
        "calibration_annotations": validate_annotation_protocol(calibration, calibration_annotations),
        "holdout_queries": validate_query_manifest(holdout, args.bundle_dir),
        "holdout_annotations": validate_annotation_protocol(holdout, holdout_annotations),
        "split_isolation": validate_calibration_holdout_isolation(calibration, holdout),
    }
    archive = _read_retrieval_archive(args.retrieval_archive)
    model_bundle = _load_clip_and_store(config, archive, args.output_dir / "formal_index.milvus.db")

    prepared_calibration = _prepare_queries(
        calibration, calibration_annotations, args.bundle_dir, archive, model_bundle, config
    )
    candidate_reports: list[dict[str, Any]] = []
    grid = config["search"]["calibration_grid"]
    objective_weights = config["search"]["calibration_objective"]
    for threshold, star_weight in itertools.product(
        grid["no_result_similarity_threshold"], grid["star_rating_weight"]
    ):
        candidate_config = {
            "no_result_similarity_threshold": float(threshold),
            "star_rating_weight": float(star_weight),
        }
        rows, ann_rows = _evaluate_prepared(prepared_calibration, candidate_config, config)
        metrics = score_search_results(
            calibration, calibration_annotations, rows, methods=SEARCH_V2_METHODS
        )
        method = metrics["methods"]["hard_filter_light_rerank"]
        objective = (
            _or_zero(method.get("ndcg_at_10"))
            * float(objective_weights["hard_filter_light_rerank_ndcg_at_10_weight"])
            + _or_zero(method.get("no_result_accuracy"))
            * float(objective_weights["hard_filter_light_rerank_no_result_accuracy_weight"])
            + _or_zero(method.get("filter_correctness"))
            * float(objective_weights["hard_filter_light_rerank_filter_correctness_weight"])
        )
        candidate_reports.append({
            "configuration": candidate_config,
            "objective": objective,
            "hard_filter_light_rerank_metrics": _without_per_query(method),
            "ann_fidelity": score_ann_fidelity(ann_rows, top_k=int(config["search"]["top_k"])),
        })
    selected = select_calibration_configuration(candidate_reports)
    selected_rows, selected_ann_rows = _evaluate_prepared(
        prepared_calibration, selected["configuration"], config
    )
    calibration_report = score_search_results(
        calibration, calibration_annotations, selected_rows, methods=SEARCH_V2_METHODS
    )
    selection_record = {
        "schema_version": "search_relevance_v2_calibration_selection",
        "selected": selected,
        "candidate_support": len(candidate_reports),
        "selection_data": "calibration_only",
        "holdout_read_for_selection": False,
        "calibration_query_lock": canonical_json_sha256(calibration),
        "calibration_annotation_lock": canonical_json_sha256(calibration_annotations),
        "objective_definition": objective_weights,
    }
    _write_json(args.output_dir / "calibration_candidates.json", candidate_reports)
    _write_jsonl(args.output_dir / "calibration_selected_results.jsonl", selected_rows)
    _write_json(args.output_dir / "calibration_selected_metrics.json", calibration_report)
    _write_json(args.output_dir / "selection.json", selection_record)

    marker = args.output_dir / "holdout_consumption_marker.json"
    marker_record = {
        "schema_version": "search_relevance_v2_holdout_consumption",
        "selection_sha256": canonical_json_sha256(selection_record),
        "holdout_query_lock": canonical_json_sha256(holdout),
        "holdout_annotation_lock": canonical_json_sha256(holdout_annotations),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "single_consumption_policy": "exclusive_marker_written_before_holdout_evaluation",
    }
    _write_json(marker, marker_record)
    prepared_holdout = _prepare_queries(
        holdout, holdout_annotations, args.bundle_dir, archive, model_bundle, config
    )
    holdout_rows, holdout_ann_rows = _evaluate_prepared(
        prepared_holdout, selected["configuration"], config
    )
    holdout_report = score_search_results(
        holdout, holdout_annotations, holdout_rows, methods=SEARCH_V2_METHODS
    )
    holdout_ann = score_ann_fidelity(holdout_ann_rows, top_k=int(config["search"]["top_k"]))
    gates = apply_search_v2_gates(
        holdout_report, holdout_ann, config["search"]["holdout_gates"]
    )
    _write_jsonl(args.output_dir / "holdout_results.jsonl", holdout_rows)
    _write_json(args.output_dir / "holdout_metrics.json", holdout_report)
    summary = {
        "schema_version": "search_relevance_evidence_v2",
        "status": "completed",
        "evidence_class": config["evidence_class"],
        "human_annotation_support": 0,
        "fresh_test_used": False,
        "historical_v1_pool_used_for_selection": False,
        "validation": validation,
        "selected_configuration": selected["configuration"],
        "calibration": {
            "query_support": len(calibration),
            "candidate_support": len(candidate_reports),
            "metrics": {name: _without_per_query(calibration_report["methods"][name]) for name in SEARCH_V2_METHODS},
            "ann_fidelity": score_ann_fidelity(selected_ann_rows, top_k=int(config["search"]["top_k"])),
        },
        "holdout": {
            "query_support": len(holdout),
            "consumed_once": True,
            "metrics": {name: _without_per_query(holdout_report["methods"][name]) for name in SEARCH_V2_METHODS},
            "ann_fidelity": holdout_ann,
            "fixed_gates": gates,
        },
        "runtime": {
            "device": model_bundle["device"],
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": model_bundle["gpu"],
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "milvus_scope": "Milvus_Lite_component_not_distributed_service",
        },
        "holdout_result_sha256": canonical_json_sha256(holdout_rows),
        "promotion_eligible_as_human_ground_truth": False,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _verify_bundle_lock(config: dict[str, Any], bundle_dir: Path) -> None:
    expected_path = Path(config["pool"]["committed_lock"])
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = json.loads((bundle_dir / "bundle_lock.json").read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("generated bundle lock differs from committed lock")


def _load_clip_and_store(config: dict[str, Any], archive: dict[str, Any], milvus_path: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config["search"]["embedding_model"]
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    store = _build_milvus_store(milvus_path, archive["vectors"], archive["metadata"], config)
    return {
        "torch": torch,
        "processor": processor,
        "model": model,
        "store": store,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _prepare_queries(
    queries: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    bundle_dir: Path,
    archive: dict[str, Any],
    model_bundle: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np
    from PIL import Image

    annotation_index = {row["query_id"]: row for row in annotations}
    vectors = archive["vectors"]
    metadata = archive["metadata"]
    top_k = int(config["search"]["top_k"])
    prepared: list[dict[str, Any]] = []
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
        exact_hits = _hits_from_indices(exact_indices[:top_k], similarities, metadata)
        milvus_hits = _milvus_hits(
            model_bundle["store"].search(query_vector.astype(float).tolist(), top_k=top_k)
        )
        filters = {
            key: value for key, value in query["requested_filters"].items()
            if key in config["search"]["supported_filters"]
        }
        eligible = [
            index for index, row in enumerate(metadata)
            if all(_norm(row.get(key)) == _norm(value) for key, value in filters.items())
        ]
        relevant_total = sum(
            _metadata_grade(row, annotation_index[query["query_id"]]) >= 2 for row in metadata
        )
        prepared.append({
            "query": query,
            "similarities": similarities,
            "exact_hits": exact_hits,
            "milvus_hits": milvus_hits,
            "eligible": eligible,
            "metadata": metadata,
            "relevant_total": relevant_total,
        })
    return prepared


def _evaluate_prepared(
    prepared: list[dict[str, Any]], candidate: dict[str, float], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    top_k = int(config["search"]["top_k"])
    threshold = float(candidate["no_result_similarity_threshold"])
    star_weight = float(candidate["star_rating_weight"])
    rows: list[dict[str, Any]] = []
    ann_rows: list[dict[str, Any]] = []
    for item in prepared:
        similarities = item["similarities"]
        metadata = item["metadata"]
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
        unsupported = query["unsupported_constraints"]
        relevant_total = item["relevant_total"]
        methods = {
            "clip_exact": _method_result(item["exact_hits"], threshold, relevant_total, unsupported),
            "clip_milvus": _method_result(item["milvus_hits"], threshold, relevant_total, unsupported),
            "structured_filter_clip": _method_result(structured_hits, threshold, relevant_total, unsupported),
            "hard_filter_light_rerank": _method_result(reranked_hits, threshold, relevant_total, unsupported),
        }
        rows.append({"query_id": query["query_id"], "methods": methods})
        ann_rows.append({
            "query_id": query["query_id"],
            "exact_ids": [row["image_id"] for row in item["exact_hits"]],
            "ann_ids": [row["image_id"] for row in item["milvus_hits"]],
        })
    return rows, ann_rows


def _without_per_query(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "per_query"}


def _or_zero(value: Any) -> float:
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

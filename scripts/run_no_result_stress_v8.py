"""Calibrate and one-time validate a synthetic dual-centroid no-result guard."""

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

from scripts.run_relevance_evidence import _hits_from_indices, _metadata_grade, _method_result, _read_retrieval_archive
from scripts.run_search_relevance_v4 import _fit_business_guard
from src.evaluation.no_result_stress_v8 import apply_no_result_stress_v8_gate
from src.evaluation.relevance_evidence import (
    canonical_json_sha256,
    file_sha256,
    load_jsonl,
    score_search_results,
    validate_annotation_protocol,
    validate_query_manifest,
)


METHODS = ("hard_filter_clip", "v4_margin_guard", "dual_centroid_guard")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--prior-v4-bundle-dir", type=Path, required=True)
    parser.add_argument("--retrieval-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    config = _load_object(args.config)
    expected_lock = _load_object(Path(config["pool"]["committed_lock"]))
    if _load_object(args.bundle_dir / "bundle_lock.json") != expected_lock:
        raise ValueError("generated v8 bundle differs from the committed lock")
    prior_lock = _load_object(Path(config["pool"]["prior_v4_lock"]))
    if _load_object(args.prior_v4_bundle_dir / "bundle_lock.json") != prior_lock:
        raise ValueError("prior v4 bundle differs from the committed lock")
    if file_sha256(args.retrieval_archive) != config["formal_release_read_only"]["retrieval_archive_sha256"]:
        raise ValueError("formal retrieval archive SHA mismatch")

    prior_queries, prior_annotations = _load_split(
        args.prior_v4_bundle_dir, prior_lock, "training"
    )
    calibration, calibration_annotations = _load_split(
        args.bundle_dir, expected_lock, "calibration"
    )
    validation_evidence = {
        "prior_v4_training_queries": validate_query_manifest(prior_queries, args.prior_v4_bundle_dir),
        "prior_v4_training_annotations": validate_annotation_protocol(prior_queries, prior_annotations),
        "calibration_queries": validate_query_manifest(calibration, args.bundle_dir),
        "calibration_annotations": validate_annotation_protocol(calibration, calibration_annotations),
        "pool_isolation": expected_lock["isolation"],
        "prior_v4_training_isolation": expected_lock["prior_v4_training_isolation"],
    }
    archive = _read_retrieval_archive(args.retrieval_archive)
    if len(archive["vectors"]) != int(config["formal_release_read_only"]["expected_index_support"]):
        raise ValueError("formal retrieval support differs from the release lock")
    model_bundle = _load_clip(config)
    prior_prepared = _prepare(
        prior_queries, prior_annotations, args.prior_v4_bundle_dir, archive, model_bundle, config
    )
    centroids = _fit_business_guard(prior_prepared, prior_annotations)
    calibration_prepared = _prepare(
        calibration, calibration_annotations, args.bundle_dir, archive, model_bundle, config
    )

    guard = config["search"]["dual_centroid_guard"]
    candidates = []
    for margin, business_similarity in itertools.product(
        guard["margin_threshold_grid"], guard["business_similarity_threshold_grid"]
    ):
        candidate = {
            "margin_threshold": float(margin),
            "business_similarity_threshold": float(business_similarity),
        }
        rows = _evaluate(calibration_prepared, centroids, candidate, config)
        report = score_search_results(calibration, calibration_annotations, rows, methods=METHODS)
        candidates.append({
            "configuration": candidate,
            "objective": _objective(report["methods"]["dual_centroid_guard"], guard["selection_objective"]),
            "candidate_metrics": _without_per_query(report["methods"]["dual_centroid_guard"]),
        })
    selected = sorted(candidates, key=lambda row: (
        -float(row["objective"]),
        float(row["configuration"]["margin_threshold"]),
        float(row["configuration"]["business_similarity_threshold"]),
    ))[0]
    calibration_rows = _evaluate(calibration_prepared, centroids, selected["configuration"], config)
    calibration_report = score_search_results(
        calibration, calibration_annotations, calibration_rows, methods=METHODS
    )
    selection = {
        "schema_version": "no_result_stress_v8_calibration_selection",
        "selection_data": "calibration_only",
        "validation_read_for_selection": False,
        "prior_v4_training_use": "business_and_non_business_centroids_only",
        "candidate_support": len(candidates),
        "objective_definition": guard["selection_objective"],
        "selected": selected,
        "calibration_query_lock": canonical_json_sha256(calibration),
        "prior_v4_training_query_lock": canonical_json_sha256(prior_queries),
        "centroid_sha256": canonical_json_sha256({
            key: [float(value) for value in vector.tolist()] for key, vector in centroids.items()
        }),
        "source_snapshot_sha256": args.source_snapshot_sha256,
        "implementation_commit_sha": _full_git_sha(args.implementation_commit),
        "fresh_test_used": False,
        "final_defined_or_consumed": False,
    }
    _write_json(args.output_dir / "calibration_candidates.json", candidates)
    _write_jsonl(args.output_dir / "calibration_results.jsonl", calibration_rows)
    _write_json(args.output_dir / "calibration_metrics.json", calibration_report)
    selection_path = args.output_dir / "selection.json"
    _write_json(selection_path, selection)

    validation_marker = {
        "schema_version": "no_result_stress_v8_validation_consumption",
        "selection_file_sha256": file_sha256(selection_path),
        "committed_validation_lock": expected_lock["search"]["validation"],
        "single_consumption_policy": "exclusive_marker_written_before_first_validation_manifest_or_annotation_open",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    validation, validation_annotations = load_validation_after_marker(
        args.bundle_dir,
        args.output_dir / "validation_consumption_marker.json",
        validation_marker,
    )
    _verify_split_lock(validation, validation_annotations, args.bundle_dir, expected_lock, "validation")
    validation_evidence.update({
        "validation_queries": validate_query_manifest(validation, args.bundle_dir),
        "validation_annotations": validate_annotation_protocol(validation, validation_annotations),
    })
    validation_prepared = _prepare(
        validation, validation_annotations, args.bundle_dir, archive, model_bundle, config
    )
    validation_rows = _evaluate(validation_prepared, centroids, selected["configuration"], config)
    validation_report = score_search_results(
        validation, validation_annotations, validation_rows, methods=METHODS
    )
    fixed_gate = apply_no_result_stress_v8_gate(
        validation_report,
        config["search"]["validation_gates"],
        candidate="dual_centroid_guard",
        baseline="v4_margin_guard",
    )
    validation_rows_path = args.output_dir / "validation_results.jsonl"
    validation_metrics_path = args.output_dir / "validation_metrics.json"
    _write_jsonl(validation_rows_path, validation_rows)
    _write_json(validation_metrics_path, validation_report)
    summary = {
        "schema_version": "no_result_stress_evidence_v8",
        "status": "COMPLETED" if fixed_gate["status"] == "PASS" else "NEGATIVE_EXPERIMENT_GATE_FAILED",
        "evidence_class": config["evidence_class"],
        "gate_class": config["gate_class"],
        "human_annotation_support": 0,
        "fresh_test_used": False,
        "final_defined_or_consumed": False,
        "ann_fidelity_scope": "NOT_MEASURED_NO_RESULT_CLASSIFIER_STRESS",
        "selected_configuration": selected["configuration"],
        "selection_file_sha256": file_sha256(selection_path),
        "configuration": {
            "config_sha256": file_sha256(args.config),
            "pool_lock_sha256": canonical_json_sha256(expected_lock),
            "retrieval_archive_sha256": file_sha256(args.retrieval_archive),
            "source_snapshot_sha256": args.source_snapshot_sha256,
            "implementation_commit_sha": _full_git_sha(args.implementation_commit),
            "embedding_model": config["search"]["embedding_model"],
            "index_support": len(archive["vectors"]),
        },
        "denominators": {
            "prior_v4_training": len(prior_queries),
            "calibration": len(calibration),
            "validation": len(validation),
            "validation_ranking": expected_lock["search"]["validation"]["ranking_support"],
            "validation_no_result": expected_lock["search"]["validation"]["no_result_support"],
            "validation_business_positive": expected_lock["search"]["validation"]["business_positive_support"],
        },
        "validation": validation_evidence,
        "calibration_metrics": _method_summaries(calibration_report),
        "validation_metrics": _method_summaries(validation_report),
        "fixed_gate": fixed_gate,
        "runtime": _runtime(model_bundle),
        "validation_result_canonical_sha256": canonical_json_sha256(validation_rows),
        "validation_result_file_sha256": file_sha256(validation_rows_path),
        "validation_metrics_file_sha256": file_sha256(validation_metrics_path),
        "promotion_eligible_as_human_ground_truth": False,
    }
    summary["artifact_sha256"] = canonical_json_sha256(summary)
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def load_validation_after_marker(
    bundle_dir: Path, marker_path: Path, marker: dict[str, Any], loader=load_jsonl
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _write_json(marker_path, marker)
    return (
        loader(bundle_dir / "search_validation_manifest.jsonl"),
        loader(bundle_dir / "search_validation_annotations.jsonl"),
    )


def _load_split(
    bundle_dir: Path, expected_lock: dict[str, Any], split: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries = load_jsonl(bundle_dir / f"search_{split}_manifest.jsonl")
    annotations = load_jsonl(bundle_dir / f"search_{split}_annotations.jsonl")
    _verify_split_lock(queries, annotations, bundle_dir, expected_lock, split)
    return queries, annotations


def _verify_split_lock(
    queries: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    bundle_dir: Path,
    expected_lock: dict[str, Any],
    split: str,
) -> None:
    query_path = bundle_dir / f"search_{split}_manifest.jsonl"
    annotation_path = bundle_dir / f"search_{split}_annotations.jsonl"
    observed = {
        "query_support": len(queries),
        "ranking_support": sum("no_result" not in row["slices"] for row in queries),
        "no_result_support": sum("no_result" in row["slices"] for row in queries),
        "business_positive_support": sum("business_positive" in row["slices"] for row in queries),
        "query_manifest_canonical_sha256": canonical_json_sha256(queries),
        "query_manifest_file_sha256": file_sha256(query_path),
        "annotation_canonical_sha256": canonical_json_sha256(annotations),
        "annotation_file_sha256": file_sha256(annotation_path),
    }
    expected = expected_lock["search"][split]
    if split == "training" and "ranking_support" not in expected:
        observed = {key: value for key, value in observed.items() if key in expected}
    if observed != expected:
        raise ValueError(f"search {split} split differs from the committed lock")


def _load_clip(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config["search"]["embedding_model"]
    return {
        "torch": torch,
        "processor": AutoProcessor.from_pretrained(model_name),
        "model": CLIPModel.from_pretrained(model_name).to(device).eval(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _prepare(
    queries: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    bundle_dir: Path,
    archive: dict[str, Any],
    model_bundle: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    from PIL import Image

    annotation_index = {row["query_id"]: row for row in annotations}
    vectors, metadata = archive["vectors"], archive["metadata"]
    prepared = []
    for query in queries:
        started = time.perf_counter()
        image = Image.open(bundle_dir / query["image"]["relative_path"]).convert("RGB")
        inputs = model_bundle["processor"](images=image, return_tensors="pt")
        inputs = {key: value.to(model_bundle["device"]) for key, value in inputs.items()}
        with model_bundle["torch"].inference_mode():
            embedding = model_bundle["model"].get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        vector = embedding[0].detach().float().cpu().numpy()
        similarities = vectors @ vector
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
            "query_vector": vector,
            "similarities": similarities,
            "metadata": metadata,
            "eligible": eligible,
            "relevant_total": relevant_total,
            "clip_encode_ms": (time.perf_counter() - started) * 1000,
        })
    return prepared


def _evaluate(
    prepared: list[dict[str, Any]],
    centroids: dict[str, Any],
    candidate: dict[str, float],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    top_k = int(config["search"]["top_k"])
    prior_margin = float(config["search"]["prior_v4_business_guard"]["margin_threshold"])
    for item in prepared:
        query = item["query"]
        similarities = item["similarities"]
        ranked = sorted(item["eligible"], key=lambda index: -float(similarities[index]))
        hits = _hits_from_indices(ranked[:top_k], similarities, item["metadata"])
        business_similarity = float(item["query_vector"] @ centroids["business"])
        non_business_similarity = float(item["query_vector"] @ centroids["non_business"])
        margin = business_similarity - non_business_similarity
        no_filters = not query["requested_filters"]
        v4_reject = no_filters and margin < prior_margin
        dual_reject = no_filters and (
            margin < float(candidate["margin_threshold"])
            or business_similarity < float(candidate["business_similarity_threshold"])
        )
        relevant_total = item["relevant_total"]
        unsupported = query["unsupported_constraints"]
        methods = {
            "hard_filter_clip": _method_result(hits, -2.0, relevant_total, unsupported),
            "v4_margin_guard": _method_result([] if v4_reject else hits, -2.0, relevant_total, unsupported),
            "dual_centroid_guard": _method_result([] if dual_reject else hits, -2.0, relevant_total, unsupported),
        }
        for method in ("v4_margin_guard", "dual_centroid_guard"):
            methods[method]["business_centroid_similarity"] = business_similarity
            methods[method]["non_business_centroid_similarity"] = non_business_similarity
            methods[method]["business_margin"] = margin
            methods[method]["guard_rejected"] = v4_reject if method == "v4_margin_guard" else dual_reject
        rows.append({"query_id": query["query_id"], "methods": methods})
    return rows


def _objective(metrics: dict[str, Any], definition: dict[str, Any]) -> float:
    return (
        float(metrics["slices"]["no_result"]["no_result_accuracy"])
        * float(definition["no_result_accuracy_weight"])
        + float(metrics["slices"]["business_positive"]["no_result_accuracy"])
        * float(definition["business_positive_acceptance_weight"])
        + float(metrics["ndcg_at_10"]) * float(definition["ndcg_at_10_weight"])
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
    }


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _full_git_sha(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 40 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("implementation commit must be a full Git SHA")
    return normalized


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

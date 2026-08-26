"""Week 8 独立查询检索锁、轻量 metadata rerank 与离线评测。"""

from __future__ import annotations

import hashlib
import io
import json
import math
import statistics
import tarfile
import time
from pathlib import Path
from typing import Any, Iterable

from src.retrieval.week8_hybrid import (
    OfflineImageChannel,
    fuse_rankings,
    metadata_ranking,
)


PARTITIONS = ("index", "development_query", "final_test_query")
METHODS = ("clip", "metadata_rerank", "hybrid_rrf", "hybrid_weighted")
REQUIRED_METADATA_FIELDS = {
    "business_id",
    "image_id",
    "business_category",
    "city",
    "star_rating",
    "price_range",
    "image_type",
    "embedding_model",
    "source_image_path",
}
TRACE_FIELDS = (
    "business_id",
    "image_id",
    "source_image_path",
    "embedding_model",
)


class Week8RetrievalError(ValueError):
    """Raised when a Week 8 retrieval identity or evaluation contract fails."""


def load_config(path: Path | str) -> dict[str, Any]:
    """Load the versioned configuration and validate fixed protocol fields."""
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week8RetrievalError(f"cannot load retrieval config: {exc}") from exc
    if config.get("schema_version") != "week8_retrieval_relevance_config_v1":
        raise Week8RetrievalError("unsupported Week 8 retrieval config schema")
    source = config.get("source")
    split = config.get("split")
    evaluation = config.get("evaluation")
    hybrid = config.get("hybrid")
    selection = config.get("selection")
    if not all(
        isinstance(item, dict) for item in (source, split, evaluation, hybrid, selection)
    ):
        raise Week8RetrievalError(
            "source, split, evaluation, hybrid, and selection are required"
        )
    dev_fraction = _finite_number(split.get("development_query_group_fraction"))
    test_fraction = _finite_number(split.get("final_test_query_group_fraction"))
    if dev_fraction <= 0 or test_fraction <= 0 or dev_fraction + test_fraction >= 1:
        raise Week8RetrievalError("query group fractions must be positive and sum below one")
    top_k_values = evaluation.get("top_k_values")
    if (
        not isinstance(top_k_values, list)
        or not top_k_values
        or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in top_k_values)
    ):
        raise Week8RetrievalError("top_k_values must contain positive integers")
    if set(selection.get("non_regression_metrics", [])) - {
        "recall_at_10",
        "filter_correctness",
        "traceable_reference_rate",
    }:
        raise Week8RetrievalError("selection non-regression metric is unsupported")
    if (
        source.get("release_id") != "trip-qwen3-vl-8b-system-repair-v1-rc1"
        or source.get("embedding_model") != "openai/clip-vit-base-patch32"
        or source.get("vector_dimension") != 512
        or source.get("expected_count") != 1000
    ):
        raise Week8RetrievalError("Week 8 retrieval must bind the formal 1,000-vector release")
    templates = split.get("template_ids")
    if (
        not isinstance(templates, dict)
        or set(templates) != set(PARTITIONS)
        or len(set(templates.values())) != len(PARTITIONS)
    ):
        raise Week8RetrievalError("partition template identities must be complete and distinct")
    if int(evaluation.get("candidate_pool_size", 0)) < max(top_k_values):
        raise Week8RetrievalError("candidate_pool_size must cover the largest top_k")
    if hybrid.get("backend_preference") not in {"auto", "milvus", "offline"}:
        raise Week8RetrievalError("hybrid backend preference is unsupported")
    if not isinstance(hybrid.get("offline_fallback_enabled"), bool):
        raise Week8RetrievalError("offline_fallback_enabled must be boolean")
    if (
        hybrid.get("milvus_lite_index_type") != "FLAT"
        or hybrid.get("milvus_remote_index_type") != "HNSW"
    ):
        raise Week8RetrievalError("Milvus Lite/remote index types must be FLAT/HNSW")
    fusion_weights = hybrid.get("weighted_fusion")
    if (
        not isinstance(fusion_weights, dict)
        or set(fusion_weights) != {"image", "metadata"}
        or not math.isclose(
            sum(_finite_number(value) for value in fusion_weights.values()),
            1.0,
            abs_tol=1e-9,
        )
    ):
        raise Week8RetrievalError("weighted fusion weights must sum to one")
    candidate_methods = selection.get("candidate_methods")
    if (
        not isinstance(candidate_methods, list)
        or not candidate_methods
        or set(candidate_methods) - (set(METHODS) - {"clip"})
    ):
        raise Week8RetrievalError("selection candidate methods are invalid")
    return config


def load_retrieval_source(
    config: dict[str, Any],
    *,
    archive_path: Path | None = None,
    retrieval_dir: Path | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, str]]:
    """Load packaged vectors/metadata without requiring Milvus or CLIP."""
    if (archive_path is None) == (retrieval_dir is None):
        raise Week8RetrievalError("provide exactly one retrieval archive or directory")
    source = config["source"]
    if archive_path is not None:
        archive = Path(archive_path)
        if not archive.is_file():
            raise Week8RetrievalError(f"retrieval archive is missing: {archive}")
        with tarfile.open(archive, "r:gz") as handle:
            vector_bytes = _read_tar_member(handle, source["archive_vectors_member"])
            metadata_bytes = _read_tar_member(handle, source["archive_metadata_member"])
        source_hashes = {
            "container_sha256": sha256_file(archive),
            "vectors_sha256": _sha256_bytes(vector_bytes),
            "metadata_sha256": _sha256_bytes(metadata_bytes),
        }
    else:
        directory = Path(retrieval_dir or "")
        vectors_path = directory / source["vectors_filename"]
        metadata_path = directory / source["metadata_filename"]
        if not vectors_path.is_file() or not metadata_path.is_file():
            raise Week8RetrievalError("retrieval directory is missing vectors or metadata")
        vector_bytes = vectors_path.read_bytes()
        metadata_bytes = metadata_path.read_bytes()
        source_hashes = {
            "container_sha256": "not_applicable_directory_source",
            "vectors_sha256": _sha256_bytes(vector_bytes),
            "metadata_sha256": _sha256_bytes(metadata_bytes),
        }

    expected_archive = source.get("expected_archive_sha256")
    if archive_path is not None and expected_archive and source_hashes["container_sha256"] != expected_archive:
        raise Week8RetrievalError("retrieval archive SHA-256 does not match the formal release")
    if source_hashes["vectors_sha256"] != source.get("expected_vectors_sha256"):
        raise Week8RetrievalError("retrieval vector SHA-256 does not match the formal release")
    if source_hashes["metadata_sha256"] != source.get("expected_metadata_sha256"):
        raise Week8RetrievalError("retrieval metadata SHA-256 does not match the formal release")

    try:
        import numpy as np

        with np.load(io.BytesIO(vector_bytes), allow_pickle=False) as payload:
            vectors = np.asarray(payload["multimodal_vector"], dtype="float32")
    except (ImportError, KeyError, OSError, ValueError) as exc:
        raise Week8RetrievalError(f"cannot load retrieval vectors: {exc}") from exc
    metadata = _parse_metadata(metadata_bytes)
    _validate_source(config, vectors, metadata)
    return vectors, metadata, source_hashes


def build_data_lock(
    config: dict[str, Any],
    vectors: Any,
    metadata: list[dict[str, Any]],
    source_hashes: dict[str, str],
    *,
    source_project_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build immutable index/dev/test rows with original-image SHA isolation."""
    _validate_source(config, vectors, metadata)
    project_root = Path(source_project_root).resolve()
    if not project_root.is_dir():
        raise Week8RetrievalError(f"source project root is missing: {project_root}")

    prepared: list[dict[str, Any]] = []
    for vector_index, row in enumerate(metadata):
        image_path = _resolve_source_image(project_root, row["source_image_path"])
        prepared.append(
            {
                "sample_id": f"week8-retrieval:{row['image_id']}",
                "source_id": f"yelp-photo:{row['image_id']}",
                "image_sha256": sha256_file(image_path),
                "group_id": row["business_id"],
                "label_provenance": "silver",
                "label_basis": "existing_release_metadata",
                "vector_index": vector_index,
                "vector_sha256": _vector_sha256(vectors[vector_index]),
                "metadata": dict(row),
            }
        )

    assignments = _assign_partitions(config, prepared)
    rows_by_partition: dict[str, list[dict[str, Any]]] = {name: [] for name in PARTITIONS}
    templates = config["split"]["template_ids"]
    for row in prepared:
        partition = assignments[row["group_id"]]
        locked = dict(row)
        locked["partition"] = partition
        locked["template_id"] = templates[partition]
        rows_by_partition[partition].append(locked)
    for rows in rows_by_partition.values():
        rows.sort(key=lambda item: item["sample_id"])
    _validate_partition_isolation(rows_by_partition)
    _validate_minimum_counts(config, rows_by_partition)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, Any]] = {}
    for partition in PARTITIONS:
        path = output / f"{partition}.jsonl"
        _write_jsonl_new(path, rows_by_partition[partition])
        files[partition] = {
            "path": path.name,
            "count": len(rows_by_partition[partition]),
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": "week8_retrieval_data_lock_v1",
        "dataset_version": config["dataset_version"],
        "experiment_id": config["experiment_id"],
        "label_provenance": "silver",
        "source_release_id": config["source"]["release_id"],
        "source_hashes": source_hashes,
        "source_image_hash_basis": "sha256_original_image_bytes",
        "split_seed": config["split"]["seed"],
        "five_dimension_isolation": "PASS",
        "files": files,
    }
    manifest_path = output / "dataset_lock.json"
    _write_json_new(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def validate_data_lock(lock_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Revalidate lock hashes, silver identity, and five-dimensional isolation."""
    root = Path(lock_dir)
    try:
        manifest = json.loads((root / "dataset_lock.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week8RetrievalError(f"cannot read retrieval data lock: {exc}") from exc
    if manifest.get("schema_version") != "week8_retrieval_data_lock_v1":
        raise Week8RetrievalError("unsupported retrieval lock schema")
    rows_by_partition: dict[str, list[dict[str, Any]]] = {}
    for partition in PARTITIONS:
        file_record = manifest.get("files", {}).get(partition, {})
        path = root / str(file_record.get("path", ""))
        if not path.is_file() or sha256_file(path) != file_record.get("sha256"):
            raise Week8RetrievalError(f"lock hash mismatch: {partition}")
        rows = _read_jsonl(path)
        if len(rows) != file_record.get("count"):
            raise Week8RetrievalError(f"lock count mismatch: {partition}")
        if any(row.get("partition") != partition for row in rows):
            raise Week8RetrievalError(f"lock partition mismatch: {partition}")
        if any(row.get("label_provenance") != "silver" for row in rows):
            raise Week8RetrievalError("retrieval metadata labels must remain silver")
        rows_by_partition[partition] = rows
    _validate_partition_isolation(rows_by_partition)
    return manifest, rows_by_partition


def evaluate_partition(
    config: dict[str, Any],
    vectors: Any,
    metadata: list[dict[str, Any]],
    rows_by_partition: dict[str, list[dict[str, Any]]],
    partition: str,
    *,
    methods: Iterable[str] = METHODS,
    image_channel: Any | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare pure CLIP and lightweight metadata rerank on one locked query split."""
    if partition not in {"development_query", "final_test_query"}:
        raise Week8RetrievalError("evaluation partition must be a query split")
    _validate_source(config, vectors, metadata)
    index_rows = rows_by_partition["index"]
    query_rows = rows_by_partition[partition]
    _verify_locked_source_rows(metadata, index_rows + query_rows)
    _verify_vector_identities(vectors, index_rows + query_rows)
    method_names = list(dict.fromkeys(methods))
    if not method_names or any(method not in METHODS for method in method_names):
        raise Week8RetrievalError("unsupported or empty retrieval method list")
    active_image_channel = image_channel or OfflineImageChannel(vectors)

    metrics: dict[str, dict[str, Any]] = {}
    all_results: list[dict[str, Any]] = []
    all_references: list[dict[str, Any]] = []
    for method in method_names:
        measured, results, references = _evaluate_method(
            config, vectors, index_rows, query_rows, method, active_image_channel
        )
        metrics[method] = measured
        all_results.extend(results)
        all_references.extend(references)
    return metrics, all_results, all_references


def select_development_method(
    config: dict[str, Any], metrics: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Lock rerank only after a strict primary improvement and all gates pass."""
    selection = config["selection"]
    baseline_name = selection["baseline_method"]
    candidate_names = selection.get("candidate_methods") or [selection["candidate_method"]]
    baseline = metrics.get(baseline_name)
    if not isinstance(baseline, dict):
        raise Week8RetrievalError("development metrics require the baseline")
    primary = selection["primary_metric"]
    required_failure = _finite_number(selection["required_failure_rate"])
    if _finite_number(baseline.get("failure_rate")) != required_failure:
        raise Week8RetrievalError("baseline failure rate gate failed")

    evaluations: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for candidate_name in candidate_names:
        candidate = metrics.get(candidate_name)
        if not isinstance(candidate, dict):
            raise Week8RetrievalError(f"development metrics missing candidate: {candidate_name}")
        candidate_failures: list[str] = []
        if _finite_number(candidate.get(primary)) <= _finite_number(baseline.get(primary)):
            candidate_failures.append(f"{primary}_not_improved")
        for metric in selection["non_regression_metrics"]:
            if _finite_number(candidate.get(metric)) < _finite_number(baseline.get(metric)):
                candidate_failures.append(f"{metric}_regressed")
        if _finite_number(candidate.get("failure_rate")) != required_failure:
            candidate_failures.append("candidate_failure_rate_gate")
        evaluations[candidate_name] = {
            "value": candidate.get(primary),
            "eligible": not candidate_failures,
            "failures": candidate_failures,
        }
        if not candidate_failures:
            eligible.append(candidate_name)
    best_candidate = max(
        candidate_names,
        key=lambda name: (_finite_number(metrics[name].get(primary)), name),
    )
    selected = (
        max(eligible, key=lambda name: (_finite_number(metrics[name].get(primary)), name))
        if eligible
        else baseline_name
    )
    failures = [] if eligible else sorted(
        {failure for result in evaluations.values() for failure in result["failures"]}
    )
    return {
        "schema_version": "week8_retrieval_development_selection_v1",
        "experiment_id": config["experiment_id"],
        "selected_method": selected,
        "candidate_locked": bool(eligible),
        "failures": failures,
        "primary_metric": primary,
        "baseline_value": baseline.get(primary),
        "candidate_value": metrics[best_candidate].get(primary),
        "best_candidate": best_candidate,
        "candidate_evaluations": evaluations,
        "selected_backend": metrics[selected].get("retrieval_backend"),
        "selected_offline_fallback": metrics[selected].get("offline_fallback"),
        "selected_fallback_reason": metrics[selected].get("fallback_reason"),
    }


def write_evaluation(
    output_dir: Path,
    *,
    partition: str,
    metrics: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    references: list[dict[str, Any]],
    selection: dict[str, Any] | None = None,
    data_lock_sha256: str | None = None,
    source_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Persist measured artifacts without overwriting an earlier run."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    metrics_payload = {
        "schema_version": "week8_retrieval_metrics_v1",
        "partition": partition,
        "methods": metrics,
    }
    _write_json_new(output / "metrics.json", metrics_payload)
    _write_jsonl_new(output / "query_results.jsonl", results)
    _write_jsonl_new(output / "business_references.jsonl", references)
    hashes = {
        "metrics_sha256": sha256_file(output / "metrics.json"),
        "query_results_sha256": sha256_file(output / "query_results.jsonl"),
        "business_references_sha256": sha256_file(output / "business_references.jsonl"),
    }
    if selection is not None:
        if not isinstance(data_lock_sha256, str) or len(data_lock_sha256) != 64:
            raise Week8RetrievalError("development selection requires the data lock SHA-256")
        if not isinstance(source_hashes, dict) or not source_hashes:
            raise Week8RetrievalError("development selection requires source hashes")
        selection.update(
            {
                "data_lock_sha256": data_lock_sha256,
                "source_hashes": dict(source_hashes),
                "development_evidence": {
                    "metrics": {
                        "path": "metrics.json",
                        "sha256": hashes["metrics_sha256"],
                    },
                    "query_results": {
                        "path": "query_results.jsonl",
                        "sha256": hashes["query_results_sha256"],
                    },
                    "business_references": {
                        "path": "business_references.jsonl",
                        "sha256": hashes["business_references_sha256"],
                    },
                },
            }
        )
        _write_json_new(output / "selection.json", selection)
        hashes["selection_sha256"] = sha256_file(output / "selection.json")
    return hashes


def validate_development_selection(
    config: dict[str, Any],
    selection_path: Path,
    *,
    lock_dir: Path,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Recompute every development binding before final-test consumption."""
    path = Path(selection_path)
    try:
        selection = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week8RetrievalError(f"cannot read development selection: {exc}") from exc
    if selection.get("schema_version") != "week8_retrieval_development_selection_v1":
        raise Week8RetrievalError("unsupported development selection schema")
    if selection.get("experiment_id") != config["experiment_id"]:
        raise Week8RetrievalError("selection experiment identity mismatch")
    if selection.get("selected_method") not in METHODS:
        raise Week8RetrievalError("selection contains an unsupported method")

    lock_path = Path(lock_dir) / "dataset_lock.json"
    if not lock_path.is_file() or sha256_file(lock_path) != selection.get("data_lock_sha256"):
        raise Week8RetrievalError("selection data lock SHA-256 mismatch")
    if selection.get("source_hashes") != source_hashes:
        raise Week8RetrievalError("selection source hashes mismatch")

    evidence = selection.get("development_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "metrics",
        "query_results",
        "business_references",
    }:
        raise Week8RetrievalError("selection development evidence is incomplete")
    for name, expected_filename in (
        ("metrics", "metrics.json"),
        ("query_results", "query_results.jsonl"),
        ("business_references", "business_references.jsonl"),
    ):
        record = evidence.get(name)
        if not isinstance(record, dict) or record.get("path") != expected_filename:
            raise Week8RetrievalError(f"selection evidence path mismatch: {name}")
        evidence_path = path.parent / expected_filename
        if not evidence_path.is_file() or sha256_file(evidence_path) != record.get("sha256"):
            raise Week8RetrievalError(f"selection evidence SHA-256 mismatch: {name}")

    try:
        metrics_payload = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week8RetrievalError(f"cannot recompute development selection: {exc}") from exc
    if (
        metrics_payload.get("schema_version") != "week8_retrieval_metrics_v1"
        or metrics_payload.get("partition") != "development_query"
        or not isinstance(metrics_payload.get("methods"), dict)
    ):
        raise Week8RetrievalError("development metrics protocol identity mismatch")
    recomputed = select_development_method(config, metrics_payload["methods"])
    decision_fields = (
        "selected_method",
        "candidate_locked",
        "failures",
        "primary_metric",
        "baseline_value",
        "candidate_value",
        "best_candidate",
        "candidate_evaluations",
        "selected_backend",
        "selected_offline_fallback",
        "selected_fallback_reason",
    )
    if any(selection.get(field) != recomputed.get(field) for field in decision_fields):
        raise Week8RetrievalError("development selection decision mismatch")
    return selection


def claim_final_test(
    marker_path: Path,
    selection: dict[str, Any],
    *,
    selection_sha256: str | None = None,
) -> None:
    """Create the single-consumption marker before reading final-test vectors."""
    marker_path = Path(marker_path)
    if marker_path.exists():
        raise Week8RetrievalError("Week 8 retrieval final test was already consumed")
    if not isinstance(selection_sha256, str) or len(selection_sha256) != 64:
        raise Week8RetrievalError("final-test marker requires the selection SHA-256")
    if not isinstance(selection.get("data_lock_sha256"), str):
        raise Week8RetrievalError("final-test marker requires the bound data lock")
    if not isinstance(selection.get("source_hashes"), dict):
        raise Week8RetrievalError("final-test marker requires the bound source hashes")
    marker = {
        "schema_version": "week8_retrieval_final_test_consumption_v1",
        "experiment_id": selection.get("experiment_id"),
        "selected_method": selection.get("selected_method"),
        "selected_backend": selection.get("selected_backend"),
        "selected_offline_fallback": selection.get("selected_offline_fallback"),
        "selection_sha256": selection_sha256,
        "data_lock_sha256": selection.get("data_lock_sha256"),
        "source_hashes": selection.get("source_hashes"),
        "status": "STARTED",
    }
    try:
        _write_json_new(marker_path, marker)
    except FileExistsError as exc:
        raise Week8RetrievalError("Week 8 retrieval final test was already consumed") from exc


def complete_final_test(marker_path: Path, final_hashes: dict[str, str]) -> dict[str, Any]:
    """Atomically complete the marker only after all final evidence is durable."""
    marker = Path(marker_path)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week8RetrievalError(f"cannot read final-test marker: {exc}") from exc
    if payload.get("schema_version") != "week8_retrieval_final_test_consumption_v1":
        raise Week8RetrievalError("unsupported final-test marker schema")
    if payload.get("status") != "STARTED":
        raise Week8RetrievalError("final-test marker is not in STARTED state")
    required = {
        "metrics_sha256",
        "query_results_sha256",
        "business_references_sha256",
    }
    if set(final_hashes) != required or any(
        not isinstance(final_hashes[name], str) or len(final_hashes[name]) != 64
        for name in required
    ):
        raise Week8RetrievalError("final-test evidence hashes are incomplete")
    completed = dict(payload)
    completed["status"] = "COMPLETED"
    completed["final_evidence"] = {
        "metrics": {
            "path": "metrics.json",
            "sha256": final_hashes["metrics_sha256"],
        },
        "query_results": {
            "path": "query_results.jsonl",
            "sha256": final_hashes["query_results_sha256"],
        },
        "business_references": {
            "path": "business_references.jsonl",
            "sha256": final_hashes["business_references_sha256"],
        },
    }
    temporary = marker.with_name(f"{marker.name}.completed.tmp")
    _write_json_new(temporary, completed)
    temporary.replace(marker)
    return completed


def _evaluate_method(
    config: dict[str, Any],
    vectors: Any,
    index_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    method: str,
    image_channel: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = config["evaluation"]
    top_k_values = sorted(set(evaluation["top_k_values"]))
    maximum_k = max(top_k_values)
    recalls: dict[int, list[float]] = {k: [] for k in top_k_values}
    ndcgs: dict[int, list[float]] = {k: [] for k in top_k_values}
    latencies: list[float] = []
    image_latencies: list[float] = []
    results: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    failures = 0
    filter_requests = filter_no_results = filter_hits = filter_correct = 0
    traceable = trace_total = 0
    source_attributed = source_total = 0
    source_channel_counts = {"image": 0, "metadata": 0}
    backend = image_channel.describe()

    for query in query_rows:
        try:
            query_vector = vectors[query["vector_index"]]
            started = time.perf_counter()
            ranked = _rank(
                config,
                vectors,
                index_rows,
                query,
                query_vector,
                method,
                maximum_k,
                image_channel=image_channel,
            )
            latencies.append((time.perf_counter() - started) * 1000.0)
            if isinstance(image_channel.last_latency_ms, (int, float)):
                image_latencies.append(float(image_channel.last_latency_ms))
            grades = {
                row["sample_id"]: _relevance_grade(config, query["metadata"], row["metadata"])
                for row in index_rows
            }
            relevant = {
                sample_id
                for sample_id, grade in grades.items()
                if grade >= _finite_number(evaluation["minimum_relevance_grade"])
            }
            ranked_ids = [hit["row"]["sample_id"] for hit in ranked]
            if relevant:
                for k in top_k_values:
                    recalls[k].append(len(relevant & set(ranked_ids[:k])) / len(relevant))
                    ndcgs[k].append(_ndcg_at_k(grades, ranked_ids, k))
            result = {
                "method": method,
                "query_sample_id": query["sample_id"],
                "partition": query["partition"],
                "evaluable_relevance": bool(relevant),
                "relevant_count": len(relevant),
                "filter_checks": [],
                "ranked": [
                    {
                        "rank": rank,
                        "sample_id": hit["row"]["sample_id"],
                        "image_id": hit["row"]["metadata"]["image_id"],
                        "clip_score": hit["clip_score"],
                        "ranking_score": hit["ranking_score"],
                        "source_channels": hit.get("source_channels", []),
                        "component_ranks": hit.get("component_ranks", {}),
                        "component_scores": hit.get("component_scores", {}),
                        "relevance_grade": grades[hit["row"]["sample_id"]],
                    }
                    for rank, hit in enumerate(ranked, start=1)
                ],
            }
            query_references: list[dict[str, Any]] = []
            for consumer in evaluation["consumer_paths"]:
                for rank, hit in enumerate(ranked, start=1):
                    reference = _traceable_reference(query, hit["row"], method, consumer, rank)
                    reference["source_channels"] = hit.get("source_channels", [])
                    reference["retrieval_backend"] = backend["backend"]
                    query_references.append(reference)

            for fields in evaluation["filter_scenarios"]:
                filters = _query_filters(query["metadata"], fields)
                if not filters:
                    continue
                filter_requests += 1
                filtered = _rank(
                    config,
                    vectors,
                    index_rows,
                    query,
                    query_vector,
                    method,
                    maximum_k,
                    filters=filters,
                    image_channel=image_channel,
                )
                if not filtered:
                    filter_no_results += 1
                filter_check = {
                    "filters": filters,
                    "no_result": not filtered,
                    "hits": [hit["row"]["sample_id"] for hit in filtered],
                    "correct": all(
                        _matches_filters(hit["row"]["metadata"], filters)
                        for hit in filtered
                    ),
                }
                result["filter_checks"].append(filter_check)
                for hit in filtered:
                    filter_hits += 1
                    filter_correct += int(_matches_filters(hit["row"]["metadata"], filters))
            results.append(result)
            references.extend(query_references)
            trace_total += len(query_references)
            traceable += sum(int(reference["traceable"]) for reference in query_references)
            for hit in ranked:
                channels = hit.get("source_channels", [])
                source_total += 1
                source_attributed += int(bool(channels))
                for channel in source_channel_counts:
                    source_channel_counts[channel] += int(channel in channels)
        except Exception as exc:  # 每个查询保留真实失败率，不静默丢弃。
            failures += 1
            results.append(
                {
                    "method": method,
                    "query_sample_id": query.get("sample_id"),
                    "partition": query.get("partition"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    metrics: dict[str, Any] = {
        "method": method,
        "query_count": len(query_rows),
        "successful_query_count": len(query_rows) - failures,
        "failure_count": failures,
        "failure_rate": failures / len(query_rows) if query_rows else 0.0,
        "relevance_support_count": len(next(iter(recalls.values()), [])),
        "filter_request_count": filter_requests,
        "filter_hit_count": filter_hits,
        "filter_correctness": filter_correct / filter_hits if filter_hits else 1.0,
        "no_result_rate": filter_no_results / filter_requests if filter_requests else 0.0,
        "traceable_reference_count": traceable,
        "reference_count": trace_total,
        "traceable_reference_rate": traceable / trace_total if trace_total else 1.0,
        "source_attribution_count": source_attributed,
        "source_result_count": source_total,
        "source_attribution_rate": source_attributed / source_total if source_total else 1.0,
        "source_channel_counts": source_channel_counts,
        "retrieval_backend": backend["backend"],
        "offline_fallback": backend["offline_fallback"],
        "fallback_reason": backend["fallback_reason"],
        "latency_mean_ms": statistics.fmean(latencies) if latencies else None,
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": _percentile(latencies, 0.95),
        "image_channel_latency_mean_ms": statistics.fmean(image_latencies)
        if image_latencies
        else None,
        "image_channel_latency_p95_ms": _percentile(image_latencies, 0.95),
    }
    for k in top_k_values:
        metrics[f"recall_at_{k}"] = statistics.fmean(recalls[k]) if recalls[k] else 0.0
        metrics[f"ndcg_at_{k}"] = statistics.fmean(ndcgs[k]) if ndcgs[k] else 0.0
    return metrics, results, references


def _rank(
    config: dict[str, Any],
    vectors: Any,
    index_rows: list[dict[str, Any]],
    query: dict[str, Any],
    query_vector: Any,
    method: str,
    top_k: int,
    *,
    filters: dict[str, str] | None = None,
    image_channel: Any,
) -> list[dict[str, Any]]:
    pool_size = int(config["evaluation"]["candidate_pool_size"])
    image_hits = image_channel.search(
        query_vector,
        index_rows,
        top_k=max(pool_size, top_k),
        filters=filters,
    )
    scored = [
        {
            "row": hit["row"],
            "clip_score": hit["image_score"],
            "source_channels": ["image"],
            "component_ranks": {"image": rank, "metadata": None},
            "component_scores": {"image": hit["image_score"], "metadata": None},
        }
        for rank, hit in enumerate(image_hits, start=1)
    ]
    if method == "clip":
        for hit in scored[:top_k]:
            hit["ranking_score"] = hit["clip_score"]
        return scored[:top_k]
    if method == "metadata_rerank":
        rerank_weights = config["evaluation"]["rerank_weights"]
        pool = scored[:pool_size]
        for hit in pool:
            bonus = 0.0
            for field, weight in rerank_weights.items():
                query_value = query["metadata"].get(field)
                if query_value != "unknown" and query_value == hit["row"]["metadata"].get(field):
                    bonus += _finite_number(weight)
            hit["ranking_score"] = hit["clip_score"] + bonus
            hit["source_channels"] = ["image", "metadata"]
            hit["component_scores"]["metadata"] = bonus
        pool.sort(
            key=lambda hit: (
                -hit["ranking_score"],
                -hit["clip_score"],
                hit["row"]["sample_id"],
            )
        )
        return pool[:top_k]
    if method in {"hybrid_rrf", "hybrid_weighted"}:
        metadata_hits = metadata_ranking(
            config,
            query["metadata"],
            index_rows,
            top_k=max(pool_size, top_k),
            filters=filters,
        )
        return fuse_rankings(
            config,
            image_hits,
            metadata_hits,
            method=method,
            top_k=top_k,
        )
    raise Week8RetrievalError(f"unsupported retrieval method: {method}")


def _assign_partitions(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, str]:
    """Keep business groups and duplicate original-image bytes in one partition."""
    parent: dict[str, str] = {row["group_id"]: row["group_id"] for row in rows}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    group_by_image: dict[str, str] = {}
    for row in rows:
        previous = group_by_image.setdefault(row["image_sha256"], row["group_id"])
        union(previous, row["group_id"])
    components: dict[str, list[str]] = {}
    for group in parent:
        components.setdefault(find(group), []).append(group)

    split = config["split"]
    dev_limit = _finite_number(split["development_query_group_fraction"])
    test_limit = dev_limit + _finite_number(split["final_test_query_group_fraction"])
    assignments: dict[str, str] = {}
    for groups in components.values():
        identity = "\x1f".join(sorted(groups))
        digest = hashlib.sha256(f"{split['seed']}\x1f{identity}".encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], "big") / 2**64
        partition = (
            "development_query"
            if score < dev_limit
            else "final_test_query"
            if score < test_limit
            else "index"
        )
        for group in groups:
            assignments[group] = partition
    return assignments


def _validate_partition_isolation(
    rows_by_partition: dict[str, list[dict[str, Any]]]
) -> None:
    dimensions = ("sample_id", "source_id", "image_sha256", "group_id", "template_id")
    seen: dict[str, dict[str, str]] = {field: {} for field in dimensions}
    for partition in PARTITIONS:
        for row in rows_by_partition.get(partition, []):
            for field in dimensions:
                value = row.get(field)
                if not isinstance(value, str) or not value:
                    raise Week8RetrievalError(f"missing isolation identity: {field}")
                prior = seen[field].get(value)
                if prior is not None and prior != partition:
                    raise Week8RetrievalError(
                        f"cross-partition {field} overlap: {value} ({prior}/{partition})"
                    )
                seen[field][value] = partition


def _validate_minimum_counts(
    config: dict[str, Any], rows_by_partition: dict[str, list[dict[str, Any]]]
) -> None:
    minimums = config["split"]["minimum_counts"]
    for partition in PARTITIONS:
        minimum = int(minimums[partition])
        actual = len(rows_by_partition[partition])
        if actual < minimum:
            raise Week8RetrievalError(
                f"{partition} count below fixed minimum: {actual} < {minimum}"
            )


def _validate_source(config: dict[str, Any], vectors: Any, metadata: list[dict[str, Any]]) -> None:
    source = config["source"]
    expected = int(source["expected_count"])
    dimension = int(source["vector_dimension"])
    if getattr(vectors, "shape", None) != (expected, dimension):
        raise Week8RetrievalError(
            f"vector shape mismatch: expected {(expected, dimension)}, got {getattr(vectors, 'shape', None)}"
        )
    if len(metadata) != expected:
        raise Week8RetrievalError(f"metadata count mismatch: {len(metadata)} != {expected}")
    image_ids: set[str] = set()
    for index, row in enumerate(metadata):
        missing = REQUIRED_METADATA_FIELDS - row.keys()
        if missing:
            raise Week8RetrievalError(f"metadata row {index} missing fields: {sorted(missing)}")
        if row["embedding_model"] != source["embedding_model"]:
            raise Week8RetrievalError("metadata embedding model mismatch")
        image_id = row["image_id"]
        if not isinstance(image_id, str) or not image_id or image_id in image_ids:
            raise Week8RetrievalError(f"invalid or duplicate image_id: {image_id}")
        image_ids.add(image_id)
        norm = math.sqrt(sum(float(value) ** 2 for value in vectors[index]))
        if not 0.999 <= norm <= 1.001:
            raise Week8RetrievalError(f"vector is not L2-normalized: {image_id}")


def _verify_vector_identities(vectors: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if _vector_sha256(vectors[row["vector_index"]]) != row.get("vector_sha256"):
            raise Week8RetrievalError(f"vector identity mismatch: {row.get('sample_id')}")


def _verify_locked_source_rows(
    metadata: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> None:
    for row in rows:
        vector_index = row.get("vector_index")
        if (
            isinstance(vector_index, bool)
            or not isinstance(vector_index, int)
            or not 0 <= vector_index < len(metadata)
        ):
            raise Week8RetrievalError(f"invalid locked vector index: {vector_index}")
        source_row = metadata[vector_index]
        if row.get("metadata") != source_row:
            raise Week8RetrievalError(f"locked metadata mismatch: {row.get('sample_id')}")
        image_id = source_row["image_id"]
        if row.get("sample_id") != f"week8-retrieval:{image_id}":
            raise Week8RetrievalError(f"locked sample identity mismatch: {image_id}")
        if row.get("source_id") != f"yelp-photo:{image_id}":
            raise Week8RetrievalError(f"locked source identity mismatch: {image_id}")


def _relevance_grade(
    config: dict[str, Any], query: dict[str, Any], candidate: dict[str, Any]
) -> float:
    grade = 0.0
    for field, weight in config["evaluation"]["relevance_weights"].items():
        query_value = query.get(field)
        if query_value != "unknown" and query_value == candidate.get(field):
            grade += _finite_number(weight)
    return grade


def _ndcg_at_k(grades: dict[str, float], ranked_ids: list[str], k: int) -> float:
    actual = [_finite_number(grades.get(sample_id, 0.0)) for sample_id in ranked_ids[:k]]
    ideal = sorted((_finite_number(value) for value in grades.values()), reverse=True)[:k]

    def dcg(values: list[float]) -> float:
        return sum((2**value - 1) / math.log2(rank + 2) for rank, value in enumerate(values))

    ideal_score = dcg(ideal)
    return dcg(actual) / ideal_score if ideal_score else 0.0


def _query_filters(metadata: dict[str, Any], fields: list[str]) -> dict[str, str]:
    filters = {}
    for field in fields:
        value = metadata.get(field)
        if isinstance(value, str) and value and value != "unknown":
            filters[field] = value
    return filters


def _matches_filters(metadata: dict[str, Any], filters: dict[str, str]) -> bool:
    return all(metadata.get(field) == value for field, value in filters.items())


def _traceable_reference(
    query: dict[str, Any], hit: dict[str, Any], method: str, consumer: str, rank: int
) -> dict[str, Any]:
    metadata = hit["metadata"]
    traceable = all(isinstance(metadata.get(field), str) and metadata[field] for field in TRACE_FIELDS)
    return {
        "consumer_path": consumer,
        "method": method,
        "query_sample_id": query["sample_id"],
        "rank": rank,
        "citation_id": f"yelp:{metadata.get('business_id')}:image:{metadata.get('image_id')}",
        "business_id": metadata.get("business_id"),
        "image_id": metadata.get("image_id"),
        "source_image_path": metadata.get("source_image_path"),
        "embedding_model": metadata.get("embedding_model"),
        "source_release": "trip-qwen3-vl-8b-system-repair-v1-rc1",
        "traceable": traceable,
    }


def _resolve_source_image(project_root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
        raise Week8RetrievalError("source_image_path must be non-empty and project-relative")
    resolved = (project_root / relative_path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise Week8RetrievalError(f"source image escapes project root: {relative_path}") from exc
    if not resolved.is_file():
        raise Week8RetrievalError(f"source image is missing: {relative_path}")
    return resolved


def _parse_metadata(content: bytes) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Week8RetrievalError(f"invalid metadata line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise Week8RetrievalError(f"metadata line {line_number} is not an object")
        rows.append(row)
    return rows


def _read_tar_member(handle: tarfile.TarFile, name: str) -> bytes:
    try:
        member = handle.getmember(name)
        extracted = handle.extractfile(member)
    except (KeyError, OSError, tarfile.TarError) as exc:
        raise Week8RetrievalError(f"retrieval archive member is missing: {name}") from exc
    if extracted is None:
        raise Week8RetrievalError(f"retrieval archive member is not a file: {name}")
    return extracted.read()


def _dot(left: Any, right: Any) -> float:
    try:
        return float(left @ right)
    except (TypeError, ValueError):
        return sum(float(a) * float(b) for a, b in zip(left, right))


def _vector_sha256(vector: Any) -> str:
    try:
        content = vector.astype("float32", copy=False).tobytes(order="C")
    except AttributeError:
        import struct

        content = b"".join(struct.pack("<f", float(value)) for value in vector)
    return _sha256_bytes(content)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Week8RetrievalError(f"expected finite number, got {value!r}")
    return float(value)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Week8RetrievalError(f"invalid JSONL line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise Week8RetrievalError(f"JSONL line {line_number} is not an object")
        rows.append(row)
    return rows


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        handle.write("\n")


def _write_jsonl_new(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")

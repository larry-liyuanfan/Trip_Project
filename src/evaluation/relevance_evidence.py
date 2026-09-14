"""Fail-closed scoring contracts for retrieval, VLM, and latency evidence.

The module intentionally keeps ANN fidelity, business relevance, model semantics,
and system timing as four separate evidence scopes.  A value from one scope must
never be promoted as evidence for another.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_QUERY_SLICES = {
    "image_similar",
    "city_business_facility_price",
    "visual_similar_business_irrelevant",
    "no_result",
    "filter_conflict",
}
SEARCH_METHODS = (
    "clip_exact",
    "clip_milvus",
    "structured_filter_clip",
    "lightweight_rerank",
)
PRODUCT_FIELDS = {
    "business_category": "scalar",
    "style_tags": "set",
    "visible_facilities": "set",
    "price_range": "scalar",
}


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON using the repository's stable UTF-8 representation."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid JSONL line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            records.append(value)
    return records


def validate_query_manifest(
    records: list[dict[str, Any]],
    asset_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate independent query provenance and return an auditable lock."""
    if not records:
        raise ValueError("query manifest must not be empty")
    query_ids: set[str] = set()
    observed_slices: set[str] = set()
    asset_hashes: set[str] = set()
    source_ids: set[str] = set()
    for record in records:
        query_id = _required_text(record, "query_id")
        if query_id in query_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        query_ids.add(query_id)
        slices = record.get("slices")
        if not isinstance(slices, list) or not all(isinstance(item, str) for item in slices):
            raise ValueError(f"{query_id}: slices must be a string list")
        observed_slices.update(slices)

        source = record.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"{query_id}: source object is required")
        for key in ("source_id", "page_url", "download_url", "license", "author"):
            _required_text(source, key, prefix=f"{query_id}.source")
        if source.get("dataset_relation") not in {
            "independent_public_source_not_yelp",
            "deterministic_synthetic_not_yelp",
        }:
            raise ValueError(f"{query_id}: source independence is not explicit")
        source_id = source["source_id"]
        source_ids.add(source_id)
        claimed_source_hash = record.get("source_record_sha256")
        actual_source_hash = canonical_json_sha256(source)
        if claimed_source_hash != actual_source_hash:
            raise ValueError(f"{query_id}: source_record_sha256 mismatch")

        image = record.get("image")
        if not isinstance(image, dict):
            raise ValueError(f"{query_id}: image object is required")
        relative_path = _required_text(image, "relative_path", prefix=f"{query_id}.image")
        image_hash = _required_sha(image, "sha256", f"{query_id}.image")
        asset_hashes.add(image_hash)
        if asset_dir is not None:
            asset_path = (Path(asset_dir) / relative_path).resolve()
            root = Path(asset_dir).resolve()
            if root not in asset_path.parents:
                raise ValueError(f"{query_id}: asset path escapes the asset directory")
            if not asset_path.is_file():
                raise ValueError(f"{query_id}: missing query asset {relative_path}")
            if file_sha256(asset_path) != image_hash:
                raise ValueError(f"{query_id}: query image sha256 mismatch")

        filters = record.get("requested_filters")
        if not isinstance(filters, dict):
            raise ValueError(f"{query_id}: requested_filters must be an object")
        unsupported = record.get("unsupported_constraints")
        if not isinstance(unsupported, list):
            raise ValueError(f"{query_id}: unsupported_constraints must be a list")
        unsigned = {key: value for key, value in record.items() if key != "query_sha256"}
        if record.get("query_sha256") != canonical_json_sha256(unsigned):
            raise ValueError(f"{query_id}: query_sha256 mismatch")

    missing = sorted(REQUIRED_QUERY_SLICES - observed_slices)
    if missing:
        raise ValueError(f"query manifest misses required slices: {missing}")
    return {
        "status": "PASS",
        "query_count": len(records),
        "query_ids_unique": True,
        "asset_sha256_unique_count": len(asset_hashes),
        "source_id_unique_count": len(source_ids),
        "required_slices": sorted(REQUIRED_QUERY_SLICES),
        "observed_slices": sorted(observed_slices),
        "source_isolation": "PASS_PUBLIC_OR_DETERMINISTIC_SYNTHETIC_NOT_YELP",
        "byte_level_index_query_collision_check": "NOT_RUN_MISSING_INDEX_IMAGE_SHA",
        "promotion_eligible_as_human_ground_truth": False,
        "manifest_sha256": canonical_json_sha256(records),
    }


def validate_annotation_protocol(
    queries: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reject weak labels presented as human review and incomplete query support."""
    expected = {record["query_id"] for record in queries}
    actual: set[str] = set()
    counts: Counter[str] = Counter()
    for record in annotations:
        query_id = _required_text(record, "query_id")
        if query_id in actual:
            raise ValueError(f"duplicate annotation query_id: {query_id}")
        actual.add(query_id)
        provenance = _required_text(record, "label_provenance")
        annotators = record.get("annotators")
        if not isinstance(annotators, list) or not all(
            isinstance(item, str) and item for item in annotators
        ):
            raise ValueError(f"{query_id}: annotators must be a non-empty string list")
        if provenance == "human":
            if len(annotators) < 2:
                raise ValueError(f"{query_id}: human labels require two annotators")
            if record.get("conflict_resolution") not in {"none", "adjudicated"}:
                raise ValueError(f"{query_id}: human conflicts require adjudication status")
        elif provenance in {"weak_programmatic_metadata", "synthetic"}:
            if any(not item.startswith("programmatic_") for item in annotators):
                raise ValueError(f"{query_id}: weak labels cannot name a human annotator")
            if record.get("conflict_resolution") != "not_applicable_single_programmatic":
                raise ValueError(f"{query_id}: weak labels require explicit conflict scope")
        else:
            raise ValueError(f"{query_id}: unsupported label provenance {provenance}")
        rules = record.get("grade_rules")
        if not isinstance(rules, dict) or not isinstance(rules.get("grades"), dict):
            raise ValueError(f"{query_id}: grade_rules.grades is required")
        counts[provenance] += 1
    if actual != expected:
        raise ValueError(
            f"annotation/query support mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return {
        "status": "PASS",
        "support": len(annotations),
        "provenance_support": dict(sorted(counts.items())),
        "human_review_support": counts["human"],
        "promotion_eligible_as_human_ground_truth": counts["human"] == len(annotations),
        "annotation_sha256": canonical_json_sha256(annotations),
    }


def validate_asset_source_registry(
    queries: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    asset_dir: Path | None = None,
) -> dict[str, Any]:
    """Bind stable Commons resolver records to the exact downloaded 960px bytes."""
    index: dict[str, dict[str, Any]] = {}
    for record in registry:
        source_id = _required_text(record, "source_id")
        if source_id in index:
            raise ValueError(f"duplicate asset registry source_id: {source_id}")
        url = _required_text(record, "exact_asset_url", prefix=source_id)
        if not url.startswith("https://upload.wikimedia.org/"):
            raise ValueError(f"{source_id}: exact asset URL must use Wikimedia upload")
        _required_sha(record, "sha256", source_id)
        _required_text(record, "relative_path", prefix=source_id)
        if not all(isinstance(record.get(key), int) and record[key] > 0 for key in ("width", "height")):
            raise ValueError(f"{source_id}: width and height must be positive integers")
        index[source_id] = record
    query_sources = {query["source"]["source_id"] for query in queries}
    if set(index) != query_sources:
        raise ValueError("asset source registry support differs from query sources")
    for query in queries:
        record = index[query["source"]["source_id"]]
        image = query["image"]
        for key in ("relative_path", "sha256", "width", "height"):
            if image.get(key) != record.get(key):
                raise ValueError(f"{query['query_id']}: asset registry {key} mismatch")
        if asset_dir is not None and file_sha256(Path(asset_dir) / record["relative_path"]) != record["sha256"]:
            raise ValueError(f"{query['query_id']}: exact registry asset SHA mismatch")
    return {
        "status": "PASS",
        "source_support": len(registry),
        "exact_asset_urls_bound": True,
        "registry_sha256": canonical_json_sha256(registry),
    }


def score_search_results(
    queries: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    results: list[dict[str, Any]],
    methods: Iterable[str] = SEARCH_METHODS,
) -> dict[str, Any]:
    """Score business semantics independently from ANN exact-neighbour fidelity."""
    query_index = {item["query_id"]: item for item in queries}
    annotation_index = {item["query_id"]: item for item in annotations}
    result_index = {item["query_id"]: item for item in results}
    if set(result_index) != set(query_index):
        raise ValueError("search result support must exactly match the query lock")
    output: dict[str, Any] = {
        "scope": "business_semantic_relevance",
        "query_support": len(queries),
        "methods": {},
        "query_manifest_sha256": canonical_json_sha256(queries),
        "annotation_sha256": canonical_json_sha256(annotations),
        "result_sha256": canonical_json_sha256(results),
    }
    for method in methods:
        per_query: list[dict[str, Any]] = []
        failures = 0
        for query_id, query in query_index.items():
            row = result_index[query_id]
            method_result = row.get("methods", {}).get(method)
            if not isinstance(method_result, dict):
                failures += 1
                per_query.append(_failed_query_metrics(query_id, query))
                continue
            hits = method_result.get("hits")
            if not isinstance(hits, list):
                failures += 1
                per_query.append(_failed_query_metrics(query_id, query))
                continue
            grades = [_weak_grade(hit, annotation_index[query_id]) for hit in hits]
            expected_no_result = "no_result" in query.get("slices", [])
            predicted_no_result = bool(method_result.get("no_result"))
            filters = query.get("requested_filters", {})
            supported_filters = {
                key: value for key, value in filters.items() if key in {"city", "business_category", "price_range"}
            }
            filter_checks = [
                all(_norm(hit.get(key)) == _norm(value) for key, value in supported_filters.items())
                for hit in hits
            ]
            relevant = [grade >= 2 for grade in grades]
            relevant_total = max(int(method_result.get("relevant_total", sum(relevant))), 0)
            per_query.append(
                {
                    "query_id": query_id,
                    "slices": query.get("slices", []),
                    "ranking_evaluable": not expected_no_result,
                    "recall_at_5": _recall(relevant[:5], relevant_total),
                    "recall_at_10": _recall(relevant[:10], relevant_total),
                    "mrr_at_10": _mrr(grades[:10]),
                    "ndcg_at_10": _ndcg(grades[:10]),
                    "no_result_correct": expected_no_result == predicted_no_result,
                    "filter_correct": all(filter_checks) if supported_filters and hits else (
                        predicted_no_result if supported_filters else True
                    ),
                    "unsupported_constraints_unapplied": set(query.get("unsupported_constraints", []))
                    == set(method_result.get("unsupported_constraints_unapplied", [])),
                    "failed": False,
                }
            )
        output["methods"][method] = _aggregate_search_method(per_query, failures)
    return output


def score_ann_fidelity(records: list[dict[str, Any]], top_k: int = 10) -> dict[str, Any]:
    """Score ANN result IDs against exact IDs; this says nothing about usefulness."""
    recalls: list[float] = []
    for row in records:
        exact = row.get("exact_ids")
        ann = row.get("ann_ids")
        if not isinstance(exact, list) or not isinstance(ann, list):
            raise ValueError("ANN fidelity rows require exact_ids and ann_ids")
        exact_set = set(exact[:top_k])
        recalls.append(len(exact_set & set(ann[:top_k])) / max(len(exact_set), 1))
    return {
        "scope": "ann_vs_exact_only",
        "metric": f"Recall@{top_k}",
        "value": statistics.fmean(recalls) if recalls else None,
        "query_support": len(recalls),
        "business_semantic_relevance_supported": False,
    }


def score_vlm_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one-factor model variants under an exact shared sample lock."""
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_variant[_required_text(record, "variant")].append(record)
    if not by_variant:
        raise ValueError("VLM comparison records must not be empty")
    locks = {
        (
            record.get("data_lock_sha256"),
            record.get("base_model"),
            record.get("base_revision"),
            record.get("prompt_sha256"),
            record.get("generation_config_sha256"),
        )
        for record in records
    }
    if len(locks) != 1:
        raise ValueError("VLM variants differ by more than the adapter factor")
    sample_sets = {
        variant: {row.get("sample_id") for row in rows}
        for variant, rows in by_variant.items()
    }
    if len({frozenset(items) for items in sample_sets.values()}) != 1:
        raise ValueError("VLM variants do not have identical sample support")

    variants: dict[str, Any] = {}
    for variant, rows in sorted(by_variant.items()):
        field_counts = {field: [0, 0, 0] for field in PRODUCT_FIELDS}
        field_known_support = Counter()
        field_unknown_support = Counter()
        field_unknown_abstentions = Counter()
        exact_matches: list[float] = []
        hallucinations = 0
        unknown_opportunities = 0
        first_attempt_valid: list[float] = []
        correction_triggered: list[float] = []
        dialogue = Counter()
        dialogue_support = 0
        for row in rows:
            gold = row.get("gold")
            prediction = row.get("prediction")
            if not isinstance(gold, dict) or not isinstance(prediction, dict):
                raise ValueError("VLM rows require gold and prediction objects")
            first_attempt_valid.append(float(row.get("first_attempt_json_valid") is True))
            correction_triggered.append(float(row.get("correction_triggered") is True))
            if row.get("scenario") == "dialogue":
                dialogue_support += 1
                for field in ("context_recall", "state_value_correct", "task_key_correct", "value_correct", "first_turn_routing_correct"):
                    dialogue[field] += int(bool(row.get(field)))
                continue
            unknown = set(gold.get("unknown_fields", []))
            row_exact = True
            for field, kind in PRODUCT_FIELDS.items():
                if kind == "set":
                    expected_values = gold.get(field, [])
                    actual_values = prediction.get(field, [])
                    expected = {
                        _norm(item) for item in expected_values if _norm(item)
                    } if isinstance(expected_values, list) else set()
                    actual = {
                        _norm(item) for item in actual_values if _norm(item)
                    } if isinstance(actual_values, list) else set()
                else:
                    expected = {_norm(gold.get(field))} - {"", "unknown"}
                    actual = {_norm(prediction.get(field))} - {"", "unknown"}
                tp = len(expected & actual)
                fp = len(actual - expected)
                fn = len(expected - actual)
                field_counts[field][0] += tp
                field_counts[field][1] += fp
                field_counts[field][2] += fn
                if field in unknown:
                    field_unknown_support[field] += 1
                    unknown_opportunities += 1
                    if actual:
                        hallucinations += 1
                    else:
                        field_unknown_abstentions[field] += 1
                else:
                    field_known_support[field] += 1
                    row_exact = row_exact and expected == actual
            exact_matches.append(float(row_exact))
        fields = {}
        for field, counts in field_counts.items():
            metrics = _prf(*counts)
            metrics["evaluable_reference_support"] = field_known_support[field]
            metrics["unknown_reference_support"] = field_unknown_support[field]
            metrics["unknown_abstention_accuracy"] = (
                field_unknown_abstentions[field] / field_unknown_support[field]
                if field_unknown_support[field] else None
            )
            if field_known_support[field] == 0:
                metrics.update({
                    "status": "NOT_APPLICABLE_NO_SUPPORTED_REFERENCE",
                    "precision": None,
                    "recall": None,
                    "f1": None,
                })
            else:
                metrics["status"] = "EVALUATED"
            fields[field] = metrics
        variants[variant] = {
            "support": len(rows),
            "product_support": len(exact_matches),
            "field_metrics": fields,
            "supported_field_exact_match": _mean(exact_matches),
            "unsupported_hallucination_rate": hallucinations / max(unknown_opportunities, 1),
            "unknown_field_opportunity_support": unknown_opportunities,
            "unknown_field_abstention_accuracy": 1 - hallucinations / max(unknown_opportunities, 1),
            "first_attempt_json_compliance": _mean(first_attempt_valid),
            "correction_trigger_rate": _mean(correction_triggered),
            "dialogue_support": dialogue_support,
            "dialogue_metrics": {
                key: dialogue[key] / max(dialogue_support, 1)
                for key in ("context_recall", "state_value_correct", "task_key_correct", "value_correct", "first_turn_routing_correct")
            },
        }
    return {
        "scope": "vlm_semantic_one_factor_comparison",
        "shared_lock": list(locks)[0],
        "variants": variants,
    }


def summarize_performance(
    records: list[dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Summarize cold/steady end-to-end timing and apply fixed, fail-closed gates."""
    if not records:
        raise ValueError("performance records must not be empty")
    cold = [row for row in records if row.get("phase") == "cold"]
    steady = [row for row in records if row.get("phase") == "steady"]
    required_stages = ("clip_encode_ms", "milvus_ms", "rerank_ms", "vlm_ms", "end_to_end_ms")
    for row in records:
        missing = [stage for stage in required_stages if not isinstance(row.get(stage), (int, float))]
        if missing:
            raise ValueError(f"performance row misses stages: {missing}")
    stage_summary = {
        phase: {
            stage: _latency_stats([float(row[stage]) for row in rows])
            for stage in required_stages
        }
        for phase, rows in (("cold", cold), ("steady", steady))
    }
    failures = sum(int(row.get("failed") is True) for row in records)
    duration_s = sum(float(row["end_to_end_ms"]) for row in steady) / 1000.0
    summary = {
        "scope": "end_to_end_stage_timing",
        "support": {"cold": len(cold), "steady": len(steady)},
        "stages": stage_summary,
        "peak_vram_mib": max((float(row.get("peak_vram_mib", 0)) for row in records), default=0),
        "throughput_queries_per_second": len(steady) / max(duration_s, 1e-12),
        "failure_rate": failures / len(records),
        "hardware": records[0].get("hardware"),
    }
    checks = {
        "steady_end_to_end_p95_ms": summary["stages"]["steady"]["end_to_end_ms"]["p95"]
        <= float(gates["max_steady_end_to_end_p95_ms"]),
        "failure_rate": summary["failure_rate"] <= float(gates["max_failure_rate"]),
        "peak_vram_mib": summary["peak_vram_mib"] <= float(gates["max_peak_vram_mib"]),
        "minimum_steady_support": len(steady) >= int(gates["min_steady_repetitions"]),
        "cold_support": len(cold) >= int(gates["min_cold_repetitions"]),
    }
    summary["fixed_gates"] = {"thresholds": gates, "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}
    return summary


def compare_performance(
    candidate_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Compare candidate and baseline on identical hardware/support without tuning gates."""
    candidate = summarize_performance(candidate_records, gates)
    baseline = summarize_performance(baseline_records, gates)
    candidate_p95 = candidate["stages"]["steady"]["end_to_end_ms"]["p95"]
    baseline_p95 = baseline["stages"]["steady"]["end_to_end_ms"]["p95"]
    ratio = candidate_p95 / max(baseline_p95, 1e-12)
    checks = {
        "candidate_absolute_gates": candidate["fixed_gates"]["status"] == "PASS",
        "same_hardware": candidate["hardware"] == baseline["hardware"],
        "same_cold_support": candidate["support"]["cold"] == baseline["support"]["cold"],
        "same_steady_support": candidate["support"]["steady"] == baseline["support"]["steady"],
        "candidate_to_baseline_p95_ratio": ratio
        <= float(gates["max_candidate_to_baseline_p95_ratio"]),
    }
    return {
        "scope": "candidate_vs_baseline_end_to_end_performance",
        "candidate": candidate,
        "baseline": baseline,
        "candidate_to_baseline_steady_p95_ratio": ratio,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _weak_grade(hit: dict[str, Any], annotation: dict[str, Any]) -> int:
    rules = annotation["grade_rules"]
    category = _norm(rules.get("target_business_category"))
    required = rules.get("required_metadata", {})
    excluded = {_norm(item) for item in rules.get("excluded_business_categories", [])}
    hit_category = _norm(hit.get("business_category"))
    if hit_category in excluded:
        return 0
    required_match = all(_norm(hit.get(key)) == _norm(value) for key, value in required.items())
    if category and hit_category == category and required_match:
        return 3
    if category and hit_category == category:
        return 2
    if hit_category and hit_category not in excluded:
        return 1
    return 0


def _aggregate_search_method(per_query: list[dict[str, Any]], failures: int) -> dict[str, Any]:
    ranking = [row for row in per_query if row["ranking_evaluable"] and not row["failed"]]
    slices: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        for slice_name in row["slices"]:
            slices[slice_name].append(row)
    return {
        "support": len(per_query),
        "ranking_support": len(ranking),
        "recall_at_5": _mean([row["recall_at_5"] for row in ranking]),
        "recall_at_10": _mean([row["recall_at_10"] for row in ranking]),
        "mrr_at_10": _mean([row["mrr_at_10"] for row in ranking]),
        "ndcg_at_10": _mean([row["ndcg_at_10"] for row in ranking]),
        "no_result_accuracy": _mean([float(row["no_result_correct"]) for row in per_query]),
        "no_result_rate": _mean([float("no_result" in row["slices"]) for row in per_query]),
        "filter_correctness": _mean([float(row["filter_correct"]) for row in per_query]),
        "unsupported_constraint_disclosure": _mean([float(row["unsupported_constraints_unapplied"]) for row in per_query]),
        "failure_rate": failures / len(per_query),
        "slices": {
            name: {
                "support": len(rows),
                "ranking_support": sum(bool(row["ranking_evaluable"] and not row["failed"]) for row in rows),
                "ndcg_at_10": _mean([row["ndcg_at_10"] for row in rows if row["ranking_evaluable"] and not row["failed"]]),
                "no_result_accuracy": _mean([float(row["no_result_correct"]) for row in rows]),
                "filter_correctness": _mean([float(row["filter_correct"]) for row in rows]),
            }
            for name, rows in sorted(slices.items())
        },
        "per_query": per_query,
    }


def _failed_query_metrics(query_id: str, query: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "slices": query.get("slices", []),
        "ranking_evaluable": "no_result" not in query.get("slices", []),
        "recall_at_5": 0.0,
        "recall_at_10": 0.0,
        "mrr_at_10": 0.0,
        "ndcg_at_10": 0.0,
        "no_result_correct": False,
        "filter_correct": False,
        "unsupported_constraints_unapplied": False,
        "failed": True,
    }


def _recall(relevant: list[bool], total: int) -> float:
    return sum(relevant) / max(total, 1)


def _mrr(grades: list[int]) -> float:
    for index, grade in enumerate(grades, start=1):
        if grade >= 2:
            return 1.0 / index
    return 0.0


def _ndcg(grades: list[int]) -> float:
    dcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))
    ideal = sorted(grades, reverse=True)
    idcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def _latency_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        "p95": _percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _norm(value: Any) -> str:
    return " ".join(value.strip().casefold().split()) if isinstance(value, str) else ""


def _required_text(record: dict[str, Any], key: str, prefix: str = "record") -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}.{key} must be non-empty text")
    return value


def _required_sha(record: dict[str, Any], key: str, prefix: str) -> str:
    value = _required_text(record, key, prefix)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{prefix}.{key} must be a lowercase SHA-256")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")

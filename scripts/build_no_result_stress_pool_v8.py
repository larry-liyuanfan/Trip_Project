"""Build a synthetic calibration/validation pool for no-result stress testing."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_exploration_pool_v4 import (
    HEIGHT,
    WIDTH,
    _build_search_split as _build_v4_search_split,
    _write_card,
    _write_json,
    _write_jsonl,
)
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256


GENERATOR_VERSION = "trip_no_result_stress_pool_v8"
SPLITS = ("calibration", "validation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-v4-lock", type=Path, required=True)
    parser.add_argument("--expected-lock", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    lock = build_pool(args.output_dir, args.prior_v4_lock)
    if args.expected_lock:
        expected = json.loads(args.expected_lock.read_text(encoding="utf-8"))
        if lock != expected:
            raise ValueError("generated no-result pool differs from committed lock")
    print(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True))


def build_pool(output_dir: Path, prior_v4_lock_path: Path) -> dict[str, Any]:
    prior_queries, prior_lock = _regenerate_prior_v4_training(prior_v4_lock_path)
    output_dir.mkdir(parents=True)
    registry: list[dict[str, Any]] = []
    manifests: dict[str, list[dict[str, Any]]] = {}
    split_locks: dict[str, Any] = {}
    for split in SPLITS:
        queries, annotations = _build_split(output_dir, split, registry)
        manifests[split] = queries
        query_path = output_dir / f"search_{split}_manifest.jsonl"
        annotation_path = output_dir / f"search_{split}_annotations.jsonl"
        _write_jsonl(query_path, queries)
        _write_jsonl(annotation_path, annotations)
        split_locks[split] = {
            "query_support": len(queries),
            "ranking_support": sum("no_result" not in row["slices"] for row in queries),
            "no_result_support": sum("no_result" in row["slices"] for row in queries),
            "business_positive_support": sum("business_positive" in row["slices"] for row in queries),
            "query_manifest_canonical_sha256": canonical_json_sha256(queries),
            "query_manifest_file_sha256": file_sha256(query_path),
            "annotation_canonical_sha256": canonical_json_sha256(annotations),
            "annotation_file_sha256": file_sha256(annotation_path),
        }
    isolation = _isolation(manifests)
    prior_isolation = _prior_isolation(manifests, prior_queries)
    lock = {
        "schema_version": "no_result_stress_pool_lock_v8",
        "generator_version": GENERATOR_VERSION,
        "evidence_class": "deterministic_synthetic_calibration_and_one_time_validation_only",
        "human_annotation_support": 0,
        "image_encoding": "binary_ppm_p6_rgb_384x256",
        "splits": list(SPLITS),
        "final_policy": "not_defined_not_generated_not_opened",
        "validation_policy": "exclusive_marker_before_first_manifest_or_annotation_open",
        "search": split_locks,
        "asset_registry_support": len(registry),
        "asset_registry_canonical_sha256": canonical_json_sha256(registry),
        "split_identity_policy": "query_id_source_id_and_image_sha256_disjoint",
        "isolation": isolation,
        "prior_v4_training_lock_sha256": canonical_json_sha256(prior_lock),
        "prior_v4_training_isolation": prior_isolation,
    }
    _write_json(output_dir / "asset_registry.json", registry)
    _write_json(output_dir / "bundle_lock.json", lock)
    return lock


def _build_split(
    output_dir: Path, split: str, registry: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing_filters = (
        ("restaurant", "Philadelphia", "budget"),
        ("hotel", "Nashville", "mid_range"),
        ("attraction", "Philadelphia", "mid_range"),
        ("restaurant", "New Orleans", "premium"),
    )
    impossible_filters = (
        ("restaurant", "Atlantis", "budget"),
        ("hotel", "El Dorado", "mid_range"),
        ("attraction", "Shangri La", "premium"),
        ("restaurant", "Emerald City", "luxury"),
    )
    non_business = {
        "calibration": ("home", "office", "shop", "warehouse", "vehicle", "landscape"),
        "validation": ("kitchen", "bedroom", "classroom", "clinic", "factory", "garage"),
    }[split]
    definitions: list[dict[str, Any]] = []
    categories = ("restaurant", "hotel", "attraction")
    for index in range(12):
        category = categories[index % len(categories)]
        definitions.append({
            "case": f"positive_open_{index:02d}",
            "visual": category,
            "detail": ("interior", "entrance", "service")[index % 3],
            "slices": ["business_positive", "ranking", "no_filter", "image_similar"],
            "filters": {},
            "target": category,
            "required": {},
            "excluded": [],
            "guard": "business",
        })
    for repeat in range(2):
        for category, city, price in existing_filters:
            definitions.append({
                "case": f"positive_filter_{repeat}_{category}_{city}",
                "visual": category,
                "detail": "travel",
                "slices": ["business_positive", "ranking", "hard_filter", "city_business_facility_price"],
                "filters": {"city": city, "business_category": category, "price_range": price},
                "target": category,
                "required": {"city": city, "price_range": price},
                "excluded": [],
                "guard": "business",
            })
    for index in range(12):
        visual = non_business[index % len(non_business)]
        definitions.append({
            "case": f"non_business_{index:02d}",
            "visual": visual,
            "detail": ("private", "ordinary", "unrelated")[index % 3],
            "slices": ["no_result", "visual_similar_business_irrelevant", "no_filter"],
            "filters": {},
            "target": "",
            "required": {},
            "excluded": ["restaurant", "hotel", "attraction"],
            "guard": "non_business",
        })
    for repeat in range(2):
        for category, city, price in impossible_filters:
            definitions.append({
                "case": f"filter_empty_{repeat}_{category}_{city}",
                "visual": category,
                "detail": "travel",
                "slices": ["no_result", "filter_empty", "hard_filter", "filter_conflict"],
                "filters": {"city": city, "business_category": category, "price_range": price},
                "target": category,
                "required": {"city": city, "price_range": price},
                "excluded": [],
                "guard": "business",
            })
    queries: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    split_offset = {"calibration": 11000, "validation": 13000}[split]
    for index, definition in enumerate(definitions, start=1):
        query_id = f"v8_{split[:3]}_search_{index:02d}"
        relative_path = f"assets/search/{split}/{query_id}.ppm"
        path = output_dir / relative_path
        _write_card(
            path,
            [
                "TRIP V8 SEARCH",
                f"SPLIT {split.upper()}",
                f"SCENE {definition['visual'].upper()}",
                f"DETAIL {definition['detail'].upper()}",
                f"VARIANT {split_offset + index}",
            ],
            split_offset + index,
        )
        image_sha = file_sha256(path)
        source = {
            "source_id": f"synthetic:{GENERATOR_VERSION}:{split}:{query_id}",
            "page_url": f"synthetic://{GENERATOR_VERSION}/{split}/{query_id}",
            "download_url": f"synthetic://{GENERATOR_VERSION}/{split}/{query_id}.ppm",
            "license": "programmatically_generated_test_asset",
            "author": GENERATOR_VERSION,
            "dataset_relation": "deterministic_synthetic_not_yelp",
        }
        query = {
            "query_id": query_id,
            "split": split,
            "slices": definition["slices"],
            "image": {"relative_path": relative_path, "sha256": image_sha, "width": WIDTH, "height": HEIGHT},
            "source": source,
            "source_record_sha256": canonical_json_sha256(source),
            "query_text": "Find a relevant indexed travel business or abstain when the request has no valid match.",
            "requested_filters": definition["filters"],
            "unsupported_constraints": [],
        }
        query["query_sha256"] = canonical_json_sha256(query)
        queries.append(query)
        annotations.append({
            "query_id": query_id,
            "label_provenance": "synthetic",
            "annotators": ["programmatic_no_result_rule_v8"],
            "conflict_resolution": "not_applicable_single_programmatic",
            "business_guard_label": definition["guard"],
            "grade_rules": {
                "target_business_category": definition["target"],
                "required_metadata": definition["required"],
                "excluded_business_categories": definition["excluded"],
                "grades": {
                    "3": "target category and all required formal metadata",
                    "2": "target category with incomplete formal metadata",
                    "1": "other non-excluded indexed business",
                    "0": "excluded or unrelated result",
                },
            },
        })
        registry.append({
            "record_id": query_id,
            "source_id": source["source_id"],
            "relative_path": relative_path,
            "sha256": image_sha,
            "width": WIDTH,
            "height": HEIGHT,
            "purpose": "search_no_result_stress",
            "split": split,
            "contains_image_bytes": False,
        })
    return queries, annotations


def _regenerate_prior_v4_training(lock_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "exploration_pool_lock_v4":
        raise ValueError("prior lock is not the v4 exploration pool lock")
    with tempfile.TemporaryDirectory(prefix="trip-v4-search-training-reference-") as temp_dir:
        root = Path(temp_dir)
        registry: list[dict[str, Any]] = []
        queries, annotations = _build_v4_search_split(root, "training", registry)
        query_path = root / "search_training_manifest.jsonl"
        annotation_path = root / "search_training_annotations.jsonl"
        _write_jsonl(query_path, queries)
        _write_jsonl(annotation_path, annotations)
        expected = lock["search"]["training"]
        observed = {
            "query_support": len(queries),
            "query_manifest_canonical_sha256": canonical_json_sha256(queries),
            "query_manifest_file_sha256": file_sha256(query_path),
            "annotation_canonical_sha256": canonical_json_sha256(annotations),
            "annotation_file_sha256": file_sha256(annotation_path),
        }
        if observed != expected:
            raise ValueError("regenerated v4 search training reference differs from its committed lock")
    return queries, lock


def _identity_sets(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        "query_id": {str(row["query_id"]) for row in rows},
        "source_id": {str(row["source"]["source_id"]) for row in rows},
        "image_sha256": {str(row["image"]["sha256"]) for row in rows},
    }


def _isolation(manifests: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    left, right = (_identity_sets(manifests[split]) for split in SPLITS)
    overlap = {kind: sorted(left[kind] & right[kind]) for kind in left}
    if any(overlap.values()):
        raise ValueError(f"v8 calibration/validation leakage: {overlap}")
    return {"status": "PASS", "calibration_vs_validation": overlap}


def _prior_isolation(
    manifests: dict[str, list[dict[str, Any]]], prior_queries: list[dict[str, Any]]
) -> dict[str, Any]:
    prior = _identity_sets(prior_queries)
    overlap = {
        split: {kind: sorted(values[kind] & prior[kind]) for kind in values}
        for split, values in ((name, _identity_sets(rows)) for name, rows in manifests.items())
    }
    if any(items for split in overlap.values() for items in split.values()):
        raise ValueError(f"v8 overlaps prior v4 training: {overlap}")
    return {"status": "PASS", "compared_prior_query_support": len(prior_queries), "overlaps": overlap}


if __name__ == "__main__":
    main()

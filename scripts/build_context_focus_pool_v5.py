"""Build a deterministic, leak-isolated synthetic VLM pool for context-focus cycle v5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_exploration_pool_v4 import (
    DIALOGUE_PROMPT,
    HEIGHT,
    PRODUCT_PROMPT,
    WIDTH,
    _write_card,
    _write_json,
    _write_jsonl,
)
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256


GENERATOR_VERSION = "trip_context_focus_pool_v5"
SPLITS = ("training", "development", "final")
COUNTS = {
    "training": {"known": 96, "unknown": 24, "multi": 24, "dialogue": 384},
    "development": {"known": 12, "unknown": 6, "multi": 6, "dialogue": 24},
    "final": {"known": 12, "unknown": 6, "multi": 6, "dialogue": 24},
}
SPLIT_OFFSET = {"training": 3000, "development": 4000, "final": 5000}
SPLIT_CITIES = {
    "training": ("Austin", "Boston", "Chicago", "Denver", "Miami", "Seattle"),
    "development": ("Portland", "Phoenix", "Dallas", "Savannah", "Orlando", "San Diego"),
    "final": ("Cleveland", "Detroit", "Houston", "Baltimore", "Charlotte", "Memphis"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-lock", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    lock = build_pool(args.output_dir)
    if args.expected_lock:
        expected = json.loads(args.expected_lock.read_text(encoding="utf-8"))
        if lock != expected:
            raise ValueError("generated context-focus pool differs from committed lock")
    print(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True))


def build_pool(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    registry: list[dict[str, Any]] = []
    vlm_locks: dict[str, Any] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        rows = _build_split(output_dir, split, registry)
        manifests[split] = rows
        manifest_path = output_dir / f"vlm_{split}_manifest.jsonl"
        _write_jsonl(manifest_path, rows)
        vlm_locks[split] = {
            "sample_support": len(rows),
            "product_support": sum(row["scenario"] == "product" for row in rows),
            "dialogue_support": sum(row["scenario"] == "dialogue" for row in rows),
            "manifest_canonical_sha256": canonical_json_sha256(rows),
            "manifest_file_sha256": file_sha256(manifest_path),
        }
    isolation = _isolation_evidence(manifests)
    lock = {
        "schema_version": "context_focus_pool_lock_v5",
        "generator_version": GENERATOR_VERSION,
        "evidence_class": "deterministic_synthetic_programmatic_exploration_only",
        "human_annotation_support": 0,
        "image_encoding": "binary_ppm_p6_rgb_384x256",
        "primary_factor": "context_focused_training_data_composition_and_support",
        "prompt_change_from_v4": False,
        "vlm": vlm_locks,
        "asset_registry_support": len(registry),
        "asset_registry_canonical_sha256": canonical_json_sha256(registry),
        "split_identity_policy": "source_id_image_sha256_sample_id_and_dialogue_text_sha256_disjoint",
        "isolation": isolation,
    }
    _write_json(output_dir / "asset_registry.json", registry)
    _write_json(output_dir / "bundle_lock.json", lock)
    return lock


def _build_split(
    output_dir: Path, split: str, registry: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    counts = COUNTS[split]
    offset = SPLIT_OFFSET[split]
    categories = ("restaurant", "hotel", "attraction")
    styles = ("minimal", "industrial", "coastal", "heritage")
    facilities = ("terrace", "spa", "bar", "garden", "lounge", "museum")
    prices = ((12, "budget"), (48, "mid_range"), (120, "premium"), (240, "luxury"))
    rows: list[dict[str, Any]] = []
    for index in range(counts["known"]):
        category = categories[(index + offset) % len(categories)]
        price_value, price_range = prices[(index // len(categories) + offset) % len(prices)]
        style = styles[(index // 2 + offset) % len(styles)]
        facility = facilities[(index // 3 + offset) % len(facilities)]
        rows.append(_product_row(
            output_dir,
            registry,
            split,
            f"v5_{split[:3]}_product_known_{index:03d}",
            [
                "TRIP V5 PRODUCT",
                f"TYPE {category.upper()}",
                f"STYLE {style.upper()}",
                f"FACILITY {facility.upper()}",
                f"PRICE {price_value}",
            ],
            {
                "business_category": category,
                "style_tags": [style],
                "visible_facilities": [facility],
                "price_range": price_range,
                "unknown_fields": [],
            },
            ["business_category", "style", "facility", "known_visible_price", "v5_retention"],
            offset + index,
        ))
    for index in range(counts["unknown"]):
        category = categories[(index + offset) % len(categories)] if index % 2 else "other"
        rows.append(_product_row(
            output_dir,
            registry,
            split,
            f"v5_{split[:3]}_product_unknown_{index:03d}",
            ["TRIP V5 PRODUCT", f"TYPE {category.upper()}", "STYLE ?", "FACILITY ?", "PRICE ?"],
            {
                "business_category": category,
                "style_tags": [],
                "visible_facilities": [],
                "price_range": "unknown",
                "unknown_fields": ["style_tags", "visible_facilities", "price_range"],
            },
            ["insufficient_visual_evidence", "price_unknown", "unknown_suppression", "v5_retention"],
            offset + 200 + index,
        ))
    for index in range(counts["multi"]):
        left = categories[(index + offset) % len(categories)]
        right = categories[(index + 1 + offset) % len(categories)]
        rows.append(_product_row(
            output_dir,
            registry,
            split,
            f"v5_{split[:3]}_product_multi_{index:03d}",
            ["TRIP V5 PRODUCT", "TYPE CONFLICT", left.upper(), right.upper(), "STYLE ?", "FACILITY ?", "PRICE ?"],
            {
                "business_category": "other",
                "style_tags": [],
                "visible_facilities": [],
                "price_range": "unknown",
                "unknown_fields": ["style_tags", "visible_facilities", "price_range"],
            },
            ["multi_subject_conflict", "insufficient_visual_evidence", "unknown_suppression", "v5_retention"],
            offset + 400 + index,
        ))
    rows.extend(_dialogue_rows(split, counts["dialogue"]))
    return rows


def _product_row(
    output_dir: Path,
    registry: list[dict[str, Any]],
    split: str,
    sample_id: str,
    lines: list[str],
    gold: dict[str, Any],
    slices: list[str],
    seed: int,
) -> dict[str, Any]:
    relative_path = f"assets/vlm/{split}/{sample_id}.ppm"
    path = output_dir / relative_path
    _write_card(path, lines, seed)
    image_sha = file_sha256(path)
    source_id = f"synthetic:{GENERATOR_VERSION}:{split}:{sample_id}"
    source = {
        "source_id": source_id,
        "page_url": f"synthetic://{GENERATOR_VERSION}/{split}/{sample_id}",
        "license": "programmatically_generated_test_asset",
        "author": GENERATOR_VERSION,
        "dataset_relation": "deterministic_synthetic_not_yelp",
    }
    registry.append({
        "record_id": sample_id,
        "source_id": source_id,
        "relative_path": relative_path,
        "sha256": image_sha,
        "width": WIDTH,
        "height": HEIGHT,
        "purpose": "vlm",
        "split": split,
        "contains_image_bytes": False,
    })
    return {
        "sample_id": sample_id,
        "split": split,
        "scenario": "product",
        "source_id": source_id,
        "source_record_sha256": canonical_json_sha256(source),
        "image_relative_path": relative_path,
        "image_sha256": image_sha,
        "slices": slices,
        "label_provenance": "synthetic_programmatic_card_v5_no_human_annotation",
        "prompt": PRODUCT_PROMPT,
        "gold": gold,
        "sample_weight": 1.0,
    }


def _dialogue_rows(split: str, count: int) -> list[dict[str, Any]]:
    cities = SPLIT_CITIES[split]
    categories = ("restaurant", "hotel", "attraction")
    prices = ("budget", "mid_range", "premium", "luxury")
    facilities = ("parking", "wifi", "pool", "breakfast")
    rows: list[dict[str, Any]] = []
    for index in range(count):
        category = categories[(index * 5 + 1) % len(categories)]
        new_category = categories[(index * 5 + 2) % len(categories)]
        city = cities[(index * 3 + 1) % len(cities)]
        old_city = cities[(index * 3 + 2) % len(cities)]
        price = prices[(index * 5 + 1) % len(prices)]
        new_price = prices[(index * 7 + 2) % len(prices)]
        facility = facilities[(index * 3 + 1) % len(facilities)]
        new_facility = facilities[(index * 5 + 2) % len(facilities)]
        ticket = f"{split[:1].upper()}{index:04d}"
        mode = index % 8
        if mode == 0:
            dialogue = (
                f"Case {ticket}. The request began as a {price} {category} in {old_city} with "
                f"{facility}. The user corrects only the city to {city}; every other constraint stays."
            )
            facts, state, task, value = [price, category, facility], "corrected", "city", city
        elif mode == 1:
            dialogue = (
                f"Case {ticket}. Keep the {category}, {city}, and {facility} constraints. The user "
                f"updates only the price range from {price} to {new_price}."
            )
            facts, state, task, value = [category, city, facility], "updated", "price_range", new_price
        elif mode == 2:
            dialogue = (
                f"Case {ticket}. The active request has city {city} and facility {facility}. Replace "
                f"the previous {category} business type with {new_category}, changing nothing else."
            )
            facts, state, task, value = [city, facility], "corrected", "business_category", new_category
        elif mode == 3:
            dialogue = (
                f"Case {ticket}. The retained request facts are {price}, {category}, and {city}. "
                f"The user adds one facility requirement: {facility}."
            )
            facts, state, task, value = [price, category, city], "updated", "visible_facility", facility
        elif mode == 4:
            dialogue = (
                f"Case {ticket}. A {price} {category} in {city} remains active. Correct only the "
                f"facility from {facility} to {new_facility}."
            )
            facts, state, task, value = [price, category, city], "corrected", "visible_facility", new_facility
        elif mode == 5:
            dialogue = (
                f"Case {ticket}. Preserve {category}, {city}, and {facility}. The user removes the "
                f"previous {price} price constraint, so the active price range is unknown."
            )
            facts, state, task, value = [category, city, facility], "corrected", "price_range", "unknown"
        elif mode == 6:
            dialogue = (
                f"Case {ticket}. Price {price}, business type {category}, and facility {facility} "
                f"are retained. The latest turn changes the city from {old_city} to {city}."
            )
            facts, state, task, value = [price, category, facility], "corrected", "city", city
        else:
            dialogue = (
                f"Case {ticket}. City {city}, price {price}, and facility {facility} are already "
                f"set. The latest turn adds business type {category}."
            )
            facts, state, task, value = [city, price, facility], "updated", "business_category", category
        source_id = f"synthetic:{GENERATOR_VERSION}:{split}:dialogue:{index:04d}"
        dialogue_sha = canonical_json_sha256({"dialogue": dialogue})
        rows.append({
            "sample_id": f"v5_{split[:3]}_dialogue_{index:04d}",
            "split": split,
            "scenario": "dialogue",
            "source_id": source_id,
            "source_record_sha256": canonical_json_sha256({"source_id": source_id, "dialogue": dialogue}),
            "dialogue_text_sha256": dialogue_sha,
            "slices": ["dialogue_state", "context_recall", "task_key_value", "first_turn_routing", "v5_context_focus"],
            "label_provenance": "synthetic_protocol_case_v5_no_human_annotation",
            "prompt": DIALOGUE_PROMPT,
            "dialogue": dialogue,
            "gold": {
                "context_facts": facts,
                "state": state,
                "task": task,
                "value": value,
                "route": "image_product_search",
            },
            "sample_weight": 1.0,
        })
    return rows


def _isolation_evidence(manifests: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    identities: dict[str, dict[str, set[str]]] = {}
    for split, rows in manifests.items():
        identities[split] = {
            "sample_id": {str(row["sample_id"]) for row in rows},
            "source_id": {str(row["source_id"]) for row in rows},
            "image_sha256": {str(row["image_sha256"]) for row in rows if row.get("image_sha256")},
            "dialogue_text_sha256": {
                str(row["dialogue_text_sha256"]) for row in rows if row.get("dialogue_text_sha256")
            },
        }
    overlaps: dict[str, dict[str, list[str]]] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            overlaps[f"{left}_vs_{right}"] = {
                kind: sorted(identities[left][kind] & identities[right][kind])
                for kind in identities[left]
            }
    if any(values for pair in overlaps.values() for values in pair.values()):
        raise ValueError(f"v5 split leakage: {overlaps}")
    return {"status": "PASS", "overlaps": overlaps}


if __name__ == "__main__":
    main()

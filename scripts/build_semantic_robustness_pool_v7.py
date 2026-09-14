"""Build a leak-isolated synthetic training/development pool for v7 robustness."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_exploration_pool_v4 import HEIGHT, WIDTH, _write_card, _write_json, _write_jsonl
from scripts.build_context_focus_pool_v5 import _build_split as _build_v5_split
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256


GENERATOR_VERSION = "trip_semantic_robustness_pool_v7"
SPLITS = ("training", "development")
COUNTS = {
    "training": {"clear": 96, "unknown": 40, "multi": 40, "negated": 40, "conflict": 40, "dialogue": 256},
    "development": {"clear": 16, "unknown": 8, "multi": 8, "negated": 8, "conflict": 8, "dialogue": 48},
}
SPLIT_OFFSET = {"training": 7000, "development": 9000}
PRODUCT_PROMPT = (
    "Read only the visible synthetic robustness card. Return exactly one JSON object with keys "
    "business_category, style_tags, visible_facilities, price_range. business_category must be "
    "restaurant, hotel, attraction, or unknown. Copy only positive visible style and facility cues "
    "as lowercase arrays. NO, NOT, REMOVED, conflicting subjects, and conflicting price tiers require "
    "abstention for the affected field. price_range must be budget, mid_range, premium, luxury, or "
    "unknown and requires exactly one explicit PRICE TIER. Do not infer absent facts."
)
DIALOGUE_PROMPT = (
    "Read the synthetic dialogue state case. Return exactly one JSON object with keys context_facts, "
    "state, task, value, route. Keep every still-active explicit fact, exclude superseded or removed "
    "values, apply only the newest change, and use route image_product_search."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-v5-lock", type=Path, required=True)
    parser.add_argument("--expected-lock", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    lock = build_pool(args.output_dir, args.prior_v5_lock)
    if args.expected_lock:
        expected = json.loads(args.expected_lock.read_text(encoding="utf-8"))
        if lock != expected:
            raise ValueError("generated v7 robustness pool differs from committed lock")
    print(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True))


def build_pool(output_dir: Path, prior_v5_lock_path: Path) -> dict[str, Any]:
    prior_rows, prior_lock = _load_prior_v5_training(prior_v5_lock_path)
    output_dir.mkdir(parents=True)
    registry: list[dict[str, Any]] = []
    manifests: dict[str, list[dict[str, Any]]] = {}
    split_locks: dict[str, Any] = {}
    for split in SPLITS:
        rows = _build_split(output_dir, split, registry)
        manifests[split] = rows
        path = output_dir / f"vlm_{split}_manifest.jsonl"
        _write_jsonl(path, rows)
        split_locks[split] = {
            "sample_support": len(rows),
            "product_support": sum(row["scenario"] == "product" for row in rows),
            "dialogue_support": sum(row["scenario"] == "dialogue" for row in rows),
            "manifest_canonical_sha256": canonical_json_sha256(rows),
            "manifest_file_sha256": file_sha256(path),
            "slice_support": _slice_support(rows),
        }
    isolation = _isolation_evidence(manifests)
    prior_isolation = _prior_cycle_isolation(manifests, prior_rows)
    lock = {
        "schema_version": "semantic_robustness_pool_lock_v7",
        "generator_version": GENERATOR_VERSION,
        "evidence_class": "deterministic_synthetic_programmatic_development_only",
        "human_annotation_support": 0,
        "image_encoding": "binary_ppm_p6_rgb_384x256",
        "primary_factor": "robustness_training_data_only",
        "prompt_fixed_within_comparison": True,
        "splits": list(SPLITS),
        "final_policy": "not_defined_not_generated_not_opened",
        "vlm": split_locks,
        "asset_registry_support": len(registry),
        "asset_registry_canonical_sha256": canonical_json_sha256(registry),
        "split_identity_policy": "source_id_image_sha256_sample_id_and_dialogue_text_sha256_disjoint",
        "isolation": isolation,
        "prior_v5_bundle_lock_sha256": canonical_json_sha256(prior_lock),
        "prior_v5_training_isolation": prior_isolation,
    }
    _write_json(output_dir / "asset_registry.json", registry)
    _write_json(output_dir / "bundle_lock.json", lock)
    return lock


def _build_split(output_dir: Path, split: str, registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = COUNTS[split]
    offset = SPLIT_OFFSET[split]
    rows: list[dict[str, Any]] = []
    builders = (
        ("clear", _clear_product),
        ("unknown", _unknown_product),
        ("multi", _multi_product),
        ("negated", _negated_product),
        ("conflict", _conflict_product),
    )
    cursor = 0
    for kind, builder in builders:
        for index in range(counts[kind]):
            lines, gold, slices = builder(index, offset)
            lines.insert(1, f"SPLIT {split.upper()}")
            sample_id = f"v7_{split[:3]}_{kind}_{index:03d}"
            rows.append(_product_row(
                output_dir, registry, split, sample_id, lines, gold, slices, offset + cursor
            ))
            cursor += 1
    rows.extend(_dialogue_rows(split, counts["dialogue"]))
    return rows


def _clear_product(index: int, offset: int) -> tuple[list[str], dict[str, Any], list[str]]:
    categories = ("restaurant", "hotel", "attraction")
    styles = ("minimal", "industrial", "coastal", "heritage", "modern", "rustic")
    facilities = ("terrace", "spa", "bar", "garden", "lounge", "museum", "pool", "parking")
    prices = ("budget", "mid_range", "premium", "luxury")
    category = categories[(index + offset) % len(categories)]
    style_a = styles[(index * 3 + offset) % len(styles)]
    style_b = styles[(index * 5 + offset + 1) % len(styles)]
    facility_a = facilities[(index * 3 + offset) % len(facilities)]
    facility_b = facilities[(index * 5 + offset + 2) % len(facilities)]
    price = prices[(index * 7 + offset) % len(prices)]
    labels = ("SUBJECT", "VENUE", "BUSINESS")
    return (
        [
            "TRIP V7 ROBUST",
            f"{labels[index % len(labels)]} {category.upper()}",
            f"STYLE {style_a.upper()} {style_b.upper()}",
            f"FACILITY {facility_a.upper()} {facility_b.upper()}",
            f"PRICE TIER {price.upper()}",
        ],
        {
            "business_category": category,
            "style_tags": sorted({style_a, style_b}),
            "visible_facilities": sorted({facility_a, facility_b}),
            "price_range": price,
            "unknown_fields": [],
        },
        ["business_category", "style", "facility", "known_visible_price", "multi_label", "v7_robustness"],
    )


def _unknown_product(index: int, offset: int) -> tuple[list[str], dict[str, Any], list[str]]:
    markers = ("NOT SHOWN", "UNCLEAR", "MISSING", "UNKNOWN")
    marker = markers[(index + offset) % len(markers)]
    return (
        ["TRIP V7 ROBUST", f"SUBJECT {marker}", f"STYLE {marker}", f"FACILITY {marker}", f"PRICE {marker}"],
        {
            "business_category": "unknown",
            "style_tags": [],
            "visible_facilities": [],
            "price_range": "unknown",
            "unknown_fields": ["business_category", "style_tags", "visible_facilities", "price_range"],
        },
        ["insufficient_visual_evidence", "price_unknown", "unknown_suppression", "v7_robustness"],
    )


def _multi_product(index: int, offset: int) -> tuple[list[str], dict[str, Any], list[str]]:
    categories = ("restaurant", "hotel", "attraction")
    left = categories[(index + offset) % len(categories)]
    right = categories[(index + offset + 1) % len(categories)]
    return (
        [
            "TRIP V7 ROBUST", f"SUBJECT {left.upper()} + {right.upper()}",
            "STYLE CONFLICT", "FACILITY CONFLICT", "PRICE TIER CONFLICT",
        ],
        {
            "business_category": "unknown",
            "style_tags": [],
            "visible_facilities": [],
            "price_range": "unknown",
            "unknown_fields": ["business_category", "style_tags", "visible_facilities", "price_range"],
        },
        ["multi_subject_conflict", "insufficient_visual_evidence", "price_unknown", "unknown_suppression", "v7_robustness"],
    )


def _negated_product(index: int, offset: int) -> tuple[list[str], dict[str, Any], list[str]]:
    categories = ("restaurant", "hotel", "attraction")
    styles = ("coastal", "heritage", "modern", "rustic")
    facilities = ("pool", "parking", "spa", "terrace")
    category = categories[(index + offset) % len(categories)]
    style = styles[(index * 3 + offset) % len(styles)]
    facility = facilities[(index * 5 + offset) % len(facilities)]
    return (
        [
            "TRIP V7 ROBUST", f"SUBJECT {category.upper()}", f"STYLE NOT {style.upper()}",
            f"FACILITY NO {facility.upper()}", "PRICE REMOVED",
        ],
        {
            "business_category": category,
            "style_tags": [],
            "visible_facilities": [],
            "price_range": "unknown",
            "unknown_fields": ["style_tags", "visible_facilities", "price_range"],
        },
        ["negated_evidence", "price_unknown", "unknown_suppression", "v7_robustness"],
    )


def _conflict_product(index: int, offset: int) -> tuple[list[str], dict[str, Any], list[str]]:
    categories = ("restaurant", "hotel", "attraction")
    styles = ("minimal", "industrial", "coastal", "heritage")
    facilities = ("terrace", "spa", "bar", "garden")
    prices = ("budget", "mid_range", "premium", "luxury")
    category = categories[(index + offset) % len(categories)]
    style = styles[(index * 3 + offset) % len(styles)]
    facility = facilities[(index * 5 + offset) % len(facilities)]
    left = prices[(index + offset) % len(prices)]
    right = prices[(index + offset + 1) % len(prices)]
    return (
        [
            "TRIP V7 ROBUST", f"SUBJECT {category.upper()}", f"STYLE {style.upper()}",
            f"FACILITY {facility.upper()}", f"PRICE TIER {left.upper()}", f"PRICE TIER {right.upper()}",
        ],
        {
            "business_category": category,
            "style_tags": [style],
            "visible_facilities": [facility],
            "price_range": "unknown",
            "unknown_fields": ["price_range"],
        },
        ["conflicting_price_evidence", "price_unknown", "unknown_suppression", "v7_robustness"],
    )


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
    source = {"source_id": source_id, "generator": GENERATOR_VERSION, "split": split, "sample_id": sample_id}
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
        "label_provenance": "synthetic_programmatic_card_v7_no_human_annotation",
        "prompt": PRODUCT_PROMPT,
        "gold": gold,
        "sample_weight": 1.0,
    }


def _dialogue_rows(split: str, count: int) -> list[dict[str, Any]]:
    cities = {
        "training": ("Austin", "Boston", "Chicago", "Denver", "Miami", "Seattle"),
        "development": ("Raleigh", "Boise", "Madison", "Tacoma", "Fresno", "Albany"),
    }[split]
    categories = ("restaurant", "hotel", "attraction")
    prices = ("budget", "mid_range", "premium", "luxury")
    facilities = ("parking", "wifi", "pool", "breakfast", "spa", "terrace")
    rows: list[dict[str, Any]] = []
    for index in range(count):
        city = cities[(index * 3 + 1) % len(cities)]
        old_city = cities[(index * 5 + 2) % len(cities)]
        category = categories[(index * 5 + 1) % len(categories)]
        new_category = categories[(index * 7 + 2) % len(categories)]
        price = prices[(index * 5 + 1) % len(prices)]
        new_price = prices[(index * 7 + 2) % len(prices)]
        facility = facilities[(index * 3 + 1) % len(facilities)]
        new_facility = facilities[(index * 5 + 2) % len(facilities)]
        case = f"V7{split[0].upper()}{index:04d}"
        mode = index % 8
        if mode == 0:
            dialogue = f"Case {case}. Keep {price}, {category}, and {facility}. The old city {old_city} is superseded; city is now {city}."
            facts, state, task, value = [price, category, facility], "corrected", "city", city
        elif mode == 1:
            dialogue = f"Case {case}. The active {category} in {city} still needs {facility}. Replace only price {price} with {new_price}."
            facts, state, task, value = [category, city, facility], "updated", "price_range", new_price
        elif mode == 2:
            dialogue = f"Case {case}. Retain city {city}, price {price}, and facility {facility}. The prior {category} is no longer active; use {new_category}."
            facts, state, task, value = [city, price, facility], "corrected", "business_category", new_category
        elif mode == 3:
            dialogue = f"Case {case}. Preserve {price}, {category}, and {city}. Add the explicit facility {facility}."
            facts, state, task, value = [price, category, city], "updated", "visible_facility", facility
        elif mode == 4:
            dialogue = f"Case {case}. City {city}, {category}, and price {price} stay fixed. Facility {facility} was removed and replaced by {new_facility}."
            facts, state, task, value = [city, category, price], "corrected", "visible_facility", new_facility
        elif mode == 5:
            dialogue = f"Case {case}. Keep {category}, {city}, and {facility}. Remove the {price} constraint; active price is unknown."
            facts, state, task, value = [category, city, facility], "corrected", "price_range", "unknown"
        elif mode == 6:
            dialogue = f"Case {case}. The latest turn says {city}, not {old_city}. Retain {price}, {category}, and {facility}."
            facts, state, task, value = [price, category, facility], "corrected", "city", city
        else:
            dialogue = f"Case {case}. Existing facts are {city}, {price}, and {facility}; append business type {category} without changing them."
            facts, state, task, value = [city, price, facility], "updated", "business_category", category
        source_id = f"synthetic:{GENERATOR_VERSION}:{split}:dialogue:{index:04d}"
        rows.append({
            "sample_id": f"v7_{split[:3]}_dialogue_{index:04d}",
            "split": split,
            "scenario": "dialogue",
            "source_id": source_id,
            "source_record_sha256": canonical_json_sha256({"source_id": source_id, "dialogue": dialogue}),
            "dialogue_text_sha256": canonical_json_sha256({"dialogue": dialogue}),
            "slices": ["dialogue_state", "context_recall", "task_key_value", "first_turn_routing", "v7_robustness"],
            "label_provenance": "synthetic_protocol_case_v7_no_human_annotation",
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


def _load_prior_v5_training(lock_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "context_focus_pool_lock_v5":
        raise ValueError("prior lock is not the v5 context-focus pool lock")
    with tempfile.TemporaryDirectory(prefix="trip-v5-training-reference-") as temp_dir:
        reference_root = Path(temp_dir)
        registry: list[dict[str, Any]] = []
        rows = _build_v5_split(reference_root, "training", registry)
        path = reference_root / "vlm_training_manifest.jsonl"
        _write_jsonl(path, rows)
        expected = lock["vlm"]["training"]
        if (
            len(rows) != expected["sample_support"]
            or canonical_json_sha256(rows) != expected["manifest_canonical_sha256"]
            or file_sha256(path) != expected["manifest_file_sha256"]
        ):
            raise ValueError("regenerated v5 training reference differs from its committed lock")
    return rows, lock


def _identities(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        "sample_id": {str(row["sample_id"]) for row in rows},
        "source_id": {str(row["source_id"]) for row in rows},
        "image_sha256": {str(row["image_sha256"]) for row in rows if row.get("image_sha256")},
        "dialogue_text_sha256": {str(row["dialogue_text_sha256"]) for row in rows if row.get("dialogue_text_sha256")},
    }


def _isolation_evidence(manifests: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    left, right = (_identities(manifests[split]) for split in SPLITS)
    overlaps = {kind: sorted(left[kind] & right[kind]) for kind in left}
    if any(overlaps.values()):
        raise ValueError(f"v7 split leakage: {overlaps}")
    return {"status": "PASS", "training_vs_development": overlaps}


def _prior_cycle_isolation(
    manifests: dict[str, list[dict[str, Any]]], prior_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    prior = _identities(prior_rows)
    overlap = {
        split: {kind: sorted(values[kind] & prior[kind]) for kind in values}
        for split, values in ((name, _identities(rows)) for name, rows in manifests.items())
    }
    if any(items for split in overlap.values() for items in split.values()):
        raise ValueError(f"v7 overlaps prior v5 training: {overlap}")
    return {"status": "PASS", "compared_prior_row_support": len(prior_rows), "overlaps": overlap}


def _slice_support(rows: list[dict[str, Any]]) -> dict[str, int]:
    names = sorted({name for row in rows for name in row.get("slices", [])})
    return {name: sum(name in row.get("slices", []) for row in rows) for name in names}


if __name__ == "__main__":
    main()

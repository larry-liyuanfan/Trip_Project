"""Build a deterministic three-way synthetic pool for exploration-only evidence v4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256


GENERATOR_VERSION = "trip_exploration_pool_v4"
WIDTH = 384
HEIGHT = 256
PRODUCT_PROMPT = (
    "Read only the visible synthetic development card. Return exactly one JSON object with keys "
    "business_category, style_tags, visible_facilities, price_range. business_category must be "
    "restaurant, hotel, attraction, or other. Use other for a TYPE CONFLICT card. style_tags and "
    "visible_facilities must be lowercase arrays. price_range follows the visible numeric PRICE: "
    "below 20 budget, 20 through 60 mid_range, 61 through 150 premium, above 150 luxury. A '?' "
    "or missing row is unknown evidence: return an empty array or unknown. Do not infer absent facts."
)
DIALOGUE_PROMPT = (
    "Read the dialogue state case. Return exactly one JSON object with keys context_facts, state, "
    "task, value, route. Keep all retained facts, use the newest explicit correction, and use route "
    "image_product_search."
)


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
            raise ValueError("generated exploration pool differs from the committed lock")
    print(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True))


def build_pool(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    registry: list[dict[str, Any]] = []
    search_locks: dict[str, Any] = {}
    vlm_locks: dict[str, Any] = {}
    for split in ("training", "development", "final"):
        queries, annotations = _build_search_split(output_dir, split, registry)
        query_path = output_dir / f"search_{split}_manifest.jsonl"
        annotation_path = output_dir / f"search_{split}_annotations.jsonl"
        _write_jsonl(query_path, queries)
        _write_jsonl(annotation_path, annotations)
        search_locks[split] = {
            "query_support": len(queries),
            "query_manifest_canonical_sha256": canonical_json_sha256(queries),
            "query_manifest_file_sha256": file_sha256(query_path),
            "annotation_canonical_sha256": canonical_json_sha256(annotations),
            "annotation_file_sha256": file_sha256(annotation_path),
        }

        vlm_rows = _build_vlm_split(output_dir, split, registry)
        vlm_path = output_dir / f"vlm_{split}_manifest.jsonl"
        _write_jsonl(vlm_path, vlm_rows)
        vlm_locks[split] = {
            "sample_support": len(vlm_rows),
            "product_support": sum(row["scenario"] == "product" for row in vlm_rows),
            "dialogue_support": sum(row["scenario"] == "dialogue" for row in vlm_rows),
            "manifest_canonical_sha256": canonical_json_sha256(vlm_rows),
            "manifest_file_sha256": file_sha256(vlm_path),
        }

    lock = {
        "schema_version": "exploration_pool_lock_v4",
        "generator_version": GENERATOR_VERSION,
        "evidence_class": "deterministic_synthetic_programmatic_exploration_only",
        "human_annotation_support": 0,
        "image_encoding": "binary_ppm_p6_rgb_384x256",
        "search": search_locks,
        "vlm": vlm_locks,
        "asset_registry_support": len(registry),
        "asset_registry_canonical_sha256": canonical_json_sha256(registry),
        "split_identity_policy": "source_id_image_sha256_and_sample_or_query_id_disjoint",
    }
    _write_json(output_dir / "asset_registry.json", registry)
    _write_json(output_dir / "bundle_lock.json", lock)
    return lock


def _build_search_split(
    output_dir: Path, split: str, registry: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supported = [
        ("restaurant", "Philadelphia", "budget"),
        ("hotel", "Nashville", "mid_range"),
        ("attraction", "Philadelphia", "mid_range"),
        ("restaurant", "New Orleans", "premium"),
    ]
    conflicts = [
        ("restaurant", "Nashville", "mid_range", "hotel"),
        ("hotel", "Philadelphia", "mid_range", "attraction"),
        ("attraction", "Nashville", "budget", "restaurant"),
        ("restaurant", "Tampa", "mid_range", "hotel"),
    ]
    absent = [
        ("restaurant", "Atlantis", "budget"),
        ("hotel", "El Dorado", "mid_range"),
        ("attraction", "Shangri-La", "premium"),
        ("restaurant", "Emerald City", "luxury"),
    ]
    definitions: list[dict[str, Any]] = []
    for repeat in range(2):
        for category, city, price in supported:
            definitions.append({
                "case": f"supported_{repeat}_{category}_{city}",
                "visual": category,
                "slices": ["image_similar", "city_business_facility_price", "hard_filter_before_rerank"],
                "filters": {"city": city, "business_category": category, "price_range": price},
                "target": category,
                "required": {"city": city, "price_range": price},
                "excluded": [],
                "guard": "business",
            })
    for index, (category, city, price, visual) in enumerate(conflicts):
        definitions.append({
            "case": f"filter_conflict_{index}",
            "visual": visual,
            "slices": ["filter_conflict", "hard_filter_before_rerank"],
            "filters": {"city": city, "business_category": category, "price_range": price},
            "target": category,
            "required": {"city": city, "price_range": price},
            "excluded": [visual],
            "guard": "business",
        })
    for index, (category, city, price) in enumerate(absent):
        definitions.append({
            "case": f"filter_empty_{index}",
            "visual": category,
            "slices": ["no_result", "hard_filter_before_rerank", "filter_empty_no_result"],
            "filters": {"city": city, "business_category": category, "price_range": price},
            "target": category,
            "required": {"city": city, "price_range": price},
            "excluded": [],
            "guard": "business",
        })
    for index, visual in enumerate(("retail", "private", "office", "retail", "private", "office", "retail", "private")):
        definitions.append({
            "case": f"visual_irrelevant_{index}",
            "visual": visual,
            "slices": ["visual_similar_business_irrelevant", "no_result", "business_guard_no_result"],
            "filters": {},
            "target": "",
            "required": {},
            "excluded": ["restaurant", "hotel", "attraction"],
            "guard": "non_business",
        })

    split_offset = {"training": 0, "development": 1000, "final": 2000}[split]
    queries: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for index, definition in enumerate(definitions, start=1):
        query_id = f"v4_{split[:3]}_search_{index:02d}"
        relative_path = f"assets/search/{split}/{query_id}.ppm"
        image_path = output_dir / relative_path
        _write_card(
            image_path,
            ["SEARCH IMAGE", f"TYPE {definition['visual'].upper()}", f"VARIANT {split_offset + index}"],
            seed=split_offset + index,
        )
        image_sha = file_sha256(image_path)
        source = _source(split, query_id)
        query = {
            "query_id": query_id,
            "split": split,
            "slices": definition["slices"],
            "image": {"relative_path": relative_path, "sha256": image_sha, "width": WIDTH, "height": HEIGHT},
            "source": source,
            "source_record_sha256": canonical_json_sha256(source),
            "query_text": "Find indexed travel businesses satisfying the explicit filters and visually similar to this image.",
            "requested_filters": definition["filters"],
            "unsupported_constraints": [],
        }
        query["query_sha256"] = canonical_json_sha256(query)
        queries.append(query)
        annotations.append({
            "query_id": query_id,
            "label_provenance": "synthetic",
            "annotators": ["programmatic_metadata_rule_v4"],
            "conflict_resolution": "not_applicable_single_programmatic",
            "business_guard_label": definition["guard"],
            "grade_rules": {
                "target_business_category": definition["target"],
                "required_metadata": definition["required"],
                "excluded_business_categories": definition["excluded"],
                "grades": {
                    "3": "target category and all required formal metadata",
                    "2": "target category with incomplete or conflicting formal metadata",
                    "1": "other non-excluded indexed business",
                    "0": "excluded visual-intent false positive or unsupported result",
                },
            },
        })
        registry.append(_asset(query_id, source["source_id"], relative_path, image_sha, "search", split))
    return queries, annotations


def _build_vlm_split(
    output_dir: Path, split: str, registry: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    counts = {
        "training": {"known": 128, "unknown": 32, "multi": 32, "dialogue": 96},
        "development": {"known": 12, "unknown": 6, "multi": 6, "dialogue": 12},
        "final": {"known": 12, "unknown": 6, "multi": 6, "dialogue": 12},
    }[split]
    offset = {"training": 0, "development": 1000, "final": 2000}[split]
    categories = ("restaurant", "hotel", "attraction")
    styles = ("modern", "rustic", "casual", "formal")
    facilities = ("wifi", "parking", "pool", "breakfast", "seating", "gallery")
    prices = ((8, "budget"), (35, "mid_range"), (90, "premium"), (300, "luxury"))
    rows: list[dict[str, Any]] = []
    for index in range(counts["known"]):
        category = categories[(index + offset) % len(categories)]
        price_value, price_range = prices[(index // len(categories) + offset) % len(prices)]
        style = styles[(index // 2 + offset) % len(styles)]
        facility = facilities[(index // 3 + offset) % len(facilities)]
        sample_id = f"v4_{split[:3]}_product_known_{index:03d}"
        rows.append(_product_row(
            output_dir, registry, split, sample_id,
            ["TRIP PRODUCT", f"TYPE {category.upper()}", f"STYLE {style.upper()}",
             f"FACILITY {facility.upper()}", f"PRICE {price_value}"],
            {
                "business_category": category,
                "style_tags": [style],
                "visible_facilities": [facility],
                "price_range": price_range,
                "unknown_fields": [],
            },
            ["business_category", "style", "facility", "known_visible_price"], offset + index,
        ))
    for index in range(counts["unknown"]):
        category = categories[(index + offset) % len(categories)] if index % 2 else "other"
        sample_id = f"v4_{split[:3]}_product_unknown_{index:03d}"
        rows.append(_product_row(
            output_dir, registry, split, sample_id,
            ["TRIP PRODUCT", f"TYPE {category.upper()}", "STYLE ?", "FACILITY ?", "PRICE ?"],
            {
                "business_category": category,
                "style_tags": [],
                "visible_facilities": [],
                "price_range": "unknown",
                "unknown_fields": ["style_tags", "visible_facilities", "price_range"],
            },
            ["insufficient_visual_evidence", "price_unknown", "unknown_suppression"], offset + 200 + index,
        ))
    for index in range(counts["multi"]):
        left = categories[(index + offset) % len(categories)]
        right = categories[(index + 1 + offset) % len(categories)]
        sample_id = f"v4_{split[:3]}_product_multi_{index:03d}"
        rows.append(_product_row(
            output_dir, registry, split, sample_id,
            ["TRIP PRODUCT", "TYPE CONFLICT", left.upper(), right.upper(), "STYLE ?", "FACILITY ?", "PRICE ?"],
            {
                "business_category": "other",
                "style_tags": [],
                "visible_facilities": [],
                "price_range": "unknown",
                "unknown_fields": ["style_tags", "visible_facilities", "price_range"],
            },
            ["multi_subject_conflict", "insufficient_visual_evidence", "unknown_suppression"], offset + 400 + index,
        ))
    rows.extend(_dialogue_rows(split, counts["dialogue"], offset))
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
    sha = file_sha256(path)
    source = _source(split, sample_id)
    registry.append(_asset(sample_id, source["source_id"], relative_path, sha, "vlm", split))
    return {
        "sample_id": sample_id,
        "split": split,
        "scenario": "product",
        "source_id": source["source_id"],
        "source_record_sha256": canonical_json_sha256(source),
        "image_relative_path": relative_path,
        "image_sha256": sha,
        "slices": slices,
        "label_provenance": "synthetic_programmatic_card_v4_no_human_annotation",
        "prompt": PRODUCT_PROMPT,
        "gold": gold,
        "sample_weight": 1.0,
    }


def _dialogue_rows(split: str, count: int, offset: int) -> list[dict[str, Any]]:
    cities = ("Tampa", "Nashville", "Philadelphia", "New Orleans", "Reno", "Tucson")
    categories = ("restaurant", "hotel", "attraction")
    prices = ("budget", "mid_range", "premium", "luxury")
    facilities = ("parking", "wifi", "pool", "breakfast")
    rows: list[dict[str, Any]] = []
    for index in range(count):
        category = categories[(index + offset) % len(categories)]
        city = cities[(index * 2 + offset) % len(cities)]
        old_city = cities[(index * 2 + 1 + offset) % len(cities)]
        price = prices[(index + offset) % len(prices)]
        facility = facilities[(index + offset) % len(facilities)]
        mode = index % 4
        if mode == 0:
            dialogue = (
                f"User first requested a {price} {category} in {old_city} with {facility}, then "
                f"corrected only the city to {city}. Return the active request."
            )
            facts, state, task, value = [price, category, facility], "corrected", "city", city
        elif mode == 1:
            dialogue = (
                f"The active request is a {category} in {city} with {facility}. The user changes "
                f"only price to {price}."
            )
            facts, state, task, value = [category, city, facility], "updated", "price_range", price
        elif mode == 2:
            dialogue = (
                f"The user asked for a hotel in {city} with {facility}, then corrected only the "
                f"business type to {category}."
            )
            facts, state, task, value = [city, facility], "corrected", "business_category", category
        else:
            dialogue = (
                f"The current request is a {price} {category} in {city}. The user adds the visible "
                f"facility requirement {facility}."
            )
            facts, state, task, value = [price, category, city], "updated", "visible_facility", facility
        source_id = f"synthetic:{GENERATOR_VERSION}:{split}:dialogue:{index:03d}"
        rows.append({
            "sample_id": f"v4_{split[:3]}_dialogue_{index:03d}",
            "split": split,
            "scenario": "dialogue",
            "source_id": source_id,
            "source_record_sha256": canonical_json_sha256({"source_id": source_id, "dialogue": dialogue}),
            "slices": ["dialogue_state", "context_recall", "task_key_value", "first_turn_routing"],
            "label_provenance": "synthetic_protocol_case_v4_no_human_annotation",
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


def _source(split: str, record_id: str) -> dict[str, str]:
    return {
        "source_id": f"synthetic:{GENERATOR_VERSION}:{split}:{record_id}",
        "page_url": f"synthetic://{GENERATOR_VERSION}/{split}/{record_id}",
        "download_url": f"synthetic://{GENERATOR_VERSION}/{split}/{record_id}.ppm",
        "license": "programmatically_generated_test_asset",
        "author": GENERATOR_VERSION,
        "dataset_relation": "deterministic_synthetic_not_yelp",
    }


def _asset(record_id: str, source_id: str, relative_path: str, sha: str, purpose: str, split: str) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "source_id": source_id,
        "relative_path": relative_path,
        "sha256": sha,
        "width": WIDTH,
        "height": HEIGHT,
        "purpose": purpose,
        "split": split,
        "contains_image_bytes": False,
    }


FONT = {
    " ": ("00000",) * 7,
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    **{
        char: tuple(rows.split("/")) for char, rows in {
            "A": "01110/10001/10001/11111/10001/10001/10001", "B": "11110/10001/10001/11110/10001/10001/11110",
            "C": "01111/10000/10000/10000/10000/10000/01111", "D": "11110/10001/10001/10001/10001/10001/11110",
            "E": "11111/10000/10000/11110/10000/10000/11111", "F": "11111/10000/10000/11110/10000/10000/10000",
            "G": "01111/10000/10000/10111/10001/10001/01111", "H": "10001/10001/10001/11111/10001/10001/10001",
            "I": "11111/00100/00100/00100/00100/00100/11111", "J": "00111/00010/00010/00010/10010/10010/01100",
            "K": "10001/10010/10100/11000/10100/10010/10001", "L": "10000/10000/10000/10000/10000/10000/11111",
            "M": "10001/11011/10101/10101/10001/10001/10001", "N": "10001/11001/10101/10011/10001/10001/10001",
            "O": "01110/10001/10001/10001/10001/10001/01110", "P": "11110/10001/10001/11110/10000/10000/10000",
            "Q": "01110/10001/10001/10001/10101/10010/01101", "R": "11110/10001/10001/11110/10100/10010/10001",
            "S": "01111/10000/10000/01110/00001/00001/11110", "T": "11111/00100/00100/00100/00100/00100/00100",
            "U": "10001/10001/10001/10001/10001/10001/01110", "V": "10001/10001/10001/10001/10001/01010/00100",
            "W": "10001/10001/10001/10101/10101/10101/01010", "X": "10001/10001/01010/00100/01010/10001/10001",
            "Y": "10001/10001/01010/00100/00100/00100/00100", "Z": "11111/00001/00010/00100/01000/10000/11111",
            "0": "01110/10001/10011/10101/11001/10001/01110", "1": "00100/01100/00100/00100/00100/00100/01110",
            "2": "01110/10001/00001/00010/00100/01000/11111", "3": "11110/00001/00001/01110/00001/00001/11110",
            "4": "00010/00110/01010/10010/11111/00010/00010", "5": "11111/10000/10000/11110/00001/00001/11110",
            "6": "01110/10000/10000/11110/10001/10001/01110", "7": "11111/00001/00010/00100/01000/01000/01000",
            "8": "01110/10001/10001/01110/10001/10001/01110", "9": "01110/10001/10001/01111/00001/00001/01110",
        }.items()
    },
}


def _write_card(path: Path, lines: list[str], seed: int) -> None:
    pixels = bytearray([244, 246, 249] * WIDTH * HEIGHT)
    accent = ((seed * 47) % 120 + 60, (seed * 71) % 120 + 60, (seed * 29) % 120 + 60)
    _rect(pixels, 0, 0, WIDTH, 18, accent)
    _rect(pixels, 0, HEIGHT - 14, WIDTH, 14, accent)
    for index, line in enumerate(lines[:7]):
        y = 27 + index * 30
        if index % 2:
            _rect(pixels, 12, y - 5, WIDTH - 24, 26, (231, 235, 241))
        _draw_text(pixels, line, 20, y, 3, (25, 31, 42))
    for index in range(5):
        _rect(pixels, WIDTH - 36 + index * 5, 3 + ((seed + index * 11) % 10), 3, 8, accent)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
        handle.write(pixels)


def _draw_text(pixels: bytearray, text: str, x: int, y: int, scale: int, color: tuple[int, int, int]) -> None:
    cursor = x
    for char in text.upper():
        glyph = FONT.get(char, FONT["?"])
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    _rect(pixels, cursor + column * scale, y + row * scale, scale, scale, color)
        cursor += 6 * scale
        if cursor + 5 * scale >= WIDTH:
            break


def _rect(pixels: bytearray, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
    for py in range(max(0, y), min(HEIGHT, y + height)):
        for px in range(max(0, x), min(WIDTH, x + width)):
            start = (py * WIDTH + px) * 3
            pixels[start:start + 3] = bytes(color)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

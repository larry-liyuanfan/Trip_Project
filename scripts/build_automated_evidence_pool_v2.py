"""Build deterministic, byte-locked synthetic Search v2 and VLM weak v3 assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256


GENERATOR_VERSION = "trip_automated_evidence_pool_v2"
WIDTH = 224
HEIGHT = 224


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
            raise ValueError(
                "generated evidence pool differs from committed lock: "
                f"expected={canonical_json_sha256(expected)} actual={canonical_json_sha256(lock)}"
            )
    print(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True))


def build_pool(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    asset_registry: list[dict[str, Any]] = []
    search_locks: dict[str, Any] = {}
    for split in ("calibration", "holdout"):
        queries, annotations = _build_search_split(output_dir, split, asset_registry)
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
    vlm_records = _build_vlm_pool(output_dir, asset_registry)
    vlm_path = output_dir / "vlm_manifest_weak_v3.jsonl"
    _write_jsonl(vlm_path, vlm_records)
    lock = {
        "schema_version": "automated_evidence_pool_lock_v2",
        "generator_version": GENERATOR_VERSION,
        "image_encoding": "binary_ppm_p6_rgb_224x224",
        "search": search_locks,
        "vlm": {
            "sample_support": len(vlm_records),
            "product_support": sum(row["scenario"] == "product" for row in vlm_records),
            "dialogue_support": sum(row["scenario"] == "dialogue" for row in vlm_records),
            "manifest_canonical_sha256": canonical_json_sha256(vlm_records),
            "manifest_file_sha256": file_sha256(vlm_path),
        },
        "asset_registry_support": len(asset_registry),
        "asset_registry_canonical_sha256": canonical_json_sha256(asset_registry),
        "human_annotation_support": 0,
        "evidence_class": "deterministic_synthetic_and_weak_programmatic_development_only",
    }
    _write_json(output_dir / "asset_registry.json", asset_registry)
    _write_json(output_dir / "bundle_lock.json", lock)
    return lock


def _build_search_split(
    output_dir: Path, split: str, asset_registry: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supported = [
        ("restaurant", "Philadelphia", "budget", "restaurant"),
        ("hotel", "Nashville", "mid_range", "hotel"),
        ("attraction", "Philadelphia", "mid_range", "attraction"),
        ("restaurant", "New Orleans", "premium", "restaurant"),
    ]
    conflicts = [
        ("restaurant", "Nashville", "mid_range", "hotel"),
        ("hotel", "Philadelphia", "mid_range", "attraction"),
        ("attraction", "Nashville", "budget", "restaurant"),
        ("restaurant", "Tampa", "mid_range", "private"),
    ]
    absent = [
        ("restaurant", "Atlantis", "budget", "restaurant"),
        ("hotel", "El Dorado", "mid_range", "hotel"),
        ("attraction", "Shangri-La", "premium", "attraction"),
        ("restaurant", "Emerald City", "luxury", "restaurant"),
    ]
    definitions: list[dict[str, Any]] = []
    for index, (category, city, price, visual) in enumerate(supported, start=1):
        definitions.append({
            "case": f"supported_{index}",
            "visual": visual,
            "slices": ["image_similar", "city_business_facility_price", "hard_filter_before_rerank"],
            "filters": {"city": city, "business_category": category, "price_range": price},
            "target": category,
            "required": {"city": city, "price_range": price},
            "excluded": [],
        })
    for index, (category, city, price, visual) in enumerate(conflicts, start=1):
        definitions.append({
            "case": f"filter_conflict_{index}",
            "visual": visual,
            "slices": ["filter_conflict", "hard_filter_before_rerank"],
            "filters": {"city": city, "business_category": category, "price_range": price},
            "target": category,
            "required": {"city": city, "price_range": price},
            "excluded": [visual] if visual in {"restaurant", "hotel", "attraction"} else [],
        })
    for index, (category, city, price, visual) in enumerate(absent, start=1):
        definitions.append({
            "case": f"no_result_{index}",
            "visual": visual,
            "slices": ["no_result", "hard_filter_before_rerank"],
            "filters": {"city": city, "business_category": category, "price_range": price},
            "target": category,
            "required": {"city": city, "price_range": price},
            "excluded": [],
        })
    for index, visual in enumerate(("private", "retail", "private", "retail"), start=1):
        definitions.append({
            "case": f"visual_irrelevant_{index}",
            "visual": visual,
            "slices": ["visual_similar_business_irrelevant", "no_result"],
            "filters": {},
            "target": "",
            "required": {},
            "excluded": ["restaurant", "hotel", "attraction"],
        })

    queries: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    split_offset = 0 if split == "calibration" else 100
    for index, definition in enumerate(definitions, start=1):
        query_id = f"v2_{split[:3]}_{index:02d}_{definition['case']}"
        relative_path = f"assets/search/{split}/{query_id}.ppm"
        path = output_dir / relative_path
        _write_scene(path, definition["visual"], split_offset + index)
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
            "image": {
                "relative_path": relative_path,
                "sha256": image_sha,
                "width": WIDTH,
                "height": HEIGHT,
            },
            "source": source,
            "source_record_sha256": canonical_json_sha256(source),
            "query_text": _query_text(definition),
            "requested_filters": definition["filters"],
            "unsupported_constraints": [],
        }
        query["query_sha256"] = canonical_json_sha256(query)
        queries.append(query)
        annotations.append({
            "query_id": query_id,
            "label_provenance": "synthetic",
            "annotators": ["programmatic_metadata_rule_v2"],
            "conflict_resolution": "not_applicable_single_programmatic",
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
        asset_registry.append(_asset_record(query_id, source["source_id"], relative_path, image_sha, "search", split))
    return queries, annotations


def _build_vlm_pool(output_dir: Path, asset_registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    known = [
        ("restaurant", 8, "budget", ["modern"], ["seating"]),
        ("hotel", 35, "mid_range", ["modern"], ["bed"]),
        ("attraction", 90, "premium", ["formal"], ["gallery"]),
        ("restaurant", 300, "luxury", ["formal"], ["counter"]),
    ]
    for index, (visual, amount, price, styles, facilities) in enumerate(known, start=1):
        sample_id = f"vlm_v3_price_{index:02d}"
        relative_path = f"assets/vlm/{sample_id}.ppm"
        path = output_dir / relative_path
        _write_scene(path, visual, 200 + index, price=amount)
        sha = file_sha256(path)
        records.append(_vlm_product(
            sample_id, relative_path, sha,
            ["known_visible_price", "business_category", "style", "facility"],
            {
                "business_category": visual,
                "style_tags": styles,
                "visible_facilities": facilities,
                "price_range": price,
                "unknown_fields": [],
            },
        ))
        asset_registry.append(_asset_record(sample_id, f"synthetic:{GENERATOR_VERSION}:{sample_id}", relative_path, sha, "vlm", None))

    pairs = (("restaurant", "hotel"), ("hotel", "attraction"), ("attraction", "restaurant"), ("retail", "hotel"))
    for index, (left, right) in enumerate(pairs, start=1):
        sample_id = f"vlm_v3_multi_{index:02d}"
        relative_path = f"assets/vlm/{sample_id}.ppm"
        path = output_dir / relative_path
        _write_scene(path, left, 300 + index, secondary=right)
        sha = file_sha256(path)
        records.append(_vlm_product(
            sample_id, relative_path, sha,
            ["multi_subject_conflict", "insufficient_visual_evidence"],
            {
                "business_category": "other",
                "style_tags": [],
                "visible_facilities": [],
                "price_range": "unknown",
                "unknown_fields": ["style_tags", "visible_facilities", "price_range"],
            },
        ))
        asset_registry.append(_asset_record(sample_id, f"synthetic:{GENERATOR_VERSION}:{sample_id}", relative_path, sha, "vlm", None))

    for index, visual in enumerate(("blank", "obscured", "private", "retail"), start=1):
        sample_id = f"vlm_v3_unknown_{index:02d}"
        relative_path = f"assets/vlm/{sample_id}.ppm"
        path = output_dir / relative_path
        _write_scene(path, visual, 400 + index)
        sha = file_sha256(path)
        records.append(_vlm_product(
            sample_id, relative_path, sha,
            ["insufficient_visual_evidence", "price_unknown"],
            {
                "business_category": "other",
                "style_tags": [],
                "visible_facilities": [],
                "price_range": "unknown",
                "unknown_fields": ["style_tags", "visible_facilities", "price_range"],
            },
        ))
        asset_registry.append(_asset_record(sample_id, f"synthetic:{GENERATOR_VERSION}:{sample_id}", relative_path, sha, "vlm", None))

    dialogue_cases = [
        ("city_correction", "User first requested a budget restaurant in Tampa, then corrected the city to Nashville. Return the active request.", ["budget", "restaurant"], "corrected", "city", "Nashville"),
        ("price_update", "The active request is a Philadelphia hotel. The user changes the price from budget to mid_range while keeping the city and business type.", ["Philadelphia", "hotel"], "updated", "price_range", "mid_range"),
        ("no_result", "The user asks for a restaurant in Atlantis and explicitly accepts a no-result response if the city is unsupported.", ["restaurant", "Atlantis", "no_result"], "active", "city", "Atlantis"),
        ("category_correction", "The user asked for a hotel in Reno, then corrected the business type to attraction without changing Reno.", ["Reno"], "corrected", "business_category", "attraction"),
        ("retain_facility", "The active request is a Nashville hotel with visible parking. The user changes only price to premium.", ["Nashville", "hotel", "parking"], "updated", "price_range", "premium"),
        ("latest_city", "The request moved from Tampa to Philadelphia and finally to New Orleans for a mid_range restaurant. Keep only the newest city.", ["mid_range", "restaurant"], "corrected", "city", "New Orleans"),
    ]
    for index, (name, dialogue, facts, state, task, value) in enumerate(dialogue_cases, start=1):
        records.append({
            "sample_id": f"vlm_v3_dialogue_{index:02d}_{name}",
            "scenario": "dialogue",
            "slices": ["dialogue_state", "context_recall", "task_key_value", "first_turn_routing"],
            "label_provenance": "synthetic_protocol_case_no_human_annotation",
            "dialogue": dialogue,
            "gold": {
                "context_facts": facts,
                "state": state,
                "task": task,
                "value": value,
                "route": "image_product_search",
            },
        })
    return records


def _vlm_product(
    sample_id: str, relative_path: str, sha: str, slices: list[str], gold: dict[str, Any]
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "scenario": "product",
        "source_id": f"synthetic:{GENERATOR_VERSION}:{sample_id}",
        "image_relative_path": relative_path,
        "image_sha256": sha,
        "slices": slices,
        "label_provenance": "weak_synthetic_programmatic_rule_v3_no_human_annotation",
        "gold": gold,
    }


def _asset_record(
    record_id: str, source_id: str, relative_path: str, sha: str, purpose: str, split: str | None
) -> dict[str, Any]:
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


def _query_text(definition: dict[str, Any]) -> str:
    filters = definition["filters"]
    if "no_result" in definition["slices"] and filters:
        return f"Find a {filters['price_range']} {filters['business_category']} in {filters['city']}; return no result if unavailable."
    if "visual_similar_business_irrelevant" in definition["slices"]:
        return "Return no travel business for this visually similar but irrelevant synthetic scene."
    return (
        f"Honor the explicit {filters['city']} {filters['price_range']} "
        f"{filters['business_category']} filters before visual ranking."
    )


def _write_scene(
    path: Path, primary: str, variant: int, price: int | None = None, secondary: str | None = None
) -> None:
    pixels = bytearray([238, 240, 244] * WIDTH * HEIGHT)
    _rect(pixels, 0, 0, WIDTH, 24, (28, 44, 70))
    _rect(pixels, 0, 190, WIDTH, 34, (48, 55, 62))
    if secondary:
        _draw_icon(pixels, primary, 4, 28, 106, 156)
        _rect(pixels, 109, 24, 6, 166, (110, 25, 25))
        _draw_icon(pixels, secondary, 115, 28, 105, 156)
    else:
        _draw_icon(pixels, primary, 18, 30, 188, 150)
    if price is not None:
        _rect(pixels, 46, 139, 132, 49, (255, 250, 210))
        _draw_price(pixels, price, 57, 146, (12, 18, 25))
    for offset in range(6):
        x = 6 + ((variant * 37 + offset * 29) % 205)
        y = 4 + ((variant * 19 + offset * 11) % 15)
        _rect(pixels, x, y, 5, 5, ((variant * 31 + offset * 17) % 255, 180, 90))
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii")
    with path.open("xb") as handle:
        handle.write(header)
        handle.write(pixels)


def _draw_icon(pixels: bytearray, kind: str, x: int, y: int, width: int, height: int) -> None:
    if kind == "restaurant":
        _rect(pixels, x + width // 5, y + height // 3, width * 3 // 5, height // 3, (205, 75, 65))
        _rect(pixels, x + width // 4, y + height * 2 // 3, 8, height // 3, (85, 55, 35))
        _rect(pixels, x + width * 3 // 4 - 8, y + height * 2 // 3, 8, height // 3, (85, 55, 35))
        _rect(pixels, x + 8, y + 12, 9, height // 2, (40, 40, 40))
        _rect(pixels, x + width - 17, y + 12, 9, height // 2, (40, 40, 40))
    elif kind == "hotel":
        _rect(pixels, x + 12, y + height // 2, width - 24, height // 3, (65, 110, 190))
        _rect(pixels, x + 20, y + height // 2 - 25, width // 3, 25, (245, 245, 250))
        _rect(pixels, x + 6, y + height // 2, 10, height // 2, (45, 55, 80))
        _rect(pixels, x + width - 16, y + height // 2, 10, height // 2, (45, 55, 80))
    elif kind == "attraction":
        _rect(pixels, x + 10, y + 25, width - 20, 12, (185, 150, 80))
        _rect(pixels, x + 4, y + height - 24, width - 8, 18, (145, 115, 65))
        for column in range(4):
            cx = x + 15 + column * max((width - 38) // 3, 1)
            _rect(pixels, cx, y + 37, 12, height - 61, (220, 195, 135))
    elif kind == "retail":
        for shelf in range(3):
            sy = y + 18 + shelf * 42
            _rect(pixels, x + 8, sy, width - 16, 8, (85, 65, 50))
            for item in range(6):
                color = ((item * 39 + shelf * 25) % 220, 90 + item * 15, 150)
                _rect(pixels, x + 14 + item * max((width - 36) // 6, 1), sy - 20, 12, 20, color)
    elif kind == "private":
        _rect(pixels, x + 25, y + 70, width - 50, 55, (75, 145, 105))
        _rect(pixels, x + 8, y + 88, 30, 50, (60, 120, 90))
        _rect(pixels, x + width - 38, y + 88, 30, 50, (60, 120, 90))
        _rect(pixels, x + width // 2 - 25, y + 132, 50, 8, (100, 70, 45))
    elif kind == "obscured":
        for stripe in range(12):
            _rect(pixels, x, y + stripe * 12, width, 7, (90 + stripe * 8, 90, 100))
    elif kind == "blank":
        _rect(pixels, x + 10, y + 10, width - 20, height - 20, (205, 208, 213))


def _draw_price(pixels: bytearray, value: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    _rect(pixels, x + 7, y, 4, 35, color)
    _rect(pixels, x, y + 6, 18, 4, color)
    _rect(pixels, x, y + 23, 18, 4, color)
    cursor = x + 27
    for digit in str(value):
        _draw_digit(pixels, int(digit), cursor, y, color)
        cursor += 24


def _draw_digit(pixels: bytearray, digit: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    segments = {
        0: "ab cdef".replace(" ", ""), 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg",
        5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg",
    }[digit]
    geometry = {
        "a": (4, 0, 14, 4), "b": (17, 3, 4, 13), "c": (17, 19, 4, 13),
        "d": (4, 31, 14, 4), "e": (0, 19, 4, 13), "f": (0, 3, 4, 13),
        "g": (4, 15, 14, 4),
    }
    for segment in segments:
        dx, dy, width, height = geometry[segment]
        _rect(pixels, x + dx, y + dy, width, height, color)


def _rect(
    pixels: bytearray, x: int, y: int, width: int, height: int, color: tuple[int, int, int]
) -> None:
    for py in range(max(y, 0), min(y + height, HEIGHT)):
        for px in range(max(x, 0), min(x + width, WIDTH)):
            offset = (py * WIDTH + px) * 3
            pixels[offset:offset + 3] = bytes(color)


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

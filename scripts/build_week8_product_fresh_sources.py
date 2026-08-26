#!/usr/bin/env python3
"""Build an isolated, unconsumed Yelp photo source overlay for Week 8 v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.alignment import build_medium_alignment, build_strong_alignment  # noqa: E402
from src.data.yelp_archives import extract_yelp_photo_files  # noqa: E402
from src.training.week7_data import (  # noqa: E402
    IDENTITY_FIELDS,
    Week7DataError,
    add_superseded_identities,
    canonical_sha256,
    load_consumed_identities,
    sha256_file,
)
class Week8FreshSourceError(ValueError):
    """Raised when the fresh-source overlay cannot satisfy its locked identity."""


def load_fresh_source_config(path: Path) -> dict[str, Any]:
    """Load only the data-build contract without importing the GPU runtime stack."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week8FreshSourceError(f"invalid Week 8 fresh-source config: {path}") from exc
    fresh = payload.get("fresh_source", {})
    if (
        payload.get("schema_version") != "week8_product_understanding_v2"
        or payload.get("week8", {}).get("source_version")
        != "week8_product_fresh_20260826_v1"
        or int(fresh.get("selected_photo_count", 0)) < 2000
        or int(fresh.get("minimum_eligible_count", 0)) < 2000
        or int(fresh.get("candidate_extract_count", 0))
        < int(fresh.get("selected_photo_count", 0))
        or not str(fresh.get("output_root") or "").startswith(
            "data/yelp/week8_product_fresh_20260826_v1"
        )
    ):
        raise Week8FreshSourceError("Week 8 v2 fresh-source identity is incomplete")
    return payload


OTA_CATEGORY_TERMS = {
    "hotel": {
        "hotels",
        "resorts",
        "bed & breakfast",
        "guest houses",
        "hostels",
    },
    "attraction": {
        "museums",
        "parks",
        "landmarks & historical buildings",
        "amusement parks",
        "zoos",
        "aquariums",
        "botanical gardens",
        "tours",
    },
    "restaurant": {
        "restaurants",
        "food",
        "cafes",
        "coffee & tea",
        "bars",
        "bakeries",
    },
}
STYLE_TERMS = {
    "casual",
    "classy",
    "cozy",
    "historic",
    "modern",
    "romantic",
    "rustic",
    "trendy",
    "upscale",
    "vintage",
}
FACILITY_TERMS = {
    "bar": ("bar", "tap"),
    "outdoor_seating": ("patio", "terrace", "outdoor seating"),
    "pool": ("pool",),
    "front_desk": ("front desk", "reception"),
    "parking": ("parking",),
    "wifi": ("wifi", "wi-fi"),
    "wheelchair_access": ("wheelchair", "accessible"),
    "gym": ("gym", "fitness"),
    "spa": ("spa",),
    "playground": ("playground",),
}


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
            count += 1
    return count


def _parquet_rows(path: Path, columns: list[str]) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise Week8FreshSourceError("pyarrow is required for fresh-source build") from exc
    if not Path(path).is_file():
        raise Week8FreshSourceError(f"required rebuilt table is missing: {path}")
    source = parquet.ParquetFile(path)
    missing = sorted(set(columns) - set(source.schema_arrow.names))
    if missing:
        raise Week8FreshSourceError(f"rebuilt table is missing columns: {missing}")
    for batch in source.iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


def _normalized_categories(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().casefold() for item in value if str(item).strip()}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return set()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return {
                str(item).strip().casefold()
                for item in parsed
                if str(item).strip()
            }
        return {item.strip().casefold() for item in stripped.split(",") if item.strip()}
    return set()


def ota_category(categories: Any) -> str | None:
    """Map only explicit Yelp OTA category labels to the three product classes."""

    normalized = _normalized_categories(categories)
    for category in ("hotel", "attraction", "restaurant"):
        if normalized & OTA_CATEGORY_TERMS[category]:
            return category
    return None


def caption_signals(caption: str) -> dict[str, Any]:
    """Count only observable caption words used to prioritize error-slice support."""

    text = str(caption or "").casefold()
    styles = sorted(term for term in STYLE_TERMS if term in text)
    facilities = sorted(
        name
        for name, terms in FACILITY_TERMS.items()
        if any(term in text for term in terms)
    )
    return {
        "styles": styles,
        "facilities": facilities,
        "richness": 2 * len(styles) + len(facilities),
    }


def _candidate_key(candidate: dict[str, Any]) -> tuple[int, str, str]:
    return (
        -int(candidate["caption_richness"]),
        str(candidate["seed_rank"]),
        str(candidate["photo_id"]),
    )


def select_ranked_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    selected_count: int,
    minimum_eligible_count: int,
    minimum_per_category: int,
    retain_all_categories_below: int = 0,
) -> list[dict[str, Any]]:
    """Select unique businesses with deterministic category coverage and richness rank."""

    candidates = list(candidates)
    if selected_count < minimum_eligible_count or minimum_eligible_count < 2000:
        raise Week8FreshSourceError("fresh source must retain at least 2000 candidates")
    by_category = {
        category: sorted(
            (row for row in candidates if row["ota_category"] == category),
            key=_candidate_key,
        )
        for category in OTA_CATEGORY_TERMS
    }
    short = {
        category: len(rows)
        for category, rows in by_category.items()
        if len(rows) < minimum_per_category
    }
    if short:
        raise Week8FreshSourceError(f"fresh OTA category support shortfall: {short}")
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for category in ("hotel", "attraction", "restaurant"):
        reserve_count = minimum_per_category
        if 0 < len(by_category[category]) < retain_all_categories_below:
            reserve_count = len(by_category[category])
        for row in by_category[category][:reserve_count]:
            selected.append(row)
            selected_ids.add(row["photo_id"])
    remaining = sorted(
        (row for row in candidates if row["photo_id"] not in selected_ids),
        key=_candidate_key,
    )
    selected.extend(remaining[: max(0, selected_count - len(selected))])
    if len(selected) != selected_count:
        raise Week8FreshSourceError(
            f"fresh source candidate shortfall: {len(selected)}/{selected_count}"
        )
    if (
        len({row["photo_id"] for row in selected}) != len(selected)
        or len({row["business_id"] for row in selected}) != len(selected)
        or len({row["source_id"] for row in selected}) != len(selected)
        or len({row["group_id"] for row in selected}) != len(selected)
    ):
        raise Week8FreshSourceError("fresh source selection is not photo/business unique")
    return sorted(selected, key=_candidate_key)


def collect_fresh_candidates(
    photos_path: Path,
    businesses_path: Path,
    consumed: dict[str, set[str]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose the best captioned, historically unconsumed photo for each OTA business."""

    business_columns = [
        "business_id",
        "name",
        "city",
        "state",
        "stars",
        "categories",
        "attributes",
        "hours",
    ]
    businesses: dict[str, dict[str, Any]] = {}
    category_counts: Counter[str] = Counter()
    for row in _parquet_rows(businesses_path, business_columns):
        business_id = str(row.get("business_id") or "")
        category = ota_category(row.get("categories"))
        if not business_id or category is None:
            continue
        group_id = f"yelp-business:{business_id}"
        if group_id in consumed["group_id"]:
            category_counts["consumed_group"] += 1
            continue
        row["ota_category"] = category
        businesses[business_id] = row
        category_counts[f"eligible_business_{category}"] += 1

    best_by_business: dict[str, dict[str, Any]] = {}
    photo_counts: Counter[str] = Counter()
    for row in _parquet_rows(
        photos_path, ["photo_id", "business_id", "caption", "label"]
    ):
        photo_counts["input_photo"] += 1
        photo_id = str(row.get("photo_id") or "")
        business_id = str(row.get("business_id") or "")
        business = businesses.get(business_id)
        native_caption = str(row.get("caption") or "").strip()
        if not photo_id or business is None:
            photo_counts["non_ota_or_ineligible_business"] += 1
            continue
        source_id = f"yelp-photo:{photo_id}"
        if source_id in consumed["source_id"]:
            photo_counts["consumed_source"] += 1
            continue
        label = str(row.get("label") or "unknown").strip() or "unknown"
        caption = native_caption or f"photo type: {label}"
        if not native_caption:
            photo_counts["fallback_label_caption"] += 1
        signals = caption_signals(caption)
        candidate = {
            "photo_id": photo_id,
            "business_id": business_id,
            "source_id": source_id,
            "group_id": f"yelp-business:{business_id}",
            "caption": caption,
            "label": label,
            "caption_source": "native" if native_caption else "photo_label_fallback",
            "ota_category": business["ota_category"],
            "caption_styles": signals["styles"],
            "caption_facilities": signals["facilities"],
            "caption_richness": signals["richness"],
            "seed_rank": hashlib.sha256(
                f"{seed}\0week8-product-fresh\0{photo_id}".encode("utf-8")
            ).hexdigest(),
            "business": business,
        }
        previous = best_by_business.get(business_id)
        if previous is None or _candidate_key(candidate) < _candidate_key(previous):
            best_by_business[business_id] = candidate
    candidates = list(best_by_business.values())
    stats = {
        "business_filter_counts": dict(sorted(category_counts.items())),
        "photo_filter_counts": dict(sorted(photo_counts.items())),
        "unique_eligible_business_photo_count": len(candidates),
        "eligible_category_counts": dict(
            sorted(Counter(row["ota_category"] for row in candidates).items())
        ),
    }
    return candidates, stats


def _validate_image(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise Week8FreshSourceError(f"extracted Yelp photo is unreadable: {path}") from exc
    if width <= 0 or height <= 0:
        raise Week8FreshSourceError(f"extracted Yelp photo has invalid dimensions: {path}")
    return int(width), int(height)


def _write_parquet(path: Path, rows: list[dict[str, Any]], schema: Any) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise Week8FreshSourceError("pyarrow is required for fresh-source output") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Week8FreshSourceError(f"refusing to overwrite fresh table: {path}")
    table = pa.Table.from_pylist(rows, schema=schema)
    parquet.write_table(table, path, compression="zstd")
    return {
        "path": path.as_posix(),
        "count": table.num_rows,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "columns": table.schema.names,
    }


def build_fresh_sources(
    root: Path,
    config_path: Path,
    *,
    rebuilt_yelp_root: Path,
    historical_root: Path,
    photos_zip: Path,
) -> dict[str, Any]:
    """Materialize the immutable Week 8 v2 source tables and image identities."""

    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    rebuilt_yelp_root = Path(rebuilt_yelp_root).resolve()
    historical_root = Path(historical_root).resolve()
    photos_zip = Path(photos_zip).resolve()
    config = load_fresh_source_config(config_path)
    fresh = config["fresh_source"]
    output_root = (root / fresh["output_root"]).resolve()
    try:
        output_root.relative_to(root)
    except ValueError as exc:
        raise Week8FreshSourceError("fresh-source output escapes the repository") from exc
    if output_root.exists():
        raise Week8FreshSourceError(f"refusing to overwrite fresh-source output: {output_root}")
    expected_zip_size = int(fresh["official_photos_zip_size"])
    if not photos_zip.is_file() or photos_zip.stat().st_size != expected_zip_size:
        raise Week8FreshSourceError("official Yelp Photos ZIP size mismatch")

    compatibility = {
        "dataset": {
            "source_paths": config["dataset"]["source_paths"],
            "seed": int(config["week8"]["seed"]),
        },
        "system_repair": {"enabled": True},
    }
    try:
        consumed, exclusion_evidence = load_consumed_identities(
            historical_root, compatibility
        )
        add_superseded_identities(
            historical_root, compatibility, consumed, exclusion_evidence
        )
    except Week7DataError as exc:
        raise Week8FreshSourceError(str(exc)) from exc
    photos_path = rebuilt_yelp_root / "interim" / "photos.parquet"
    businesses_path = rebuilt_yelp_root / "interim" / "business.parquet"
    candidates, filter_stats = collect_fresh_candidates(
        photos_path,
        businesses_path,
        consumed,
        seed=int(config["week8"]["seed"]),
    )
    candidate_pool = select_ranked_candidates(
        candidates,
        selected_count=int(fresh["candidate_extract_count"]),
        minimum_eligible_count=int(fresh["minimum_eligible_count"]),
        minimum_per_category=int(fresh["minimum_per_ota_category"]),
        retain_all_categories_below=int(fresh["retain_all_categories_below"]),
    )

    output_root.mkdir(parents=True, exist_ok=False)
    incomplete_path = output_root / "BUILD_INCOMPLETE.json"
    _write_json_new(
        incomplete_path,
        {
            "status": "INCOMPLETE",
            "build_id": fresh["build_id"],
            "config_sha256": sha256_file(config_path),
        },
    )
    raw_dir = output_root / "raw"
    requested_ids = {row["photo_id"] for row in candidate_pool}
    extracted = extract_yelp_photo_files(photos_zip, raw_dir, requested_ids)
    photos_dir = raw_dir / "photos"
    extracted_ids = {
        path.stem for path in photos_dir.glob("*.jpg") if path.is_file()
    }
    if (
        extracted_ids != requested_ids
        or int(extracted.get("extracted_photo_count", -1)) != len(candidate_pool)
    ):
        raise Week8FreshSourceError(
            f"fresh photo extraction incomplete: {len(extracted_ids)}/{len(candidate_pool)}"
        )

    validated_candidates = []
    hash_rejections: Counter[str] = Counter()
    candidate_rejection_rows = []
    seen_candidate_hashes: set[str] = set()
    for row in candidate_pool:
        image = photos_dir / f"{row['photo_id']}.jpg"
        try:
            width, height = _validate_image(image)
        except Week8FreshSourceError:
            hash_rejections["unreadable_image"] += 1
            candidate_rejection_rows.append(
                {
                    "photo_id": row["photo_id"],
                    "source_id": row["source_id"],
                    "group_id": row["group_id"],
                    "reason": "unreadable_image",
                    "image_sha256": None,
                }
            )
            continue
        digest = sha256_file(image)
        if digest in consumed["image_sha256"]:
            hash_rejections["historically_consumed"] += 1
            candidate_rejection_rows.append(
                {
                    "photo_id": row["photo_id"],
                    "source_id": row["source_id"],
                    "group_id": row["group_id"],
                    "reason": "historically_consumed_image_sha256",
                    "image_sha256": digest,
                }
            )
            continue
        if digest in seen_candidate_hashes:
            hash_rejections["candidate_duplicate"] += 1
            candidate_rejection_rows.append(
                {
                    "photo_id": row["photo_id"],
                    "source_id": row["source_id"],
                    "group_id": row["group_id"],
                    "reason": "candidate_duplicate_image_sha256",
                    "image_sha256": digest,
                }
            )
            continue
        seen_candidate_hashes.add(digest)
        row = dict(row)
        row["image_sha256"] = digest
        row["image_width"] = width
        row["image_height"] = height
        validated_candidates.append(row)
    selected = select_ranked_candidates(
        validated_candidates,
        selected_count=int(fresh["selected_photo_count"]),
        minimum_eligible_count=int(fresh["minimum_eligible_count"]),
        minimum_per_category=int(fresh["minimum_per_ota_category"]),
        retain_all_categories_below=int(fresh["retain_all_categories_below"]),
    )

    photo_rows = []
    image_index = []
    identity_rows = []
    seen_hashes: set[str] = set()
    for row in selected:
        photo_id = row["photo_id"]
        image = photos_dir / f"{photo_id}.jpg"
        width = int(row["image_width"])
        height = int(row["image_height"])
        digest = str(row["image_sha256"])
        if digest in seen_hashes:
            raise Week8FreshSourceError(f"selected fresh photo hash duplicated: {photo_id}")
        seen_hashes.add(digest)
        relative_image = (
            Path(fresh["output_root"]) / "raw" / "photos" / f"{photo_id}.jpg"
        ).as_posix()
        photo_rows.append(
            {
                "photo_id": photo_id,
                "business_id": row["business_id"],
                "caption": row["caption"],
                "label": row["label"],
                "image_path": relative_image,
            }
        )
        image_index.append(
            {
                "photo_id": photo_id,
                "business_id": row["business_id"],
                "image_path": relative_image,
                "image_valid": True,
            }
        )
        identity_rows.append(
            {
                "photo_id": photo_id,
                "business_id": row["business_id"],
                "source_id": row["source_id"],
                "group_id": row["group_id"],
                "image_sha256": digest,
                "ota_category": row["ota_category"],
                "caption_richness": row["caption_richness"],
                "caption_styles": row["caption_styles"],
                "caption_facilities": row["caption_facilities"],
                "caption_source": row["caption_source"],
                "seed_rank": row["seed_rank"],
                "image_path": relative_image,
                "image_width": width,
                "image_height": height,
            }
        )

    selected_businesses = [row["business"] for row in selected]
    strong_rows = build_strong_alignment(photo_rows, image_index)
    medium_rows = build_medium_alignment(
        photo_rows, image_index, selected_businesses
    )
    if len(strong_rows) != len(selected) or len(medium_rows) != len(selected):
        raise Week8FreshSourceError("fresh alignment row counts changed")

    import pyarrow as pa

    photos_schema = pa.schema(
        [
            ("photo_id", pa.string()),
            ("business_id", pa.string()),
            ("caption", pa.string()),
            ("label", pa.string()),
            ("image_path", pa.string()),
        ]
    )
    strong_schema = pa.schema(
        [
            ("pair_id", pa.string()),
            ("photo_id", pa.string()),
            ("business_id", pa.string()),
            ("image_path", pa.string()),
            ("caption", pa.string()),
            ("label", pa.string()),
            ("alignment_type", pa.string()),
        ]
    )
    medium_schema = pa.schema(
        [
            ("pair_id", pa.string()),
            ("photo_id", pa.string()),
            ("business_id", pa.string()),
            ("image_path", pa.string()),
            ("business_description", pa.string()),
            ("attribute_dimension_labels", pa.list_(pa.string())),
            ("alignment_type", pa.string()),
        ]
    )
    table_evidence = {
        "interim/photos.parquet": _write_parquet(
            output_root / "interim" / "photos.parquet",
            photo_rows,
            photos_schema,
        ),
        "processed/strong_image_caption_pairs.parquet": _write_parquet(
            output_root / "processed" / "strong_image_caption_pairs.parquet",
            strong_rows,
            strong_schema,
        ),
        "processed/image_business_attribute_pairs.parquet": _write_parquet(
            output_root / "processed" / "image_business_attribute_pairs.parquet",
            medium_rows,
            medium_schema,
        ),
    }
    identity_path = output_root / "fresh_identity_manifest.jsonl"
    _write_jsonl_new(identity_path, identity_rows)
    rejection_path = output_root / "candidate_rejections.jsonl"
    _write_jsonl_new(rejection_path, candidate_rejection_rows)
    category_counts = dict(
        sorted(Counter(row["ota_category"] for row in identity_rows).items())
    )
    image_hash_aggregate = canonical_sha256(
        [
            {"photo_id": row["photo_id"], "image_sha256": row["image_sha256"]}
            for row in identity_rows
        ]
    )
    dimension_counts = {
        "source_id": len({row["source_id"] for row in identity_rows}),
        "group_id": len({row["group_id"] for row in identity_rows}),
        "image_sha256": len({row["image_sha256"] for row in identity_rows}),
    }
    if any(value != len(selected) for value in dimension_counts.values()):
        raise Week8FreshSourceError("fresh identity dimensions are not unique")
    manifest = {
        "schema_version": "week8_product_fresh_source_v1",
        "status": "COMPLETED",
        "build_id": fresh["build_id"],
        "source_version": config["week8"]["source_version"],
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "seed": int(config["week8"]["seed"]),
        "source": "Yelp Open Dataset official Photos ZIP and rebuilt metadata",
        "source_url": "https://business.yelp.com/external-assets/files/Yelp-Photos.zip",
        "official_photos_zip": {
            "size": photos_zip.stat().st_size,
            "sha256": sha256_file(photos_zip),
        },
        "rebuilt_inputs": {
            "interim/photos.parquet": {
                "size": photos_path.stat().st_size,
                "sha256": sha256_file(photos_path),
            },
            "interim/business.parquet": {
                "size": businesses_path.stat().st_size,
                "sha256": sha256_file(businesses_path),
            },
        },
        "historical_exclusion_evidence": exclusion_evidence,
        "historical_consumed_dimension_counts": {
            field: len(consumed[field]) for field in IDENTITY_FIELDS
        },
        "filter_statistics": filter_stats,
        "candidate_extract_count": len(candidate_pool),
        "validated_candidate_count": len(validated_candidates),
        "candidate_rejection_counts": dict(sorted(hash_rejections.items())),
        "candidate_rejections": {
            "path": rejection_path.relative_to(output_root).as_posix(),
            "count": len(candidate_rejection_rows),
            "sha256": sha256_file(rejection_path),
        },
        "selected_photo_count": len(selected),
        "minimum_eligible_count": int(fresh["minimum_eligible_count"]),
        "category_counts": category_counts,
        "caption_source_counts": dict(
            sorted(Counter(row["caption_source"] for row in identity_rows).items())
        ),
        "identity_unique_counts": dimension_counts,
        "historical_identity_overlap_counts": {
            "source_id": 0,
            "group_id": 0,
            "image_sha256": 0,
        },
        "image_hash_aggregate_sha256": image_hash_aggregate,
        "identity_manifest": {
            "path": identity_path.relative_to(output_root).as_posix(),
            "count": len(identity_rows),
            "sha256": sha256_file(identity_path),
        },
        "raw_extract_manifest_sha256": sha256_file(
            raw_dir / "extract_photo_manifest.json"
        ),
        "tables": table_evidence,
    }
    manifest_path = output_root / "fresh_source_manifest.json"
    _write_json_new(manifest_path, manifest)
    incomplete_path.unlink()
    result = dict(manifest)
    result["manifest_path"] = manifest_path.relative_to(root).as_posix()
    result["manifest_sha256"] = sha256_file(manifest_path)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/week8/product_understanding_v2.json",
    )
    value.add_argument("--rebuilt-yelp-root", type=Path, required=True)
    value.add_argument("--historical-root", type=Path, required=True)
    value.add_argument("--photos-zip", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    result = build_fresh_sources(
        ROOT,
        args.config,
        rebuilt_yelp_root=args.rebuilt_yelp_root,
        historical_root=args.historical_root,
        photos_zip=args.photos_zip,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

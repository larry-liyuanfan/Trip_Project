"""Prepare the bounded Week 1 OTA sample from Yelp Open Dataset JSONL files."""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


BUSINESS_FILENAMES = [
    "yelp_academic_dataset_business.json",
    "business.json",
]
REVIEW_FILENAMES = [
    "yelp_academic_dataset_review.json",
    "review.json",
]
PHOTO_FILENAMES = [
    "photos.json",
    "photo.json",
    "yelp_academic_dataset_photo.json",
]

OTA_CATEGORY_KEYWORDS = {
    "restaurants",
    "food",
    "cafes",
    "coffee & tea",
    "hotels",
    "hotel",
    "travel services",
    "active life",
    "arts & entertainment",
    "museums",
    "landmarks & historical buildings",
    "shopping",
    "nightlife",
    "parks",
    "local flavor",
}


def prepare_yelp_subset(
    raw_dir: Path,
    output_dir: Path,
    max_businesses: int | None = None,
    max_reviews_per_business: int = 5,
    include_closed: bool = False,
) -> dict[str, Any]:
    """Build sample catalog, review, photo, and manifest artifacts."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    business_path = find_first_existing(raw_dir, BUSINESS_FILENAMES)
    review_path = find_first_existing(raw_dir, REVIEW_FILENAMES)
    photo_path = find_first_existing(raw_dir, PHOTO_FILENAMES, required=False)

    businesses = select_ota_businesses(
        iter_jsonl(business_path),
        max_businesses=max_businesses,
        include_closed=include_closed,
    )
    business_by_id = {business["business_id"]: business for business in businesses}
    poi_ids = {business_id: f"yelp_{business_id}" for business_id in business_by_id}

    catalog_records = [business_to_poi_record(business) for business in businesses]
    review_records = select_reviews(
        iter_jsonl(review_path),
        poi_ids=poi_ids,
        max_reviews_per_business=max_reviews_per_business,
    )
    multimodal_records = (
        select_photo_records(iter_jsonl(photo_path), poi_ids=poi_ids) if photo_path else []
    )

    write_jsonl(output_dir / "poi_catalog.jsonl", catalog_records)
    write_jsonl(output_dir / "reviews.jsonl", review_records)
    write_jsonl(output_dir / "multimodal_items.jsonl", multimodal_records)

    manifest = {
        "source": "Yelp Open Dataset",
        "raw_dir": str(raw_dir),
        "business_count": len(catalog_records),
        "review_count": len(review_records),
        "multimodal_item_count": len(multimodal_records),
        "max_reviews_per_business": max_reviews_per_business,
        "outputs": [
            "poi_catalog.jsonl",
            "reviews.jsonl",
            "multimodal_items.jsonl",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def select_ota_businesses(
    businesses: Iterable[dict[str, Any]],
    max_businesses: int | None,
    include_closed: bool,
) -> list[dict[str, Any]]:
    """Select open OTA-relevant businesses up to the optional sample cap."""
    selected: list[dict[str, Any]] = []
    for business in businesses:
        if not include_closed and business.get("is_open") == 0:
            continue
        categories = parse_categories(business.get("categories"))
        if not is_ota_relevant(categories):
            continue
        selected.append(business)
        if max_businesses is not None and len(selected) >= max_businesses:
            break
    return selected


def business_to_poi_record(business: dict[str, Any]) -> dict[str, Any]:
    """Convert a Yelp business row to the sample retrieval catalog schema."""
    categories = parse_categories(business.get("categories"))
    primary_category = infer_primary_category(categories)
    name = business.get("name", "")
    city = business.get("city", "")
    state = business.get("state", "")
    tags = sorted(set(categories))
    description_parts = [name, primary_category, city, state]
    return {
        "poi_id": f"yelp_{business['business_id']}",
        "source": "yelp_open_dataset",
        "business_id": business["business_id"],
        "name": name,
        "category": primary_category,
        "city": city,
        "state": state,
        "latitude": business.get("latitude"),
        "longitude": business.get("longitude"),
        "rating": business.get("stars"),
        "review_count": business.get("review_count"),
        "tags": tags,
        "attributes": business.get("attributes") or {},
        "description": " ".join(str(part) for part in description_parts if part),
    }


def select_reviews(
    reviews: Iterable[dict[str, Any]],
    poi_ids: dict[str, str],
    max_reviews_per_business: int,
) -> list[dict[str, Any]]:
    """Collect a bounded number of reviews for each selected business."""
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for review in reviews:
        business_id = review.get("business_id")
        if business_id not in poi_ids:
            continue
        if counts.get(business_id, 0) >= max_reviews_per_business:
            continue
        counts[business_id] = counts.get(business_id, 0) + 1
        selected.append(
            {
                "review_id": review.get("review_id"),
                "poi_id": poi_ids[business_id],
                "business_id": business_id,
                "rating": review.get("stars"),
                "text": review.get("text", ""),
                "date": review.get("date"),
                "source": "yelp_open_dataset",
            }
        )
    return selected


def select_photo_records(
    photos: Iterable[dict[str, Any]],
    poi_ids: dict[str, str],
) -> list[dict[str, Any]]:
    """Link selected business IDs to photo metadata and expected local paths."""
    records: list[dict[str, Any]] = []
    for photo in photos:
        business_id = photo.get("business_id")
        photo_id = photo.get("photo_id")
        if business_id not in poi_ids or not photo_id:
            continue
        records.append(
            {
                "item_id": f"yelp_photo_{photo_id}",
                "poi_id": poi_ids[business_id],
                "business_id": business_id,
                "photo_id": photo_id,
                "image_path": f"data/yelp/raw/photos/{photo_id}.jpg",
                "caption": photo.get("caption", ""),
                "label": photo.get("label", ""),
                "source": "yelp_open_dataset",
            }
        )
    return records


def parse_categories(categories: Any) -> list[str]:
    """Normalize Yelp category strings or lists to lowercase values."""
    if not categories:
        return []
    if isinstance(categories, list):
        return [str(category).strip().lower() for category in categories if str(category).strip()]
    return [category.strip().lower() for category in str(categories).split(",") if category.strip()]


def is_ota_relevant(categories: list[str]) -> bool:
    """Return whether any category belongs to the OTA-oriented allowlist."""
    return any(category in OTA_CATEGORY_KEYWORDS for category in categories)


def infer_primary_category(categories: list[str]) -> str:
    """Map detailed Yelp categories to a stable sample POI type."""
    priority = [
        ("Cafe", {"cafes", "coffee & tea"}),
        ("Restaurant", {"restaurants", "food"}),
        ("Hotel", {"hotels", "hotel"}),
        ("Attraction", {"arts & entertainment", "museums", "landmarks & historical buildings", "parks"}),
        ("Shopping", {"shopping"}),
        ("Nightlife", {"nightlife"}),
    ]
    category_set = set(categories)
    for label, matches in priority:
        if category_set & matches:
            return label
    return "POI"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield non-empty JSONL objects from a UTF-8 source."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records as UTF-8 JSON Lines with stable newline behavior."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_first_existing(
    directory: Path,
    filenames: list[str],
    required: bool = True,
) -> Path | None:
    """Resolve accepted Yelp filename variants or raise when required."""
    for filename in filenames:
        path = directory / filename
        if path.exists():
            return path
    if required:
        expected = ", ".join(filenames)
        raise FileNotFoundError(f"Expected one of [{expected}] under {directory}")
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for bounded sample preparation."""
    parser = argparse.ArgumentParser(description="Prepare a small OTA subset from Yelp Open Dataset JSONL files.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/yelp/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/yelp/processed/ota_subset_v1"))
    parser.add_argument("--max-businesses", type=int, default=200)
    parser.add_argument("--max-reviews-per-business", type=int, default=5)
    parser.add_argument("--include-closed", action="store_true")
    return parser


def main() -> None:
    """Build the sample and print its manifest."""
    args = build_arg_parser().parse_args()
    manifest = prepare_yelp_subset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        max_businesses=args.max_businesses,
        max_reviews_per_business=args.max_reviews_per_business,
        include_closed=args.include_closed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

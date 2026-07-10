from collections import Counter
from typing import Any


def build_dataset_statistics(
    businesses: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    photos: list[dict[str, Any]],
    image_index: list[dict[str, Any]],
    strong: list[dict[str, Any]],
    medium: list[dict[str, Any]],
    weak: list[dict[str, Any]],
) -> dict[str, Any]:
    categories = Counter()
    business_by_id = {}
    cities = set()
    for business in businesses:
        if business.get("business_id"):
            business_by_id[str(business["business_id"])] = business
        if business.get("city"):
            cities.add(str(business["city"]))
        categories.update(business.get("categories") or [])
    valid_images = [row for row in image_index if row.get("image_valid") in {True, "True", "true", 1, "1"}]
    # Baseline balance metrics must describe the actual strong-supervision set,
    # not every photo metadata row (many photos have blank captions).
    label_distribution = Counter(str(pair.get("label")) for pair in strong if pair.get("label"))
    caption_lengths = [len(str(pair.get("caption")).strip()) for pair in strong if str(pair.get("caption") or "").strip()]
    weak_categories = Counter()
    for weak_row in weak:
        business = business_by_id.get(str(weak_row.get("business_id")))
        if not business:
            continue
        weak_categories.update((business.get("categories") or [])[:1])
    return {
        "business_count": len(businesses),
        "review_count": len(reviews),
        "photo_metadata_count": len(photos),
        "city_count": len(cities),
        "valid_image_count": len(valid_images),
        "valid_image_ratio": len(valid_images) / len(photos) if photos else 0,
        "strong_pairs": len(strong),
        "medium_pairs": len(medium),
        "weak_pairs": len(weak),
        "top_categories": categories.most_common(20),
        "photo_label_distribution": dict(label_distribution),
        "caption_length_stats": caption_length_stats(caption_lengths),
        "weak_group_top_categories": weak_categories.most_common(20),
        "businesses_with_reviews": len({row.get("business_id") for row in reviews if row.get("business_id")}),
        "businesses_with_valid_images": len({row.get("business_id") for row in valid_images if row.get("business_id")}),
    }


def caption_length_stats(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {
            "caption_count": 0,
            "min_chars": None,
            "mean_chars": None,
            "max_chars": None,
        }
    return {
        "caption_count": len(lengths),
        "min_chars": min(lengths),
        "mean_chars": sum(lengths) / len(lengths),
        "max_chars": max(lengths),
    }

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
    for business in businesses:
        categories.update(business.get("categories") or [])
    valid_images = [row for row in image_index if row.get("image_valid") in {True, "True", "true", 1, "1"}]
    label_distribution = Counter(str(photo.get("label")) for photo in photos if photo.get("label"))
    caption_lengths = [len(str(photo.get("caption")).strip()) for photo in photos if str(photo.get("caption") or "").strip()]
    return {
        "business_count": len(businesses),
        "review_count": len(reviews),
        "photo_metadata_count": len(photos),
        "valid_image_count": len(valid_images),
        "valid_image_ratio": len(valid_images) / len(photos) if photos else 0,
        "strong_pairs": len(strong),
        "medium_pairs": len(medium),
        "weak_pairs": len(weak),
        "top_categories": categories.most_common(20),
        "photo_label_distribution": dict(label_distribution),
        "caption_length_stats": caption_length_stats(caption_lengths),
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

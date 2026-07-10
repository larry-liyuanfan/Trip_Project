from collections import defaultdict
from typing import Any, Iterable

from src.data.text_templates import business_description, mapping_value


def build_strong_alignment(
    photos: Iterable[dict[str, Any]],
    image_index: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid_images = _valid_image_by_photo_id(image_index)
    rows = []
    for photo in photos:
        photo_id = str(photo.get("photo_id"))
        image = valid_images.get(photo_id)
        caption = str(photo.get("caption") or "").strip()
        # Caption-bearing rows are the actual single-image/single-text supervision set.
        if not image or not caption:
            continue
        rows.append(
            {
                "pair_id": f"strong_{photo_id}",
                "photo_id": photo_id,
                "business_id": photo.get("business_id"),
                "image_path": image.get("image_path"),
                "caption": caption,
                "label": photo.get("label") or "",
                "alignment_type": "strong",
            }
        )
    return rows


def build_medium_alignment(
    photos: Iterable[dict[str, Any]],
    image_index: Iterable[dict[str, Any]],
    businesses: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid_images = _valid_image_by_photo_id(image_index)
    business_by_id = {str(row.get("business_id")): row for row in businesses if row.get("business_id")}
    rows = []
    for photo in photos:
        photo_id = str(photo.get("photo_id"))
        business = business_by_id.get(str(photo.get("business_id")))
        image = valid_images.get(photo_id)
        if not business or not image:
            continue
        rows.append(
            {
                "pair_id": f"medium_{photo_id}",
                "photo_id": photo_id,
                "business_id": business.get("business_id"),
                "image_path": image.get("image_path"),
                "business_description": business_description(business),
                # Keep explicit attribute labels so VLM tasks can target parking,
                # ambience, hours, and service attributes instead of only free text.
                "attribute_dimension_labels": attribute_dimension_labels(business),
                "alignment_type": "medium",
            }
        )
    return rows


def attribute_dimension_labels(business: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    attributes = mapping_value(business.get("attributes"))
    labels.extend(str(key) for key, value in attributes.items() if value not in {None, "", "None"})
    if mapping_value(business.get("hours")):
        labels.append("hours")
    return sorted(set(labels))


def build_weak_alignment(
    photos: Iterable[dict[str, Any]],
    image_index: Iterable[dict[str, Any]],
    reviews: Iterable[dict[str, Any]],
    max_reviews_per_business: int,
    max_images_per_business: int,
) -> list[dict[str, Any]]:
    valid_images = _valid_image_by_photo_id(image_index)
    photos_by_business: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reviews_by_business: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for photo in photos:
        photo_id = str(photo.get("photo_id"))
        if photo_id in valid_images and photo.get("business_id"):
            photos_by_business[str(photo["business_id"])].append(photo)
    for review in reviews:
        if review.get("business_id"):
            reviews_by_business[str(review["business_id"])].append(review)
    rows = []
    for business_id in sorted(set(photos_by_business) & set(reviews_by_business)):
        selected_photos = photos_by_business[business_id][:max_images_per_business]
        selected_reviews = reviews_by_business[business_id][:max_reviews_per_business]
        rows.append(
            {
                "pair_id": f"weak_{business_id}",
                "business_id": business_id,
                "photo_ids": [photo.get("photo_id") for photo in selected_photos],
                "image_paths": [valid_images[str(photo.get("photo_id"))].get("image_path") for photo in selected_photos],
                "review_ids": [review.get("review_id") for review in selected_reviews],
                "review_texts": [review.get("text") for review in selected_reviews],
                "alignment_type": "weak",
            }
        )
    return rows


def _valid_image_by_photo_id(image_index: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("photo_id")): row
        for row in image_index
        if row.get("photo_id") is not None and row.get("image_valid") in {True, "True", "true", 1, "1"}
    }

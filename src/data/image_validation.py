"""Validate local Yelp images while retaining missing and corruption reasons."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


def validate_photo_images(photo_rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collect validated index rows for tests and small in-memory callers."""
    rows: list[dict[str, Any]] = []
    summary = stream_validate_photo_images(photo_rows, rows.append)
    return rows, summary


def stream_validate_photo_images(
    photo_rows: Iterable[dict[str, Any]],
    row_sink: Callable[[dict[str, Any]], None],
) -> dict[str, int]:
    """Validate photos one at a time and emit image-index rows immediately."""
    summary = {"total_images": 0, "valid_images": 0, "missing_images": 0, "corrupted_images": 0}
    for photo in photo_rows:
        summary["total_images"] += 1
        row, status = validate_photo_image(photo)
        if status == "valid":
            summary["valid_images"] += 1
        elif status == "missing":
            summary["missing_images"] += 1
        else:
            summary["corrupted_images"] += 1
        row_sink(row)
    return summary


def iter_validated_photo_images(
    photo_rows: Iterable[dict[str, Any]],
    workers: int = 1,
) -> Iterable[tuple[dict[str, Any], str]]:
    """Validate one bounded photo batch, optionally overlapping file I/O."""
    if workers < 1:
        raise ValueError("image validation workers must be at least 1")
    if workers == 1:
        for photo in photo_rows:
            yield validate_photo_image(photo)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(validate_photo_image, photo_rows)


def validate_photo_image(photo: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return one index row plus a compact status for streaming summaries."""
    path = Path(str(photo.get("image_path", "")))
    row = {
        "photo_id": photo.get("photo_id"),
        "business_id": photo.get("business_id"),
        "image_path": str(path).replace("\\", "/"),
        "image_valid": False,
        "image_width": None,
        "image_height": None,
        "validation_error": None,
    }
    if not path.exists():
        row["validation_error"] = "missing"
        return row, "missing"
    try:
        with Image.open(path) as image:
            # `load()` forces Pillow to decode image data, which catches corrupt
            # files without reopening every image just to read its dimensions.
            image.load()
            row["image_width"], row["image_height"] = image.size
        row["image_valid"] = True
        return row, "valid"
    except Exception as exc:
        row["validation_error"] = f"unreadable: {exc}"
        return row, "corrupted"

from pathlib import Path
from typing import Any, Iterable

from PIL import Image


def validate_photo_images(photo_rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    summary = {"total_images": 0, "valid_images": 0, "missing_images": 0, "corrupted_images": 0}
    for photo in photo_rows:
        summary["total_images"] += 1
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
            summary["missing_images"] += 1
        else:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    row["image_width"], row["image_height"] = image.size
                row["image_valid"] = True
                summary["valid_images"] += 1
            except Exception as exc:
                row["validation_error"] = f"unreadable: {exc}"
                summary["corrupted_images"] += 1
        rows.append(row)
    return rows, summary

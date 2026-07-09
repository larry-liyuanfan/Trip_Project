from pathlib import Path
from typing import Any, Iterable


def parse_photo_records(records: Iterable[dict[str, Any]], image_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        photo_id = record.get("photo_id")
        if not photo_id:
            continue
        rows.append(
            {
                "photo_id": photo_id,
                "business_id": record.get("business_id"),
                "caption": record.get("caption") or "",
                "label": record.get("label") or "",
                "image_path": str(image_root / f"{photo_id}.jpg").replace("\\", "/"),
            }
        )
    return rows

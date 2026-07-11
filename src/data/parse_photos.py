"""Parse Yelp photo metadata and map each photo ID to its local JPEG path."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterable


def parse_photo_records(records: Iterable[dict[str, Any]], image_root: Path) -> list[dict[str, Any]]:
    """Collect parsed photo rows for bounded callers and tests."""
    rows: list[dict[str, Any]] = []
    for record in records:
        row = parse_photo_record(record, image_root)
        if row is not None:
            rows.append(row)
    return rows


def parse_photo_record(record: dict[str, Any], image_root: Path) -> dict[str, Any] | None:
    """Convert one metadata row, rejecting records without a `photo_id`."""
    photo_id = record.get("photo_id")
    if not photo_id:
        return None
    return {
        "photo_id": photo_id,
        "business_id": record.get("business_id"),
        "caption": record.get("caption") or "",
        "label": record.get("label") or "",
        "image_path": str(image_root / f"{photo_id}.jpg").replace("\\", "/"),
    }


def stream_photo_records(
    records: Iterable[dict[str, Any]],
    image_root: Path,
    row_sink: Callable[[dict[str, Any]], None],
) -> dict[str, int]:
    """Parse photo metadata without retaining the full 200K-row table in Python."""
    summary = {"input_photos": 0, "parsed_photos": 0, "missing_photo_id": 0}
    for record in records:
        summary["input_photos"] += 1
        row = parse_photo_record(record, image_root)
        if row is None:
            summary["missing_photo_id"] += 1
            continue
        row_sink(row)
        summary["parsed_photos"] += 1
    return summary

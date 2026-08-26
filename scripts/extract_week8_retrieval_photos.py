#!/usr/bin/env python3
"""Extract the formal Week 8 retrieval images into an isolated source overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.yelp_archives import extract_yelp_photo_files  # noqa: E402


class Week8RetrievalPhotoError(ValueError):
    """Raised when the isolated retrieval photo identity is incomplete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_photo_ids(metadata_path: Path, expected_count: int) -> set[str]:
    photo_ids: set[str] = set()
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            photo_id = row.get("image_id")
            expected_path = f"data/yelp/raw/photos/{photo_id}.jpg"
            if (
                not isinstance(photo_id, str)
                or not photo_id
                or row.get("source_image_path") != expected_path
                or photo_id in photo_ids
            ):
                raise Week8RetrievalPhotoError(
                    f"invalid retrieval photo identity at metadata line {line_number}"
                )
            photo_ids.add(photo_id)
    if len(photo_ids) != expected_count:
        raise Week8RetrievalPhotoError(
            f"retrieval photo count mismatch: {len(photo_ids)} != {expected_count}"
        )
    return photo_ids


def extract_overlay(
    *,
    metadata_path: Path,
    photos_zip: Path,
    output_root: Path,
    expected_count: int = 1000,
    expected_zip_size: int = 7_447_210_067,
) -> dict[str, Any]:
    metadata_path = Path(metadata_path).resolve()
    photos_zip = Path(photos_zip).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise Week8RetrievalPhotoError("refusing to overwrite retrieval photo overlay")
    if not photos_zip.is_file() or photos_zip.stat().st_size != expected_zip_size:
        raise Week8RetrievalPhotoError("official Yelp Photos ZIP size mismatch")
    photo_ids = load_photo_ids(metadata_path, expected_count)
    raw_dir = output_root / "data" / "yelp" / "raw"
    extracted = extract_yelp_photo_files(photos_zip, raw_dir, photo_ids)
    photos_dir = raw_dir / "photos"
    extracted_ids = {
        path.stem for path in photos_dir.glob("*.jpg") if path.is_file()
    }
    if extracted_ids != photo_ids or extracted.get("extracted_photo_count") != expected_count:
        missing = sorted(photo_ids - extracted_ids)
        raise Week8RetrievalPhotoError(
            f"retrieval photo extraction incomplete: missing {len(missing)}"
        )
    image_hashes = {
        photo_id: sha256_file(photos_dir / f"{photo_id}.jpg")
        for photo_id in sorted(photo_ids)
    }
    aggregate = hashlib.sha256(
        "".join(f"{key}:{value}\n" for key, value in image_hashes.items()).encode("utf-8")
    ).hexdigest()
    result = {
        "schema_version": "week8_retrieval_photo_overlay_v1",
        "status": "COMPLETED",
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "photos_zip": str(photos_zip),
        "photos_zip_size": photos_zip.stat().st_size,
        "requested_photo_count": expected_count,
        "extracted_photo_count": len(extracted_ids),
        "image_identity_sha256": aggregate,
        "raw_extract_manifest_sha256": sha256_file(
            raw_dir / "extract_photo_manifest.json"
        ),
    }
    manifest_path = output_root / "week8_retrieval_photo_overlay.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    result["manifest_sha256"] = sha256_file(manifest_path)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--metadata", type=Path, required=True)
    value.add_argument("--photos-zip", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--expected-count", type=int, default=1000)
    value.add_argument("--expected-zip-size", type=int, default=7_447_210_067)
    return value


def main() -> int:
    args = parser().parse_args()
    result = extract_overlay(
        metadata_path=args.metadata,
        photos_zip=args.photos_zip,
        output_root=args.output_root,
        expected_count=args.expected_count,
        expected_zip_size=args.expected_zip_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

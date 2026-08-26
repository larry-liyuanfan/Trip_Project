#!/usr/bin/env python3
"""Rebuild only the Yelp tables required by the Week 8 product data lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_yelp_alignment import run_alignment  # noqa: E402
from scripts.parse_yelp_json import run_parse  # noqa: E402


class Week8YelpRebuildError(ValueError):
    """Raised when an official archive or immutable rebuild target is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_unique_member(archive: Path, suffix: str, destination: Path) -> dict[str, Any]:
    """Extract one exact-suffix member without unpacking either full Yelp archive."""
    with zipfile.ZipFile(archive) as source:
        matches = [name for name in source.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise Week8YelpRebuildError(
                f"expected one {suffix} member in {archive}, found {len(matches)}"
            )
        member = matches[0]
        info = source.getinfo(member)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open(info) as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
    return {
        "archive": str(archive),
        "member": member,
        "compressed_size": info.compress_size,
        "uncompressed_size": info.file_size,
        "output": str(destination),
        "output_sha256": sha256_file(destination),
    }


def extract_unique_tar_member_from_zip(
    archive: Path,
    *,
    tar_basename: str,
    member_basename: str,
    destination: Path,
) -> dict[str, Any]:
    """Stream one file from the official ZIP -> TAR layout without unpacking either archive."""

    with zipfile.ZipFile(archive) as source:
        tar_matches = [
            name for name in source.namelist() if Path(name).name == tar_basename
        ]
        if len(tar_matches) != 1:
            raise Week8YelpRebuildError(
                f"expected one {tar_basename} member in {archive}, found {len(tar_matches)}"
            )
        tar_member = tar_matches[0]
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.name}.partial")
        if destination.exists() or partial.exists():
            raise Week8YelpRebuildError(f"refusing to overwrite extracted member: {destination}")
        matches: list[dict[str, Any]] = []
        try:
            with source.open(tar_member) as tar_stream, tarfile.open(
                fileobj=tar_stream, mode="r|"
            ) as nested:
                for info in nested:
                    if not info.isfile() or Path(info.name).name != member_basename:
                        continue
                    matches.append({"member": info.name, "uncompressed_size": info.size})
                    extracted = nested.extractfile(info)
                    if extracted is None:
                        raise Week8YelpRebuildError(
                            f"cannot read {info.name} from {tar_member}"
                        )
                    if len(matches) == 1:
                        with partial.open("xb") as writer:
                            shutil.copyfileobj(extracted, writer, length=1024 * 1024)
            if len(matches) != 1:
                raise Week8YelpRebuildError(
                    f"expected one {member_basename} member in {tar_member}, found {len(matches)}"
                )
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    return {
        "archive": str(archive),
        "outer_member": tar_member,
        "member": matches[0]["member"],
        "uncompressed_size": matches[0]["uncompressed_size"],
        "output": str(destination),
        "output_sha256": sha256_file(destination),
    }


def rebuild(args: argparse.Namespace) -> dict[str, Any]:
    json_zip = args.json_zip.resolve()
    photos_zip = args.photos_zip.resolve()
    image_root = args.existing_image_root.resolve()
    output = args.output_yelp_root.resolve()
    manifest_path = args.manifest.resolve()
    if output.exists() or manifest_path.exists():
        raise Week8YelpRebuildError("refusing to overwrite Week 8 Yelp rebuild output")
    for archive, expected_size in (
        (json_zip, args.json_zip_size),
        (photos_zip, args.photos_zip_size),
    ):
        if not archive.is_file() or archive.stat().st_size != expected_size:
            raise Week8YelpRebuildError(
                f"official archive size mismatch: {archive}"
            )
        if not zipfile.is_zipfile(archive):
            raise Week8YelpRebuildError(f"invalid ZIP archive: {archive}")
    if not image_root.is_dir():
        raise Week8YelpRebuildError(f"existing Yelp image root missing: {image_root}")

    raw = output / "raw"
    business_json = raw / "yelp_academic_dataset_business.json"
    photo_json = raw / "photos.json"
    empty_review_json = raw / "week8_empty_reviews.json"
    extracted = [
        extract_unique_tar_member_from_zip(
            json_zip,
            tar_basename="yelp_dataset.tar",
            member_basename="yelp_academic_dataset_business.json",
            destination=business_json,
        ),
        extract_unique_tar_member_from_zip(
            photos_zip,
            tar_basename="yelp_photos.tar",
            member_basename="photos.json",
            destination=photo_json,
        ),
    ]
    empty_review_json.open("x", encoding="utf-8").close()
    raw.mkdir(parents=True, exist_ok=True)
    os.symlink(image_root, raw / "photos", target_is_directory=True)

    config = {
        "paths": {
            "business_json": str(business_json),
            "review_json": str(empty_review_json),
            "photo_json": str(photo_json),
            "image_root": str(raw / "photos"),
            "interim_dir": str(output / "interim"),
            "processed_dir": str(output / "processed"),
            "logs_dir": str(output / "logs"),
            "validation_dir": str(output / "validation"),
            "report_path": str(output / "rebuild_report.md"),
        },
        "output": {"format": "parquet", "chunk_size": 50_000},
        "image_validation": {"workers": args.workers, "batch_size": 512},
        "review_filters": {"min_text_length": 20, "reject_symbol_only": True},
        "processing_limits": {
            "max_businesses": None,
            "max_reviews": 0,
            "max_photos": None,
        },
        "weak_alignment": {"max_reviews_per_business": 1, "max_images_per_business": 1},
        "clip_denoising": {"enabled": False, "threshold": 0.25},
    }
    parse_summary = run_parse(config)
    alignment_summary = run_alignment(config)
    required = [
        output / "interim" / "photos.parquet",
        output / "processed" / "strong_image_caption_pairs.parquet",
        output / "processed" / "image_business_attribute_pairs.parquet",
    ]
    if any(not path.is_file() for path in required):
        raise Week8YelpRebuildError("required Week 8 source table was not generated")
    manifest = {
        "schema_version": "week8_yelp_source_rebuild_v1",
        "source": "Yelp Open Dataset official archives",
        "source_urls": {
            "json": "https://business.yelp.com/external-assets/files/Yelp-JSON.zip",
            "photos": "https://business.yelp.com/external-assets/files/Yelp-Photos.zip",
        },
        "archives": {
            str(json_zip): {"size": json_zip.stat().st_size, "sha256": sha256_file(json_zip)},
            str(photos_zip): {"size": photos_zip.stat().st_size, "sha256": sha256_file(photos_zip)},
        },
        "extracted_members": extracted,
        "existing_image_root": str(image_root),
        "existing_image_count": sum(1 for path in image_root.iterdir() if path.is_file()),
        "parse_summary": parse_summary,
        "alignment_summary": alignment_summary,
        "required_tables": {
            path.relative_to(output).as_posix(): {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in required
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--json-zip", type=Path, required=True)
    value.add_argument("--photos-zip", type=Path, required=True)
    value.add_argument("--existing-image-root", type=Path, required=True)
    value.add_argument("--output-yelp-root", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--json-zip-size", type=int, default=4_345_335_132)
    value.add_argument("--photos-zip-size", type=int, default=7_447_210_067)
    value.add_argument("--workers", type=int, default=16)
    return value


def main() -> int:
    result = rebuild(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

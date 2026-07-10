import argparse
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any


JSON_DATASET_FILES = {
    "yelp_academic_dataset_business.json",
    "yelp_academic_dataset_checkin.json",
    "yelp_academic_dataset_review.json",
    "yelp_academic_dataset_tip.json",
    "yelp_academic_dataset_user.json",
}
PHOTO_METADATA_FILES = {"photos.json", "photo.json", "yelp_academic_dataset_photo.json"}


def extract_yelp_archives(
    json_zip_path: Path,
    raw_dir: Path,
    photos_zip_path: Path | None = None,
    include_photo_files: bool = False,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: list[dict[str, Any]] = []
    documentation_files: list[dict[str, Any]] = []
    documentation_files.extend(extract_documentation_from_zip(Path(json_zip_path), raw_dir))
    extracted_files.extend(
        extract_selected_members_from_zip_tar(
            zip_path=Path(json_zip_path),
            raw_dir=raw_dir,
            wanted_names=JSON_DATASET_FILES,
            include_photo_files=False,
        )
    )
    if photos_zip_path:
        documentation_files.extend(extract_documentation_from_zip(Path(photos_zip_path), raw_dir))
        extracted_files.extend(
            extract_selected_members_from_zip_tar(
                zip_path=Path(photos_zip_path),
                raw_dir=raw_dir,
                wanted_names=PHOTO_METADATA_FILES,
                include_photo_files=include_photo_files,
            )
        )

    manifest = {
        "source": "Yelp Open Dataset",
        "raw_dir": str(raw_dir),
        "json_zip_path": str(json_zip_path),
        "photos_zip_path": str(photos_zip_path) if photos_zip_path else None,
        "include_photo_files": include_photo_files,
        "extracted_files": extracted_files,
        "documentation_files": documentation_files,
    }
    (raw_dir / "extract_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def extract_documentation_from_zip(zip_path: Path, raw_dir: Path) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    docs_dir = raw_dir / "docs"
    with zipfile.ZipFile(zip_path) as zip_archive:
        for member_name in zip_archive.namelist():
            normalized = normalized_member_name(member_name)
            if is_macos_resource(normalized) or normalized.endswith("/"):
                continue
            filename = Path(normalized).name.lower()
            if "documentation" not in filename and "tos" not in filename:
                continue
            output_name = Path(normalized).name
            target_path = safe_output_path(raw_dir, f"docs/{output_name}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_archive.open(member_name) as source, target_path.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted.append(
                {
                    "archive": str(zip_path),
                    "zip_member": member_name,
                    "output_name": f"docs/{output_name}",
                    "output_path": str(target_path),
                    "size": target_path.stat().st_size,
                }
            )
    return extracted


def extract_yelp_photo_files(
    photos_zip_path: Path,
    raw_dir: Path,
    photo_ids: set[str],
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted_names = {f"photos/{photo_id}.jpg" for photo_id in photo_ids}
    extracted_files = extract_selected_members_from_zip_tar(
        zip_path=Path(photos_zip_path),
        raw_dir=raw_dir,
        wanted_names=wanted_names,
        include_photo_files=False,
    )
    manifest = {
        "source": "Yelp Open Dataset",
        "raw_dir": str(raw_dir),
        "photos_zip_path": str(photos_zip_path),
        "requested_photo_count": len(photo_ids),
        "extracted_photo_count": len(extracted_files),
        "extracted_files": extracted_files,
    }
    (raw_dir / "extract_photo_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_photo_ids_from_multimodal_items(path: Path) -> set[str]:
    photo_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            photo_id = record.get("photo_id")
            if photo_id:
                photo_ids.add(str(photo_id))
    return photo_ids


def extract_selected_members_from_zip_tar(
    zip_path: Path,
    raw_dir: Path,
    wanted_names: set[str],
    include_photo_files: bool,
) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    remaining = set(wanted_names)
    with zipfile.ZipFile(zip_path) as zip_archive:
        tar_names = [
            name
            for name in zip_archive.namelist()
            if name.endswith(".tar") and not is_macos_resource(name)
        ]
        if not tar_names:
            raise FileNotFoundError(f"No dataset tar found inside {zip_path}")

        for tar_name in tar_names:
            with zip_archive.open(tar_name) as tar_stream:
                with tarfile.open(fileobj=tar_stream, mode="r|*") as tar_archive:
                    for member in tar_archive:
                        member_name = normalized_member_name(member.name)
                        if not member.isfile():
                            continue
                        should_extract = member_name in remaining or (
                            include_photo_files and member_name.startswith("photos/")
                        )
                        if not should_extract:
                            continue

                        target_path = safe_output_path(raw_dir, member_name)
                        extracted_stream = tar_archive.extractfile(member)
                        if extracted_stream is None:
                            continue
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with target_path.open("wb") as output:
                            shutil.copyfileobj(extracted_stream, output, length=1024 * 1024)
                        if member_name in remaining:
                            remaining.remove(member_name)
                        extracted.append(
                            {
                                "archive": str(zip_path),
                                "tar_member": member.name,
                                "output_name": member_name,
                                "output_path": str(target_path),
                                "size": member.size,
                            }
                        )
                        if not include_photo_files and not remaining:
                            break
            if not include_photo_files and not remaining:
                break
    return extracted


def normalized_member_name(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def is_macos_resource(name: str) -> bool:
    normalized = normalized_member_name(name)
    return normalized.startswith("__MACOSX/") or "/._" in normalized or normalized.startswith("._")


def safe_output_path(raw_dir: Path, member_name: str) -> Path:
    target = (raw_dir / member_name).resolve()
    raw_root = raw_dir.resolve()
    if target != raw_root and raw_root not in target.parents:
        raise ValueError(f"Refusing to extract archive member outside raw dir: {member_name}")
    return target


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Yelp Open Dataset zip/tar archives for Week 1 data prep.")
    parser.add_argument("--json-zip", type=Path, default=Path("data/Yelp-JSON.zip"))
    parser.add_argument("--photos-zip", type=Path, default=Path("data/Yelp-Photos.zip"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/yelp/raw"))
    parser.add_argument("--include-photo-files", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    photos_zip_path = args.photos_zip if args.photos_zip.exists() else None
    manifest = extract_yelp_archives(
        json_zip_path=args.json_zip,
        photos_zip_path=photos_zip_path,
        raw_dir=args.raw_dir,
        include_photo_files=args.include_photo_files,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

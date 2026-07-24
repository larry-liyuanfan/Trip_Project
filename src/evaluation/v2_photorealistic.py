"""Replace pending Week 3 v2 after-sales candidates with reviewed photo assets."""

import copy
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.evaluation.annotation_workflow import export_packet
from src.evaluation.manifests import (
    ManifestValidationError,
    build_exclusion_rows,
    load_configured_manifests,
    validate_exclusion_manifest,
    validate_manifest_record,
    validate_release_provenance,
    write_jsonl,
)


TARGET_STRATA = ("hygiene_stain", "facility_damage")
RECIPE_VERSION = "week3_after_sales_photorealistic_v1"
_TRAILING_NUMBER = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class PhotoReplacementPlan:
    """Fully validated replacement bytes and traceability information."""

    manifests: dict[str, list[dict[str, Any]]]
    exclusion_rows: list[dict[str, Any]]
    packet_rows: list[dict[str, Any]]
    mappings: tuple[dict[str, str], ...]
    reserve_paths: tuple[str, ...]


def plan_after_sales_photo_replacement(
    *,
    root: Path,
    config: dict[str, Any],
    image_dir: Path,
) -> PhotoReplacementPlan:
    """Map reviewed photo assets onto pending v2 records without creating labels."""
    project_root = Path(root).resolve()
    photo_root = Path(image_dir).resolve()
    if project_root not in photo_root.parents:
        raise ManifestValidationError("replacement image directory must be inside the project")
    if config.get("dataset_version") != "week3_evaluation_v2":
        raise ManifestValidationError("photo replacement requires week3_evaluation_v2")

    manifests = load_configured_manifests(config, root=project_root)
    all_current = [record for rows in manifests.values() for record in rows]
    validate_exclusion_manifest(
        all_current,
        project_root / config["paths"]["exclusion_manifest"],
    )

    updated = copy.deepcopy(manifests)
    after_sales = updated["after_sales"]
    mappings: list[dict[str, str]] = []
    used_paths: set[Path] = set()
    for stratum in TARGET_STRATA:
        targets = sorted(
            (
                record
                for record in after_sales
                if record["sampling_stratum"] == stratum
                and record["annotation_status"] == "pending"
            ),
            key=lambda record: record["sample_id"],
        )
        prefix = "hygiene_" if stratum == "hygiene_stain" else "facility_"
        candidates = sorted(
            (
                path
                for path in photo_root.glob(f"{prefix}*.png")
                if path.is_file()
            ),
            key=_natural_photo_key,
        )
        if len(candidates) < len(targets):
            raise ManifestValidationError(
                f"{stratum} requires {len(targets)} photos, found {len(candidates)}"
            )
        for ordinal, (record, photo_path) in enumerate(
            zip(targets, candidates[: len(targets)], strict=True),
            start=1,
        ):
            sha256, perceptual_hash = fingerprint_photo(photo_path)
            relative_path = photo_path.relative_to(project_root).as_posix()
            source_token = f"{stratum}:{ordinal:04d}"
            old_path = record["input"]["images"][0]["path"]
            record["image_sha256"] = sha256
            record["input"]["images"][0] = {
                "path": relative_path,
                "sha256": sha256,
                "perceptual_hash": perceptual_hash,
            }
            record["source_id"] = f"synthetic:photorealistic:{source_token}"
            record["source_type"] = "business_synthetic"
            record["source_license"] = "project_business_synthetic"
            record["provenance"] = {
                "group_id": f"photorealistic-event:{source_token}",
                "source_uri": f"synthetic://week3/photorealistic/{source_token}",
                "source_version": RECIPE_VERSION,
                "synthetic_recipe_version": RECIPE_VERSION,
                "constraint_template_id": None,
                "pii_review_status": "not_applicable",
            }
            record["notes"] = _append_note(
                record.get("notes"),
                "Replaced with a visually reviewed photorealistic candidate; "
                "human annotation remains required.",
            )
            validate_manifest_record(record, root=project_root)
            used_paths.add(photo_path)
            mappings.append(
                {
                    "sample_id": record["sample_id"],
                    "stratum": stratum,
                    "old_path": old_path,
                    "new_path": relative_path,
                    "sha256": sha256,
                    "perceptual_hash": perceptual_hash,
                }
            )

    all_updated = [record for rows in updated.values() for record in rows]
    exclusion_rows = build_exclusion_rows(all_updated)
    validate_release_provenance(all_updated)
    packet_rows = export_packet(
        updated["after_sales"],
        scenario="after_sales",
        stage="annotation",
        include_suggestions=True,
    )
    reserve_paths = tuple(
        path.relative_to(project_root).as_posix()
        for path in sorted(photo_root.glob("*.png"), key=_natural_photo_key)
        if path not in used_paths
    )
    return PhotoReplacementPlan(
        manifests=updated,
        exclusion_rows=exclusion_rows,
        packet_rows=packet_rows,
        mappings=tuple(mappings),
        reserve_paths=reserve_paths,
    )


def apply_after_sales_photo_replacement(
    plan: PhotoReplacementPlan,
    *,
    root: Path,
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Atomically replace local data artifacts and retain a full backup."""
    project_root = Path(root).resolve()
    if not run_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ManifestValidationError("run_id contains unsupported characters")
    manifest_relative = config["scenarios"]["after_sales"]["manifest_path"]
    exclusion_relative = config["paths"]["exclusion_manifest"]
    packet_relative = (
        Path(config["paths"]["codings_dir"])
        / "week3_v2"
        / "after_sales_annotation_packet.jsonl"
    ).as_posix()
    audit_relative = (
        Path(config["paths"]["sampling_logs_dir"])
        / f"after_sales_photorealistic_{run_id}.json"
    ).as_posix()
    backup_root = (
        project_root / "data" / "eval" / "backups" / f"photorealistic-{run_id}"
    )
    if backup_root.exists():
        raise ManifestValidationError(f"backup already exists: {backup_root}")

    rows_by_path = {
        manifest_relative: plan.manifests["after_sales"],
        exclusion_relative: plan.exclusion_rows,
        packet_relative: plan.packet_rows,
    }
    originals = {
        relative: project_root / relative
        for relative in rows_by_path
    }
    for relative, path in originals.items():
        if not path.exists():
            raise ManifestValidationError(f"required live artifact is missing: {relative}")

    backup_root.mkdir(parents=True, exist_ok=False)
    try:
        for relative, path in originals.items():
            backup_path = backup_root / relative
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)
        for relative, rows in rows_by_path.items():
            live_path = project_root / relative
            temporary = live_path.with_name(live_path.name + f".{run_id}.tmp")
            if temporary.exists():
                raise ManifestValidationError(f"temporary output already exists: {temporary}")
            write_jsonl(temporary, rows)
            os.replace(temporary, live_path)

        all_records = [
            record
            for records in plan.manifests.values()
            for record in records
        ]
        validate_exclusion_manifest(
            all_records,
            project_root / exclusion_relative,
        )
        audit = {
            "status": "applied",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "recipe_version": RECIPE_VERSION,
            "replacement_count": len(plan.mappings),
            "reserve_paths": list(plan.reserve_paths),
            "mappings": list(plan.mappings),
            "artifact_sha256": {
                relative: hashlib.sha256((project_root / relative).read_bytes()).hexdigest()
                for relative in rows_by_path
            },
            "backup_path": backup_root.relative_to(project_root).as_posix(),
        }
        audit_path = project_root / audit_relative
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if audit_path.exists():
            raise ManifestValidationError(f"audit output already exists: {audit_path}")
        temporary_audit = audit_path.with_suffix(audit_path.suffix + ".tmp")
        temporary_audit.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_audit, audit_path)
        return audit
    except Exception:
        for relative in rows_by_path:
            backup_path = backup_root / relative
            if backup_path.exists():
                live_path = project_root / relative
                live_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_path, live_path)
        raise


def fingerprint_photo(path: Path) -> tuple[str, str]:
    """Return byte SHA-256 and a stable 64-bit difference hash."""
    photo_path = Path(path)
    payload = photo_path.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    try:
        with Image.open(photo_path) as image:
            image.load()
            if image.format != "PNG" or image.width < 640 or image.height < 640:
                raise ManifestValidationError(
                    f"replacement photo must be a readable PNG of at least 640px: {photo_path}"
                )
            grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(grayscale.get_flattened_data())
    except (OSError, UnidentifiedImageError) as exc:
        raise ManifestValidationError(f"replacement photo is unreadable: {photo_path}") from exc
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return sha256, f"{bits:016x}"


def _natural_photo_key(path: Path) -> tuple[int, str]:
    match = _TRAILING_NUMBER.search(path.stem)
    return (int(match.group(1)) if match else 10**9, path.name)


def _append_note(existing: Any, addition: str) -> str:
    prefix = existing.strip() if isinstance(existing, str) else ""
    return " ".join(part for part in (prefix, addition) if part)

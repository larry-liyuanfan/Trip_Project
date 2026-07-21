"""Safely plan and apply a targeted refresh of Week 3 synthetic evidence."""

import copy
import hashlib
import io
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError

from src.evaluation.annotation_workflow import export_packet
from src.evaluation.candidates import (
    CandidateDeduplicator,
    render_synthetic_evidence_image,
)
from src.evaluation.manifests import (
    ManifestValidationError,
    build_exclusion_rows,
    load_configured_manifests,
    read_jsonl_objects,
    validate_exclusion_manifest,
    write_jsonl,
)


SYNTHETIC_SOURCE_PATTERN = re.compile(
    r"^synthetic:(attraction_closure|transport_delay):(\d{4})$"
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TARGET_STRATA = ("attraction_closure", "transport_delay")


@dataclass(frozen=True)
class PlannedImage:
    sample_id: str
    relative_path: str
    old_sha256: str
    new_sha256: str
    new_perceptual_hash: str
    png_bytes: bytes


@dataclass(frozen=True)
class SyntheticEvidenceRefreshPlan:
    run_id: str
    recipe_version: str
    images: tuple[PlannedImage, ...]
    manifests: dict[str, list[dict[str, Any]]]
    exclusion_rows: list[dict[str, Any]]
    packet_rows: dict[str, list[dict[str, Any]]]
    live_artifact_hashes: dict[str, str]


def plan_synthetic_evidence_refresh(
    *,
    root: Path,
    config: dict[str, Any],
    run_id: str,
) -> SyntheticEvidenceRefreshPlan:
    """Validate current artifacts and return all v2 bytes without writing."""
    project_root = Path(root).resolve()
    _validate_run_id(run_id)
    recipe_version = _recipe_version(config)
    manifests = load_configured_manifests(config, root=project_root)
    after_sales = manifests.get("after_sales")
    if not isinstance(after_sales, list):
        raise ManifestValidationError("configured manifests require after_sales")

    targets = [
        record
        for record in after_sales
        if record.get("source_type") == "business_synthetic"
    ]
    _validate_targets(targets, config)
    target_ids = {record["sample_id"] for record in targets}
    _validate_no_target_drafts(project_root, config, target_ids)
    _validate_packets_are_pristine(project_root, config, after_sales)

    all_current = [record for records in manifests.values() for record in records]
    validate_exclusion_manifest(
        all_current,
        project_root / config["paths"]["exclusion_manifest"],
    )

    planned_images: list[PlannedImage] = []
    replacements: dict[str, dict[str, str]] = {}
    for record in targets:
        match = SYNTHETIC_SOURCE_PATTERN.fullmatch(record["source_id"])
        if match is None:
            raise ManifestValidationError(
                f"synthetic source_id pattern is invalid: {record['source_id']}"
            )
        issue_type, index_text = match.groups()
        image, _ = render_synthetic_evidence_image(
            issue_type=issue_type,
            index=int(index_text),
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        png_bytes = buffer.getvalue()
        new_sha256, new_perceptual_hash = _fingerprint_png_bytes(png_bytes)
        image_row = record["input"]["images"][0]
        planned_images.append(
            PlannedImage(
                sample_id=record["sample_id"],
                relative_path=image_row["path"],
                old_sha256=image_row["sha256"],
                new_sha256=new_sha256,
                new_perceptual_hash=new_perceptual_hash,
                png_bytes=png_bytes,
            )
        )
        replacements[record["sample_id"]] = {
            "sha256": new_sha256,
            "perceptual_hash": new_perceptual_hash,
        }

    updated_manifests = copy.deepcopy(manifests)
    for record in updated_manifests["after_sales"]:
        replacement = replacements.get(record["sample_id"])
        if replacement is None:
            continue
        record["image_sha256"] = replacement["sha256"]
        record["input"]["images"][0]["sha256"] = replacement["sha256"]
        record["input"]["images"][0]["perceptual_hash"] = replacement[
            "perceptual_hash"
        ]
        record["provenance"]["source_version"] = recipe_version
        record["provenance"]["synthetic_recipe_version"] = recipe_version

    all_updated = [
        record for records in updated_manifests.values() for record in records
    ]
    exclusion_rows = build_exclusion_rows(all_updated)
    _validate_perceptual_independence(all_updated)

    base_packet = export_packet(
        updated_manifests["after_sales"],
        scenario="after_sales",
        stage="annotation",
    )
    suggested_packet = export_packet(
        updated_manifests["after_sales"],
        scenario="after_sales",
        stage="annotation",
        include_suggestions=True,
    )
    codings_dir = Path(config["paths"]["codings_dir"])
    packet_rows = {
        (codings_dir / "after_sales_annotation_packet.jsonl").as_posix(): base_packet,
        (
            codings_dir / "after_sales_annotation_suggested.jsonl"
        ).as_posix(): suggested_packet,
    }
    affected_paths = [image.relative_path for image in planned_images]
    affected_paths.extend(
        [
            config["scenarios"]["after_sales"]["manifest_path"],
            config["paths"]["exclusion_manifest"],
            *packet_rows,
        ]
    )
    live_artifact_hashes = {
        relative_path: hashlib.sha256(
            _safe_root_path(project_root, relative_path).read_bytes()
        ).hexdigest()
        for relative_path in affected_paths
    }
    return SyntheticEvidenceRefreshPlan(
        run_id=run_id,
        recipe_version=recipe_version,
        images=tuple(planned_images),
        manifests=updated_manifests,
        exclusion_rows=exclusion_rows,
        packet_rows=packet_rows,
        live_artifact_hashes=live_artifact_hashes,
    )


def execute_synthetic_evidence_refresh(
    plan: SyntheticEvidenceRefreshPlan,
    *,
    root: Path,
    config: dict[str, Any],
    replace: Callable[[Path, Path], None] = os.replace,
) -> dict[str, Any]:
    """Apply one validated plan or restore every replaced live artifact."""
    project_root = Path(root).resolve()
    _validate_run_id(plan.run_id)
    if plan.recipe_version != _recipe_version(config):
        raise ManifestValidationError("refresh plan recipe version no longer matches config")
    for relative_path, expected_hash in plan.live_artifact_hashes.items():
        live_path = _safe_root_path(project_root, relative_path)
        actual_hash = hashlib.sha256(live_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ManifestValidationError(
                f"live artifact changed after refresh planning: {relative_path}"
            )

    staging_relative = Path("data/eval/.staging") / plan.run_id
    backup_relative = (
        Path("data/eval/backups") / f"synthetic-evidence-v1-{plan.run_id}"
    )
    audit_relative = Path("data/eval/logs/after_sales_synthetic_refresh_v2.json")
    staging_root = _safe_root_path(project_root, staging_relative.as_posix())
    backup_root = _safe_root_path(project_root, backup_relative.as_posix())
    audit_path = _safe_root_path(project_root, audit_relative.as_posix())
    for path, label in (
        (staging_root, "staging"),
        (backup_root, "backup"),
        (audit_path, "audit"),
    ):
        if path.exists():
            raise ManifestValidationError(
                f"synthetic refresh {label} path already exists: {path}"
            )

    staged_rows = {
        config["scenarios"]["after_sales"]["manifest_path"]: plan.manifests[
            "after_sales"
        ],
        config["paths"]["exclusion_manifest"]: plan.exclusion_rows,
        **plan.packet_rows,
    }
    replaced_paths: list[str] = []
    rollback_failures: list[str] = []
    try:
        for planned_image in plan.images:
            staged_path = _safe_root_path(
                staging_root,
                planned_image.relative_path,
            )
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_bytes(planned_image.png_bytes)
            staged_sha256, staged_perceptual_hash = _fingerprint_png_bytes(
                staged_path.read_bytes()
            )
            if (
                staged_sha256 != planned_image.new_sha256
                or staged_perceptual_hash != planned_image.new_perceptual_hash
            ):
                raise ManifestValidationError(
                    f"staged image fingerprint mismatch: {planned_image.relative_path}"
                )
        for relative_path, rows in staged_rows.items():
            staged_path = _safe_root_path(staging_root, relative_path)
            write_jsonl(staged_path, rows)

        backup_root.mkdir(parents=True, exist_ok=False)
        for relative_path in plan.live_artifact_hashes:
            live_path = _safe_root_path(project_root, relative_path)
            backup_path = _safe_root_path(backup_root, relative_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(live_path, backup_path)
        _write_json_atomically(
            backup_root / "backup_manifest.json",
            {
                "run_id": plan.run_id,
                "recipe_version": plan.recipe_version,
                "artifact_hashes": plan.live_artifact_hashes,
            },
        )

        replacement_order = [image.relative_path for image in plan.images]
        replacement_order.extend(staged_rows)
        for relative_path in replacement_order:
            staged_path = _safe_root_path(staging_root, relative_path)
            live_path = _safe_root_path(project_root, relative_path)
            live_path.parent.mkdir(parents=True, exist_ok=True)
            replace(staged_path, live_path)
            replaced_paths.append(relative_path)

        _validate_applied_refresh(project_root, config, plan)
        audit = {
            "status": "completed",
            "run_id": plan.run_id,
            "recipe_version": plan.recipe_version,
            "target_count": len(plan.images),
            "backup_path": backup_relative.as_posix(),
            "images": [
                {
                    "sample_id": image.sample_id,
                    "path": image.relative_path,
                    "old_sha256": image.old_sha256,
                    "new_sha256": image.new_sha256,
                    "new_perceptual_hash": image.new_perceptual_hash,
                }
                for image in plan.images
            ],
        }
        _write_json_atomically(audit_path, audit)
    except Exception as exc:
        for relative_path in reversed(replaced_paths):
            try:
                backup_path = _safe_root_path(backup_root, relative_path)
                rollback_path = _safe_root_path(
                    staging_root,
                    (Path("rollback") / relative_path).as_posix(),
                )
                rollback_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_path, rollback_path)
                os.replace(rollback_path, _safe_root_path(project_root, relative_path))
            except Exception as rollback_exc:
                rollback_failures.append(f"{relative_path}: {rollback_exc}")
        if audit_path.exists():
            audit_path.unlink()
        if staging_root.exists():
            shutil.rmtree(staging_root)
        if rollback_failures:
            raise ManifestValidationError(
                "synthetic evidence refresh failed and rollback failed; "
                f"backup retained at {backup_root}; failures={rollback_failures}"
            ) from exc
        raise ManifestValidationError(
            f"synthetic evidence refresh failed and rolled back: {exc}"
        ) from exc

    if staging_root.exists():
        shutil.rmtree(staging_root)
    return {
        "status": "completed",
        "run_id": plan.run_id,
        "recipe_version": plan.recipe_version,
        "target_count": len(plan.images),
        "backup_path": backup_relative.as_posix(),
        "audit_path": audit_relative.as_posix(),
    }


def _validate_applied_refresh(
    root: Path,
    config: dict[str, Any],
    plan: SyntheticEvidenceRefreshPlan,
) -> None:
    manifests = load_configured_manifests(config, root=root)
    all_records = [record for records in manifests.values() for record in records]
    actual_exclusions = validate_exclusion_manifest(
        all_records,
        root / config["paths"]["exclusion_manifest"],
    )
    if actual_exclusions != plan.exclusion_rows:
        raise ManifestValidationError("applied exclusion registry differs from refresh plan")
    for relative_path, expected in plan.packet_rows.items():
        if read_jsonl_objects(root / relative_path) != expected:
            raise ManifestValidationError(
                f"applied annotation packet differs from refresh plan: {relative_path}"
            )


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise ManifestValidationError(f"temporary JSON path already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_root_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestValidationError(
            f"synthetic refresh path must be repository-relative: {relative_path}"
        )
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ManifestValidationError(
            f"synthetic refresh path escapes its root: {relative_path}"
        ) from exc
    return resolved


def _validate_targets(
    targets: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    quotas = config["scenarios"]["after_sales"]["sampling"]["quotas"]
    expected_counts = {stratum: int(quotas[stratum]) for stratum in TARGET_STRATA}
    actual_counts = {
        stratum: sum(record.get("sampling_stratum") == stratum for record in targets)
        for stratum in TARGET_STRATA
    }
    if actual_counts != expected_counts:
        raise ManifestValidationError(
            "synthetic target count does not match configured quotas: "
            f"expected {expected_counts}, got {actual_counts}"
        )
    for record in targets:
        match = SYNTHETIC_SOURCE_PATTERN.fullmatch(str(record.get("source_id", "")))
        if match is None or match.group(1) != record.get("sampling_stratum"):
            raise ManifestValidationError(
                f"synthetic source_id pattern is invalid: {record.get('source_id')}"
            )
        if record.get("review_status") != "pending":
            raise ManifestValidationError(
                f"refusing to refresh non-pending review: {record['sample_id']}"
            )
        if record.get("annotation_status") == "completed":
            raise ManifestValidationError(
                f"refusing to refresh completed annotation: {record['sample_id']}"
            )
        if (
            record.get("annotation_status") != "pending"
            or record.get("annotation") is not None
            or record.get("annotator") is not None
            or record.get("reviewer") is not None
        ):
            raise ManifestValidationError(
                f"refusing to refresh non-pristine human state: {record['sample_id']}"
            )


def _validate_no_target_drafts(
    root: Path,
    config: dict[str, Any],
    target_ids: set[str],
) -> None:
    drafts_dir = root / config["paths"]["codings_dir"] / "ui_drafts"
    if not drafts_dir.exists():
        return
    for draft_path in sorted(drafts_dir.glob("*.jsonl")):
        for row in read_jsonl_objects(draft_path):
            sample_id = row.get("sample_id")
            if sample_id in target_ids:
                raise ManifestValidationError(
                    f"UI draft conflicts with synthetic refresh: {sample_id} in {draft_path}"
                )


def _validate_packets_are_pristine(
    root: Path,
    config: dict[str, Any],
    after_sales: list[dict[str, Any]],
) -> None:
    codings_dir = root / config["paths"]["codings_dir"]
    packet_checks = (
        (
            codings_dir / "after_sales_annotation_packet.jsonl",
            export_packet(
                after_sales,
                scenario="after_sales",
                stage="annotation",
            ),
        ),
        (
            codings_dir / "after_sales_annotation_suggested.jsonl",
            export_packet(
                after_sales,
                scenario="after_sales",
                stage="annotation",
                include_suggestions=True,
            ),
        ),
    )
    for path, expected in packet_checks:
        actual = read_jsonl_objects(path)
        if actual != expected:
            raise ManifestValidationError(
                f"refusing to overwrite manual packet content: {path}"
            )


def _fingerprint_png_bytes(png_bytes: bytes) -> tuple[str, str]:
    sha256 = hashlib.sha256(png_bytes).hexdigest()
    try:
        with Image.open(io.BytesIO(png_bytes)) as image:
            image.load()
            if image.size != (960, 640) or image.mode != "RGB":
                raise ManifestValidationError(
                    "synthetic v2 image must be a 960x640 RGB PNG"
                )
            grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(grayscale.get_flattened_data())
    except (OSError, UnidentifiedImageError) as exc:
        raise ManifestValidationError("synthetic v2 image is unreadable") from exc
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return sha256, f"{bits:016x}"


def _validate_perceptual_independence(records: list[dict[str, Any]]) -> None:
    deduplicator = CandidateDeduplicator(max_perceptual_distance=4)
    for record in records:
        images = record["input"]["images"]
        perceptual_hashes = [image.get("perceptual_hash") for image in images]
        if any(not isinstance(value, str) for value in perceptual_hashes):
            raise ManifestValidationError(
                f"missing image perceptual_hash: {record['sample_id']}"
            )
        provenance = record.get("provenance") or {}
        if not deduplicator.accept(
            source_id=record["source_id"],
            group_id=provenance.get("group_id", record["sample_id"]),
            image_hashes=[image["sha256"] for image in images],
            perceptual_hashes=perceptual_hashes,
        ):
            raise ManifestValidationError(
                "refreshed manifests contain an exact or perceptual collision: "
                f"{record['sample_id']}"
            )


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ManifestValidationError(
            "synthetic refresh run_id must contain only letters, digits, dot, underscore, or hyphen"
        )


def _recipe_version(config: dict[str, Any]) -> str:
    candidate_sources = config.get("candidate_sources")
    if not isinstance(candidate_sources, dict):
        raise ManifestValidationError("config requires candidate_sources")
    value = candidate_sources.get("after_sales_synthetic_recipe_version")
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(
            "config requires candidate_sources.after_sales_synthetic_recipe_version"
        )
    return value.strip()

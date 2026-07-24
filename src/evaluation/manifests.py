"""Contracts and isolation helpers for Week 3 evaluation manifests."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "sample_id",
    "scenario",
    "source_type",
    "source_id",
    "source_license",
    "image_sha256",
    "input",
    "split",
    "dataset_version",
    "annotation_status",
    "annotator",
    "review_status",
    "reviewer",
    "file_status",
    "annotation",
    "notes",
}

SCENARIOS = {"image_product_search", "after_sales", "itinerary_planning"}
ANNOTATION_STATUSES = {"pending", "in_progress", "completed"}
REVIEW_STATUSES = {"pending", "validated", "rejected"}
FILE_STATUSES = {"pending", "valid", "missing", "unreadable"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PERCEPTUAL_HASH_PATTERN = re.compile(r"^[0-9a-f]{16}$")
PII_REVIEW_STATUSES = {"not_applicable", "pending", "redacted", "verified"}

ANNOTATION_FIELDS = {
    "image_product_search": {
        "business_category",
        "style_tags",
        "visible_facilities",
        "price_range",
    },
    "after_sales": {
        "issue_type",
        "severity",
        "key_information",
        "ocr_ground_truth",
    },
    "itinerary_planning": {
        "reference_images",
        "text_constraints",
        "style_preferences",
        "hard_constraints",
        "soft_constraints",
        "required_itinerary_elements",
    },
}

BUSINESS_CATEGORIES = {"hotel", "attraction", "restaurant", "unknown"}
PRICE_RANGES = {"budget", "mid_range", "premium", "luxury", "unknown"}
ISSUE_TYPES = {
    "hygiene_stain",
    "facility_damage",
    "attraction_closure",
    "transport_delay",
    "other",
    "unknown",
}
SEVERITIES = {"low", "medium", "high", "critical", "unknown"}


class ManifestValidationError(ValueError):
    """Raised when an evaluation manifest record violates its contract."""


class EvaluationCollisionError(ValueError):
    """Raised when a training candidate overlaps the evaluation registry."""


def validate_manifest_record(
    record: dict[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """Return a validated copy of one evaluation manifest record."""
    if not isinstance(record, dict):
        raise ManifestValidationError("manifest record must be a JSON object")
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ManifestValidationError(f"missing required fields: {sorted(missing)}")

    for field in (
        "sample_id",
        "source_type",
        "source_id",
        "source_license",
        "dataset_version",
    ):
        _require_nonempty_string(record[field], field)

    scenario = record["scenario"]
    if scenario not in SCENARIOS:
        raise ManifestValidationError(f"invalid scenario: {scenario!r}")
    if record["split"] != "evaluation":
        raise ManifestValidationError("split must be 'evaluation'")
    if record["annotation_status"] not in ANNOTATION_STATUSES:
        raise ManifestValidationError("invalid annotation_status")
    if record["review_status"] not in REVIEW_STATUSES:
        raise ManifestValidationError("invalid review_status")
    if record["file_status"] not in FILE_STATUSES:
        raise ManifestValidationError("invalid file_status")

    input_images = _validate_input(scenario, record["input"])
    if "provenance" in record:
        _validate_provenance(record["provenance"])
    image_sha256 = record["image_sha256"]
    if not isinstance(image_sha256, str) or SHA256_PATTERN.fullmatch(image_sha256) is None:
        raise ManifestValidationError("image_sha256 must be 64 lowercase hexadecimal characters")
    if image_sha256 != input_images[0]["sha256"]:
        raise ManifestValidationError("image_sha256 must match the first input image sha256")
    if root is not None:
        _validate_input_image_bytes(input_images, root=Path(root))

    annotator = record["annotator"]
    if annotator is not None:
        _require_nonempty_string(annotator, "annotator")
    if record["notes"] is not None and not isinstance(record["notes"], str):
        raise ManifestValidationError("notes must be a string or null")

    annotation_status = record["annotation_status"]
    if annotation_status == "pending":
        if annotator is not None or record["annotation"] is not None:
            raise ManifestValidationError("pending annotation must have null annotator and annotation")
    elif annotation_status == "in_progress":
        if annotator is None or record["annotation"] is not None:
            raise ManifestValidationError("in_progress annotation requires an annotator and null annotation")
    else:
        if annotator is None:
            raise ManifestValidationError("completed annotation requires an annotator")
        _validate_annotation(scenario, record["annotation"])

    review_status = record["review_status"]
    reviewer = record["reviewer"]
    if review_status != "pending" and annotation_status != "completed":
        raise ManifestValidationError(f"{review_status} review requires a completed annotation")
    if review_status == "pending":
        if reviewer is not None:
            raise ManifestValidationError("pending review must have a null reviewer")
    else:
        _require_nonempty_string(reviewer, "reviewer")
        if reviewer == annotator:
            raise ManifestValidationError("review requires an independent reviewer")
    if review_status == "validated" and record["file_status"] != "valid":
        raise ManifestValidationError("validated review requires a completed annotation and valid file")
    return dict(record)


def load_manifest(path: Path, *, root: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate a UTF-8 JSONL manifest without hiding malformed rows."""
    path = Path(path)
    if not path.exists():
        raise ManifestValidationError(f"manifest does not exist: {path}")

    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ManifestValidationError(
                    f"invalid JSON on line {line_number}: {exc}"
                ) from exc
            try:
                record = validate_manifest_record(payload, root=root)
            except ManifestValidationError as exc:
                raise ManifestValidationError(f"line {line_number}: {exc}") from exc
            sample_id = record["sample_id"]
            if sample_id in sample_ids:
                raise ManifestValidationError(f"duplicate sample_id on line {line_number}: {sample_id}")
            sample_ids.add(sample_id)
            records.append(record)
    return records


def load_configured_manifests(
    config: dict[str, Any],
    *,
    root: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load every configured manifest and enforce its declared scenario boundary."""
    configured: dict[str, list[dict[str, Any]]] = {}
    project_root = Path(root)
    for scenario, settings in config["scenarios"].items():
        manifest_path = settings["manifest_path"]
        records = load_manifest(project_root / manifest_path, root=project_root)
        for record in records:
            if record["scenario"] != scenario:
                raise ManifestValidationError(
                    f"{manifest_path} contains scenario {record['scenario']!r}, "
                    f"expected {scenario!r}"
                )
        configured[scenario] = records
    return configured


def summarize_counts(
    records: list[dict[str, Any]],
    target_count: int,
    tested_sample_ids: set[str] | None = None,
) -> dict[str, int]:
    """Report target, candidate, annotated, validated, and persisted-test counts separately."""
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 0:
        raise ManifestValidationError("target_count must be a non-negative integer")

    validated_records = [validate_manifest_record(record) for record in records]
    sample_ids = [record["sample_id"] for record in validated_records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ManifestValidationError("duplicate sample_id in manifest records")

    annotated = [
        record for record in validated_records if record["annotation_status"] == "completed"
    ]
    validated = [
        record
        for record in annotated
        if is_release_eligible(record)
    ]
    persisted_ids = tested_sample_ids or set()
    tested_count = sum(record["sample_id"] in persisted_ids for record in validated)
    return {
        "target_count": target_count,
        "candidate_count": len(validated_records),
        "annotated_count": len(annotated),
        "validated_count": len(validated),
        "tested_count": tested_count,
    }


def is_release_eligible(record: dict[str, Any]) -> bool:
    """Return whether a structurally valid human-completed record may run."""
    if record.get("annotation_status") != "completed":
        return False
    if record.get("file_status") != "valid" or record.get("review_status") == "rejected":
        return False
    return isinstance(record.get("annotation"), dict)


def build_exclusion_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build stable exclusions for every registered evaluation candidate."""
    validated_records = [validate_manifest_record(record) for record in records]
    sample_ids = [record["sample_id"] for record in validated_records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ManifestValidationError("duplicate sample_id in evaluation registry")
    source_ids: set[str] = set()
    image_hashes: set[str] = set()
    for record in validated_records:
        source_id = record["source_id"]
        if source_id in source_ids:
            raise ManifestValidationError(
                f"duplicate source_id in evaluation registry: {source_id}"
            )
        source_ids.add(source_id)
        for image in record["input"]["images"]:
            image_hash = image["sha256"]
            if image_hash in image_hashes:
                raise ManifestValidationError(
                    f"duplicate image_sha256 in evaluation registry: {image_hash}"
                )
            image_hashes.add(image_hash)
    rows = []
    for record in validated_records:
        for image in record["input"]["images"]:
            row = {
                "sample_id": record["sample_id"],
                "scenario": record["scenario"],
                "source_id": record["source_id"],
                "image_path": image["path"],
                "image_sha256": image["sha256"],
                "dataset_version": record["dataset_version"],
            }
            provenance = record.get("provenance")
            if isinstance(provenance, dict):
                row.update(
                    {
                        "group_id": provenance["group_id"],
                        "source_uri": provenance["source_uri"],
                        "source_version": provenance["source_version"],
                        "synthetic_recipe_version": provenance[
                            "synthetic_recipe_version"
                        ],
                        "constraint_template_id": provenance[
                            "constraint_template_id"
                        ],
                        "pii_review_status": provenance["pii_review_status"],
                    }
                )
            if "perceptual_hash" in image:
                row["image_perceptual_hash"] = image["perceptual_hash"]
            rows.append(row)
    return sorted(rows, key=lambda row: (row["sample_id"], row["image_path"]))


def validate_exclusion_manifest(
    records: list[dict[str, Any]],
    exclusion_path: Path,
) -> list[dict[str, Any]]:
    """Reject cross-manifest conflicts and a missing or stale exclusion registry."""
    expected = build_exclusion_rows(records)
    actual = read_jsonl_objects(exclusion_path)
    if actual != expected:
        raise ManifestValidationError(
            "evaluation exclusion manifest is stale; rebuild it explicitly with "
            "prepare_week3_evaluation.py"
        )
    return actual


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    """Read strict UTF-8 JSONL objects without accepting non-finite constants."""
    path = Path(path)
    if not path.exists():
        raise ManifestValidationError(f"required JSONL file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ManifestValidationError(
                    f"invalid JSON in {path} line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ManifestValidationError(
                    f"{path} line {line_number} must be a JSON object"
                )
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSON objects as UTF-8 JSONL, including an honest empty file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if not isinstance(row, dict):
                raise ManifestValidationError("JSONL rows must be objects")
            try:
                serialized = json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            except ValueError as exc:
                raise ManifestValidationError(
                    f"JSONL row contains a non-finite number: {exc}"
                ) from exc
            handle.write(serialized + "\n")


def reject_evaluation_collisions(
    training_candidates: list[dict[str, Any]],
    exclusion_rows: list[dict[str, Any]],
    *,
    max_perceptual_distance: int = 4,
) -> None:
    """Reject exact, grouped, templated, or perceptually similar evaluation overlap."""
    excluded_sources = {
        row["source_id"] for row in exclusion_rows if isinstance(row.get("source_id"), str)
    }
    excluded_hashes = {
        row["image_sha256"]
        for row in exclusion_rows
        if isinstance(row.get("image_sha256"), str)
    }
    excluded_groups = {
        row["group_id"]
        for row in exclusion_rows
        if isinstance(row.get("group_id"), str)
    }
    excluded_templates = {
        row["constraint_template_id"]
        for row in exclusion_rows
        if isinstance(row.get("constraint_template_id"), str)
    }
    excluded_perceptual_hashes = {
        row["image_perceptual_hash"]
        for row in exclusion_rows
        if isinstance(row.get("image_perceptual_hash"), str)
    }
    _validate_perceptual_distance(max_perceptual_distance)
    collisions: list[str] = []
    for index, candidate in enumerate(training_candidates):
        if not isinstance(candidate, dict):
            raise ManifestValidationError(f"training candidate {index} must be a JSON object")
        source_id = candidate.get("source_id")
        _require_nonempty_string(source_id, f"training candidate {index} source_id")
        image_sha256 = candidate.get("image_sha256")
        if image_sha256 is not None and (
            not isinstance(image_sha256, str)
            or SHA256_PATTERN.fullmatch(image_sha256) is None
        ):
            raise ManifestValidationError(
                f"training candidate {index} image_sha256 must be null or a valid SHA-256"
            )
        if source_id in excluded_sources:
            collisions.append(f"candidate {index} source_id={source_id}")
        if image_sha256 is not None and image_sha256 in excluded_hashes:
            collisions.append(f"candidate {index} image_sha256={image_sha256}")
        group_id = candidate.get("group_id")
        if group_id is not None:
            _require_nonempty_string(group_id, f"training candidate {index} group_id")
            if group_id in excluded_groups:
                collisions.append(f"candidate {index} group_id={group_id}")
        constraint_template_id = candidate.get("constraint_template_id")
        if constraint_template_id is not None:
            _require_nonempty_string(
                constraint_template_id,
                f"training candidate {index} constraint_template_id",
            )
            if constraint_template_id in excluded_templates:
                collisions.append(
                    f"candidate {index} constraint_template_id={constraint_template_id}"
                )
        perceptual_hash = candidate.get("image_perceptual_hash")
        if perceptual_hash is not None:
            _validate_perceptual_hash(
                perceptual_hash,
                f"training candidate {index} image_perceptual_hash",
            )
            if any(
                perceptual_hash_distance(perceptual_hash, excluded)
                <= max_perceptual_distance
                for excluded in excluded_perceptual_hashes
            ):
                collisions.append(
                    f"candidate {index} image_perceptual_hash={perceptual_hash}"
                )

    if collisions:
        raise EvaluationCollisionError(
            "training candidates overlap evaluation exclusions: " + "; ".join(collisions)
        )


def validate_release_provenance(
    records: list[dict[str, Any]],
    *,
    max_perceptual_distance: int = 4,
) -> list[dict[str, Any]]:
    """Apply the stronger provenance and near-duplicate gate required for live runs."""
    _validate_perceptual_distance(max_perceptual_distance)
    validated: list[dict[str, Any]] = []
    group_ids: set[str] = set()
    image_fingerprints: list[tuple[str, str]] = []
    for record in records:
        checked = validate_manifest_record(record)
        provenance = checked.get("provenance")
        if not isinstance(provenance, dict):
            raise ManifestValidationError(
                f"release provenance is required for {checked['sample_id']}"
            )
        _validate_provenance(provenance)
        if checked.get("review_status") == "rejected":
            raise ManifestValidationError(
                f"rejected record cannot enter a release: {checked['sample_id']}"
            )
        group_id = provenance["group_id"]
        if group_id in group_ids:
            raise ManifestValidationError(f"duplicate group_id in release set: {group_id}")
        group_ids.add(group_id)
        for image in checked["input"]["images"]:
            fingerprint = image.get("perceptual_hash")
            _validate_perceptual_hash(
                fingerprint,
                f"{checked['sample_id']} input image perceptual_hash",
            )
            for previous_sample, previous in image_fingerprints:
                if (
                    perceptual_hash_distance(fingerprint, previous)
                    <= max_perceptual_distance
                ):
                    raise ManifestValidationError(
                        "near-duplicate image fingerprints in release set: "
                        f"{previous_sample} and {checked['sample_id']}"
                    )
            image_fingerprints.append((checked["sample_id"], fingerprint))
        validated.append(checked)
    return validated


def perceptual_hash_distance(first: str, second: str) -> int:
    """Return Hamming distance between two 64-bit hexadecimal fingerprints."""
    _validate_perceptual_hash(first, "first perceptual hash")
    _validate_perceptual_hash(second, "second perceptual hash")
    return (int(first, 16) ^ int(second, 16)).bit_count()


def _require_nonempty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field} must be a non-empty string")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _validate_input(scenario: str, input_payload: Any) -> list[dict[str, str]]:
    if not isinstance(input_payload, dict):
        raise ManifestValidationError("input must be a JSON object")
    expected_fields = {"images", "text_constraints"}
    missing = expected_fields - input_payload.keys()
    extra = input_payload.keys() - expected_fields
    if missing:
        raise ManifestValidationError(f"input missing required fields: {sorted(missing)}")
    if extra:
        raise ManifestValidationError(f"input contains unsupported fields: {sorted(extra)}")

    images = input_payload["images"]
    if not isinstance(images, list) or not images:
        raise ManifestValidationError("input.images must be a non-empty array")
    if scenario in {"image_product_search", "after_sales"} and len(images) != 1:
        raise ManifestValidationError(f"{scenario} requires exactly one input image")

    image_paths: set[str] = set()
    image_hashes: set[str] = set()
    validated_images: list[dict[str, str]] = []
    for index, image in enumerate(images):
        if not isinstance(image, dict) or not {"path", "sha256"}.issubset(image):
            raise ManifestValidationError(
                f"input.images[{index}] must contain path and sha256"
            )
        extra_image_fields = set(image) - {"path", "sha256", "perceptual_hash"}
        if extra_image_fields:
            raise ManifestValidationError(
                f"input.images[{index}] contains unsupported fields: "
                f"{sorted(extra_image_fields)}"
            )
        image_path = image["path"]
        _require_repository_relative_path(image_path, f"input.images[{index}].path")
        image_hash = image["sha256"]
        if not isinstance(image_hash, str) or SHA256_PATTERN.fullmatch(image_hash) is None:
            raise ManifestValidationError(
                f"input.images[{index}].sha256 must be 64 lowercase hexadecimal characters"
            )
        if image_path in image_paths:
            raise ManifestValidationError(f"duplicate input image path: {image_path}")
        if image_hash in image_hashes:
            raise ManifestValidationError(f"duplicate input image sha256: {image_hash}")
        image_paths.add(image_path)
        image_hashes.add(image_hash)
        validated_image = {"path": image_path, "sha256": image_hash}
        if "perceptual_hash" in image:
            _validate_perceptual_hash(
                image["perceptual_hash"],
                f"input.images[{index}].perceptual_hash",
            )
            validated_image["perceptual_hash"] = image["perceptual_hash"]
        validated_images.append(validated_image)

    text_constraints = input_payload["text_constraints"]
    if scenario == "itinerary_planning":
        _require_nonempty_string(text_constraints, "input.text_constraints")
    elif text_constraints is not None:
        raise ManifestValidationError(
            f"input.text_constraints must be null for {scenario}"
        )
    return validated_images


def _require_repository_relative_path(value: Any, field: str) -> None:
    _require_nonempty_string(value, field)
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ManifestValidationError(f"{field} must be repository-relative")


def calculate_image_sha256(root: Path, repository_relative_path: str) -> str:
    """Calculate SHA-256 from a repository-contained image file."""
    _require_repository_relative_path(repository_relative_path, "image path")
    resolved_root = Path(root).resolve()
    resolved_path = (resolved_root / repository_relative_path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ManifestValidationError(
            f"image path escapes repository root: {repository_relative_path}"
        ) from exc
    if not resolved_path.is_file():
        raise ManifestValidationError(f"input image does not exist: {repository_relative_path}")

    digest = hashlib.sha256()
    with resolved_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_input_image_bytes(images: list[dict[str, str]], *, root: Path) -> None:
    for image in images:
        actual_hash = calculate_image_sha256(root, image["path"])
        if actual_hash != image["sha256"]:
            raise ManifestValidationError(
                f"input image sha256 does not match image bytes: {image['path']}"
            )


def _require_string_list(value: Any, field: str, *, nonempty: bool = False) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ManifestValidationError(f"{field} must be an array of non-empty strings")
    if nonempty and not value:
        raise ManifestValidationError(f"{field} must not be empty")


def _validate_provenance(provenance: Any) -> None:
    if not isinstance(provenance, dict):
        raise ManifestValidationError("provenance must be a JSON object")
    expected = {
        "source_uri",
        "source_version",
        "group_id",
        "synthetic_recipe_version",
        "constraint_template_id",
        "pii_review_status",
    }
    if set(provenance) != expected:
        raise ManifestValidationError(
            "provenance must contain exactly " + ", ".join(sorted(expected))
        )
    for field in ("source_version", "group_id"):
        _require_nonempty_string(provenance[field], f"provenance.{field}")
    for field in ("source_uri", "synthetic_recipe_version", "constraint_template_id"):
        value = provenance[field]
        if value is not None:
            _require_nonempty_string(value, f"provenance.{field}")
    if provenance["pii_review_status"] not in PII_REVIEW_STATUSES:
        raise ManifestValidationError("invalid provenance.pii_review_status")


def _validate_perceptual_hash(value: Any, field: str) -> None:
    if not isinstance(value, str) or PERCEPTUAL_HASH_PATTERN.fullmatch(value) is None:
        raise ManifestValidationError(f"{field} must be 16 lowercase hexadecimal characters")


def _validate_perceptual_distance(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 64:
        raise ManifestValidationError("max_perceptual_distance must be between 0 and 64")


def _validate_annotation(scenario: str, annotation: Any) -> None:
    if not isinstance(annotation, dict):
        raise ManifestValidationError("completed annotation must be a JSON object")
    expected = ANNOTATION_FIELDS[scenario]
    missing = expected - annotation.keys()
    extra = annotation.keys() - expected
    if missing:
        raise ManifestValidationError(f"annotation missing required fields: {sorted(missing)}")
    if extra:
        raise ManifestValidationError(f"annotation contains unsupported fields: {sorted(extra)}")

    if scenario == "image_product_search":
        if annotation["business_category"] not in BUSINESS_CATEGORIES:
            raise ManifestValidationError("invalid business_category")
        if annotation["price_range"] not in PRICE_RANGES:
            raise ManifestValidationError("invalid price_range")
        _require_string_list(annotation["style_tags"], "style_tags")
        _require_string_list(annotation["visible_facilities"], "visible_facilities")
        return

    if scenario == "after_sales":
        if annotation["issue_type"] not in ISSUE_TYPES:
            raise ManifestValidationError("invalid issue_type")
        if annotation["severity"] not in SEVERITIES:
            raise ManifestValidationError("invalid severity")
        _require_string_list(annotation["key_information"], "key_information")
        ocr_ground_truth = annotation["ocr_ground_truth"]
        if ocr_ground_truth is not None:
            _require_string_list(ocr_ground_truth, "ocr_ground_truth")
        return

    _require_string_list(annotation["reference_images"], "reference_images", nonempty=True)
    for field in (
        "text_constraints",
        "style_preferences",
        "hard_constraints",
        "soft_constraints",
        "required_itinerary_elements",
    ):
        _require_string_list(annotation[field], field)

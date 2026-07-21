"""Transactional transformations for human Week 3 annotations."""

import copy
from typing import Any

from src.evaluation.annotation_suggestions import build_deterministic_suggestion
from src.evaluation.manifests import ManifestValidationError, validate_manifest_record


ANNOTATION_FIELDS = {
    "image_product_search": [
        "business_category",
        "style_tags",
        "visible_facilities",
        "price_range",
    ],
    "after_sales": ["issue_type", "severity", "key_information", "ocr_ground_truth"],
    "itinerary_planning": [
        "reference_images",
        "text_constraints",
        "style_preferences",
        "hard_constraints",
        "soft_constraints",
        "required_itinerary_elements",
    ],
}


def export_packet(
    records: list[dict[str, Any]],
    *,
    scenario: str,
    stage: str,
    include_suggestions: bool = False,
) -> list[dict[str, Any]]:
    """Build one immutable human annotation packet."""
    if stage != "annotation":
        raise ManifestValidationError("the Week 3 workflow is annotation-only")
    rows: list[dict[str, Any]] = []
    for record in records:
        if stage == "annotation":
            if record["annotation_status"] != "pending":
                continue
            row = {
                "sample_id": record["sample_id"],
                "annotator": None,
                "annotation": {
                    field: None for field in ANNOTATION_FIELDS[scenario]
                },
                "context": {
                    "input": record["input"],
                    "sampling_stratum": record.get("sampling_stratum"),
                    "source_type": record["source_type"],
                    "notes": record["notes"],
                },
            }
            if include_suggestions:
                row["context"]["deterministic_suggestion"] = (
                    build_deterministic_suggestion(record)
                )
            rows.append(row)
    return rows


def _index_submissions(
    records: list[dict[str, Any]], submissions: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    record_index = {record.get("sample_id"): record for record in records}
    if len(record_index) != len(records):
        raise ManifestValidationError("manifest contains duplicate sample_id")
    submission_index: dict[str, dict[str, Any]] = {}
    for row in submissions:
        if not isinstance(row, dict):
            raise ManifestValidationError("workflow submissions must be JSON objects")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ManifestValidationError("workflow submission requires sample_id")
        if sample_id in submission_index:
            raise ManifestValidationError(f"duplicate workflow submission: {sample_id}")
        if sample_id not in record_index:
            raise ManifestValidationError(f"unknown sample_id in workflow submission: {sample_id}")
        submission_index[sample_id] = row
    return record_index, submission_index


def apply_annotations(
    records: list[dict[str, Any]], submissions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply human labels only after validating the complete submission batch."""
    _, submission_index = _index_submissions(records, submissions)
    updated = copy.deepcopy(records)
    for record in updated:
        row = submission_index.get(record["sample_id"])
        if row is None:
            continue
        if set(row) != {"sample_id", "annotator", "annotation"}:
            raise ManifestValidationError(
                "annotation submission must contain exactly sample_id, annotator, annotation"
            )
        if record["annotation_status"] != "pending" or record["review_status"] != "pending":
            raise ManifestValidationError(
                f"sample is not pending annotation: {record['sample_id']}"
            )
        record["annotation_status"] = "completed"
        record["annotator"] = row["annotator"]
        record["annotation"] = copy.deepcopy(row["annotation"])
        validate_manifest_record(record)
    return updated

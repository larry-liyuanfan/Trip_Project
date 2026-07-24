"""Non-gold deterministic suggestions for Week 3 human annotation packets."""

import re
from typing import Any

from src.evaluation.candidates import synthetic_evidence_text
from src.evaluation.manifests import ManifestValidationError


SUGGESTION_CONTRACT_VERSION = "week3_deterministic_annotation_suggestion_v1"
SUGGESTION_METHOD = "source_metadata_rules_v1"
PRODUCT_STRATA = {"hotel", "attraction", "restaurant"}
AFTER_SALES_STRATA = {
    "hygiene_stain",
    "facility_damage",
    "attraction_closure",
    "transport_delay",
}
SYNTHETIC_SOURCE_PATTERN = re.compile(
    r"^synthetic:(attraction_closure|transport_delay):(\d{4})$"
)


def _field_suggestion(
    value: Any,
    *,
    basis: str,
    confidence: str,
) -> dict[str, Any]:
    return {
        "value": value,
        "confidence": confidence,
        "basis": basis,
        "requires_human_confirmation": True,
    }


def build_deterministic_suggestion(record: dict[str, Any]) -> dict[str, Any]:
    """Build source/rule hints that cannot be applied as gold annotations."""
    scenario = record.get("scenario")
    if scenario == "image_product_search":
        fields, unsupported = _product_suggestion(record)
    elif scenario == "after_sales":
        fields, unsupported = _after_sales_suggestion(record)
    elif scenario == "itinerary_planning":
        fields, unsupported = _itinerary_suggestion(record)
    else:
        raise ManifestValidationError(f"unsupported suggestion scenario: {scenario!r}")
    return {
        "contract_version": SUGGESTION_CONTRACT_VERSION,
        "method": SUGGESTION_METHOD,
        "non_gold": True,
        "field_suggestions": fields,
        "unsupported_fields": unsupported,
        "warning": (
            "Suggestions come from source metadata and deterministic rules only; "
            "the annotator must inspect the original input and enter independent gold labels."
        ),
    }


def _product_suggestion(
    record: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    fields: dict[str, dict[str, Any]] = {}
    unsupported = ["price_range", "style_tags", "visible_facilities"]
    stratum = record.get("sampling_stratum")
    if stratum in PRODUCT_STRATA:
        fields["business_category"] = _field_suggestion(
            stratum,
            basis=f"sampling_stratum={stratum}",
            confidence="high",
        )
    else:
        unsupported.insert(0, "business_category")
    return fields, unsupported


def _after_sales_suggestion(
    record: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    fields: dict[str, dict[str, Any]] = {}
    unsupported = ["severity"]
    stratum = record.get("sampling_stratum")
    if stratum in AFTER_SALES_STRATA:
        fields["issue_type"] = _field_suggestion(
            stratum,
            basis=f"sampling_stratum={stratum}",
            confidence="high",
        )
    else:
        unsupported.append("issue_type")

    source_id = record.get("source_id")
    match = SYNTHETIC_SOURCE_PATTERN.fullmatch(source_id or "")
    if match and record.get("source_type") == "business_synthetic":
        issue_type, index_text = match.groups()
        heading, status, detail = synthetic_evidence_text(
            issue_type=issue_type,
            index=int(index_text),
        )
        detail_parts = [part.strip() for part in detail.split("|")]
        basis = (
            f"source_id={source_id}; "
            f"synthetic_recipe_version={record.get('provenance', {}).get('synthetic_recipe_version')}"
        )
        fields["key_information"] = _field_suggestion(
            detail_parts,
            basis=basis,
            confidence="high",
        )
        fields["ocr_ground_truth"] = _field_suggestion(
            [heading, status, detail],
            basis=basis,
            confidence="high",
        )
    else:
        unsupported.extend(["key_information", "ocr_ground_truth"])
    return fields, unsupported


def _itinerary_suggestion(
    record: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    input_payload = record.get("input")
    if not isinstance(input_payload, dict):
        raise ManifestValidationError("itinerary suggestion requires input")
    images = input_payload.get("images")
    raw_constraints = input_payload.get("text_constraints")
    if not isinstance(images, list) or not isinstance(raw_constraints, str):
        raise ManifestValidationError(
            "itinerary suggestion requires input images and text_constraints"
        )
    image_paths = [
        image["path"]
        for image in images
        if isinstance(image, dict) and isinstance(image.get("path"), str)
    ]
    atomic_constraints = _split_constraints(raw_constraints)
    hard_constraints = [
        value for value in atomic_constraints if _is_hard_constraint(value)
    ]
    soft_constraints = [
        value for value in atomic_constraints if _is_soft_constraint(value)
    ]
    required_elements = ["daily_schedule"]
    if any("预算" in value for value in atomic_constraints):
        required_elements.append("budget_check")
    if any("结束" in value for value in atomic_constraints):
        required_elements.append("end_time_check")
    if any("用餐" in value for value in atomic_constraints):
        required_elements.append("meals")
    if any("交通" in value for value in atomic_constraints):
        required_elements.append("transport")
    basis = "input.text_constraints deterministic punctuation and keyword parsing"
    fields = {
        "reference_images": _field_suggestion(
            image_paths,
            basis="input.images ordered paths",
            confidence="high",
        ),
        "text_constraints": _field_suggestion(
            atomic_constraints,
            basis=basis,
            confidence="high",
        ),
        "hard_constraints": _field_suggestion(
            hard_constraints,
            basis=basis,
            confidence="medium",
        ),
        "soft_constraints": _field_suggestion(
            soft_constraints,
            basis=basis,
            confidence="medium",
        ),
        "required_itinerary_elements": _field_suggestion(
            required_elements,
            basis=basis,
            confidence="medium",
        ),
    }
    return fields, ["style_preferences"]


def _split_constraints(raw_constraints: str) -> list[str]:
    parts = []
    for part in re.split(r"[，；。]+", raw_constraints):
        normalized = part.strip()
        if normalized.startswith("并包含"):
            normalized = normalized[1:]
        if normalized:
            parts.append(normalized)
    return parts


def _is_hard_constraint(value: str) -> bool:
    return bool(
        re.fullmatch(r"\d+天行程", value)
        or re.fullmatch(r"预算不超过\d+元", value)
        or re.fullmatch(r"最后一天\d{1,2}:\d{2}前结束", value)
        or value.startswith("包含")
        or value.startswith("避免")
    )


def _is_soft_constraint(value: str) -> bool:
    return "节奏" in value or "优先" in value or "结合" in value

"""Gold-independent product consistency and weak-reference provenance audit.

These checks establish internal consistency, not that a model saw a fact correctly.
"""

from collections import Counter
from typing import Any


PRODUCT_FIELDS = ("business_category", "style_tags", "visible_facilities", "price_range")


def product_consistency_errors(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["not_an_object"]
    unknown = payload.get("unknown_fields")
    if not isinstance(unknown, list) or any(not isinstance(item, str) for item in unknown):
        return ["invalid_unknown_fields"]
    errors = []
    for field in PRODUCT_FIELDS:
        value = payload.get(field)
        if field in unknown and value not in (None, "unknown", []):
            errors.append(f"known_value_marked_unknown:{field}")
        if field in {"business_category", "price_range"} and value == "unknown" and field not in unknown:
            errors.append(f"unknown_value_not_declared:{field}")
    for field in unknown:
        if field not in PRODUCT_FIELDS:
            errors.append(f"unrecognized_unknown_field:{field}")
    return errors


def audit_product_references(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep every sample; expose metadata and contradictions without relabeling it."""
    problems: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    affected = []
    metadata_count = 0
    for row in rows:
        target = row["target"]
        labels[str(row.get("label_source", "unspecified"))] += 1
        errors = product_consistency_errors(target)
        problems.update(errors)
        inferred = " ".join(target.get("inferred_attributes", []))
        provenance = str(row.get("target_provenance", {}))
        mixed_metadata = "元数据" in inferred or "business_metadata" in provenance
        metadata_count += int(mixed_metadata)
        if mixed_metadata and any(
            row.get("target_provenance", {}).get(field) == "caption_lexical_silver"
            for field in ("style_tags", "visible_facilities")
        ) and "风格或设施包含" in inferred:
            errors.append("metadata_mislabeled_as_caption_provenance")
            problems["metadata_mislabeled_as_caption_provenance"] += 1
        if errors:
            affected.append({"sample_id": row["sample_id"], "errors": errors})
    return {
        "protocol": "product_reference_semantics_audit_v1",
        "sample_count": len(rows),
        "label_source_counts": dict(labels),
        "metadata_proxy_samples": metadata_count,
        "issue_counts": dict(problems),
        "affected_samples": affected,
        "visual_accuracy_claim_supported": bool(rows) and not metadata_count and not problems and all(
            row.get("visual_accuracy_claim_supported") is True for row in rows
        ),
        "interpretation": "Metadata/caption silver agreement is not image-grounded accuracy.",
    }

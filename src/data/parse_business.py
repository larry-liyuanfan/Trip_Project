"""Parse Yelp businesses and stabilize variable nested attributes for tables."""

import ast
import json
from collections.abc import Callable
from typing import Any, Iterable


SELECTED_ATTRIBUTE_KEYS = {
    "RestaurantsPriceRange2",
    "OutdoorSeating",
    "BusinessParking",
    "WiFi",
    "RestaurantsTakeOut",
    "RestaurantsDelivery",
    "RestaurantsReservations",
    "GoodForKids",
    "Ambience",
}


def parse_business_record(record: dict[str, Any]) -> dict[str, Any]:
    """Extract core business fields and selected flattened attributes."""
    attributes = record.get("attributes") or {}
    parsed = {
        "business_id": record.get("business_id"),
        "name": record.get("name"),
        "address": record.get("address"),
        "city": record.get("city"),
        "state": record.get("state"),
        "postal_code": record.get("postal_code"),
        "latitude": record.get("latitude"),
        "longitude": record.get("longitude"),
        "stars": record.get("stars"),
        "review_count": record.get("review_count"),
        "is_open": record.get("is_open"),
        "categories": parse_categories(record.get("categories")),
        "attributes": attributes,
        "hours": record.get("hours") or {},
    }
    parsed.update(flatten_selected_attributes(attributes))
    return parsed


def parse_business_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse a bounded iterable into an in-memory business table."""
    return [parse_business_record(record) for record in records]


def stream_business_records(
    records: Iterable[dict[str, Any]],
    row_sink: Callable[[dict[str, Any]], None],
) -> dict[str, int]:
    """Parse business JSONL rows directly into a caller-owned table writer."""
    summary = {"input_businesses": 0, "parsed_businesses": 0, "missing_business_id": 0}
    for record in records:
        summary["input_businesses"] += 1
        row = parse_business_record(record)
        if not row.get("business_id"):
            summary["missing_business_id"] += 1
            continue
        # Yelp attributes and hours have different nested keys per business.
        # Store them as JSON text so every Parquet chunk shares one schema.
        row_sink(serialize_business_nested_fields(row))
        summary["parsed_businesses"] += 1
    return summary


def serialize_business_nested_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Serialize variable attributes and hours for a stable Parquet schema."""
    serialized = dict(row)
    for key in ("attributes", "hours"):
        value = serialized.get(key) or {}
        serialized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return serialized


def parse_categories(categories: Any) -> list[str]:
    """Normalize comma-separated or list categories to lowercase labels."""
    if not categories:
        return []
    if isinstance(categories, list):
        return [str(category).strip().lower() for category in categories if str(category).strip()]
    return [category.strip().lower() for category in str(categories).split(",") if category.strip()]


def flatten_selected_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Flatten only attribute families needed by downstream OTA tasks."""
    flattened: dict[str, Any] = {}
    for key, value in attributes.items():
        if key not in SELECTED_ATTRIBUTE_KEYS:
            continue
        parsed_value = _parse_possible_literal(value)
        if isinstance(parsed_value, dict):
            for child_key, child_value in parsed_value.items():
                flattened[f"attr_{key}_{child_key}"] = child_value
        else:
            flattened[f"attr_{key}"] = parsed_value
    return flattened


def _parse_possible_literal(value: Any) -> Any:
    """Safely decode Yelp's stringified list and mapping attribute values."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if not stripped.startswith(("{", "[")):
        return value
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return value

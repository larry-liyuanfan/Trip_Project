import ast
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
    return [parse_business_record(record) for record in records]


def parse_categories(categories: Any) -> list[str]:
    if not categories:
        return []
    if isinstance(categories, list):
        return [str(category).strip().lower() for category in categories if str(category).strip()]
    return [category.strip().lower() for category in str(categories).split(",") if category.strip()]


def flatten_selected_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
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

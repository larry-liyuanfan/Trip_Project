import json
from typing import Any


def business_description(business: dict[str, Any]) -> str:
    parts = [
        business.get("name"),
        ", ".join(business.get("categories") or []),
        business.get("city"),
        business.get("state"),
        f"rated {business.get('stars')}" if business.get("stars") is not None else None,
    ]
    attrs = mapping_value(business.get("attributes"))
    if attrs:
        attr_text = ", ".join(f"{key}: {value}" for key, value in sorted(attrs.items())[:5])
        parts.append(attr_text)
    hours = mapping_value(business.get("hours"))
    if hours:
        parts.append("hours: " + ", ".join(f"{day} {value}" for day, value in sorted(hours.items())))
    return " | ".join(str(part) for part in parts if part)


def mapping_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}

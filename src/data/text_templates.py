from typing import Any


def business_description(business: dict[str, Any]) -> str:
    parts = [
        business.get("name"),
        ", ".join(business.get("categories") or []),
        business.get("city"),
        business.get("state"),
        f"rated {business.get('stars')}" if business.get("stars") is not None else None,
    ]
    attrs = business.get("attributes") or {}
    if attrs:
        attr_text = ", ".join(f"{key}: {value}" for key, value in sorted(attrs.items())[:5])
        parts.append(attr_text)
    return " | ".join(str(part) for part in parts if part)

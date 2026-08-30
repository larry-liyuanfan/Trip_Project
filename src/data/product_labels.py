"""Versioned caption silver and merchant metadata; neither is human visual gold."""

import ast
import json
import math
import re
from typing import Any

LABEL_PROTOCOL = "caption_evidence_v2"
STYLES = ("casual", "classy", "cozy", "historic", "modern", "romantic", "rustic", "trendy", "upscale", "vintage")
FACILITIES = {
    "bar": ("bar",), "outdoor_seating": ("patio", "terrace", "outdoor seating"),
    "pool": ("pool",), "front_desk": ("front desk", "reception"), "parking": ("parking",),
    "wifi": ("wifi", "wi-fi"), "wheelchair_access": ("wheelchair ramp", "wheelchair access", "accessible entrance"),
    "gym": ("gym", "fitness"), "spa": ("spa",), "playground": ("playground",),
}


def affirmative_term(text: str, term: str) -> bool:
    """完整词匹配，否定只作用于当前分句；不把 mushroom 当作 room。"""
    for clause in re.split(r"[.;!?，。；]|\bbut\b", text.casefold()):
        for match in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", clause):
            before, after = clause[:match.start()], clause[match.end():]
            if re.search(r"\b(?:no|not|without|neither|lack(?:s|ing)?|absent)\b", before):
                continue
            if re.match(r"\s+(?:(?:is|are|was)\s+)?(?:not|unavailable|absent)\b", after):
                continue
            return True
    return False


def _literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value.strip())
        except (ValueError, SyntaxError, TypeError):
            pass
    return value.strip()


def business_attributes(value: Any) -> dict[str, Any]:
    """Parse real mappings or the bounded attribute portion of legacy descriptions."""
    parsed = _literal(value)
    if isinstance(parsed, dict):
        return {str(k): _literal(v) for k, v in parsed.items()}
    if not isinstance(value, str):
        return {}
    result = {}
    # 已知属性键作为边界，字典内逗号不会拆散 BusinessParking/Ambience。
    names = "BusinessParking|BikeParking|WiFi|Alcohol|OutdoorSeating|WheelchairAccessible|Ambience|RestaurantsPriceRange2"
    matches = list(re.finditer(r"(?:^|[|,]\s*)\s*(" + names + r")\s*:\s*", value))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(value)
        raw = value[match.end():end].split(" | ", 1)[0].strip(" ,")
        result[match.group(1)] = _literal(raw)
    return result


def merchant_tags(value: Any) -> tuple[list[str], list[str]]:
    attrs = business_attributes(value)
    ambience = attrs.get("Ambience", {})
    styles = [tag for tag in STYLES if isinstance(ambience, dict) and _literal(ambience.get(tag)) is True]
    parking = attrs.get("BusinessParking", {})
    checks = {
        "parking": isinstance(parking, dict) and any(_literal(v) is True for v in parking.values()),
        "wifi": attrs.get("WiFi") in ("free", "paid"),
        "bar": attrs.get("Alcohol") in ("full_bar", "beer_and_wine"),
        "outdoor_seating": attrs.get("OutdoorSeating") is True,
        "wheelchair_access": attrs.get("WheelchairAccessible") is True,
    }
    return sorted(styles), sorted(k for k, present in checks.items() if present)


def caption_labels(caption: str) -> dict[str, Any]:
    categories = {
        "hotel": ("hotel", "resort", "hotel room", "guest room"),
        "restaurant": ("restaurant", "cafe", "bar", "dining"),
        "attraction": ("museum", "park", "attraction", "landmark"),
    }
    found = [name for name, terms in categories.items() if any(affirmative_term(caption, t) for t in terms)]
    category = found[0] if len(found) == 1 else "unknown"
    styles = [tag for tag in STYLES if affirmative_term(caption, tag)]
    facilities = [tag for tag, terms in FACILITIES.items() if any(affirmative_term(caption, t) for t in terms)]
    return {"business_category": category, "style_tags": sorted(styles), "visible_facilities": sorted(facilities),
            "price_range": "unknown", "observed_evidence": [caption] if caption else [],
            "inferred_attributes": [], "unknown_fields": sorted(
                (["business_category"] if category == "unknown" else [])
                + (["style_tags"] if not styles else []) + (["visible_facilities"] if not facilities else [])
                + ["price_range"]), "confidence": 0.5}


def silver_row(source: dict[str, Any]) -> dict[str, Any]:
    """新版本引用身份不改变；商家属性单列，永不并入图片 target。"""
    caption = str(source.get("caption") or "")
    weight = float(source.get("sample_weight", 0.5))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("silver weight must be finite and non-negative")
    styles, facilities = merchant_tags(source.get("attributes", source.get("business_description", "")))
    target = caption_labels(caption)
    slices = ["business_category_" + target["business_category"], "price_without_visual_evidence", "should_use_unknown",
              "facility_visible" if target["visible_facilities"] else "facility_should_be_empty",
              "style_visible" if target["style_tags"] else "style_should_be_empty",
              "semantic_label_even_when_schema_complete"]
    return {**source, "target": target, "error_slices": slices, "label_protocol": LABEL_PROTOCOL,
            "label_source": "programmatic_silver", "sample_weight": min(weight, 0.5),
            "target_provenance": {key: "caption_lexical_silver_v2" for key in
                                  ("business_category", "style_tags", "visible_facilities", "price_range")},
            "merchant_metadata": {"style_tags": styles, "facilities": facilities},
            "visual_accuracy_claim_supported": False}

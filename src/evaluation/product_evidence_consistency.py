"""Gold-independent checks that a positive label names its own visible object."""
from collections import Counter
import re

from src.inference.product_observation import negated_label_fact


FACILITY_OBJECT_TERMS = {
    "seating": ("chair", "chairs", "bench", "benches", "stool", "stools", "seat", "seats",
                "sofa", "sofas", "couch", "couches", "booth", "booths", "椅", "凳", "沙发", "座位"),
    "dining_tables": ("table", "tables", "餐桌", "桌"),
    "counter": ("counter", "counters", "柜台", "吧台"),
    "bar": ("bar", "beer tap", "beer taps", "drink-serving", "drink service", "beverage station",
            "drink equipment", "酒吧", "酒水台", "啤酒龙头"),
    "pool": ("swimming pool", "泳池", "游泳池"),
    "front_desk": ("front desk", "reception desk", "前台"),
    "parking": ("parking", "parked", "parking space", "parking spaces", "parking lot", "停车", "车位"),
    "wifi_sign": ("wifi", "wi-fi", "无线网络"),
    "wheelchair_access": ("wheelchair", "accessible ramp", "accessibility ramp", "curb cut", "无障碍", "轮椅"),
    "gym": ("gym", "fitness", "treadmill", "exercise equipment", "weights", "健身", "跑步机"),
    "spa": ("spa", "massage table", "treatment room", "水疗", "按摩床"),
    "playground": ("playground", "play structure", "slide", "swings", "游乐场", "滑梯", "秋千"),
    "elevator": ("elevator", "lift doors", "电梯"),
    "restroom_sign": ("restroom", "toilet sign", "washroom", "卫生间", "洗手间"),
}


def _contains_term(text, term):
    if term.isascii():
        return bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I))
    return term in text


def evidence_consistency_errors(observation):
    """Return deterministic diagnostics without reading a reference label."""
    errors = []
    for index, item in enumerate(observation.get("facility_evidence", [])):
        label, fact = item.get("label"), item.get("fact", "")
        if negated_label_fact(label, fact):
            errors.append(f"facility[{index}]:{label}:negated")
            continue
        if label == "outdoor_seating":
            location = any(_contains_term(fact, term) for term in ("outdoor", "patio", "terrace", "outside", "室外", "户外", "露台"))
            furnishing = any(_contains_term(fact, term) for term in FACILITY_OBJECT_TERMS["seating"] + FACILITY_OBJECT_TERMS["dining_tables"])
            supported = location and furnishing
        else:
            terms = FACILITY_OBJECT_TERMS.get(label)
            supported = terms is None or any(_contains_term(fact, term) for term in terms)
        if not supported:
            errors.append(f"facility[{index}]:{label}:object_mismatch")
    for index, item in enumerate(observation.get("style_evidence", [])):
        label, fact = item.get("label"), item.get("fact", "")
        if label == "natural" and re.search(r"\b(?:mural|painting|picture|photo|print)\b|壁画|照片|画作", fact, re.I):
            errors.append(f"style[{index}]:natural:depicted_not_real")
        if label == "traditional" and re.fullmatch(
                r"\s*(?:wall[- ]mounted\s+)?(?:framed\s+)?(?:photos?|pictures?|art(?:work)?)\s*(?:on\s+walls?)?\s*",
                fact, re.I):
            errors.append(f"style[{index}]:traditional:generic_frame_only")
    return errors


def summarize_evidence_consistency(records):
    """Summarize target-free evidence checks over immutable model records."""
    counts = Counter()
    successful = 0
    labels = 0
    samples_with_errors = 0
    for record in records:
        observation = record.get("observation")
        if not isinstance(observation, dict):
            continue
        successful += 1
        labels += len(observation.get("facility_evidence", []))
        labels += len(observation.get("style_evidence", []))
        errors = evidence_consistency_errors(observation)
        samples_with_errors += bool(errors)
        for error in errors:
            counts[error.rsplit(":", 1)[-1]] += 1
    return {
        "protocol": "product_evidence_consistency_v1",
        "target_free": True,
        "selection_use": "diagnostic_only",
        "records_read": len(records),
        "successful_observations": successful,
        "positive_evidence_labels": labels,
        "inconsistent_evidence_labels": sum(counts.values()),
        "samples_with_errors": samples_with_errors,
        "error_counts": dict(sorted(counts.items())),
    }

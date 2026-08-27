"""Minimal deterministic business checks, separate from JSON/Schema acceptance."""

import json
import re
from typing import Any


class BusinessValidationError(ValueError):
    pass


def requested_days(text: str) -> int | None:
    match = re.search(r"(?<![\d年月])([0-9]+|[一二两三四五六七八九十]+)\s*(?:天|日(?=游|行程|旅游|旅行|安排|计划|[，。；,;\s]|$)|days?\b)", text, re.I)
    if not match:
        return None
    value = match.group(1)
    if value.isdigit():
        return int(value) if len(value) < 5 else None
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if "十" in value:
        left, right = value.split("十", 1)
        return (digits.get(left, 1) * 10 + digits.get(right, 0)) if len(left) <= 1 and len(right) <= 1 else None
    return digits.get(value)


def itinerary_business_errors(payload: dict[str, Any], text: str) -> list[str]:
    errors = []
    days = payload.get("itinerary", [])
    expected = requested_days(text)
    if expected is not None and len(days) != expected:
        errors.append(f"day_count_expected_{expected}_got_{len(days)}")
    if [day.get("day_index") for day in days] != list(range(1, len(days) + 1)):
        errors.append("day_indices_must_be_contiguous_from_1")
    serialized = json.dumps(payload, ensure_ascii=False)
    if re.search(r"简短摘要|简短活动|原始约束短语|photo type|budget constraint|requirement type|占位|TODO|TBD|placeholder", serialized, re.I):
        errors.append("template_placeholder_content")
    # 城市/交通/截止时间等明确条件必须在检查表中有证据，不接受仅复制约束文字。
    checks = payload.get("constraint_check", [])
    explicit = re.findall(r"(?:上海|北京|广州|深圳|杭州|成都|Shanghai|Beijing|公共交通|public transport|\d{1,2}:\d{2})", text, re.I)
    explicit.extend(match.group(1).strip() for match in re.finditer(r"(?:城市|目的地)[:：]\s*([^；，。;,]+)", text))
    explicit.extend(clause.strip() for clause in re.split(r"[；;，,。]", text)
                    if re.search(r"必须|不要|不去|不得|不超过|预算(?:上限|不超|为|[:：])", clause))
    for term in dict.fromkeys(explicit):
        matching = [item for item in checks if term.casefold() in str(item.get("constraint", "")).casefold()]
        if not matching or not any(item.get("status") in {"satisfied", "met"} and item.get("evidence") for item in matching):
            errors.append(f"explicit_constraint_not_verified:{term}")
    if any(item.get("constraint_type") == "hard" and item.get("status") != "satisfied" for item in checks):
        errors.append("hard_constraint_not_satisfied")
    if any(not day.get("activities") for day in days):
        errors.append("day_without_activities")
    return errors

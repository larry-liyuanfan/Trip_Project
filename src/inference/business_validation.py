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
    errors.extend(_activity_constraint_errors(days, text))
    return errors


def _clock_minutes(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", value)
    if not match:
        return None
    hour, minute = map(int, match.groups())
    return hour * 60 + minute if hour < 24 and minute < 60 else None


def _activity_constraint_errors(days: list[dict], text: str) -> list[str]:
    """检查可直接核对的计划内容，不能让模型的 satisfied 自证替代执行证据。"""
    errors = []
    deadline_match = re.search(r"(\d{1,2}:\d{2})\s*(?:之?前)\s*(?:结束|返回|回到)", text)
    deadline = _clock_minutes(deadline_match.group(1)) if deadline_match else None
    public_only = bool(re.search(r"公共交通|public transport", text, re.I))
    activities = [activity for day in days for activity in day.get("activities", [])]
    for day in days:
        previous_end = None
        for activity in day.get("activities", []):
            start, end = (_clock_minutes(activity.get(key)) for key in ("start_time", "end_time"))
            if any(activity.get(key) is not None and _clock_minutes(activity[key]) is None
                   for key in ("start_time", "end_time")):
                errors.append("invalid_activity_time")
            if start is not None and end is not None and start >= end:
                errors.append("activity_time_order_invalid")
            if previous_end is not None and start is not None and start < previous_end:
                errors.append("activity_time_overlap")
            previous_end = end
            if deadline is not None:
                if end is None:
                    errors.append("deadline_not_verifiable_without_activity_end")
                elif end > deadline:
                    errors.append("activity_ends_after_requested_deadline")
            if public_only:
                transport = str(activity.get("transport") or "")
                if re.search(r"出租|打车|自驾|包车|taxi|private car|drive", transport, re.I):
                    errors.append("private_transport_violates_public_transport")
                if not re.search(r"公共交通|公交|地铁|步行|火车|轻轨|电车|public transport|bus|metro|subway|walk|train|tram", transport, re.I):
                    errors.append("public_transport_not_verifiable")
    activity_text = " ".join(str(item.get(key) or "") for item in activities for key in ("place_name", "activity"))
    # 仅处理明确的地点包含/排除语法；未覆盖的自然语言条件不伪装成完整语义证明。
    for match in re.finditer(r"(?:必须包含|必须去|必须参观|必去)\s*([^，。；,;]+)", text):
        place = match.group(1).strip()
        if place not in activity_text:
            errors.append(f"required_place_missing_from_activities:{place}")
    for match in re.finditer(r"(?:不去|不要去|不得去|不参观)\s*([^，。；,;]+)", text):
        place = match.group(1).strip()
        if place in activity_text:
            errors.append(f"excluded_place_in_activities:{place}")
    return list(dict.fromkeys(errors))

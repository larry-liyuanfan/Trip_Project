"""Week 6 单场景训练目标的确定性质量审计。"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.evaluation.schema_validation import SchemaValidationError, validate_output
from src.training.week6_qlora import Week6TrainingError, validate_training_row


_CONSTRAINT_PREFIX = "原始文字约束："
_CLAUSE_SPLIT = re.compile(r"[，；。]")
_DAY_PATTERN = re.compile(r"计划(?P<days>\d+)天")


def _text_items(messages: list[dict[str, Any]], role: str) -> Iterable[str]:
    for message in messages:
        if message.get("role") != role:
            continue
        content = message.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    yield item["text"]


def extract_itinerary_constraints(messages: list[dict[str, Any]]) -> str:
    """从版本化用户消息中提取原始文字约束。"""
    matches = [
        text[len(_CONSTRAINT_PREFIX) :].strip()
        for text in _text_items(messages, "user")
        if text.startswith(_CONSTRAINT_PREFIX)
    ]
    if len(matches) != 1 or not matches[0]:
        raise Week6TrainingError("itinerary row requires one original constraint block")
    return matches[0]


def expected_itinerary_elements(constraint_text: str) -> set[str]:
    """按训练 Prompt 的显式映射推导必须出现的行程元素。"""
    expected = {"daily_schedule"}
    if "用餐" in constraint_text or "每日餐" in constraint_text:
        expected.add("meals")
    if "交通" in constraint_text:
        expected.add("transport")
    if "预算" in constraint_text:
        expected.add("budget_check")
    if "截止" in constraint_text or "结束时间" in constraint_text:
        expected.add("end_time_check")
    return expected


def repair_itinerary_target(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    """按显式文字约束生成派生 silver 目标，并保留图片相关字段。"""
    validate_training_row(row, scenario="itinerary_planning")
    constraint_text = extract_itinerary_constraints(row["messages"])
    day_match = _DAY_PATTERN.search(constraint_text)
    if day_match is None:
        raise Week6TrainingError("cannot repair itinerary target without trip days")
    days = int(day_match.group("days"))
    if not 1 <= days <= 14:
        raise Week6TrainingError("itinerary repair supports 1 to 14 days")
    clauses = [
        clause.strip()
        for clause in _CLAUSE_SPLIT.split(constraint_text)
        if clause.strip()
    ]
    if not clauses or len(clauses) > 12:
        raise Week6TrainingError("itinerary constraint clause count is unsupported")

    assistant_texts = list(_text_items(row["messages"][-1:], "assistant"))
    if len(assistant_texts) != 1:
        raise Week6TrainingError("itinerary target repair requires one assistant output")
    try:
        source_output = json.loads(assistant_texts[0])
        validate_output(root, "itinerary_planning", source_output, "v2")
    except (json.JSONDecodeError, SchemaValidationError) as exc:
        raise Week6TrainingError("itinerary target repair requires a valid source output") from exc

    hard_constraints = [
        clause
        for clause in clauses
        if clause.startswith("计划") or "出行" in clause or "预算" in clause
    ]
    soft_constraints = [clause for clause in clauses if clause not in hard_constraints]
    ordered_elements = [
        name
        for name in (
            "daily_schedule",
            "meals",
            "transport",
            "budget_check",
            "end_time_check",
        )
        if name in expected_itinerary_elements(constraint_text)
    ]
    transport = "公共交通" if "公共交通" in constraint_text else None
    repaired = {
        "style_preferences": list(source_output.get("style_preferences", []))[:3],
        "hard_constraints": hard_constraints,
        "soft_constraints": soft_constraints,
        "required_itinerary_elements": ordered_elements,
        "itinerary": [
            {
                "day_index": day_index,
                "date": None,
                "summary": f"第{day_index}天按约束安排行程",
                "activities": [
                    {
                        "start_time": None,
                        "end_time": None,
                        "place_name": None,
                        "activity": "按预算、交通和用餐约束安排活动",
                        "transport": transport,
                        "source_evidence": [],
                    }
                ],
            }
            for day_index in range(1, days + 1)
        ],
        "constraint_check": [
            {
                "constraint": clause,
                "constraint_type": "hard" if clause in hard_constraints else "soft",
                "status": "unknown",
                "evidence": None,
            }
            for clause in clauses
        ],
        "observed_evidence": list(source_output.get("observed_evidence", []))[:3],
        "unknown_fields": list(source_output.get("unknown_fields", [])),
        "confidence": source_output.get("confidence"),
    }
    validate_output(root, "itinerary_planning", repaired, "v2")
    return repaired


def audit_itinerary_target(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    """审计一条行程训练目标，不修改输入或推断图片语义。"""
    validate_training_row(row, scenario="itinerary_planning")
    messages = row["messages"]
    constraint_text = extract_itinerary_constraints(messages)
    clauses = {
        clause.strip()
        for clause in _CLAUSE_SPLIT.split(constraint_text)
        if clause.strip()
    }
    day_match = _DAY_PATTERN.search(constraint_text)
    expected_days = int(day_match.group("days")) if day_match else None

    assistant_texts = list(_text_items(messages[-1:], "assistant"))
    parsed: dict[str, Any] | None = None
    json_valid = False
    schema_valid = False
    if len(assistant_texts) == 1:
        try:
            candidate = json.loads(assistant_texts[0])
            json_valid = isinstance(candidate, dict)
            if json_valid:
                parsed = candidate
                validate_output(root, "itinerary_planning", parsed, "v2")
                schema_valid = True
        except (json.JSONDecodeError, SchemaValidationError):
            pass

    output = parsed or {}
    hard = output.get("hard_constraints")
    soft = output.get("soft_constraints")
    output_constraints = (
        {str(value) for value in [*(hard or []), *(soft or [])]}
        if isinstance(hard, list) and isinstance(soft, list)
        else set()
    )
    checks = output.get("constraint_check")
    checked_constraints = (
        {str(item.get("constraint")) for item in checks if isinstance(item, dict)}
        if isinstance(checks, list)
        else set()
    )
    itinerary = output.get("itinerary")
    actual_days = len(itinerary) if isinstance(itinerary, list) else None
    day_indices = (
        [item.get("day_index") for item in itinerary if isinstance(item, dict)]
        if isinstance(itinerary, list)
        else []
    )
    actual_elements = output.get("required_itinerary_elements")
    actual_element_set = (
        {str(value) for value in actual_elements}
        if isinstance(actual_elements, list)
        else set()
    )
    expected_elements = expected_itinerary_elements(constraint_text)

    checks_by_name = {
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "expected_days_parsed": expected_days is not None,
        "day_count_match": expected_days is not None and actual_days == expected_days,
        "day_indices_sequential": (
            expected_days is not None
            and day_indices == list(range(1, expected_days + 1))
        ),
        "constraint_text_exact_match": output_constraints == clauses,
        "constraint_check_exact_coverage": (
            bool(output_constraints) and checked_constraints == output_constraints
        ),
        "required_elements_complete": expected_elements <= actual_element_set,
    }
    return {
        "sample_id": row["sample_id"],
        "label_source": row["label_source"],
        "sample_weight": float(row["sample_weight"]),
        "expected_days": expected_days,
        "actual_days": actual_days,
        "expected_constraints": sorted(clauses),
        "output_constraints": sorted(output_constraints),
        "expected_itinerary_elements": sorted(expected_elements),
        "output_itinerary_elements": sorted(actual_element_set),
        "checks": checks_by_name,
        "passed": all(checks_by_name.values()),
    }


def summarize_itinerary_targets(
    root: Path,
    rows: Iterable[dict[str, Any]],
    *,
    max_examples_per_issue: int = 5,
) -> dict[str, Any]:
    """按标签来源聚合审计结果并保留有界错误样例。"""
    if max_examples_per_issue < 0:
        raise Week6TrainingError("max_examples_per_issue cannot be negative")
    totals: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    issue_examples: dict[str, list[str]] = {}
    for row in rows:
        audit = audit_itinerary_target(root, row)
        source = audit["label_source"]
        source_counts = by_source.setdefault(source, Counter())
        totals["rows"] += 1
        source_counts["rows"] += 1
        if audit["passed"]:
            totals["passed"] += 1
            source_counts["passed"] += 1
        for name, passed in audit["checks"].items():
            if passed:
                totals[name] += 1
                source_counts[name] += 1
                continue
            issue = f"failed_{name}"
            totals[issue] += 1
            source_counts[issue] += 1
            examples = issue_examples.setdefault(issue, [])
            if len(examples) < max_examples_per_issue:
                examples.append(audit["sample_id"])
    if not totals["rows"]:
        raise Week6TrainingError("target audit requires at least one row")
    return {
        "status": "ok",
        "scenario": "itinerary_planning",
        "rows": totals["rows"],
        "passed": totals["passed"],
        "counts": dict(sorted(totals.items())),
        "by_label_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(by_source.items())
        },
        "issue_examples": dict(sorted(issue_examples.items())),
    }

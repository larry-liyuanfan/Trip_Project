"""Week 5 人工结果的只读中文审阅视图。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.week5_dataset import SCENARIOS, read_jsonl


KEY_ZH = {
    "business_category": "商家类别",
    "style_tags": "风格标签",
    "visible_facilities": "可见设施",
    "price_range": "价格档次",
    "observed_evidence": "可观察证据",
    "inferred_attributes": "推断属性",
    "unknown_fields": "无法判断字段",
    "confidence": "置信度",
    "issue_type": "问题类型",
    "severity": "严重程度",
    "issue_location": "问题位置",
    "key_information": "关键信息",
    "ocr_text": "图片文字 OCR",
    "style_preferences": "风格偏好",
    "hard_constraints": "硬性约束",
    "soft_constraints": "软性约束",
    "required_itinerary_elements": "必需行程要素",
    "itinerary": "行程",
    "day_index": "第几天",
    "date": "日期",
    "summary": "当日概要",
    "activities": "活动",
    "start_time": "开始时间",
    "end_time": "结束时间",
    "place_name": "地点名称",
    "activity": "活动内容",
    "transport": "交通方式",
    "source_evidence": "来源证据",
    "constraint_check": "约束检查",
    "constraint": "约束",
    "constraint_type": "约束类型",
    "status": "满足状态",
    "evidence": "依据",
}

VALUE_ZH = {
    "hotel": "酒店",
    "attraction": "景点",
    "restaurant": "餐厅",
    "other": "其他",
    "unknown": "无法判断",
    "budget": "经济型",
    "mid_range": "中档",
    "premium": "高档",
    "luxury": "奢华",
    "hygiene_stain": "卫生污渍",
    "facility_damage": "设施损坏",
    "attraction_closure": "景点关闭",
    "transport_delay": "交通延误",
    "low": "低",
    "medium": "中",
    "high": "高",
    "critical": "严重",
    "hard": "硬性",
    "soft": "软性",
    "satisfied": "已满足",
    "violated": "未满足",
}


def localized_value(value: Any) -> Any:
    """翻译稳定字段名和受控枚举，不改写自由文本或规范 JSON。"""
    if isinstance(value, dict):
        return {KEY_ZH.get(key, key): localized_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [localized_value(item) for item in value]
    if isinstance(value, str) and value in VALUE_ZH:
        return f"{VALUE_ZH[value]}（{value}）"
    return value


def export_localized_annotations(root: Path, config: dict[str, Any]) -> dict[str, int]:
    """导出最新人工 revision 的中文审阅镜像；原 annotations 保持字节不变。"""
    output_dir = root / config["paths"]["output_dir"]
    localized_dir = output_dir / "localized_annotations"
    localized_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for scenario in SCENARIOS:
        latest: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(output_dir / "annotations" / f"{scenario}.jsonl"):
            sample_id = str(row["sample_id"])
            if int(row.get("revision", 0)) >= int(
                latest.get(sample_id, {}).get("revision", 0)
            ):
                latest[sample_id] = row
        destination = localized_dir / f"{scenario}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for sample_id in sorted(latest):
                row = latest[sample_id]
                canonical = json.dumps(
                    row["human_annotation"], ensure_ascii=False, sort_keys=True
                )
                payload = {
                    "sample_id": sample_id,
                    "scenario": scenario,
                    "annotation_revision": int(row["revision"]),
                    "canonical_annotation_sha256": hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest(),
                    "localized_annotation": localized_value(row["human_annotation"]),
                }
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        counts[scenario] = len(latest)
    return counts

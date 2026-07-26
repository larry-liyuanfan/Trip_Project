"""Week 4 few-shot prompt rendering over immutable Week 3 v2 records."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from src.evaluation.prompting import render_standard_prompt
from src.evaluation.scenarios import SCENARIOS


WEEK4_PROMPT_VERSIONS = {"fewshot_4_v2": 4, "fewshot_7_v2": 7}


class Week4PromptError(ValueError):
    """Raised when the frozen few-shot selection or records are invalid."""


def load_week4_selection(path: Path | str) -> dict[str, Any]:
    """Load and validate the checked-in example and pilot selection."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") not in {
        "week4_prompt_selection_v1",
        "week4_prompt_selection_v2",
    }:
        raise Week4PromptError("unsupported Week 4 selection version")
    if payload["version"] == "week4_prompt_selection_v2":
        if payload.get("example_dataset_version") != "week4_demo_dev_v1":
            raise Week4PromptError("v2 examples must use week4_demo_dev_v1")
        if payload.get("pilot_dataset_version") != "week3_evaluation_v2":
            raise Week4PromptError("v2 pilot must use week3_evaluation_v2")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, dict) or set(scenarios) != set(SCENARIOS):
        raise Week4PromptError("selection must contain exactly the three scenarios")
    for scenario, settings in scenarios.items():
        positives = settings.get("positive_example_ids")
        boundaries = settings.get("boundary_example_ids")
        pilot = settings.get("pilot_sample_ids")
        if not isinstance(positives, list) or len(positives) != 5:
            raise Week4PromptError(f"{scenario} requires five positive examples")
        if not isinstance(boundaries, list) or len(boundaries) != 2:
            raise Week4PromptError(f"{scenario} requires two boundary examples")
        if not isinstance(pilot, list) or not pilot:
            raise Week4PromptError(f"{scenario} requires fixed pilot samples")
        combined = positives + boundaries + pilot
        if any(not isinstance(value, str) or not value for value in combined):
            raise Week4PromptError(f"{scenario} sample IDs must be non-empty strings")
        if len(set(combined)) != len(combined):
            raise Week4PromptError(f"{scenario} example and pilot IDs must be disjoint")
    return payload


def validate_selection_records(
    selection: dict[str, Any],
    example_records_by_id: dict[str, dict[str, Any]],
    pilot_records_by_id: dict[str, dict[str, Any]] | None = None,
) -> None:
    """将示例绑定到 development 金标，并验证其与 evaluation pilot 隔离。"""
    pilot_records = pilot_records_by_id or example_records_by_id
    example_version = selection.get(
        "example_dataset_version",
        selection.get("dataset_version", "week3_evaluation_v2"),
    )
    pilot_version = selection.get(
        "pilot_dataset_version",
        selection.get("dataset_version", "week3_evaluation_v2"),
    )
    selected_examples: list[dict[str, Any]] = []
    selected_pilots: list[dict[str, Any]] = []
    for scenario, settings in selection["scenarios"].items():
        example_ids = (
            settings["positive_example_ids"] + settings["boundary_example_ids"]
        )
        for sample_id in example_ids:
            record = example_records_by_id.get(sample_id)
            if record is None:
                raise Week4PromptError(f"selection example is missing: {sample_id}")
            if record.get("scenario") != scenario:
                raise Week4PromptError(f"selection scenario mismatch: {sample_id}")
            if record.get("dataset_version") != example_version:
                raise Week4PromptError(
                    f"selection example dataset mismatch: {sample_id}"
                )
            if selection["version"] == "week4_prompt_selection_v2":
                if record.get("split") != "development":
                    raise Week4PromptError(
                        f"selection example must be development: {sample_id}"
                    )
            if record.get("annotation_status") != "completed":
                raise Week4PromptError(f"selection gold is incomplete: {sample_id}")
            if not isinstance(record.get("annotation"), dict):
                raise Week4PromptError(f"selection annotation is invalid: {sample_id}")
            selected_examples.append(record)
        for sample_id in settings["pilot_sample_ids"]:
            record = pilot_records.get(sample_id)
            if record is None:
                raise Week4PromptError(f"selection pilot is missing: {sample_id}")
            if record.get("scenario") != scenario:
                raise Week4PromptError(f"selection scenario mismatch: {sample_id}")
            if record.get("dataset_version") != pilot_version:
                raise Week4PromptError(
                    f"selection pilot dataset mismatch: {sample_id}"
                )
            if selection["version"] == "week4_prompt_selection_v2":
                if record.get("split") != "evaluation":
                    raise Week4PromptError(
                        f"selection pilot must be evaluation: {sample_id}"
                    )
            if record.get("annotation_status") != "completed":
                raise Week4PromptError(f"selection gold is incomplete: {sample_id}")
            selected_pilots.append(record)
    _reject_example_pilot_collisions(selected_examples, selected_pilots)


def _reject_example_pilot_collisions(
    examples: list[dict[str, Any]],
    pilots: list[dict[str, Any]],
) -> None:
    """拒绝 sample、来源、图片或来源组任一层面的 demo/test 重叠。"""
    for identity_name, extractor in (
        ("sample_id", lambda row: {row["sample_id"]}),
        ("source_id", lambda row: {row["source_id"]}),
        (
            "image_sha256",
            lambda row: {image["sha256"] for image in row["input"]["images"]},
        ),
        (
            "group_id",
            lambda row: (
                {row["provenance"]["group_id"]}
                if isinstance(row.get("provenance"), dict)
                and row["provenance"].get("group_id")
                else set()
            ),
        ),
    ):
        example_values = set().union(*(extractor(row) for row in examples))
        pilot_values = set().union(*(extractor(row) for row in pilots))
        overlap = example_values & pilot_values
        if overlap:
            raise Week4PromptError(
                f"example/pilot {identity_name} collision: {sorted(overlap)}"
            )


def validate_demo_dev_isolation(
    demo_dev_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
) -> None:
    """验证整个 development 池与最终 evaluation 集完全隔离。"""
    if not demo_dev_records:
        raise Week4PromptError("demo/dev pool is empty")
    if any(
        record.get("dataset_version") != "week4_demo_dev_v1"
        or record.get("split") != "development"
        for record in demo_dev_records
    ):
        raise Week4PromptError("demo/dev pool version or split is invalid")
    _reject_example_pilot_collisions(demo_dev_records, evaluation_records)


def example_ids_for_variant(
    selection: dict[str, Any],
    scenario: str,
    prompt_version: str,
) -> list[str]:
    """Return exactly 3+1 or 5+2 frozen example IDs."""
    settings = selection["scenarios"][scenario]
    positives = settings["positive_example_ids"]
    boundaries = settings["boundary_example_ids"]
    if prompt_version == "fewshot_4_v2":
        return positives[:3] + boundaries[:1]
    if prompt_version == "fewshot_7_v2":
        return positives + boundaries
    raise Week4PromptError(f"unsupported few-shot prompt version: {prompt_version}")


def render_week4_request(
    root: Path,
    scenario: str,
    input_context: dict[str, Any],
    *,
    prompt_version: str,
    selection: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Render optimized instructions and frozen few-shot turns."""
    if prompt_version not in WEEK4_PROMPT_VERSIONS:
        raise Week4PromptError(f"unsupported Week 4 prompt: {prompt_version}")
    rendered = render_standard_prompt(
        root,
        scenario,
        input_context,
        version="week4_optimized_v2",
    )
    example_ids = example_ids_for_variant(selection, scenario, prompt_version)
    example_records = [records_by_id[sample_id] for sample_id in example_ids]
    collage_path, collage_sha256 = _build_example_collage(
        root,
        scenario,
        prompt_version,
        selection["version"],
        example_records,
    )
    messages = [copy.deepcopy(rendered["messages"][0])]
    messages.append(
        {
            "role": "user",
            "content": _collage_example_parts(
                collage_path,
                example_records,
            ),
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "examples": [
                        {
                            "example_index": index,
                            "output": _demonstration_output(
                                scenario, record["annotation"]
                            ),
                        }
                        for index, record in enumerate(example_records, start=1)
                    ]
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    )
    messages.append(copy.deepcopy(rendered["messages"][1]))
    rendered["messages"] = messages
    rendered["prompt_version"] = prompt_version
    rendered["example_ids"] = example_ids
    rendered["example_count"] = len(example_ids)
    rendered["example_collage_path"] = str(collage_path.relative_to(root)).replace(
        "\\", "/"
    )
    rendered["example_collage_sha256"] = collage_sha256
    return rendered


def _demonstration_output(
    scenario: str,
    annotation: dict[str, Any],
) -> dict[str, Any]:
    """行程示例只保留人工金标字段，避免重复长行程挤占固定上下文。"""
    output = annotation_to_output(scenario, annotation)
    if scenario != "itinerary_planning":
        return output
    return {
        field: output[field]
        for field in (
            "style_preferences",
            "hard_constraints",
            "soft_constraints",
            "required_itinerary_elements",
        )
    }


def annotation_to_output(scenario: str, annotation: dict[str, Any]) -> dict[str, Any]:
    """Map existing gold fields into a Schema-valid demonstration response."""
    if scenario == "image_product_search":
        return {
            "business_category": annotation.get("business_category", "unknown"),
            "style_tags": list(annotation.get("style_tags") or []),
            "visible_facilities": list(annotation.get("visible_facilities") or []),
            "price_range": annotation.get("price_range", "unknown"),
            "observed_evidence": [],
            "inferred_attributes": [],
            "unknown_fields": _unknown_fields(
                annotation,
                ("business_category", "price_range"),
            ),
            "confidence": None,
        }
    if scenario == "after_sales":
        ocr = annotation.get("ocr_ground_truth")
        return {
            "issue_type": annotation.get("issue_type", "unknown"),
            "severity": annotation.get("severity", "unknown"),
            "issue_location": None,
            "key_information": list(annotation.get("key_information") or []),
            "ocr_text": list(ocr) if isinstance(ocr, list) and ocr else None,
            "observed_evidence": [],
            "unknown_fields": _unknown_fields(annotation, ("issue_type", "severity")),
            "confidence": None,
        }
    if scenario != "itinerary_planning":
        raise Week4PromptError(f"unsupported scenario: {scenario}")
    hard = list(annotation.get("hard_constraints") or [])
    soft = list(annotation.get("soft_constraints") or [])
    days = _requested_days(hard)
    return {
        "style_preferences": list(annotation.get("style_preferences") or []),
        "hard_constraints": hard,
        "soft_constraints": soft,
        "required_itinerary_elements": list(
            annotation.get("required_itinerary_elements") or []
        ),
        "itinerary": [
            {
                "day_index": day_index,
                "date": None,
                "summary": "按已识别约束安排",
                "activities": [
                    {
                        "start_time": None,
                        "end_time": None,
                        "place_name": None,
                        "activity": "按已识别约束安排",
                        "transport": None,
                        "source_evidence": [],
                    }
                ],
            }
            for day_index in range(1, days + 1)
        ],
        "constraint_check": [
            {
                "constraint": value,
                "constraint_type": constraint_type,
                "status": "unknown",
                "evidence": None,
            }
            for constraint_type, values in (("hard", hard), ("soft", soft))
            for value in values
        ],
        "observed_evidence": [],
        "unknown_fields": [],
        "confidence": None,
    }


def _collage_example_parts(
    collage_path: Path,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    descriptions = ["拼图中的编号与以下已完成金标示例一一对应："]
    for index, record in enumerate(records, start=1):
        text_constraints = record["input"].get("text_constraints")
        suffix = (
            f"；原始文字约束：{text_constraints}"
            if isinstance(text_constraints, str) and text_constraints
            else ""
        )
        descriptions.append(f"示例 {index}{suffix}")
    normalized = str(collage_path).replace("\\", "/")
    return [
        {"type": "text", "text": "\n".join(descriptions)},
        {
            "type": "image_url",
            "image_url": {"url": f"file://{normalized}"},
        },
    ]


def _build_example_collage(
    root: Path,
    scenario: str,
    prompt_version: str,
    selection_version: str,
    records: list[dict[str, Any]],
) -> tuple[Path, str]:
    """Build one deterministic collage so the fixed two-image runtime limit holds."""
    cell_size = 384
    columns = 2
    rows = (len(records) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_size, rows * cell_size), "white")
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records, start=1):
        image_ref = record["input"]["images"][0]["path"]
        image_path = Path(root) / image_ref
        with Image.open(image_path) as source:
            fitted = ImageOps.fit(
                source.convert("RGB"),
                (cell_size, cell_size),
                method=Image.Resampling.LANCZOS,
            )
        x = ((index - 1) % columns) * cell_size
        y = ((index - 1) // columns) * cell_size
        canvas.paste(fitted, (x, y))
        draw.rectangle((x, y, x + 52, y + 28), fill="white")
        draw.text((x + 8, y + 6), str(index), fill="black")
    output = (
        Path(root)
        / "outputs/week4/example_collages"
        / f"{scenario}_{prompt_version}_{selection_version}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with Image.open(output) as existing:
            if existing.convert("RGB").tobytes() != canvas.tobytes():
                raise Week4PromptError(f"existing collage differs: {output}")
    else:
        canvas.save(output, format="PNG", optimize=False)
    return output, hashlib.sha256(output.read_bytes()).hexdigest()


def _unknown_fields(
    annotation: dict[str, Any],
    scalar_fields: tuple[str, ...],
) -> list[str]:
    return [
        field
        for field in scalar_fields
        if annotation.get(field) in {None, "unknown"}
    ]


def _requested_days(hard_constraints: list[Any]) -> int:
    for value in hard_constraints:
        if not isinstance(value, str):
            continue
        match = re.search(r"([1-4])", value)
        if match:
            return int(match.group(1))
    return 1

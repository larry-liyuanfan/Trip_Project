"""Versioned baseline and standardized prompt loading for Week 3 evaluation."""

import json
from pathlib import Path
from typing import Any

from src.data.yelp_paths import parse_simple_yaml
from src.evaluation.scenarios import SCENARIOS
from src.evaluation.schema_validation import load_output_schema


class PromptConfigurationError(ValueError):
    """Raised when a prompt version or scenario configuration is invalid."""


def load_baseline_prompt(
    root: Path,
    scenario: str,
    version: str = "baseline_minimal_v1",
) -> str:
    """Load one physically separate minimal baseline task sentence."""
    _require_scenario(scenario)
    prompt_path = (
        Path(root)
        / "configs"
        / "evaluation"
        / "prompts"
        / version
        / f"{scenario}.txt"
    )
    if not prompt_path.is_file():
        raise PromptConfigurationError(f"baseline prompt does not exist: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise PromptConfigurationError(f"baseline prompt is empty: {prompt_path}")
    return prompt


def render_baseline_request(
    root: Path,
    scenario: str,
    input_context: dict[str, Any],
    version: str = "baseline_minimal_v1",
) -> dict[str, Any]:
    """Render an unoptimized baseline as an explicit multimodal request."""
    prompt = load_baseline_prompt(root, scenario, version)
    content = [{"type": "text", "text": prompt}]
    content.extend(_render_input_parts(scenario, input_context))
    return {
        "prompt_version": version,
        "scenario": scenario,
        "messages": [{"role": "user", "content": content}],
    }


def render_standard_prompt(
    root: Path,
    scenario: str,
    input_context: dict[str, Any],
    version: str = "standardized_v1",
) -> dict[str, Any]:
    """Render four prompt layers and OpenAI-compatible chat messages."""
    _require_scenario(scenario)
    prompt_root = Path(root) / "configs" / "evaluation" / "prompts" / version
    common = _load_yaml_mapping(prompt_root / "common.yaml")
    scenario_spec = _load_yaml_mapping(prompt_root / f"{scenario}.yaml")
    if common.get("prompt_version") != version:
        raise PromptConfigurationError(f"common prompt version must be {version}")
    if scenario_spec.get("scenario") != scenario:
        raise PromptConfigurationError(f"scenario prompt must declare {scenario}")

    system_role = _require_text(common, "system_role")
    task_instruction = _require_text(scenario_spec, "task_instruction")
    schema_name = _require_text(scenario_spec, "schema_name")
    output_schema = load_output_schema(root, scenario)
    if output_schema.get("$id") != schema_name:
        raise PromptConfigurationError(
            f"schema_name {schema_name!r} does not match loaded Schema $id"
        )
    serialized_schema = json.dumps(
        output_schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    output_template = _require_text(common, "output_constraint")
    output_constraint = output_template.format(
        schema_name=schema_name,
        schema_contract=serialized_schema,
    )
    serialized_context = json.dumps(
        input_context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    layers = {
        "system_role": system_role,
        "task_instruction": task_instruction,
        "input_context": serialized_context,
        "output_constraint": output_constraint,
    }
    user_content = [{"type": "text", "text": f"任务说明：{task_instruction}"}]
    user_content.extend(_render_input_parts(scenario, input_context))
    user_content.append({"type": "text", "text": f"输出约束：{output_constraint}"})
    return {
        "prompt_version": version,
        "scenario": scenario,
        "schema_name": schema_name,
        "output_schema": output_schema,
        "layers": layers,
        "messages": [
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_content},
        ],
    }


def _render_input_parts(
    scenario: str,
    input_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Place images in manifest order, then append the raw text constraint."""
    if not isinstance(input_context, dict):
        raise PromptConfigurationError("input_context must be a mapping")
    images = input_context.get("images")
    if not isinstance(images, list):
        raise PromptConfigurationError("input_context.images must be an array")
    if scenario in {"image_product_search", "after_sales"} and len(images) != 1:
        raise PromptConfigurationError(f"{scenario} requires exactly one input image")
    if scenario == "itinerary_planning" and not images:
        raise PromptConfigurationError(
            "itinerary_planning requires at least one input image"
        )

    parts: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        if not isinstance(image, dict):
            raise PromptConfigurationError("each input image must be a mapping")
        path = image.get("path")
        if not isinstance(path, str) or not path.strip():
            raise PromptConfigurationError("each input image path must be non-empty")
        parts.append(
            {"type": "text", "text": f"参考图片占位符 <image_{index}>"}
        )
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": _as_image_url(path.strip())},
            }
        )

    text_constraints = input_context.get("text_constraints")
    if scenario == "itinerary_planning":
        if not isinstance(text_constraints, str) or not text_constraints.strip():
            raise PromptConfigurationError(
                "itinerary_planning requires non-empty input_context.text_constraints"
            )
        parts.append(
            {
                "type": "text",
                "text": f"原始文字约束：{text_constraints.strip()}",
            }
        )
    elif text_constraints is not None:
        raise PromptConfigurationError(
            f"{scenario} requires null input_context.text_constraints"
        )
    return parts


def _as_image_url(path: str) -> str:
    """Preserve URLs and make repository-relative image paths explicit file URLs."""
    if "://" in path or path.startswith("data:"):
        return path
    normalized = path.replace("\\", "/")
    return f"file://{normalized}"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PromptConfigurationError(f"standard prompt file does not exist: {path}")
    payload = parse_simple_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PromptConfigurationError(f"standard prompt file must be a mapping: {path}")
    return payload


def _require_text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PromptConfigurationError(f"prompt field {field} must be a non-empty string")
    return value.strip()


def _require_scenario(scenario: str) -> None:
    if scenario not in SCENARIOS:
        raise PromptConfigurationError(f"unsupported evaluation scenario: {scenario}")

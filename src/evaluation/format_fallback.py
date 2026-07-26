"""Bounded JSON parsing and existing-Schema validation fallback."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.evaluation.schema_validation import SchemaValidationError, validate_output


FENCE_PATTERN = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*\s*\Z",
    re.IGNORECASE,
)


def parse_with_schema_fallback(
    root: Path,
    scenario: str,
    raw_output: str,
) -> dict[str, Any]:
    """Remove one optional fence, parse JSON, and validate without repair."""
    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be text")
    match = FENCE_PATTERN.fullmatch(raw_output)
    candidate = match.group("body") if match else raw_output
    result: dict[str, Any] = {
        "scenario": scenario,
        "raw_output": raw_output,
        "fence_removed": match is not None,
        "parsed_output": None,
        "json_valid": False,
        "schema_valid": False,
        "error": None,
    }
    try:
        parsed = json.loads(candidate, parse_constant=_reject_constant)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        result["error"] = f"json_parse_error: {exc}"
        return result
    result["parsed_output"] = parsed
    result["json_valid"] = True
    try:
        validate_output(
            root,
            scenario,
            parsed,
            "v2" if scenario == "itinerary_planning" else "v1",
        )
    except SchemaValidationError as exc:
        result["error"] = f"schema_validation_error: {exc}"
        return result
    result["schema_valid"] = True
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")

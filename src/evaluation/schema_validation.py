"""Load and validate Week 3 scenario output schemas without GPU dependencies."""

import json
import math
from pathlib import Path
from typing import Any

from src.evaluation.scenarios import SCENARIOS


class SchemaValidationError(ValueError):
    """Raised when a schema file or output instance violates its contract."""


def load_output_schema(
    root: Path,
    scenario: str,
    version: str = "v1",
) -> dict[str, Any]:
    """Load one versioned scenario JSON Schema from repository configuration."""
    if scenario not in SCENARIOS:
        raise SchemaValidationError(f"unsupported evaluation scenario: {scenario}")
    schema_path = (
        Path(root)
        / "configs"
        / "evaluation"
        / "schemas"
        / f"{scenario}_{version}.schema.json"
    )
    if not schema_path.is_file():
        raise SchemaValidationError(f"output schema does not exist: {schema_path}")
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"invalid JSON Schema: {schema_path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"JSON Schema must be an object: {schema_path}")
    return payload


def validate_output(
    root: Path,
    scenario: str,
    payload: Any,
    version: str = "v1",
) -> Any:
    """Validate one parsed output against the project's supported Schema subset."""
    schema = load_output_schema(root, scenario, version)
    _validate_instance(schema, payload, path="$")
    return payload


def _validate_instance(schema: dict[str, Any], value: Any, *, path: str) -> None:
    expected_type = schema.get("type")
    if expected_type is not None:
        allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_matches_type(value, item) for item in allowed_types):
            expected = "|".join(str(item) for item in allowed_types)
            raise SchemaValidationError(f"{path} must have type {expected}")
        if value is None:
            return

    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path} must match enum {schema['enum']}")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [field for field in required if field not in value]
        if missing:
            raise SchemaValidationError(f"{path} missing required properties: {missing}")
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise SchemaValidationError(
                    f"{path} additional properties are not allowed: {extras}"
                )
        for field, child in value.items():
            if field in properties:
                _validate_instance(properties[field], child, path=f"{path}.{field}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SchemaValidationError(f"{path} has fewer than minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaValidationError(f"{path} has more than maxItems")
        if schema.get("uniqueItems"):
            serialized = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in value]
            if len(serialized) != len(set(serialized)):
                raise SchemaValidationError(f"{path} array items must be unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_instance(item_schema, item, path=f"{path}[{index}]")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise SchemaValidationError(f"{path} is shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise SchemaValidationError(f"{path} is longer than maxLength")

    if _matches_type(value, "number"):
        if isinstance(value, float) and not math.isfinite(value):
            raise SchemaValidationError(f"{path} must be a finite number")
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(f"{path} is above maximum")


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    raise SchemaValidationError(f"unsupported JSON Schema type: {expected}")

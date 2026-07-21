"""Validation helpers for model-server readiness evidence."""

from typing import Any


class ReadinessValidationError(ValueError):
    """Raised when a readiness response is not a usable model listing."""


def parse_models_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or payload.get("object") != "list":
        raise ReadinessValidationError("/v1/models payload must be an object list")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ReadinessValidationError("/v1/models payload must contain at least one model")
    model_ids: list[str] = []
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(model_id, str) or not model_id.strip():
            raise ReadinessValidationError("each /v1/models item requires a non-empty id")
        model_ids.append(model_id)
    return model_ids

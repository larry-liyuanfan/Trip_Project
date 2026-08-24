"""OpenAI-compatible multimodal client, response normalization, and fallback."""

import json
import os
from pathlib import Path
from typing import Any

import requests

from src.inference.prompts import get_image_understanding_prompt
from src.inference.transport_utils import normalize_image_url, strip_json_fence
from src.inference.schemas import (
    ImageUnderstandingRequest,
    ImageUnderstandingResponse,
    StructuredImageInfo,
)


class OpenAICompatibleClient:
    """Call a local vLLM or hosted OpenAI-compatible multimodal endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        """Configure the endpoint, served model name, and request timeout."""
        self.base_url = (
            base_url
            or os.getenv("MODEL_API_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or os.getenv("VLLM_BASE_URL")
            or ""
        ).rstrip("/")
        self.model_name = (
            model_name
            or os.getenv("MODEL_NAME")
            or os.getenv("VLLM_MODEL_NAME")
            or "Qwen/Qwen3-VL-8B-Instruct"
        )
        self.api_key = api_key or _read_api_key()
        self.timeout_seconds = timeout_seconds
        production = os.getenv("APP_ENV", "").strip().lower() == "production"
        self.fallback_enabled = _env_flag(
            "MODEL_FALLBACK_ENABLED",
            default=not production,
        )

    def understand_images(
        self, request: ImageUnderstandingRequest
    ) -> ImageUnderstandingResponse:
        """Call the configured endpoint, otherwise return deterministic output."""
        if not self.base_url:
            if self.fallback_enabled:
                return fallback_image_understanding(request)
            raise RuntimeError("model endpoint is not configured")

        payload = self._build_chat_payload(request)
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = requests.post(
                self.chat_completions_url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return parse_model_response(content)
        except Exception as exc:
            if not self.fallback_enabled:
                raise RuntimeError("model request failed") from exc
            fallback = fallback_image_understanding(request)
            fallback.raw_model_output = f"model request failed; fallback used: {exc}"
            return fallback

    @property
    def chat_completions_url(self) -> str:
        """Build the chat-completions URL for both root and versioned base URLs."""
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def _build_chat_payload(self, request: ImageUnderstandingRequest) -> dict[str, Any]:
        """Encode text and image parts in the OpenAI multimodal message format."""
        content: list[dict[str, Any]] = [
            {"type": "text", "text": get_image_understanding_prompt(request.prompt_version)}
        ]
        if request.user_text:
            content.append({"type": "text", "text": request.user_text})
        for image_url in request.image_urls:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": normalize_image_url(image_url)},
                }
            )

        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 512,
            "enable_thinking": False,
        }


# Preserve the existing public name used by routes and historical tests.
VLLMClient = OpenAICompatibleClient


def _read_api_key() -> str:
    """Read a model API key from the environment or a mounted secret file."""
    direct_key = os.getenv("MODEL_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if direct_key:
        return direct_key.strip()

    key_file = os.getenv("MODEL_API_KEY_FILE") or os.getenv("DASHSCOPE_API_KEY_FILE")
    if not key_file:
        return ""
    return Path(key_file).read_text(encoding="utf-8").strip()


def _env_flag(name: str, *, default: bool) -> bool:
    """Parse a conventional boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_model_response(content: str) -> ImageUnderstandingResponse:
    """Parse model JSON while preserving raw text when structured parsing fails."""
    json_content = strip_json_fence(content)
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError:
        return ImageUnderstandingResponse(
            image_summary=content,
            structured_info=StructuredImageInfo(),
            confidence=0.3,
            raw_model_output=content,
        )

    structured = normalize_structured_info(data.get("structured_info", data))
    return ImageUnderstandingResponse(
        image_summary=data.get("image_summary", ""),
        structured_info=StructuredImageInfo(**structured),
        confidence=float(data.get("confidence", structured.get("confidence", 0.5))),
        raw_model_output=content,
    )


def normalize_structured_info(structured: dict[str, Any]) -> dict[str, Any]:
    """Coerce model-dependent field shapes into the stable Pydantic contract."""
    normalized = dict(structured)
    normalized["objects"] = normalize_objects(normalized.get("objects", []))
    for key in ["style", "ocr_text", "location_clues", "travel_intent"]:
        normalized[key] = ensure_list(normalized.get(key, []))
    for key in ["merchant_type", "poi_type", "scene"]:
        normalized[key] = normalize_optional_text(normalized.get(key))
    normalized.pop("confidence", None)
    normalized.pop("image_summary", None)
    return normalized


def normalize_optional_text(value: Any) -> str | None:
    """将模型偶发返回的空列表或单元素列表收敛为可选文本。"""
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, list):
        return str(value[0]) if value[0] is not None else None
    return str(value)


def normalize_objects(value: Any) -> list[str]:
    """Reduce string or object-style visual detections to object labels."""
    items = value if isinstance(value, list) else [value]
    objects: list[str] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            object_type = item.get("type") or item.get("name") or item.get("label")
            if object_type:
                objects.append(str(object_type))
        else:
            objects.append(str(item))
    return objects


def ensure_list(value: Any) -> list[str]:
    """Normalize an optional scalar or sequence into a string list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def fallback_image_understanding(
    request: ImageUnderstandingRequest,
) -> ImageUnderstandingResponse:
    """Return predictable structured signals for tests without live vLLM."""
    joined = " ".join(request.image_urls + [request.user_text or ""]).lower()
    if "museum" in joined:
        return ImageUnderstandingResponse(
            image_summary="Image likely shows a museum or cultural attraction.",
            structured_info=StructuredImageInfo(
                objects=["building", "exhibition", "indoor space"],
                merchant_type="museum",
                poi_type="attraction",
                scene="museum exhibition",
                style=["cultural", "educational"],
                travel_intent=["museum visit", "city walk"],
            ),
            confidence=0.62,
        )

    return ImageUnderstandingResponse(
        image_summary="Image likely shows a cozy cafe scene suitable for OTA visual search.",
        structured_info=StructuredImageInfo(
            objects=["coffee", "table", "indoor seating"],
            merchant_type="cafe",
            poi_type="food_and_drink",
            scene="indoor cafe",
            style=["cozy", "minimal", "warm lighting"],
            ocr_text=[],
            location_clues=[],
            travel_intent=["coffee break", "afternoon tea", "solo work"],
        ),
        confidence=0.78,
    )

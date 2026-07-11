"""OpenAI-compatible vLLM client, response normalization, and local fallback."""

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests

from src.inference.prompts import get_image_understanding_prompt
from src.inference.schemas import (
    ImageUnderstandingRequest,
    ImageUnderstandingResponse,
    StructuredImageInfo,
)


class VLLMClient:
    """Thin OpenAI-compatible vLLM client with a deterministic local fallback."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        """Configure the endpoint, served model name, and request timeout."""
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL") or "").rstrip("/")
        self.model_name = model_name or os.getenv("VLLM_MODEL_NAME", "Qwen3-VL-2B-Instruct")
        self.timeout_seconds = timeout_seconds

    def understand_images(
        self, request: ImageUnderstandingRequest
    ) -> ImageUnderstandingResponse:
        """Call live vLLM when configured, otherwise return deterministic output."""
        if not self.base_url:
            return fallback_image_understanding(request)

        payload = self._build_chat_payload(request)
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return parse_model_response(content)
        except Exception as exc:
            fallback = fallback_image_understanding(request)
            fallback.raw_model_output = f"vLLM request failed; fallback used: {exc}"
            return fallback

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
        }


def normalize_image_url(image_url: str) -> str:
    """Convert an existing local file URL to a data URL for container access."""
    if not image_url.startswith("file://"):
        return image_url

    parsed = urlparse(image_url)
    raw_path = _file_url_to_path_text(parsed.netloc, unquote(parsed.path))
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / raw_path
    if not path.exists():
        return image_url

    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _file_url_to_path_text(netloc: str, path: str) -> str:
    """Normalize POSIX, relative, UNC-like, and Windows-drive file URL forms."""
    if netloc in ("", "localhost"):
        if len(path) >= 3 and path[0] == "/" and path[2] == ":":
            return path[1:]
        return path.lstrip("/") if not Path(path).is_absolute() else path
    if len(netloc) == 2 and netloc[1] == ":":
        return f"{netloc}{path}"
    return f"{netloc}{path}"


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


def strip_json_fence(content: str) -> str:
    """Remove an optional Markdown JSON fence around model output."""
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def normalize_structured_info(structured: dict[str, Any]) -> dict[str, Any]:
    """Coerce model-dependent field shapes into the stable Pydantic contract."""
    normalized = dict(structured)
    normalized["objects"] = normalize_objects(normalized.get("objects", []))
    for key in ["style", "ocr_text", "location_clues", "travel_intent"]:
        normalized[key] = ensure_list(normalized.get(key, []))
    normalized.pop("confidence", None)
    normalized.pop("image_summary", None)
    return normalized


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

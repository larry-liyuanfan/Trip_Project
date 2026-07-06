import json
import os
from typing import Any

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
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL") or "").rstrip("/")
        self.model_name = model_name or os.getenv("VLLM_MODEL_NAME", "Qwen3-VL-2B-Instruct")
        self.timeout_seconds = timeout_seconds

    def understand_images(
        self, request: ImageUnderstandingRequest
    ) -> ImageUnderstandingResponse:
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
        content: list[dict[str, Any]] = [
            {"type": "text", "text": get_image_understanding_prompt(request.prompt_version)}
        ]
        if request.user_text:
            content.append({"type": "text", "text": request.user_text})
        for image_url in request.image_urls:
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 512,
        }


def parse_model_response(content: str) -> ImageUnderstandingResponse:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return ImageUnderstandingResponse(
            image_summary=content,
            structured_info=StructuredImageInfo(),
            confidence=0.3,
            raw_model_output=content,
        )

    structured = data.get("structured_info", data)
    return ImageUnderstandingResponse(
        image_summary=data.get("image_summary", ""),
        structured_info=StructuredImageInfo(**structured),
        confidence=float(data.get("confidence", structured.get("confidence", 0.5))),
        raw_model_output=content,
    )


def fallback_image_understanding(
    request: ImageUnderstandingRequest,
) -> ImageUnderstandingResponse:
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


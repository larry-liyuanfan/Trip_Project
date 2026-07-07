import os
from typing import Any

from fastapi import APIRouter

from src.inference.client import VLLMClient
from src.inference.schemas import ImageUnderstandingRequest

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "ota-multimodal-search-planning",
        "model": os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "backend": "vLLM",
        "version": "0.1.0",
    }


@router.post("/v1/image-understanding")
def image_understanding(request: ImageUnderstandingRequest) -> dict[str, Any]:
    response = VLLMClient().understand_images(request)
    return response.model_dump()

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.inference.client import VLLMClient
from src.inference.schemas import (
    ImageUnderstandingRequest,
    TravelPlanningRequest,
    VisualSearchRequest,
)
from src.planning.itinerary_planner import build_itinerary
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.index_builder import load_jsonl

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "ota-multimodal-search-planning",
        "model": "Qwen3-VL-2B-Instruct",
        "backend": "vLLM",
        "version": "0.1.0",
    }


@router.post("/v1/image-understanding")
def image_understanding(request: ImageUnderstandingRequest) -> dict[str, Any]:
    response = VLLMClient().understand_images(request)
    return response.model_dump()


@router.post("/v1/visual-search")
def visual_search(request: VisualSearchRequest) -> dict[str, Any]:
    understanding = VLLMClient().understand_images(
        ImageUnderstandingRequest(
            image_urls=request.image_urls,
            user_text=request.query_text,
            language="zh",
        )
    )
    query_terms = " ".join(
        [
            request.query_text,
            understanding.structured_info.merchant_type or "",
            understanding.structured_info.scene or "",
            " ".join(understanding.structured_info.style),
        ]
    )
    catalog = _load_sample_catalog()
    results = HybridRetriever(catalog).search(query_terms, top_k=request.top_k)
    return {
        "query_understanding": {
            "merchant_type": understanding.structured_info.merchant_type,
            "scene": understanding.structured_info.scene,
            "intent": request.query_text,
        },
        "results": results,
    }


@router.post("/v1/travel-planning")
def travel_planning(request: TravelPlanningRequest) -> dict[str, Any]:
    catalog = _load_sample_catalog()
    query = " ".join(request.reviews + request.preferences.get("interests", []))
    candidates = HybridRetriever(catalog).search(query, top_k=4) or catalog[:2]
    return build_itinerary(candidates, request.preferences)


def _load_sample_catalog() -> list[dict[str, Any]]:
    path = Path("data/samples/poi_catalog.jsonl")
    if not path.exists():
        return []
    return load_jsonl(path)


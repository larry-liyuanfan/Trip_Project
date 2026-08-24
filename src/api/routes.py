"""Thin API routes that delegate inference, retrieval, and planning work."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.inference.client import VLLMClient
from src.inference.schemas import (
    DialogueRequest,
    ImageUnderstandingRequest,
    TaskRequest,
    TravelPlanningRequest,
    VisualSearchRequest,
)
from src.inference.system_runtime import (
    ModelGenerationError,
    RuntimeConfigurationError,
    ScenarioService,
    build_service,
)
from src.planning.itinerary_planner import build_itinerary
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.index_builder import load_jsonl

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Report process liveness without claiming dependency readiness."""
    return {
        "status": "ok",
        "service": "ota-multimodal-search-planning",
        "model": os.getenv("MODEL_NAME")
        or os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"),
        "backend": os.getenv("MODEL_PROVIDER", "transformers-peft"),
        "version": "1.0.0-rc1",
    }


@router.get("/ready")
def readiness() -> dict[str, Any]:
    """Require the exact release identity and live model backend."""
    try:
        payload = get_scenario_service().ready()
    except RuntimeConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if payload["status"] != "ready":
        raise HTTPException(status_code=503, detail=payload)
    return payload


@router.post("/v1/image-understanding")
def image_understanding(request: ImageUnderstandingRequest) -> dict[str, Any]:
    """Extract structured travel signals from one or more images."""
    response = VLLMClient().understand_images(request)
    return response.model_dump()


@router.post("/v1/tasks/image-product-search")
def image_product_search(request: TaskRequest) -> dict[str, Any]:
    """Run the release-locked image product understanding contract."""
    return _run_scenario("image_product_search", request)


@router.post("/v1/tasks/after-sales")
def after_sales(request: TaskRequest) -> dict[str, Any]:
    """Run the release-locked after-sales evidence contract."""
    return _run_scenario("after_sales", request)


@router.post("/v1/tasks/itinerary-planning")
def itinerary_planning(request: TaskRequest) -> dict[str, Any]:
    """Run the release-locked multimodal itinerary contract."""
    return _run_scenario("itinerary_planning", request)


@router.post("/v1/dialogue")
def dialogue(request: DialogueRequest) -> dict[str, Any]:
    """Expose dialogue only under the approved beta quality tier."""
    if not _env_flag("ENABLE_BETA_DIALOGUE", default=False):
        raise HTTPException(status_code=404, detail="DIALOGUE_BETA is disabled")
    try:
        return get_scenario_service().run_dialogue(request).model_dump()
    except RuntimeConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/v1/visual-search")
def visual_search(request: VisualSearchRequest) -> dict[str, Any]:
    """Combine VLM-derived terms with the current hybrid retrieval baseline."""
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
    """Retrieve candidate POIs and build a preference-aware sample itinerary."""
    catalog = _load_sample_catalog()
    query = " ".join(request.reviews + request.preferences.get("interests", []))
    candidates = HybridRetriever(catalog).search(query, top_k=4) or catalog[:2]
    return build_itinerary(candidates, request.preferences)


def _load_sample_catalog() -> list[dict[str, Any]]:
    """Load the lightweight checked-in catalog used before a full index exists."""
    path = Path("data/samples/poi_catalog.jsonl")
    if not path.exists():
        return []
    return load_jsonl(path)


@lru_cache(maxsize=1)
def get_scenario_service() -> ScenarioService:
    """Create one lazy process-local model service."""
    return build_service()


def _run_scenario(scenario: str, request: TaskRequest) -> dict[str, Any]:
    try:
        return get_scenario_service().run_task(scenario, request).model_dump()
    except RuntimeConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

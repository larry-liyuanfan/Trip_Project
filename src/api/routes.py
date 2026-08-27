"""Thin API routes that delegate inference, retrieval, and planning work."""

import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, HTTPException

from src.inference.client import VLLMClient
from src.inference.schemas import (
    DialogueRequest,
    ImageUnderstandingRequest,
    TaskRequest,
    SingleImageTaskRequest,
    ItineraryTaskRequest,
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
from src.retrieval.clip_embeddings import CLIPImageEncoder
from src.retrieval.milvus_vectors import OTAMilvusVectorStore, load_milvus_config
from src.retrieval.visual_search import VisualSearchService
from src.retrieval.query_inputs import user_query_attributes, unapplied_query_text

router = APIRouter()
_scenario_service_lock = Lock()
_visual_search_service_lock = Lock()


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
    retrieval_checks = _retrieval_readiness()
    payload["checks"].update(retrieval_checks)
    if not all(check["ok"] for check in retrieval_checks.values()):
        payload["status"] = "not_ready"
    if payload["status"] != "ready":
        raise HTTPException(status_code=503, detail=payload)
    return payload


@router.post("/v1/image-understanding")
def image_understanding(request: ImageUnderstandingRequest) -> dict[str, Any]:
    """Extract structured travel signals from one or more images."""
    try:
        response = VLLMClient().understand_images(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="legacy model endpoint is unavailable") from exc
    return response.model_dump()


@router.post("/v1/tasks/image-product-search")
def image_product_search(request: SingleImageTaskRequest) -> dict[str, Any]:
    """Run the release-locked image product understanding contract."""
    return _run_scenario("image_product_search", request)


@router.post("/v1/tasks/after-sales")
def after_sales(request: SingleImageTaskRequest) -> dict[str, Any]:
    """Run the release-locked after-sales evidence contract."""
    return _run_scenario("after_sales", request)


@router.post("/v1/tasks/itinerary-planning")
def itinerary_planning(request: ItineraryTaskRequest) -> dict[str, Any]:
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
    if os.getenv("APP_ENV", "").strip().lower() == "production":
        if not request.image_urls and not request.query_text.strip():
            raise HTTPException(status_code=422, detail="search requires an image or query text")
        if len(request.image_urls) > 1 or (request.retrieval_mode == "embedding" and not request.image_urls):
            raise HTTPException(
                status_code=422,
                detail="embedding search requires one image; keyword/hybrid accept zero or one",
            )
        try:
            image_path = _local_image_path(request.image_urls[0]) if request.image_urls else None
        except RuntimeConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            service = get_visual_search_service()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"visual-search dependencies are unavailable: {exc}",
            ) from exc
        filters = {key: getattr(request, key) for key in ("city", "business_category", "price_range") if getattr(request, key)}
        try:
            results = service.search(
                image_path,
                top_k=request.top_k,
                filters=filters,
                query_text=request.query_text,
                retrieval_mode=request.retrieval_mode,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"visual search failed: {exc}") from exc
        query_attributes = user_query_attributes(request.query_text, filters)
        unapplied = unapplied_query_text(request.query_text, query_attributes)
        return {
            "retrieval_mode": "clip_milvus_hnsw_cosine" if request.retrieval_mode == "embedding" else request.retrieval_mode,
            "query_text": request.query_text,
            "query_attributes": query_attributes,
            "query_status": "PARTIAL_UNSUPPORTED_CONSTRAINTS" if unapplied else "COMPLETED",
            "unapplied_query_text": unapplied,
            "text_interpretation": "structured_category_price_and_explicit_city_only",
            "embedding_model": "openai/clip-vit-base-patch32",
            "results": results,
        }
    try:
        understanding = VLLMClient().understand_images(
            ImageUnderstandingRequest(
                image_urls=request.image_urls,
                user_text=request.query_text,
                language="zh",
            )
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="legacy model endpoint is unavailable") from exc
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
    if os.getenv("APP_ENV", "").strip().lower() == "production":
        raise HTTPException(
            status_code=404,
            detail="sample-catalog planner is disabled in production; use /v1/tasks/itinerary-planning",
        )
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


def get_scenario_service() -> ScenarioService:
    """Create one lazy process-local model service."""
    # lru_cache 本身允许并发首次 miss 重复执行；必须在缓存查找外串行化。
    with _scenario_service_lock:
        return _cached_scenario_service()


@lru_cache(maxsize=1)
def _cached_scenario_service() -> ScenarioService:
    service = build_service()
    service.retrieval_runner = _run_dialogue_retrieval
    return service


def _run_dialogue_retrieval(text, images, state):
    try:
        return visual_search(VisualSearchRequest(query_text=text, image_urls=images,
                            city=state.get("city"), retrieval_mode="hybrid"))
    except HTTPException as exc:
        raise ModelGenerationError(f"retrieval failed ({exc.status_code}): {exc.detail}") from exc


def get_visual_search_service() -> VisualSearchService:
    """Build the configured CLIP/Milvus production retrieval service."""
    with _visual_search_service_lock:
        return _cached_visual_search_service()


@lru_cache(maxsize=1)
def _cached_visual_search_service() -> VisualSearchService:
    config_path = Path(
        os.getenv("MILVUS_CONFIG", "docker/system/milvus_system.yaml")
    )
    config = load_milvus_config(config_path)
    encoder = CLIPImageEncoder(
        device=os.getenv("CLIP_DEVICE", "auto"),
        batch_size=int(os.getenv("CLIP_BATCH_SIZE", "20")),
    )
    return VisualSearchService(encoder, OTAMilvusVectorStore(config))


def _retrieval_readiness() -> dict[str, dict[str, Any]]:
    try:
        service = get_visual_search_service()
    except Exception as exc:
        reason = f"retrieval initialization failed: {exc}"
        return {
            "clip": {"ok": False, "detail": reason},
            "milvus": {"ok": False, "detail": reason},
        }
    clip_ready, clip_reason = service.ready()
    try:
        milvus_ready, milvus_reason = service.store.ready()
    except Exception as exc:
        milvus_ready, milvus_reason = False, f"Milvus readiness failed: {exc}"
    return {
        "clip": {"ok": clip_ready, "detail": clip_reason},
        "milvus": {"ok": milvus_ready, "detail": milvus_reason},
    }


def _run_scenario(scenario: str, request: TaskRequest) -> dict[str, Any]:
    from pydantic import ValidationError
    try:
        contract = ItineraryTaskRequest if scenario == "itinerary_planning" else SingleImageTaskRequest
        request = contract.model_validate(request.model_dump())
        return get_scenario_service().run_task(scenario, request).model_dump()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _local_image_path(value: str) -> Path:
    if value.startswith("file://"):
        parsed = urlparse(value)
        path_text = unquote(parsed.path)
        if len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":":
            path_text = path_text[1:]
        path = Path(path_text)
    elif "://" in value or value.startswith("data:"):
        raise RuntimeConfigurationError(
            "production visual search accepts mounted local image paths only"
        )
    else:
        path = Path(value)
    resolved = path if path.is_absolute() else Path.cwd() / path
    if not resolved.is_file():
        raise RuntimeConfigurationError(f"visual-search image does not exist: {resolved}")
    return resolved.resolve()

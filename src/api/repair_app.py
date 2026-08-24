"""GPU-local inference service for immutable system-repair experiments."""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.schemas import TaskRequest
from src.inference.system_runtime import (
    ModelGenerationError,
    ReleaseSettings,
    RuntimeConfigurationError,
    ScenarioService,
    TransformersPeftBackend,
)


class RepairCompletionRequest(BaseModel):
    """Bounded OpenAI-compatible request used by the prompt pilot."""

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int = Field(default=3072, ge=1, le=4096)
    response_format: dict[str, Any] | None = None


def create_repair_app(
    *,
    settings: ReleaseSettings | None = None,
    backend: TransformersPeftBackend | None = None,
) -> FastAPI:
    """Create a fail-closed app without retrieval or legacy fallback routes."""

    active_settings = settings or ReleaseSettings.load()
    active_backend = backend or TransformersPeftBackend(active_settings)
    service = ScenarioService(active_settings, active_backend)
    app = FastAPI(title="Trip system repair inference", version="1")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "release_id": active_settings.release_id}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        ready, reason = active_backend.ready()
        if not ready:
            raise HTTPException(status_code=503, detail=reason)
        return {
            "object": "list",
            "data": [
                {
                    "id": active_settings.adapter_name,
                    "object": "model",
                    "owned_by": "trip-project",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def completion(request: RepairCompletionRequest) -> dict[str, Any]:
        if request.model != active_settings.adapter_name:
            raise HTTPException(status_code=422, detail="served model identity mismatch")
        started = time.time()
        try:
            result = active_backend.generate_with_usage(
                request.messages,
                response_format=request.response_format,
                max_new_tokens=request.max_tokens,
            )
        except (RuntimeConfigurationError, ModelGenerationError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "id": f"repair-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(started),
            "model": active_settings.adapter_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.input_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            },
        }

    def run_task(scenario: str, request: TaskRequest) -> dict[str, Any]:
        try:
            return service.run_task(scenario, request).model_dump()
        except RuntimeConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ModelGenerationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/tasks/image-product-search")
    def product(request: TaskRequest) -> dict[str, Any]:
        return run_task("image_product_search", request)

    @app.post("/v1/tasks/after-sales")
    def after_sales(request: TaskRequest) -> dict[str, Any]:
        return run_task("after_sales", request)

    @app.post("/v1/tasks/itinerary-planning")
    def itinerary(request: TaskRequest) -> dict[str, Any]:
        return run_task("itinerary_planning", request)

    return app


app = create_repair_app()

"""Typed API contracts for image understanding, retrieval, and planning."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ImageUnderstandingRequest(BaseModel):
    """Input images, optional user context, language, and prompt version."""
    image_urls: list[str] = Field(default_factory=list)
    user_text: str | None = None
    language: Literal["zh", "en"] = "zh"
    prompt_version: str = "prompt_image_understanding_v1"


class StructuredImageInfo(BaseModel):
    """Normalized visual and travel-search signals extracted by the VLM."""
    objects: list[str] = Field(default_factory=list)
    merchant_type: str | None = None
    poi_type: str | None = None
    scene: str | None = None
    style: list[str] = Field(default_factory=list)
    ocr_text: list[str] = Field(default_factory=list)
    location_clues: list[str] = Field(default_factory=list)
    travel_intent: list[str] = Field(default_factory=list)


class ImageUnderstandingResponse(BaseModel):
    """Structured image understanding plus confidence and optional raw output."""
    image_summary: str
    structured_info: StructuredImageInfo
    confidence: float = 0.0
    raw_model_output: str | None = None


class VisualSearchRequest(BaseModel):
    """Multimodal search input and requested result count."""
    image_urls: list[str] = Field(default_factory=list)
    query_text: str = ""
    city: str | None = None
    top_k: int = 5
    retrieval_mode: Literal["keyword", "embedding", "hybrid"] = "hybrid"


class TravelPlanningRequest(BaseModel):
    """Images, reviews, and preferences used to construct an itinerary."""
    image_urls: list[str] = Field(default_factory=list)
    reviews: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)


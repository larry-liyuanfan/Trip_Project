"""Typed API contracts for image understanding, retrieval, and planning."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    image_urls: list[str] = Field(default_factory=list, max_length=8)
    query_text: str = Field(default="", max_length=4000)
    city: str | None = Field(default=None, max_length=128)
    top_k: int = Field(default=5, ge=1, le=100, strict=True)
    retrieval_mode: Literal["keyword", "embedding", "hybrid"] = "hybrid"


class TravelPlanningRequest(BaseModel):
    """Images, reviews, and preferences used to construct an itinerary."""
    image_urls: list[str] = Field(default_factory=list)
    reviews: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)


class TaskRequest(BaseModel):
    """Input contract shared by the three production scenario endpoints."""

    image_urls: list[str] = Field(min_length=1, max_length=8)
    text_context: str | None = Field(default=None, max_length=4000)


class ModelAttempt(BaseModel):
    """One immutable raw generation attempt retained for diagnosis."""

    attempt: int
    raw_output: str
    error: str | None = None
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class TaskResponse(BaseModel):
    """Validated business result and model provenance."""

    scenario: Literal[
        "image_product_search",
        "after_sales",
        "itinerary_planning",
    ]
    result: dict[str, Any]
    schema_valid: bool
    prompt_version: str
    model: str
    adapter: str
    release_id: str
    attempts: list[ModelAttempt]
    total_latency_ms: float


class DialogueTurn(BaseModel):
    """One bounded dialogue turn supplied by the caller."""

    role: Literal["user", "assistant", "tool"]
    content: str = Field(min_length=1, max_length=8000)


class DialogueRequest(BaseModel):
    """Caller-managed dialogue history and explicit state."""

    messages: list[DialogueTurn] = Field(min_length=1, max_length=32)
    image_urls: list[str] = Field(default_factory=list, max_length=8)
    state: dict[str, Any] = Field(default_factory=dict)


class DialogueModelOutput(BaseModel):
    """Schema enforced on the model's beta dialogue response."""

    model_config = ConfigDict(extra="forbid")

    reply: str = Field(min_length=1, max_length=8000)
    state_updates: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class DialogueResponse(BaseModel):
    """Validated beta dialogue output with the merged caller-visible state."""

    reply: str
    state: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    quality_tier: Literal["DIALOGUE_BETA"] = "DIALOGUE_BETA"
    model: str
    adapter: str
    release_id: str
    attempts: list[ModelAttempt]
    total_latency_ms: float
    execution_mode: Literal[
        "MODEL_GENERATED_CONTRACT",
        "DETERMINISTIC_CONTRACT",
    ] = "MODEL_GENERATED_CONTRACT"
    semantic_fallback_status: Literal[
        "NOT_USED",
        "SUCCEEDED",
        "FAILED_SAFE",
    ] = "NOT_USED"


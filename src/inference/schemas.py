"""Typed API contracts for image understanding, retrieval, and planning."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    business_category: Literal["hotel", "restaurant", "attraction"] | None = None
    price_range: Literal["budget", "mid_range", "premium", "luxury"] | None = None


class TravelPlanningRequest(BaseModel):
    """Images, reviews, and preferences used to construct an itinerary."""
    image_urls: list[str] = Field(default_factory=list)
    reviews: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)


class TaskRequest(BaseModel):
    """Input contract shared by the three production scenario endpoints."""

    image_urls: list[str] = Field(min_length=1, max_length=8)
    text_context: str | None = Field(default=None, max_length=4000)

    @field_validator("image_urls")
    @classmethod
    def nonempty_images(cls, value):
        if any(not image.strip() for image in value):
            raise ValueError("image paths must not be empty")
        return value


class SingleImageTaskRequest(TaskRequest):
    image_urls: list[str] = Field(min_length=1, max_length=1)
    text_context: None = None


class ItineraryTaskRequest(TaskRequest):
    text_context: str = Field(min_length=1, max_length=4000)

    @field_validator("text_context")
    @classmethod
    def nonblank_constraints(cls, value):
        if not value.strip():
            raise ValueError("itinerary requires non-empty text constraints")
        return value


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
    business_valid: bool | None = None
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
    image_urls: list[str] = Field(default_factory=list, max_length=8)


class DialogueRequest(BaseModel):
    """Caller-managed dialogue history and explicit state."""

    messages: list[DialogueTurn] = Field(min_length=1, max_length=32)
    image_urls: list[str] = Field(default_factory=list, max_length=8)
    state: dict[str, Any] = Field(default_factory=dict)
    task: Literal["auto", "product", "itinerary", "retrieval", "conversation", "state_update"] = "auto"

    @model_validator(mode="after")
    def validate_image_history(self):
        images = [*self.image_urls, *(image for turn in self.messages for image in turn.image_urls)]
        if any(not image.strip() for image in images) or len(images) > 8:
            raise ValueError("dialogue accepts at most eight non-empty image references across all turns")
        if any(turn.image_urls and turn.role != "user" for turn in self.messages):
            raise ValueError("image references must belong to user turns")
        return self


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
    task_status: Literal["COMPLETED", "NOT_COMPLETED", "STATE_UPDATED"] = "NOT_COMPLETED"
    task_result: dict[str, Any] | None = None
    task_error: str | None = None
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


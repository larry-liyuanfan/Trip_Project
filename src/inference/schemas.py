from typing import Literal

from pydantic import BaseModel, Field


class ImageUnderstandingRequest(BaseModel):
    image_urls: list[str] = Field(default_factory=list)
    user_text: str | None = None
    language: Literal["zh", "en"] = "zh"
    prompt_version: str = "prompt_image_understanding_v1"


class StructuredImageInfo(BaseModel):
    objects: list[str] = Field(default_factory=list)
    merchant_type: str | None = None
    poi_type: str | None = None
    scene: str | None = None
    style: list[str] = Field(default_factory=list)
    ocr_text: list[str] = Field(default_factory=list)
    location_clues: list[str] = Field(default_factory=list)
    travel_intent: list[str] = Field(default_factory=list)


class ImageUnderstandingResponse(BaseModel):
    image_summary: str
    structured_info: StructuredImageInfo
    confidence: float = 0.0
    raw_model_output: str | None = None


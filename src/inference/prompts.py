"""Versioned prompts used by multimodal image-understanding requests."""

IMAGE_UNDERSTANDING_PROMPTS = {
    "prompt_image_understanding_v1": """You are an OTA multimodal search assistant.
Extract structured travel search signals from the image(s).
Return JSON with: objects, merchant_type, poi_type, scene, style, ocr_text,
location_clues, travel_intent, confidence, and image_summary.
Focus on restaurants, cafes, hotels, attractions, products, and travel POIs.""",
}


def get_image_understanding_prompt(version: str = "prompt_image_understanding_v1") -> str:
    """Resolve a prompt version with a stable default for unknown versions."""
    return IMAGE_UNDERSTANDING_PROMPTS.get(
        version, IMAGE_UNDERSTANDING_PROMPTS["prompt_image_understanding_v1"]
    )

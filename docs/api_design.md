# API Design

## GET /health

Response:

```json
{
  "status": "ok",
  "service": "ota-multimodal-search-planning",
  "model": "Qwen3-VL-2B-Instruct",
  "backend": "vLLM",
  "version": "0.1.0"
}
```

## POST /v1/image-understanding

Request:

```json
{
  "image_urls": ["file://data/samples/images/cafe_001.jpg"],
  "user_text": "这张图可能适合什么旅行场景？",
  "language": "zh",
  "prompt_version": "prompt_image_understanding_v1"
}
```

Response:

```json
{
  "image_summary": "Image likely shows a cozy cafe scene suitable for OTA visual search.",
  "structured_info": {
    "objects": ["coffee", "table", "indoor seating"],
    "merchant_type": "cafe",
    "poi_type": "food_and_drink",
    "scene": "indoor cafe",
    "style": ["cozy", "minimal", "warm lighting"],
    "ocr_text": [],
    "location_clues": [],
    "travel_intent": ["coffee break", "afternoon tea", "solo work"]
  },
  "confidence": 0.78
}
```

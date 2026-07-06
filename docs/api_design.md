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

## POST /v1/visual-search

Request:

```json
{
  "image_urls": ["file://data/samples/images/cafe_001.jpg"],
  "query_text": "找类似适合下午茶的咖啡店",
  "city": "Shanghai",
  "top_k": 5,
  "retrieval_mode": "hybrid"
}
```

Response:

```json
{
  "query_understanding": {
    "merchant_type": "cafe",
    "scene": "indoor cafe",
    "intent": "找类似适合下午茶的咖啡店"
  },
  "results": []
}
```

## POST /v1/travel-planning

Request:

```json
{
  "image_urls": [
    "file://data/samples/images/cafe_001.jpg",
    "file://data/samples/images/museum_001.jpg"
  ],
  "reviews": ["环境安静，适合下午坐一会儿。"],
  "preferences": {
    "city": "Shanghai",
    "duration": "1 day",
    "budget": "medium",
    "pace": "relaxed",
    "interests": ["coffee", "museum", "city walk"]
  }
}
```

Response:

```json
{
  "itinerary": [
    {
      "time": "10:00-12:00",
      "poi_name": "Sample Museum",
      "poi_type": "Museum",
      "reason": "Matches user interest in museum."
    }
  ],
  "summary": "Relaxed 1 day itinerary for Shanghai.",
  "assumptions": ["Sample POI catalog is used until Yelp data is integrated."],
  "confidence": 0.72
}
```


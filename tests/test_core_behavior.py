import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.api.routes import health, image_understanding
from src.inference.client import VLLMClient
from src.inference.client import parse_model_response
from src.inference.schemas import ImageUnderstandingRequest
from src.planning.itinerary_planner import build_itinerary
from src.retrieval.keyword_retriever import KeywordRetriever


class CoreBehaviorTest(unittest.TestCase):
    def test_health_endpoint_reports_service_metadata(self):
        body = health()

        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "ota-multimodal-search-planning")
        self.assertEqual(body["backend"], "vLLM")

    def test_health_endpoint_reports_configured_model_name(self):
        with patch.dict("os.environ", {"VLLM_MODEL_NAME": "Qwen/Qwen2.5-VL-3B-Instruct"}):
            body = health()

        self.assertEqual(body["model"], "Qwen/Qwen2.5-VL-3B-Instruct")

    def test_image_understanding_returns_structured_fields_without_live_vllm(self):
        payload = image_understanding(
            ImageUnderstandingRequest(
                image_urls=["file://data/samples/images/cafe_001.jpg"],
                user_text="这张图可能适合什么旅行场景？",
                language="zh",
            )
        )

        self.assertIn("image_summary", payload)
        self.assertEqual(payload["structured_info"]["merchant_type"], "cafe")
        self.assertEqual(payload["structured_info"]["poi_type"], "food_and_drink")
        self.assertGreaterEqual(payload["confidence"], 0)
        self.assertLessEqual(payload["confidence"], 1)

    def test_vllm_payload_encodes_local_file_urls_as_data_urls(self):
        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            image_path.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
                b"\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01"
                b"\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
            )

            payload = VLLMClient()._build_chat_payload(
                ImageUnderstandingRequest(image_urls=[image_path.as_uri()])
            )

        image_part = payload["messages"][0]["content"][-1]
        self.assertTrue(image_part["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_parse_model_response_accepts_fenced_json_and_scalar_lists(self):
        response = parse_model_response(
            """```json
{
  "objects": [{"type": "cafe", "confidence": 0.7}],
  "merchant_type": "cafe",
  "poi_type": "food_and_drink",
  "scene": "indoor",
  "style": "warm",
  "ocr_text": "Cafe",
  "location_clues": "Shanghai",
  "travel_intent": "coffee break",
  "confidence": 0.8,
  "image_summary": "A cafe image."
}
```"""
        )

        self.assertEqual(response.image_summary, "A cafe image.")
        self.assertEqual(response.structured_info.objects, ["cafe"])
        self.assertEqual(response.structured_info.style, ["warm"])
        self.assertEqual(response.structured_info.ocr_text, ["Cafe"])
        self.assertEqual(response.confidence, 0.8)

    def test_keyword_retriever_returns_ranked_matching_pois(self):
        catalog = [
            {
                "poi_id": "poi_001",
                "name": "Sample Cafe",
                "category": "Cafe",
                "tags": ["coffee", "quiet", "afternoon tea"],
                "description": "Cozy cafe for relaxed city walks.",
            },
            {
                "poi_id": "poi_002",
                "name": "Sample Museum",
                "category": "Museum",
                "tags": ["art", "exhibition"],
                "description": "Museum with cultural exhibitions.",
            },
        ]
        retriever = KeywordRetriever(catalog)

        results = retriever.search("quiet coffee afternoon", top_k=1)

        self.assertEqual(results[0]["poi_id"], "poi_001")
        self.assertGreater(results[0]["score"], 0)
        self.assertIn("coffee", results[0]["matched_reasons"])

    def test_travel_planning_uses_preferences_and_candidates(self):
        candidates = [
            {"poi_id": "poi_002", "name": "Sample Museum", "category": "Museum"},
            {"poi_id": "poi_001", "name": "Sample Cafe", "category": "Cafe"},
        ]

        itinerary = build_itinerary(
            candidates=candidates,
            preferences={
                "city": "Shanghai",
                "duration": "1 day",
                "pace": "relaxed",
                "interests": ["coffee", "museum"],
            },
        )

        self.assertTrue(itinerary["summary"].startswith("Relaxed 1 day itinerary"))
        self.assertEqual(len(itinerary["itinerary"]), 2)
        self.assertEqual(itinerary["itinerary"][0]["poi_name"], "Sample Museum")

    def test_sample_catalog_exists_and_is_jsonl(self):
        catalog_path = Path("data/samples/poi_catalog.jsonl")

        self.assertTrue(catalog_path.exists())
        self.assertTrue(catalog_path.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    unittest.main()

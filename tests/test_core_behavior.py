import unittest
from pathlib import Path

from src.api.routes import health, image_understanding
from src.inference.schemas import ImageUnderstandingRequest
from src.planning.itinerary_planner import build_itinerary
from src.retrieval.keyword_retriever import KeywordRetriever


class CoreBehaviorTest(unittest.TestCase):
    def test_health_endpoint_reports_service_metadata(self):
        body = health()

        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "ota-multimodal-search-planning")
        self.assertEqual(body["backend"], "vLLM")

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

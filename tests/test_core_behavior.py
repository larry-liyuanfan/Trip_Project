import unittest
import json
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.api.routes import health, image_understanding
from src.inference.client import VLLMClient
from src.inference.client import parse_model_response
from src.inference.schemas import ImageUnderstandingRequest
from src.data.yelp_archives import extract_yelp_archives, extract_yelp_photo_files
from src.planning.itinerary_planner import build_itinerary
from src.retrieval.keyword_retriever import KeywordRetriever
from src.data.yelp_open_dataset import prepare_yelp_subset


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

    def test_prepare_yelp_subset_filters_ota_businesses_and_writes_outputs(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            output_dir = Path(tmpdir) / "processed"
            raw_dir.mkdir()
            (raw_dir / "yelp_academic_dataset_business.json").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "business_id": "biz_cafe",
                                "name": "Sample Yelp Cafe",
                                "city": "Shanghai",
                                "state": "SH",
                                "stars": 4.5,
                                "review_count": 20,
                                "categories": "Cafes, Restaurants, Coffee & Tea",
                                "attributes": {"OutdoorSeating": "True"},
                                "is_open": 1,
                            }
                        ),
                        json.dumps(
                            {
                                "business_id": "biz_auto",
                                "name": "Sample Auto Repair",
                                "city": "Shanghai",
                                "state": "SH",
                                "stars": 4.0,
                                "review_count": 10,
                                "categories": "Auto Repair",
                                "is_open": 1,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            (raw_dir / "yelp_academic_dataset_review.json").write_text(
                json.dumps(
                    {
                        "review_id": "rev_1",
                        "business_id": "biz_cafe",
                        "stars": 5,
                        "text": "Quiet cafe near the museum, good for a relaxed afternoon.",
                        "date": "2024-01-01",
                    }
                ),
                encoding="utf-8",
            )
            (raw_dir / "photos.json").write_text(
                json.dumps(
                    {
                        "photo_id": "photo_1",
                        "business_id": "biz_cafe",
                        "caption": "latte and window seat",
                        "label": "inside",
                    }
                ),
                encoding="utf-8",
            )

            manifest = prepare_yelp_subset(raw_dir=raw_dir, output_dir=output_dir)

            catalog = [
                json.loads(line)
                for line in (output_dir / "poi_catalog.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            reviews = [
                json.loads(line)
                for line in (output_dir / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            multimodal = [
                json.loads(line)
                for line in (output_dir / "multimodal_items.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(manifest["business_count"], 1)
        self.assertEqual(catalog[0]["poi_id"], "yelp_biz_cafe")
        self.assertEqual(catalog[0]["category"], "Cafe")
        self.assertIn("coffee & tea", catalog[0]["tags"])
        self.assertEqual(reviews[0]["poi_id"], "yelp_biz_cafe")
        self.assertEqual(multimodal[0]["image_path"], "data/yelp/raw/photos/photo_1.jpg")

    def test_extract_yelp_archives_handles_gzipped_tar_files_inside_zip(self):
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            json_zip = workspace / "Yelp-JSON.zip"
            photos_zip = workspace / "Yelp-Photos.zip"
            raw_dir = workspace / "raw"

            self._write_zip_with_gzipped_tar(
                json_zip,
                "Yelp JSON/yelp_dataset.tar",
                {
                    "yelp_academic_dataset_business.json": {"business_id": "biz_1"},
                    "yelp_academic_dataset_review.json": {"review_id": "rev_1"},
                    "yelp_academic_dataset_user.json": {"user_id": "ignored"},
                },
            )
            self._write_zip_with_gzipped_tar(
                photos_zip,
                "Yelp Photos/yelp_photos.tar",
                {
                    "photos.json": {"photo_id": "photo_1", "business_id": "biz_1"},
                    "photos/photo_1.jpg": b"fake-image",
                },
            )

            manifest = extract_yelp_archives(
                json_zip_path=json_zip,
                photos_zip_path=photos_zip,
                raw_dir=raw_dir,
                include_photo_files=False,
            )

            self.assertTrue((raw_dir / "yelp_academic_dataset_business.json").exists())
            self.assertTrue((raw_dir / "yelp_academic_dataset_review.json").exists())
            self.assertTrue((raw_dir / "photos.json").exists())
            self.assertFalse((raw_dir / "photos" / "photo_1.jpg").exists())
            self.assertEqual(
                sorted(record["output_name"] for record in manifest["extracted_files"]),
                [
                    "photos.json",
                    "yelp_academic_dataset_business.json",
                    "yelp_academic_dataset_review.json",
                ],
            )

    def test_extract_yelp_photo_files_extracts_only_requested_images(self):
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            photos_zip = workspace / "Yelp-Photos.zip"
            raw_dir = workspace / "raw"

            self._write_zip_with_gzipped_tar(
                photos_zip,
                "Yelp Photos/yelp_photos.tar",
                {
                    "photos/photo_1.jpg": b"selected",
                    "photos/photo_2.jpg": b"ignored",
                },
            )

            manifest = extract_yelp_photo_files(
                photos_zip_path=photos_zip,
                raw_dir=raw_dir,
                photo_ids={"photo_1"},
            )

            self.assertEqual((raw_dir / "photos" / "photo_1.jpg").read_bytes(), b"selected")
            self.assertFalse((raw_dir / "photos" / "photo_2.jpg").exists())
            self.assertEqual(manifest["requested_photo_count"], 1)
            self.assertEqual(manifest["extracted_photo_count"], 1)

    def _write_zip_with_gzipped_tar(
        self,
        zip_path: Path,
        tar_name: str,
        records: dict[str, dict[str, object] | bytes],
    ) -> None:
        tar_buffer = BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as archive:
            for member_name, payload in records.items():
                if isinstance(payload, bytes):
                    body = payload
                else:
                    body = (json.dumps(payload) + "\n").encode("utf-8")
                info = tarfile.TarInfo(member_name)
                info.size = len(body)
                archive.addfile(info, BytesIO(body))
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(tar_name, tar_buffer.getvalue())
            archive.writestr(f"__MACOSX/{tar_name}", b"ignored")


if __name__ == "__main__":
    unittest.main()

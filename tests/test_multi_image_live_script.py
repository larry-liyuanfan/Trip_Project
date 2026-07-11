import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.test_multi_image_understanding import build_payload, collect_default_image_urls


class MultiImageLiveScriptTest(unittest.TestCase):
    def test_collect_default_image_urls_prefers_existing_yelp_subset_images(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_a = root / "data" / "yelp" / "raw" / "photos" / "photo_a.jpg"
            image_b = root / "data" / "yelp" / "raw" / "photos" / "photo_b.jpg"
            image_a.parent.mkdir(parents=True)
            image_a.write_bytes(b"image-a")
            image_b.write_bytes(b"image-b")
            multimodal_items = root / "data" / "yelp" / "processed" / "ota_subset_v1" / "multimodal_items.jsonl"
            multimodal_items.parent.mkdir(parents=True)
            multimodal_items.write_text(
                "\n".join(
                    [
                        json.dumps({"image_path": str(image_a)}),
                        json.dumps({"image_path": str(root / "missing.jpg")}),
                        json.dumps({"image_path": str(image_b)}),
                    ]
                ),
                encoding="utf-8",
            )

            urls = collect_default_image_urls(project_root=root)

        self.assertEqual(len(urls), 2)
        self.assertTrue(urls[0].startswith("file://"))
        self.assertTrue(urls[1].startswith("file://"))
        self.assertIn("photo_a.jpg", urls[0])
        self.assertIn("photo_b.jpg", urls[1])

    def test_build_payload_marks_request_as_multi_image_live_check(self):
        payload = build_payload(["file:///a.jpg", "file:///b.jpg"])

        self.assertEqual(payload["image_urls"], ["file:///a.jpg", "file:///b.jpg"])
        self.assertEqual(payload["language"], "zh")
        self.assertIn("multi-image", payload["user_text"])


if __name__ == "__main__":
    unittest.main()

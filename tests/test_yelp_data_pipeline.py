import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image


class YelpDataPipelineTest(unittest.TestCase):
    def test_business_parser_extracts_fields_and_flattens_attributes(self):
        from src.data.parse_business import parse_business_record

        record = parse_business_record(
            {
                "business_id": "biz_1",
                "name": "Demo Cafe",
                "city": "New Orleans",
                "state": "LA",
                "stars": 4.5,
                "review_count": 12,
                "categories": "Cafes, Coffee & Tea",
                "attributes": {
                    "RestaurantsPriceRange2": "2",
                    "OutdoorSeating": "True",
                    "BusinessParking": "{'garage': False, 'street': True}",
                },
                "hours": {"Monday": "8:00-17:00"},
            }
        )

        self.assertEqual(record["business_id"], "biz_1")
        self.assertEqual(record["categories"], ["cafes", "coffee & tea"])
        self.assertEqual(record["attr_RestaurantsPriceRange2"], "2")
        self.assertEqual(record["attr_OutdoorSeating"], "True")
        self.assertEqual(record["attr_BusinessParking_street"], True)

    def test_review_parser_filters_invalid_text_and_counts_stats(self):
        from src.data.parse_reviews import parse_review_records

        reviews = [
            {"review_id": "empty", "business_id": "biz_1", "text": "", "stars": 1},
            {"review_id": "symbols", "business_id": "biz_1", "text": "!!!", "stars": 1},
            {"review_id": "short", "business_id": "biz_1", "text": "ok", "stars": 3},
            {"review_id": "valid", "business_id": "biz_1", "text": "Great quiet cafe for breakfast.", "stars": 5},
        ]

        parsed, stats, summary = parse_review_records(reviews, min_text_length=10)

        self.assertEqual([row["review_id"] for row in parsed], ["valid"])
        self.assertEqual(stats[0]["business_id"], "biz_1")
        self.assertEqual(stats[0]["valid_review_count"], 1)
        self.assertEqual(summary["filtered_empty"], 1)
        self.assertEqual(summary["filtered_symbol_only"], 1)
        self.assertEqual(summary["filtered_too_short"], 1)

    def test_photo_parser_maps_image_paths(self):
        from src.data.parse_photos import parse_photo_records

        rows = parse_photo_records(
            [{"photo_id": "p1", "business_id": "biz_1", "caption": "front door", "label": "outside"}],
            image_root=Path("data/yelp/raw/photos"),
        )

        self.assertEqual(rows[0]["image_path"], "data/yelp/raw/photos/p1.jpg")
        self.assertEqual(rows[0]["caption"], "front door")

    def test_image_validation_marks_valid_missing_and_corrupted_images(self):
        from src.data.image_validation import validate_photo_images

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid = root / "valid.jpg"
            corrupt = root / "corrupt.jpg"
            Image.new("RGB", (2, 3), "white").save(valid)
            corrupt.write_bytes(b"not an image")

            rows, summary = validate_photo_images(
                [
                    {"photo_id": "valid", "image_path": str(valid)},
                    {"photo_id": "missing", "image_path": str(root / "missing.jpg")},
                    {"photo_id": "corrupt", "image_path": str(corrupt)},
                ]
            )

        by_id = {row["photo_id"]: row for row in rows}
        self.assertTrue(by_id["valid"]["image_valid"])
        self.assertEqual(by_id["valid"]["image_width"], 2)
        self.assertEqual(by_id["valid"]["image_height"], 3)
        self.assertFalse(by_id["missing"]["image_valid"])
        self.assertFalse(by_id["corrupt"]["image_valid"])
        self.assertEqual(summary["valid_images"], 1)

    def test_alignment_builders_filter_and_limit_pairs(self):
        from src.data.alignment import (
            build_medium_alignment,
            build_strong_alignment,
            build_weak_alignment,
        )

        businesses = [
            {"business_id": "biz_1", "name": "Demo Cafe", "categories": ["cafes"], "attributes": {"OutdoorSeating": "True"}},
        ]
        photos = [
            {"photo_id": "p1", "business_id": "biz_1", "caption": "latte", "label": "food"},
            {"photo_id": "p2", "business_id": "biz_1", "caption": "door", "label": "outside"},
            {"photo_id": "bad", "business_id": "biz_1", "caption": "bad", "label": "inside"},
        ]
        image_index = [
            {"photo_id": "p1", "image_path": "p1.jpg", "image_valid": True},
            {"photo_id": "p2", "image_path": "p2.jpg", "image_valid": True},
            {"photo_id": "bad", "image_path": "bad.jpg", "image_valid": False},
        ]
        reviews = [
            {"review_id": "r1", "business_id": "biz_1", "text": "quiet breakfast"},
            {"review_id": "r2", "business_id": "biz_1", "text": "busy lunch"},
            {"review_id": "r3", "business_id": "biz_1", "text": "good coffee"},
        ]

        strong = build_strong_alignment(photos, image_index)
        medium = build_medium_alignment(photos, image_index, businesses)
        weak = build_weak_alignment(
            photos,
            image_index,
            reviews,
            max_reviews_per_business=2,
            max_images_per_business=1,
        )

        self.assertEqual([row["photo_id"] for row in strong], ["p1", "p2"])
        self.assertEqual(medium[0]["business_id"], "biz_1")
        self.assertIn("Demo Cafe", medium[0]["business_description"])
        self.assertEqual(len(weak), 1)
        self.assertEqual(len(weak[0]["review_ids"]), 2)
        self.assertEqual(len(weak[0]["photo_ids"]), 1)

    def test_clip_denoising_skip_path_does_not_fail(self):
        from src.data.clip_denoising import run_clip_denoising

        summary, rows = run_clip_denoising([{"pair_id": "pair_1"}], {"enabled": False})

        self.assertEqual(rows, [])
        self.assertEqual(summary["status"], "skipped")

    def test_dataset_statistics_include_image_ratio_labels_and_caption_lengths(self):
        from src.data.statistics import build_dataset_statistics

        stats = build_dataset_statistics(
            businesses=[{"business_id": "biz_1", "categories": ["cafes"]}],
            reviews=[{"business_id": "biz_1"}],
            photos=[
                {"photo_id": "p1", "business_id": "biz_1", "caption": "front door", "label": "outside"},
                {"photo_id": "p2", "business_id": "biz_1", "caption": "", "label": "food"},
            ],
            image_index=[
                {"photo_id": "p1", "business_id": "biz_1", "image_valid": True},
                {"photo_id": "p2", "business_id": "biz_1", "image_valid": False},
            ],
            strong=[],
            medium=[],
            weak=[],
        )

        self.assertEqual(stats["valid_image_ratio"], 0.5)
        self.assertEqual(stats["photo_label_distribution"], {"outside": 1, "food": 1})
        self.assertEqual(stats["caption_length_stats"]["caption_count"], 1)
        self.assertEqual(stats["caption_length_stats"]["min_chars"], 10)
        self.assertEqual(stats["caption_length_stats"]["max_chars"], 10)

    def test_report_generator_writes_todo_for_missing_stats(self):
        from scripts.generate_yelp_report import render_report

        report = render_report({"output_format": "parquet"}, {})

        self.assertIn("TODO", report)
        self.assertIn("Strong alignment", report)

    def test_report_generator_uses_actual_clip_status_and_quality_statistics(self):
        from scripts.generate_yelp_report import render_report

        report = render_report(
            {"paths": {}, "processing_limits": {"max_reviews": 10000}},
            {
                "valid_image_ratio": 0.25,
                "photo_label_distribution": {"food": 2},
                "caption_length_stats": {"caption_count": 2, "min_chars": 4, "mean_chars": 6.0, "max_chars": 8},
            },
            clip={"status": "skipped", "reason": "CLIP dependencies unavailable", "input_pairs": 10, "retained_pairs": 0},
        )

        self.assertIn("Valid image ratio: 0.25", report)
        self.assertIn("Photo label distribution: {'food': 2}", report)
        self.assertIn("Caption length statistics: {'caption_count': 2, 'min_chars': 4, 'mean_chars': 6.0, 'max_chars': 8}", report)
        self.assertIn("CLIP denoising: skipped", report)
        self.assertIn("Denoising before/after weak pairs: 10 -> 0", report)
        self.assertNotIn("CLIP denoising is currently disabled and skipped cleanly.", report)

    def test_table_csv_fallback_round_trips_scalar_and_structured_values(self):
        from src.data.jsonl_utils import read_table, write_table

        with TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "rows.parquet"
            write_table(
                table_path,
                [
                    {
                        "business_id": "biz_1",
                        "photo_id": "photo_1",
                        "image_valid": True,
                        "rating": 4.5,
                        "tags": ["cafes", "coffee"],
                        "attributes": {"OutdoorSeating": "True"},
                    }
                ],
                output_format="parquet",
            )

            rows = read_table(table_path)

        self.assertEqual(rows[0]["business_id"], "biz_1")
        self.assertEqual(rows[0]["photo_id"], "photo_1")
        self.assertTrue(rows[0]["image_valid"])
        self.assertEqual(rows[0]["rating"], 4.5)
        self.assertEqual(rows[0]["tags"], ["cafes", "coffee"])
        self.assertEqual(rows[0]["attributes"], {"OutdoorSeating": "True"})

    def test_pipeline_validation_checks_outputs_columns_counts_and_image_paths(self):
        from src.data.jsonl_utils import write_json, write_table
        from src.data.pipeline_validation import validate_week2_outputs

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            interim = root / "data" / "yelp" / "interim"
            processed = root / "data" / "yelp" / "processed"
            image = root / "data" / "yelp" / "raw" / "photos" / "p1.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (1, 1), "white").save(image)
            config = {
                "paths": {
                    "interim_dir": str(interim),
                    "processed_dir": str(processed),
                    "report_path": str(root / "reports" / "report.md"),
                },
                "output": {"format": "csv"},
            }
            write_table(interim / "business.csv", [{"business_id": "biz_1", "name": "Demo", "categories": ["cafes"], "stars": 4.5}], "csv")
            write_table(interim / "reviews.csv", [{"review_id": "r1", "business_id": "biz_1", "text": "good coffee", "stars": 5}], "csv")
            write_table(interim / "photos.csv", [{"photo_id": "p1", "business_id": "biz_1", "caption": "cup", "label": "food", "image_path": str(image)}], "csv")
            write_table(interim / "photo_image_index.csv", [{"photo_id": "p1", "business_id": "biz_1", "image_path": str(image), "image_valid": True}], "csv")
            write_table(interim / "review_business_stats.csv", [{"business_id": "biz_1", "valid_review_count": 1}], "csv")
            write_table(processed / "strong_image_caption_pairs.csv", [{"business_id": "biz_1", "photo_id": "p1", "image_path": str(image), "caption": "cup", "label": "food"}], "csv")
            write_table(processed / "image_business_attribute_pairs.csv", [{"business_id": "biz_1", "photo_id": "p1", "image_path": str(image), "business_description": "Demo cafe"}], "csv")
            write_table(processed / "business_level_weak_pairs.csv", [{"business_id": "biz_1", "photo_ids": ["p1"], "image_paths": [str(image)], "review_texts": ["good coffee"]}], "csv")
            write_json(processed / "dataset_statistics.json", {"business_count": 1, "review_count": 1, "photo_metadata_count": 1, "valid_image_count": 1, "strong_pairs": 1, "medium_pairs": 1, "weak_pairs": 1})
            (root / "reports").mkdir()
            (root / "reports" / "report.md").write_text(
                "Businesses parsed: 1\nReviews parsed: 1\nPhoto metadata entries parsed: 1\nValid local images: 1\nStrong pairs: 1\nMedium pairs: 1\nWeak groups: 1\nCSV fallback\n",
                encoding="utf-8",
            )

            result = validate_week2_outputs(config)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["counts"]["strong_pairs"], 1)
        self.assertFalse(result["errors"])

    def test_parse_and_alignment_scripts_run_on_tiny_fixture(self):
        from scripts.build_yelp_alignment import run_alignment
        from scripts.parse_yelp_json import run_parse

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "data" / "yelp" / "raw"
            raw.mkdir(parents=True)
            photos_dir = raw / "photos"
            photos_dir.mkdir()
            Image.new("RGB", (1, 1), "white").save(photos_dir / "p1.jpg")
            (raw / "yelp_academic_dataset_business.json").write_text(
                json.dumps(
                    {
                        "business_id": "biz_1",
                        "name": "Demo Cafe",
                        "categories": "Cafes",
                        "stars": 4.0,
                        "review_count": 1,
                        "attributes": {"OutdoorSeating": "True"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (raw / "yelp_academic_dataset_review.json").write_text(
                json.dumps(
                    {
                        "review_id": "r1",
                        "business_id": "biz_1",
                        "text": "A calm cafe with good coffee.",
                        "stars": 5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (raw / "photos.json").write_text(
                json.dumps(
                    {
                        "photo_id": "p1",
                        "business_id": "biz_1",
                        "caption": "coffee cup",
                        "label": "food",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = {
                "paths": {
                    "business_json": str(raw / "yelp_academic_dataset_business.json"),
                    "review_json": str(raw / "yelp_academic_dataset_review.json"),
                    "photo_json": str(raw / "photos.json"),
                    "image_root": str(photos_dir),
                    "interim_dir": str(root / "data" / "yelp" / "interim"),
                    "processed_dir": str(root / "data" / "yelp" / "processed"),
                    "logs_dir": str(root / "data" / "yelp" / "logs"),
                    "validation_dir": str(root / "data" / "yelp" / "validation"),
                    "report_path": str(root / "reports" / "report.md"),
                },
                "output": {"format": "csv"},
                "review_filters": {"min_text_length": 10, "reject_symbol_only": True},
                "weak_alignment": {"max_reviews_per_business": 2, "max_images_per_business": 2},
                "clip_denoising": {"enabled": False},
            }

            parse_summary = run_parse(config)
            alignment_summary = run_alignment(config)

        self.assertEqual(parse_summary["business_count"], 1)
        self.assertEqual(alignment_summary["strong_pairs"], 1)

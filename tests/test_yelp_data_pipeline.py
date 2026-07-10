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

    def test_review_parser_can_stream_rows_to_sink_without_returning_review_list(self):
        from src.data.parse_reviews import stream_review_records

        captured = []
        stats, summary = stream_review_records(
            [
                {"review_id": "r1", "business_id": "biz_1", "text": "Good coffee and calm seating.", "stars": 5},
                {"review_id": "r2", "business_id": "biz_1", "text": "!!!", "stars": 1},
                {"review_id": "r3", "business_id": "biz_2", "text": "Nice service and clean table.", "stars": 4},
            ],
            row_sink=captured.append,
            min_text_length=10,
            reject_symbol_only=True,
        )

        self.assertEqual([row["review_id"] for row in captured], ["r1", "r3"])
        self.assertEqual(summary["input_reviews"], 3)
        self.assertEqual(summary["valid_reviews"], 2)
        self.assertEqual(summary["filtered_symbol_only"], 1)
        self.assertEqual(
            sorted((row["business_id"], row["valid_review_count"]) for row in stats),
            [("biz_1", 1), ("biz_2", 1)],
        )

    def test_review_parser_filters_rows_without_join_identifiers(self):
        from src.data.parse_reviews import stream_review_records

        captured = []
        _, summary = stream_review_records(
            [
                {"review_id": "missing_business", "text": "A review with no business identifier."},
                {"business_id": "biz_1", "text": "A review with no review identifier."},
                {"review_id": "valid", "business_id": "biz_1", "text": "A valid review for alignment."},
            ],
            row_sink=captured.append,
        )

        self.assertEqual([row["review_id"] for row in captured], ["valid"])
        self.assertEqual(summary["filtered_missing_identifier"], 2)

    def test_business_parser_streams_rows_to_a_sink(self):
        from src.data.parse_business import stream_business_records

        captured = []
        summary = stream_business_records(
            [
                {"business_id": "biz_1", "name": "Cafe", "categories": "Cafes"},
                {"business_id": "biz_2", "name": "Restaurant", "categories": "Restaurants"},
            ],
            row_sink=captured.append,
        )

        self.assertEqual(summary["parsed_businesses"], 2)
        self.assertEqual([row["business_id"] for row in captured], ["biz_1", "biz_2"])

    def test_business_stream_serializes_variable_nested_fields_for_stable_parquet_schema(self):
        from src.data.parse_business import stream_business_records

        captured = []
        stream_business_records(
            [
                {"business_id": "biz_1", "attributes": {"WiFi": "free"}, "hours": {"Monday": "9:00-17:00"}},
                {"business_id": "biz_2", "attributes": {"OutdoorSeating": "True"}, "hours": {"Friday": "10:00-20:00"}},
            ],
            row_sink=captured.append,
        )

        self.assertEqual(captured[0]["attributes"], '{"WiFi": "free"}')
        self.assertEqual(captured[1]["hours"], '{"Friday": "10:00-20:00"}')

    def test_photo_parser_maps_image_paths(self):
        from src.data.parse_photos import parse_photo_records

        rows = parse_photo_records(
            [{"photo_id": "p1", "business_id": "biz_1", "caption": "front door", "label": "outside"}],
            image_root=Path("data/yelp/raw/photos"),
        )

        self.assertEqual(rows[0]["image_path"], "data/yelp/raw/photos/p1.jpg")
        self.assertEqual(rows[0]["caption"], "front door")

    def test_photo_parser_streams_rows_to_a_sink(self):
        from src.data.parse_photos import stream_photo_records

        captured = []
        summary = stream_photo_records(
            [{"photo_id": "p1", "business_id": "biz_1", "caption": "front door", "label": "outside"}],
            image_root=Path("data/yelp/raw/photos"),
            row_sink=captured.append,
        )

        self.assertEqual(summary["parsed_photos"], 1)
        self.assertEqual(captured[0]["photo_id"], "p1")

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

    def test_image_validation_streams_index_rows_to_a_sink(self):
        from src.data.image_validation import stream_validate_photo_images

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "valid.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            captured = []
            summary = stream_validate_photo_images(
                [{"photo_id": "p1", "business_id": "biz_1", "image_path": str(image_path)}],
                row_sink=captured.append,
            )

        self.assertEqual(summary["valid_images"], 1)
        self.assertTrue(captured[0]["image_valid"])

    def test_image_validation_can_process_a_bounded_batch_with_multiple_workers(self):
        from src.data.image_validation import iter_validated_photo_images

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid = root / "valid.jpg"
            Image.new("RGB", (2, 2), "white").save(valid)
            results = list(
                iter_validated_photo_images(
                    [
                        {"photo_id": "valid", "image_path": str(valid)},
                        {"photo_id": "missing", "image_path": str(root / "missing.jpg")},
                    ],
                    workers=2,
                )
            )

        self.assertEqual([status for _, status in results], ["valid", "missing"])

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
            {"photo_id": "no_caption", "business_id": "biz_1", "caption": "", "label": "inside"},
            {"photo_id": "bad", "business_id": "biz_1", "caption": "bad", "label": "inside"},
        ]
        image_index = [
            {"photo_id": "p1", "image_path": "p1.jpg", "image_valid": True},
            {"photo_id": "p2", "image_path": "p2.jpg", "image_valid": True},
            {"photo_id": "no_caption", "image_path": "no_caption.jpg", "image_valid": True},
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
        self.assertIn("OutdoorSeating", medium[0]["attribute_dimension_labels"])
        self.assertEqual(len(weak), 1)
        self.assertEqual(len(weak[0]["review_ids"]), 2)
        self.assertEqual(len(weak[0]["photo_ids"]), 1)

    def test_clip_denoising_skip_path_does_not_fail(self):
        from src.data.clip_denoising import run_clip_denoising

        summary, rows = run_clip_denoising([{"pair_id": "pair_1"}], {"enabled": False})

        self.assertEqual(rows, [])
        self.assertEqual(summary["status"], "skipped")

    def test_clip_denoising_retains_scored_image_review_rows(self):
        from src.data.clip_denoising import run_clip_denoising

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "photo.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            summary, rows = run_clip_denoising(
                [
                    {
                        "business_id": "biz_1",
                        "photo_ids": ["photo_1"],
                        "image_paths": [str(image_path)],
                        "review_ids": ["review_keep", "review_drop"],
                        "review_texts": ["bright cafe interior", "unrelated review"],
                    }
                ],
                {"enabled": True, "threshold": 0.5, "model_id": "test/clip"},
                score_candidates=lambda candidates, _: [0.8, 0.2],
            )

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["input_pairs"], 2)
        self.assertEqual(summary["retained_pairs"], 1)
        self.assertEqual(rows[0]["photo_id"], "photo_1")
        self.assertEqual(rows[0]["review_id"], "review_keep")
        self.assertEqual(rows[0]["clip_similarity"], 0.8)
        self.assertEqual(rows[0]["alignment_type"], "weak_denoised")

    def test_dataset_statistics_include_image_ratio_labels_and_caption_lengths(self):
        from src.data.statistics import build_dataset_statistics

        stats = build_dataset_statistics(
            businesses=[
                {"business_id": "biz_1", "categories": ["cafes"], "city": "New Orleans"},
                {"business_id": "biz_2", "categories": ["restaurants"], "city": "Philadelphia"},
            ],
            reviews=[{"business_id": "biz_1"}],
            photos=[
                {"photo_id": "p1", "business_id": "biz_1", "caption": "front door", "label": "outside"},
                {"photo_id": "p2", "business_id": "biz_1", "caption": "", "label": "food"},
            ],
            image_index=[
                {"photo_id": "p1", "business_id": "biz_1", "image_valid": True},
                {"photo_id": "p2", "business_id": "biz_1", "image_valid": False},
            ],
            strong=[{"caption": "front door", "label": "outside"}],
            medium=[],
            weak=[{"business_id": "biz_1"}],
        )

        self.assertEqual(stats["valid_image_ratio"], 0.5)
        self.assertEqual(stats["city_count"], 2)
        self.assertEqual(stats["photo_label_distribution"], {"outside": 1})
        self.assertEqual(stats["weak_group_top_categories"], [("cafes", 1)])
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
                "city_count": 2,
                "valid_image_ratio": 0.25,
                "photo_label_distribution": {"food": 2},
                "caption_length_stats": {"caption_count": 2, "min_chars": 4, "mean_chars": 6.0, "max_chars": 8},
                "weak_group_top_categories": [("cafes", 1)],
            },
            clip={"status": "skipped", "reason": "CLIP dependencies unavailable", "input_pairs": 10, "retained_pairs": 0},
        )

        self.assertIn("# Yelp多模态数据处理说明报告（第一部分）", report)
        self.assertIn("覆盖城市数量: 2", report)
        self.assertIn("Valid image ratio: 0.25", report)
        self.assertIn("弱对齐品类覆盖: [('cafes', 1)]", report)
        self.assertIn("Photo label distribution: {'food': 2}", report)
        self.assertIn("Caption length statistics: {'caption_count': 2, 'min_chars': 4, 'mean_chars': 6.0, 'max_chars': 8}", report)
        self.assertIn("CLIP denoising: skipped", report)
        self.assertIn("Denoising before/after weak pairs: 10 -> 0", report)
        self.assertIn("CLIP runtime: model=TODO, device=TODO", report)
        self.assertNotIn("CLIP denoising is currently disabled and skipped cleanly.", report)

    def test_report_generator_includes_review_filter_counts_and_clip_threshold(self):
        from scripts.generate_yelp_report import render_report

        report = render_report(
            {"paths": {}, "processing_limits": {"max_reviews": None}},
            {},
            validation={
                "review_filters": {
                    "input_reviews": 10,
                    "valid_reviews": 7,
                    "filtered_empty": 1,
                    "filtered_too_short": 1,
                    "filtered_symbol_only": 1,
                    "filtered_missing_identifier": 0,
                }
            },
            clip={"status": "completed", "threshold": 0.25},
        )

        self.assertIn(
            "Review filter counts: input=10, valid=7, empty=1, too_short=1, symbol_only=1, missing_identifier=0",
            report,
        )
        self.assertIn("CLIP threshold: 0.25", report)

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

    def test_table_stream_writer_flushes_chunks(self):
        from src.data.jsonl_utils import TableStreamWriter, read_table

        with TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "reviews.parquet"
            writer = TableStreamWriter(
                table_path,
                output_format="parquet",
                fieldnames=["review_id", "business_id", "text"],
                chunk_size=2,
            )
            writer.write({"review_id": "r1", "business_id": "b1", "text": "one"})
            writer.write({"review_id": "r2", "business_id": "b1", "text": "two"})
            writer.write({"review_id": "r3", "business_id": "b2", "text": "three"})
            summary = writer.close()

            rows = read_table(table_path)

        self.assertEqual(summary["rows"], 3)
        self.assertEqual([row["review_id"] for row in rows], ["r1", "r2", "r3"])

    def test_table_stream_writer_writes_schema_for_empty_table(self):
        from src.data.jsonl_utils import TableStreamWriter, read_table

        with TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "empty.parquet"
            writer = TableStreamWriter(
                table_path,
                output_format="parquet",
                fieldnames=["business_id", "clip_similarity"],
            )
            summary = writer.close()

            rows = read_table(table_path)
            self.assertTrue(table_path.exists())
            self.assertEqual(summary["rows"], 0)
            self.assertEqual(rows, [])

    def test_table_stream_writer_uses_explicit_schema_across_null_and_error_batches(self):
        import pyarrow as pa

        from src.data.jsonl_utils import TableStreamWriter, read_table

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "image_index.parquet"
            writer = TableStreamWriter(
                path,
                output_format="parquet",
                fieldnames=["photo_id", "image_valid", "validation_error"],
                chunk_size=1,
                parquet_schema=pa.schema(
                    [
                        ("photo_id", pa.string()),
                        ("image_valid", pa.bool_()),
                        ("validation_error", pa.string()),
                    ]
                ),
            )
            writer.write({"photo_id": "p1", "image_valid": True, "validation_error": None})
            writer.write({"photo_id": "p2", "image_valid": False, "validation_error": "unreadable"})
            writer.close()
            rows = read_table(path)

        self.assertEqual(rows[1]["validation_error"], "unreadable")

    def test_clip_runner_writes_empty_output_when_denoising_is_disabled(self):
        from scripts.run_clip_denoising import run_with_config
        from src.data.jsonl_utils import read_table, write_table

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed = root / "processed"
            write_table(
                processed / "business_level_weak_pairs.csv",
                [
                    {
                        "business_id": "biz_1",
                        "photo_ids": ["p1"],
                        "image_paths": ["missing.jpg"],
                        "review_ids": ["r1"],
                        "review_texts": ["quiet cafe"],
                    }
                ],
                "csv",
            )
            summary = run_with_config(
                {
                    "paths": {
                        "interim_dir": str(root / "interim"),
                        "processed_dir": str(processed),
                        "logs_dir": str(root / "logs"),
                        "validation_dir": str(root / "validation"),
                        "report_path": str(root / "report.md"),
                    },
                    "output": {"format": "csv"},
                    "clip_denoising": {"enabled": False, "output_filename": "weak_pairs_denoised"},
                }
            )
            output_path = processed / "weak_pairs_denoised.csv"

            self.assertEqual(summary["status"], "skipped")
            self.assertTrue(output_path.exists())
            self.assertEqual(read_table(output_path), [])

    def test_clip_validation_checks_retained_table_and_summary_count(self):
        from src.data.clip_denoising import DENOISED_PAIR_FIELDS
        from src.data.jsonl_utils import write_json, write_table
        from src.data.pipeline_validation import validate_clip_denoising_output

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed = root / "processed"
            image_path = root / "photo.jpg"
            Image.new("RGB", (1, 1), "white").save(image_path)
            write_table(
                processed / "weak_pairs_denoised.csv",
                [
                    {
                        "business_id": "biz_1",
                        "photo_id": "p1",
                        "image_path": str(image_path),
                        "review_id": "r1",
                        "review_text": "quiet cafe",
                        "clip_similarity": 0.7,
                        "clip_model": "test/clip",
                        "alignment_type": "weak_denoised",
                    }
                ],
                "csv",
            )
            write_json(processed / "clip_denoising_summary.json", {"status": "completed", "retained_pairs": 1})
            errors = validate_clip_denoising_output(
                processed,
                "csv",
                {"enabled": True, "output_filename": "weak_pairs_denoised"},
                set(DENOISED_PAIR_FIELDS),
            )

        self.assertEqual(errors, [])

    def test_pipeline_table_inspection_uses_parquet_metadata_for_counts_and_columns(self):
        from src.data.jsonl_utils import write_table
        from src.data.pipeline_validation import inspect_table

        with TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "rows.parquet"
            write_table(
                table_path,
                [
                    {"review_id": "r1", "business_id": "b1", "text": "one"},
                    {"review_id": "r2", "business_id": "b2", "text": "two"},
                ],
                "parquet",
            )
            count, columns = inspect_table(table_path)

        self.assertEqual(count, 2)
        self.assertEqual(columns, {"review_id", "business_id", "text"})

    def test_pipeline_table_inspection_preserves_nested_parquet_column_names(self):
        from src.data.jsonl_utils import write_table
        from src.data.pipeline_validation import inspect_table

        with TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "nested.parquet"
            write_table(
                table_path,
                [{"business_id": "b1", "categories": ["cafes"], "photo_ids": ["p1"]}],
                "parquet",
            )
            _, columns = inspect_table(table_path)

        self.assertEqual(columns, {"business_id", "categories", "photo_ids"})

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
            write_table(processed / "image_business_attribute_pairs.csv", [{"business_id": "biz_1", "photo_id": "p1", "image_path": str(image), "business_description": "Demo cafe", "attribute_dimension_labels": ["OutdoorSeating"]}], "csv")
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

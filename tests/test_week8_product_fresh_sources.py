import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet

from scripts.build_week8_product_fresh_sources import (
    Week8FreshSourceError,
    build_fresh_sources,
    caption_signals,
    collect_fresh_candidates,
    ota_category,
    select_extraction_candidates,
    select_ranked_candidates,
)
from src.training.week8_product import load_week8_product_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/week8/product_understanding_v2.json"
V5_CONFIG = ROOT / "configs/week8/product_understanding_v5.json"


def candidate(index: int, category: str, richness: int = 0) -> dict:
    return {
        "photo_id": f"photo-{index:04d}",
        "business_id": f"business-{index:04d}",
        "source_id": f"yelp-photo:photo-{index:04d}",
        "group_id": f"yelp-business:business-{index:04d}",
        "ota_category": category,
        "caption_richness": richness,
        "seed_rank": f"{index:064x}",
    }


class Week8ProductFreshSourceTests(unittest.TestCase):
    def test_v2_config_keeps_counts_and_points_only_to_isolated_sources(self):
        config = load_week8_product_config(CONFIG)
        self.assertEqual(config["schema_version"], "week8_product_understanding_v2")
        self.assertEqual(config["dataset"]["development_count"], 60)
        self.assertEqual(config["dataset"]["test_count"], 60)
        self.assertEqual(config["dataset"]["continuation_train_count"], 400)
        required = sum(
            config["dataset"][name]
            for name in ("development_count", "test_count", "continuation_train_count")
        )
        self.assertGreaterEqual(config["fresh_source"]["selected_photo_count"], required)
        self.assertGreater(
            config["fresh_source"]["candidate_extract_count"],
            config["fresh_source"]["selected_photo_count"],
        )
        for name in ("photos", "strong_pairs", "medium_pairs"):
            self.assertIn(
                "data/yelp/week8_product_fresh_20260826_v1/",
                config["dataset"]["source_paths"][name],
            )
        self.assertTrue(
            config["experiment_identity"]["development_run_id"].endswith("_v2")
        )

    def test_ota_category_and_caption_richness_use_explicit_terms(self):
        self.assertEqual(ota_category(["Hotels", "Restaurants"]), "hotel")
        self.assertEqual(ota_category('["Museums", "Shopping"]'), "attraction")
        self.assertEqual(ota_category("Restaurants, Italian"), "restaurant")
        self.assertIsNone(ota_category(["Shopping", "Doctors"]))
        signals = caption_signals(
            "Cozy modern hotel reception with pool, parking and accessible entrance"
        )
        self.assertEqual(signals["styles"], ["cozy", "modern"])
        self.assertIn("front_desk", signals["facilities"])
        self.assertIn("pool", signals["facilities"])
        self.assertGreater(signals["richness"], 4)

    def test_selection_is_unique_richness_ranked_and_has_category_support(self):
        categories = ("hotel", "attraction", "restaurant")
        rows = [
            candidate(index, categories[index % 3], richness=index % 7)
            for index in range(2400)
        ]
        selected = select_ranked_candidates(
            rows,
            selected_count=2000,
            minimum_eligible_count=2000,
            minimum_per_category=200,
        )
        self.assertEqual(len(selected), 2000)
        self.assertEqual(len({row["photo_id"] for row in selected}), 2000)
        self.assertEqual(len({row["business_id"] for row in selected}), 2000)
        counts = {
            category: sum(row["ota_category"] == category for row in selected)
            for category in categories
        }
        self.assertTrue(all(value >= 200 for value in counts.values()))
        self.assertGreaterEqual(
            selected[0]["caption_richness"], selected[-1]["caption_richness"]
        )

    def test_selection_retains_every_scarce_category_candidate(self):
        rows = [candidate(index, "restaurant", richness=3) for index in range(2100)]
        rows.extend(candidate(3000 + index, "hotel") for index in range(13))
        rows.extend(candidate(4000 + index, "attraction") for index in range(7))
        selected = select_ranked_candidates(
            rows,
            selected_count=2000,
            minimum_eligible_count=2000,
            minimum_per_category=1,
            retain_all_categories_below=200,
        )
        self.assertEqual(sum(row["ota_category"] == "hotel" for row in selected), 13)
        self.assertEqual(
            sum(row["ota_category"] == "attraction" for row in selected), 7
        )

    def test_v5_category_specific_minimums_match_observed_feasible_support(self):
        rows = [candidate(index, "restaurant", richness=3) for index in range(600)]
        rows.extend(candidate(3000 + index, "hotel") for index in range(13))
        rows.extend(candidate(4000 + index, "attraction") for index in range(7))
        selected = select_ranked_candidates(
            rows,
            selected_count=620,
            minimum_eligible_count=520,
            minimum_per_category={
                "hotel": 13,
                "attraction": 7,
                "restaurant": 500,
            },
            retain_all_categories_below=1000,
        )
        self.assertEqual(
            Counter(row["ota_category"] for row in selected),
            Counter({"hotel": 13, "attraction": 7, "restaurant": 600}),
        )

    def test_candidate_collection_uses_one_best_unconsumed_photo_per_business(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            businesses = [
                {
                    "business_id": "b1",
                    "name": "Hotel One",
                    "city": "X",
                    "state": "Y",
                    "stars": 4.0,
                    "categories": ["hotels"],
                    "attributes": "{}",
                    "hours": "{}",
                },
                {
                    "business_id": "b2",
                    "name": "Cafe Two",
                    "city": "X",
                    "state": "Y",
                    "stars": 4.0,
                    "categories": ["restaurants"],
                    "attributes": "{}",
                    "hours": "{}",
                },
            ]
            photos = [
                {
                    "photo_id": "p1",
                    "business_id": "b1",
                    "caption": "hotel room",
                    "label": "inside",
                },
                {
                    "photo_id": "p2",
                    "business_id": "b1",
                    "caption": "modern cozy hotel reception with pool",
                    "label": "inside",
                },
                {
                    "photo_id": "p3",
                    "business_id": "b2",
                    "caption": "cozy cafe bar",
                    "label": "inside",
                },
                {
                    "photo_id": "p4",
                    "business_id": "b2",
                    "caption": "",
                    "label": "inside",
                },
            ]
            business_path = root / "business.parquet"
            photo_path = root / "photos.parquet"
            parquet.write_table(pa.Table.from_pylist(businesses), business_path)
            parquet.write_table(pa.Table.from_pylist(photos), photo_path)
            consumed = {
                "sample_id": set(),
                "source_id": {"yelp-photo:p3"},
                "image_sha256": set(),
                "group_id": set(),
                "constraint_template_id": set(),
            }
            rows, stats = collect_fresh_candidates(
                photo_path, business_path, consumed, seed=20260826
            )
            self.assertEqual({row["photo_id"] for row in rows}, {"p2", "p4"})
            hotel = next(row for row in rows if row["photo_id"] == "p2")
            restaurant = next(row for row in rows if row["photo_id"] == "p4")
            self.assertEqual(hotel["ota_category"], "hotel")
            self.assertEqual(restaurant["caption"], "photo type: inside")
            self.assertEqual(restaurant["caption_source"], "photo_label_fallback")
            self.assertEqual(stats["photo_filter_counts"]["consumed_source"], 1)

    def test_v5_keeps_bounded_fallback_photos_before_hash_validation(self):
        config = load_week8_product_config(V5_CONFIG)
        self.assertEqual(config["schema_version"], "week8_product_understanding_v5")
        self.assertEqual(
            config["week8"]["source_version"],
            "week8_product_fresh_20260827_v2",
        )
        self.assertEqual(config["fresh_source"]["max_candidates_per_business"], 8)
        self.assertEqual(
            config["fresh_source"]["minimum_per_ota_category_by_category"],
            {"hotel": 13, "attraction": 7, "restaurant": 500},
        )
        split_minimums = config["dataset"]["split_category_minimums"]
        self.assertEqual(
            sum(split_minimums[split]["hotel"] for split in split_minimums), 13
        )
        self.assertEqual(
            sum(split_minimums[split]["attraction"] for split in split_minimums),
            7,
        )

    def test_v6_records_observed_post_hash_category_ceiling(self):
        config = load_week8_product_config(
            Path("configs/week8/product_understanding_v6.json")
        )

        self.assertEqual(config["schema_version"], "week8_product_understanding_v6")
        self.assertEqual(
            config["week8"]["source_version"],
            "week8_product_fresh_20260827_v3",
        )
        self.assertEqual(
            config["fresh_source"]["minimum_per_ota_category_by_category"],
            {"hotel": 1, "attraction": 7, "restaurant": 500},
        )
        split_minimums = config["dataset"]["split_category_minimums"]
        self.assertEqual(split_minimums["test"]["hotel"], 1)
        self.assertEqual(split_minimums["development"]["hotel"], 0)
        self.assertEqual(split_minimums["train"]["hotel"], 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            businesses = [
                {
                    "business_id": "hotel-1",
                    "name": "Hotel One",
                    "city": "X",
                    "state": "Y",
                    "stars": 4.0,
                    "categories": ["hotels"],
                    "attributes": "{}",
                    "hours": "{}",
                }
            ]
            photos = [
                {
                    "photo_id": "hotel-primary",
                    "business_id": "hotel-1",
                    "caption": "modern hotel lobby with pool and parking",
                    "label": "inside",
                },
                {
                    "photo_id": "hotel-fallback",
                    "business_id": "hotel-1",
                    "caption": "hotel room",
                    "label": "inside",
                },
                {
                    "photo_id": "hotel-third",
                    "business_id": "hotel-1",
                    "caption": "hotel entrance",
                    "label": "outside",
                },
            ]
            business_path = root / "business.parquet"
            photo_path = root / "photos.parquet"
            parquet.write_table(pa.Table.from_pylist(businesses), business_path)
            parquet.write_table(pa.Table.from_pylist(photos), photo_path)
            consumed = {field: set() for field in (
                "sample_id",
                "source_id",
                "image_sha256",
                "group_id",
                "constraint_template_id",
            )}
            rows, stats = collect_fresh_candidates(
                photo_path,
                business_path,
                consumed,
                seed=20260827,
                max_candidates_per_business=2,
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["business_id"] for row in rows}, {"hotel-1"})
            self.assertIn("hotel-primary", {row["photo_id"] for row in rows})
            self.assertEqual(stats["unique_eligible_business_count"], 1)
            self.assertEqual(stats["bounded_eligible_photo_candidate_count"], 2)

    def test_post_hash_selection_falls_back_within_business(self):
        rows = [candidate(index, "restaurant", richness=3) for index in range(600)]
        primary = candidate(8000, "hotel", richness=9)
        primary["business_id"] = "hotel-business"
        primary["group_id"] = "yelp-business:hotel-business"
        fallback = candidate(8001, "hotel", richness=2)
        fallback["business_id"] = "hotel-business"
        fallback["group_id"] = "yelp-business:hotel-business"
        attraction = candidate(9000, "attraction", richness=2)
        pool = select_extraction_candidates(
            [primary, fallback, attraction, *rows],
            selected_count=602,
            minimum_per_category=1,
            retain_all_categories_below=10,
        )
        # 模拟最佳图因历史 image_sha256 冲突被拒绝；同 business 的下一张仍可入选。
        validated = [row for row in pool if row["photo_id"] != primary["photo_id"]]
        selected = select_ranked_candidates(
            validated,
            selected_count=520,
            minimum_eligible_count=520,
            minimum_per_category=1,
            retain_all_categories_below=10,
        )
        selected_ids = {row["photo_id"] for row in selected}
        self.assertNotIn(primary["photo_id"], selected_ids)
        self.assertIn(fallback["photo_id"], selected_ids)
        self.assertEqual(
            len({row["business_id"] for row in selected}), len(selected)
        )

    def test_extraction_pool_covers_businesses_before_adding_fallback_layers(self):
        rows = []
        index = 0
        for category in ("hotel", "attraction", "restaurant"):
            for business in range(3):
                for photo_rank in range(3):
                    row = candidate(index, category, richness=10 - photo_rank)
                    row["business_id"] = f"{category}-business-{business}"
                    row["group_id"] = f"yelp-business:{row['business_id']}"
                    rows.append(row)
                    index += 1
        selected = select_extraction_candidates(
            rows,
            selected_count=12,
            minimum_per_category=2,
            retain_all_categories_below=0,
        )
        for category in ("hotel", "attraction", "restaurant"):
            category_rows = [row for row in selected if row["ota_category"] == category]
            self.assertEqual(len(category_rows), 4)
            self.assertEqual(len({row["business_id"] for row in category_rows}), 2)

    def test_builder_refuses_to_overwrite_versioned_output(self):
        original = json.loads(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = copy.deepcopy(original)
            payload["fresh_source"]["output_root"] = (
                "data/yelp/week8_product_fresh_20260826_v1/test-existing"
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            output = root / payload["fresh_source"]["output_root"]
            output.mkdir(parents=True)
            with self.assertRaisesRegex(Week8FreshSourceError, "overwrite"):
                build_fresh_sources(
                    root,
                    config_path,
                    rebuilt_yelp_root=root / "rebuilt",
                    historical_root=root / "historical",
                    photos_zip=root / "photos.zip",
                )


if __name__ == "__main__":
    unittest.main()

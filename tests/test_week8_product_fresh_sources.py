import copy
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet

from scripts.build_week8_product_fresh_sources import (
    Week8FreshSourceError,
    build_fresh_sources,
    caption_signals,
    collect_fresh_candidates,
    ota_category,
    select_ranked_candidates,
)
from src.training.week8_product import load_week8_product_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/week8/product_understanding_v2.json"


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
        self.assertGreaterEqual(config["fresh_source"]["selected_photo_count"], 2000)
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
            self.assertEqual([row["photo_id"] for row in rows], ["p2"])
            self.assertEqual(rows[0]["ota_category"], "hotel")
            self.assertEqual(stats["photo_filter_counts"]["consumed_source"], 1)

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

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.prompting import render_standard_prompt
from src.training.week8_product import (
    _collect_week8_v2_sources,
    load_week8_product_config,
    product_error_slices,
    product_silver_target,
    select_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/week8/product_understanding_v1.json"
ACTIVE_CONFIG = ROOT / "configs/week8/product_understanding_v4.json"


def summary(composite: float, latency: float = 1000.0) -> dict:
    support = {
        "business_category_accuracy": 60,
        "price_range_accuracy": 0,
        "style_f1": 20,
        "facility_f1": 20,
        "label_completeness": 60,
        "json_compliance": 60,
        "schema_pass": 60,
    }
    return {
        "scenarios": {
            "image_product_search": {
                "composite": composite,
                "aggregate": {"json_compliance": 1.0, "schema_pass": 1.0},
                "metric_support": support,
            }
        },
        "failure_rate": 0.0,
        "latency_ms_mean": latency,
    }


class Week8ProductTests(unittest.TestCase):
    def test_v3_config_pins_completed_fresh_source_manifest(self):
        config = load_week8_product_config(ACTIVE_CONFIG)
        self.assertEqual(config["schema_version"], "week8_product_understanding_v4")
        self.assertEqual(
            config["fresh_source"]["manifest_sha256"],
            "582f7e4700078f41234082d16043a09c59f248a36f9300995663c705525ce195",
        )
        self.assertEqual(
            config["dataset"]["split_category_minimums"]["test"]["hotel"], 1
        )
        self.assertEqual(
            config["dataset"]["split_field_support_minimums"]["test"],
            {"style": 25, "facility": 30},
        )

    @patch("src.training.week8_product._collect_repair_public_sources")
    def test_v2_split_allocation_preserves_scarce_category_support(self, collect):
        pool = []
        for category, count in (("hotel", 13), ("attraction", 7), ("restaurant", 30)):
            category_text = {
                "hotel": "Hotels",
                "attraction": "Museums",
                "restaurant": "Restaurants",
            }[category]
            pool.extend(
                {
                    "source_id": f"{category}-{index}",
                    "business_description": f"name | {category_text} | attrs",
                }
                for index in range(count)
            )
        collect.return_value = {"development": pool, "test": [], "train": []}
        selected = _collect_week8_v2_sources(
            Path("."),
            {},
            {},
            {"development": 14, "test": 14, "train": 12},
            {
                "development": {"hotel": 5, "attraction": 3, "restaurant": 2},
                "test": {"hotel": 5, "attraction": 3, "restaurant": 2},
                "train": {"hotel": 3, "attraction": 1, "restaurant": 2},
            },
            {},
            50,
        )
        all_ids = [row["source_id"] for rows in selected.values() for row in rows]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(
            sum("Hotels" in row["business_description"] for row in selected["test"]),
            5,
        )
        self.assertEqual(
            sum("Museums" in row["business_description"] for row in selected["train"]),
            1,
        )

    def test_config_locks_three_approved_prompt_roles_and_release_adapter(self):
        config = load_week8_product_config(CONFIG)
        self.assertEqual(
            set(config["prompts"]),
            {"current_release", "compact_field_check", "visual_evidence_guard"},
        )
        self.assertEqual(
            config["model"]["adapter_model_sha256"],
            "c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a",
        )
        self.assertEqual(config["week8"]["label_policy"], "programmatic_silver_only")

    def test_silver_target_never_promotes_metadata_price_to_visual_truth(self):
        target = product_silver_target(
            {
                "caption": "modern hotel lobby with a front desk",
                "business_description": "x|Hotels|RestaurantsPriceRange2: 4",
                "repair_mode": True,
            }
        )
        self.assertEqual(target["business_category"], "hotel")
        self.assertEqual(target["price_range"], "unknown")
        self.assertIn("price_range", target["unknown_fields"])
        self.assertNotIn("价位来自商家元数据", " ".join(target["inferred_attributes"]))

    def test_error_slices_cover_multilabel_unknown_and_ambiguous_subjects(self):
        target = {
            "business_category": "restaurant",
            "style_tags": ["modern", "cozy"],
            "visible_facilities": ["bar"],
        }
        slices = product_error_slices(
            {"caption": "several people near many restaurant tables"}, target
        )
        self.assertIn("multiple_or_ambiguous_subject", slices)
        self.assertIn("style_multilabel", slices)
        self.assertIn("facility_visible", slices)
        self.assertIn("should_use_unknown", slices)

    def test_prompt_assets_render_schema_bound_short_evidence_contract(self):
        for version in (
            "week8_product_field_check_v1",
            "week8_product_evidence_guard_v1",
        ):
            rendered = render_standard_prompt(
                ROOT,
                "image_product_search",
                {"images": [{"path": "data/samples/images/cafe_001.jpg"}], "text_constraints": None},
                version,
            )
            self.assertEqual(rendered["prompt_version"], version)
            self.assertEqual(rendered["schema_name"], "image_product_search_v1.schema.json")
            self.assertEqual(rendered["response_format"], {"type": "json_object"})
            combined = json.dumps(rendered["messages"], ensure_ascii=False)
            self.assertIn("observed_evidence", combined)
            self.assertNotIn("思维链", combined)

    def test_selection_requires_strict_improvement_and_non_regression(self):
        config = load_week8_product_config(CONFIG)
        summaries = {
            "current_release": summary(0.70),
            "compact_field_check": summary(0.72, 1100.0),
            "visual_evidence_guard": summary(0.75, 1200.0),
        }
        selected = select_prompt(config, summaries)
        self.assertEqual(selected["status"], "PROMPT_LOCKED")
        self.assertEqual(selected["selected_role"], "visual_evidence_guard")

        summaries["visual_evidence_guard"] = summary(0.70)
        summaries["compact_field_check"] = summary(0.69)
        blocked = select_prompt(config, summaries)
        self.assertEqual(blocked["status"], "SFT_ALLOWED_NO_PROMPT_WINNER")
        self.assertIsNone(blocked["selected_role"])


if __name__ == "__main__":
    unittest.main()

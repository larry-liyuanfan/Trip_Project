import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.prompting import render_standard_prompt
from src.training.week8_product import (
    load_week8_product_config,
    product_error_slices,
    product_silver_target,
    select_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/week8/product_understanding_v1.json"


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

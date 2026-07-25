import json
import tempfile
import unittest
from pathlib import Path


class Week4PromptOptimizationTest(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def test_selection_has_required_examples_and_disjoint_fixed_pilot(self):
        from src.evaluation.week4_prompting import load_week4_selection

        selection = load_week4_selection(
            self.PROJECT_ROOT
            / "configs/evaluation/week4_prompt_selection_v1.json"
        )
        for scenario, settings in selection["scenarios"].items():
            with self.subTest(scenario=scenario):
                self.assertEqual(len(settings["positive_example_ids"]), 5)
                self.assertEqual(len(settings["boundary_example_ids"]), 2)
                combined = (
                    settings["positive_example_ids"]
                    + settings["boundary_example_ids"]
                    + settings["pilot_sample_ids"]
                )
                self.assertEqual(len(combined), len(set(combined)))

    def test_four_and_seven_shot_use_exact_positive_boundary_counts(self):
        from src.evaluation.week4_prompting import (
            example_ids_for_variant,
            load_week4_selection,
        )

        selection = load_week4_selection(
            self.PROJECT_ROOT
            / "configs/evaluation/week4_prompt_selection_v1.json"
        )
        for scenario, settings in selection["scenarios"].items():
            four = example_ids_for_variant(selection, scenario, "fewshot_4_v1")
            seven = example_ids_for_variant(selection, scenario, "fewshot_7_v1")
            self.assertEqual(
                four,
                settings["positive_example_ids"][:3]
                + settings["boundary_example_ids"][:1],
            )
            self.assertEqual(
                seven,
                settings["positive_example_ids"]
                + settings["boundary_example_ids"],
            )

    def test_gold_mapping_does_not_invent_product_or_after_sales_fields(self):
        from src.evaluation.week4_prompting import annotation_to_output

        product = annotation_to_output(
            "image_product_search",
            {
                "business_category": "unknown",
                "price_range": "unknown",
                "style_tags": [],
                "visible_facilities": [],
            },
        )
        self.assertEqual(product["business_category"], "unknown")
        self.assertEqual(product["price_range"], "unknown")
        self.assertEqual(product["observed_evidence"], [])
        self.assertIsNone(product["confidence"])

        after_sales = annotation_to_output(
            "after_sales",
            {
                "issue_type": "facility_damage",
                "severity": "critical",
                "key_information": [],
                "ocr_ground_truth": None,
            },
        )
        self.assertEqual(after_sales["issue_type"], "facility_damage")
        self.assertEqual(after_sales["severity"], "critical")
        self.assertIsNone(after_sales["ocr_text"])
        self.assertIsNone(after_sales["issue_location"])

    def test_format_fallback_only_removes_optional_fence_and_never_repairs(self):
        from src.evaluation.format_fallback import parse_with_schema_fallback

        valid = {
            "business_category": "unknown",
            "style_tags": [],
            "visible_facilities": [],
            "price_range": "unknown",
            "observed_evidence": [],
            "inferred_attributes": [],
            "unknown_fields": [],
            "confidence": None,
        }
        raw = "```json\n" + json.dumps(valid) + "\n```"
        parsed = parse_with_schema_fallback(
            self.PROJECT_ROOT,
            "image_product_search",
            raw,
        )
        self.assertEqual(parsed["raw_output"], raw)
        self.assertTrue(parsed["fence_removed"])
        self.assertTrue(parsed["schema_valid"])

        missing = dict(valid)
        missing.pop("price_range")
        invalid = parse_with_schema_fallback(
            self.PROJECT_ROOT,
            "image_product_search",
            json.dumps(missing),
        )
        self.assertTrue(invalid["json_valid"])
        self.assertFalse(invalid["schema_valid"])
        self.assertNotIn("price_range", invalid["parsed_output"])

    def test_week4_prompt_wording_forbids_long_reasoning(self):
        directory = (
            self.PROJECT_ROOT
            / "configs/evaluation/prompts/week4_optimized_v1"
        )
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(directory.glob("*.yaml"))
        )
        self.assertIn("observed_evidence", combined)
        self.assertIn("constraint_check", combined)
        self.assertIn("不输出长篇推理", combined)
        self.assertNotIn("chain-of-thought", combined.lower())


if __name__ == "__main__":
    unittest.main()

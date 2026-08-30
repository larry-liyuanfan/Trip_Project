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
            / "configs/evaluation/week4_prompt_selection_v2.json"
        )
        self.assertEqual(
            selection["example_dataset_version"],
            "week4_demo_dev_v1",
        )
        self.assertEqual(
            selection["pilot_dataset_version"],
            "week3_evaluation_v2",
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
            / "configs/evaluation/week4_prompt_selection_v2.json"
        )
        for scenario, settings in selection["scenarios"].items():
            four = example_ids_for_variant(selection, scenario, "fewshot_4_v2")
            seven = example_ids_for_variant(selection, scenario, "fewshot_7_v2")
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

    def test_v2_selection_rejects_demo_test_identity_collision(self):
        from src.evaluation.week4_prompting import (
            Week4PromptError,
            validate_selection_records,
        )

        scenario = "image_product_search"
        selection = {
            "version": "week4_prompt_selection_v2",
            "example_dataset_version": "week4_demo_dev_v1",
            "pilot_dataset_version": "week3_evaluation_v2",
            "scenarios": {
                scenario: {
                    "positive_example_ids": [f"demo-{index}" for index in range(5)],
                    "boundary_example_ids": ["demo-5", "demo-6"],
                    "pilot_sample_ids": ["pilot-1"],
                }
            },
        }
        examples = {
            sample_id: self._selection_record(
                sample_id,
                scenario,
                "week4_demo_dev_v1",
                "development",
                f"source-demo-{index}",
                "a" * 64 if index == 0 else f"{index:064x}",
            )
            for index, sample_id in enumerate(
                selection["scenarios"][scenario]["positive_example_ids"]
                + selection["scenarios"][scenario]["boundary_example_ids"]
            )
        }
        pilots = {
            "pilot-1": self._selection_record(
                "pilot-1",
                scenario,
                "week3_evaluation_v2",
                "evaluation",
                "source-pilot",
                "a" * 64,
            )
        }

        with self.assertRaisesRegex(Week4PromptError, "image_sha256 collision"):
            validate_selection_records(selection, examples, pilots)

    @staticmethod
    def _selection_record(
        sample_id,
        scenario,
        dataset_version,
        split,
        source_id,
        image_sha256,
    ):
        return {
            "sample_id": sample_id,
            "scenario": scenario,
            "dataset_version": dataset_version,
            "split": split,
            "source_id": source_id,
            "input": {"images": [{"sha256": image_sha256}]},
            "annotation_status": "completed",
            "annotation": {},
        }

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
            / "configs/evaluation/prompts/week4_optimized_v2"
        )
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(directory.glob("*.yaml"))
        )
        self.assertIn("observed_evidence", combined)
        self.assertIn("constraint_check", combined)
        self.assertIn("不输出长篇推理", combined)
        self.assertNotIn("chain-of-thought", combined.lower())

    def test_itinerary_v2_demonstration_keeps_only_existing_gold_fields(self):
        from src.evaluation.week4_prompting import _demonstration_output

        output = _demonstration_output(
            "itinerary_planning",
            {
                "style_preferences": ["自然"],
                "hard_constraints": ["2天"],
                "soft_constraints": ["慢节奏"],
                "required_itinerary_elements": ["daily_schedule"],
            },
        )

        self.assertEqual(
            set(output),
            {
                "style_preferences",
                "hard_constraints",
                "soft_constraints",
                "required_itinerary_elements",
            },
        )
        self.assertNotIn("itinerary", output)
        self.assertNotIn("constraint_check", output)

    def test_model_request_errors_make_week4_run_ineligible(self):
        from src.evaluation.week4_runner import (
            Week4RunError,
            _ensure_no_model_request_errors,
        )

        with self.assertRaisesRegex(Week4RunError, "run is ineligible"):
            _ensure_no_model_request_errors(
                [
                    {
                        "sample_id": "sample-1",
                        "scenario": "itinerary_planning",
                        "error": "model_request_error: HTTP 400",
                    }
                ]
            )

    def test_baseline_comparison_has_no_cross_track_business_delta(self):
        from src.evaluation.week4_analysis import _build_runtime_comparisons

        optimized = [
            {
                "scenario": "itinerary_planning",
                "prompt_version": "standardized_v2",
                "json_compliance": 0.9,
                "schema_compliance": 0.87,
                "mean_total_tokens": 1900.0,
                "mean_latency_ms": 11000.0,
                "p95_latency_ms": 52000.0,
            }
        ]
        baseline = [
            {
                "scenario": "itinerary_planning",
                "json_valid": False,
                "schema_valid": False,
                "latency_ms": 7000.0,
                "token_usage": {"total_tokens": 1234},
            }
        ]

        row = _build_runtime_comparisons(
            optimized,
            baseline,
            baseline_run_id="qwen37-baseline",
        )[0]

        self.assertFalse(row["business_metrics_comparable"])
        self.assertNotIn("business_quality_delta", row)
        self.assertIsNone(row["baseline_mean_total_tokens"])
        self.assertEqual(row["baseline_token_status"], "PENDING_not_recorded")
        self.assertEqual(row["baseline_mean_latency_ms"], 7000.0)
        self.assertEqual(row["baseline_run_id"], "qwen37-baseline")


if __name__ == "__main__":
    unittest.main()

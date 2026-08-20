import unittest
from pathlib import Path

from src.evaluation.metrics import (
    SCENARIO_METRIC_NAMES,
    WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL,
    aggregate_scenario_scores,
    load_metric_aliases,
    score_sample,
    score_sample_with_gold_evaluable_support,
)
from src.training.week7_evaluation import _weighted_metric


ROOT = Path(__file__).resolve().parents[1]


def result(
    scenario: str, parsed_output: dict | None, *, structured_valid: bool,
) -> dict:
    return {
        "run_id": "protocol-v4-fixture",
        "sample_id": f"{scenario}-001",
        "scenario": scenario,
        "model_name": "fixture",
        "prompt_version": "week7_locked_v1",
        "parsed_output": parsed_output,
        "json_valid": structured_valid,
        "schema_valid": structured_valid,
        "latency_ms": 1.0,
    }


def aggregate_support(score: dict) -> dict[str, int]:
    aggregate = aggregate_scenario_scores([score])
    support = {}
    for metric_name in SCENARIO_METRIC_NAMES[score["scenario"]]:
        aggregate_name = (
            f"{metric_name}_macro"
            if f"{metric_name}_macro" in aggregate
            else metric_name
        )
        support[metric_name] = int(
            aggregate.get(
                f"{aggregate_name}_support_count", aggregate["sample_count"]
            )
        )
    return support


class Week7ProtocolV4SupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aliases = load_metric_aliases(
            ROOT / "configs/evaluation/metric_aliases_v1.json"
        )

    def assert_output_invariant_support(
        self,
        scenario: str,
        annotation: dict,
        perfect_output: dict,
        wrong_output: dict,
    ) -> tuple[dict, dict, dict]:
        invalid = score_sample_with_gold_evaluable_support(
            result(scenario, None, structured_valid=False),
            annotation,
            self.aliases,
        )
        perfect = score_sample_with_gold_evaluable_support(
            result(scenario, perfect_output, structured_valid=True),
            annotation,
            self.aliases,
        )
        wrong = score_sample_with_gold_evaluable_support(
            result(scenario, wrong_output, structured_valid=True),
            annotation,
            self.aliases,
        )
        expected_support = invalid["gold_evaluable_metric_support"]
        self.assertEqual(expected_support, perfect["gold_evaluable_metric_support"])
        self.assertEqual(expected_support, wrong["gold_evaluable_metric_support"])
        self.assertEqual(aggregate_support(invalid), aggregate_support(perfect))
        self.assertEqual(aggregate_support(invalid), aggregate_support(wrong))
        self.assertEqual(
            invalid["metric_support_protocol"],
            WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL,
        )
        return invalid, perfect, wrong

    def test_product_support_depends_only_on_gold_evaluability(self) -> None:
        annotation = {
            "business_category": "unknown",
            "price_range": "unknown",
            "style_tags": [],
            "visible_facilities": [],
        }
        invalid, perfect, _ = self.assert_output_invariant_support(
            "image_product_search",
            annotation,
            {
                "business_category": "unknown",
                "price_range": "unknown",
                "style_tags": [],
                "visible_facilities": [],
            },
            {
                "business_category": "hotel",
                "price_range": "premium",
                "style_tags": ["modern"],
                "visible_facilities": ["pool"],
            },
        )
        expected = invalid["gold_evaluable_metric_support"]
        self.assertFalse(expected["business_category_accuracy"])
        self.assertFalse(expected["price_range_accuracy"])
        self.assertTrue(expected["style_f1"])
        self.assertTrue(expected["facility_f1"])
        self.assertFalse(expected["label_completeness"])
        self.assertIsNone(invalid["label_completeness"])
        self.assertIsNone(perfect["label_completeness"])

    def test_protocol_does_not_change_legacy_invalid_product_support(self) -> None:
        annotation = {
            "business_category": "unknown",
            "price_range": "unknown",
            "style_tags": [],
            "visible_facilities": [],
        }
        raw_result = result(
            "image_product_search", None, structured_valid=False,
        )
        legacy = score_sample(raw_result, annotation, self.aliases)
        protocol = score_sample_with_gold_evaluable_support(
            raw_result, annotation, self.aliases,
        )
        self.assertEqual(legacy["label_completeness"], 0.0)
        self.assertIsNone(protocol["label_completeness"])

    def test_product_empty_lists_marked_unknown_are_not_evaluable(self) -> None:
        annotation = {
            "business_category": "unknown",
            "price_range": "unknown",
            "style_tags": [],
            "visible_facilities": [],
            "unknown_fields": [
                "business_category",
                "price_range",
                "style_tags",
                "visible_facilities",
            ],
        }
        invalid, perfect, wrong = self.assert_output_invariant_support(
            "image_product_search",
            annotation,
            {
                "business_category": "unknown",
                "price_range": "unknown",
                "style_tags": [],
                "visible_facilities": [],
            },
            {
                "business_category": "hotel",
                "price_range": "premium",
                "style_tags": ["modern"],
                "visible_facilities": ["pool"],
            },
        )
        self.assertFalse(any(invalid["gold_evaluable_metric_support"].values()))
        for score in (invalid, perfect, wrong):
            for metric_name in SCENARIO_METRIC_NAMES["image_product_search"]:
                self.assertIsNone(score[metric_name])
            self.assertEqual(score["multilabel_counts"], {})

    def test_after_sales_support_is_output_invariant(self) -> None:
        annotation = {
            "issue_type": "facility_damage",
            "severity": "high",
            "key_information": [],
            "ocr_ground_truth": None,
        }
        invalid, _, _ = self.assert_output_invariant_support(
            "after_sales",
            annotation,
            {
                "issue_type": "facility_damage",
                "severity": "high",
                "key_information": [],
                "ocr_text": None,
            },
            {
                "issue_type": "wrong",
                "severity": "low",
                "key_information": ["wrong"],
                "ocr_text": ["wrong"],
            },
        )
        support = invalid["gold_evaluable_metric_support"]
        self.assertTrue(support["key_information_f1"])
        self.assertFalse(support["ocr_recall"])
        self.assertFalse(support["ocr_exact_match"])

    def test_itinerary_support_is_output_invariant(self) -> None:
        annotation = {
            "hard_constraints": ["two days"],
            "soft_constraints": [],
            "required_itinerary_elements": None,
        }
        invalid, _, _ = self.assert_output_invariant_support(
            "itinerary_planning",
            annotation,
            {
                "hard_constraints": ["two days"],
                "soft_constraints": [],
                "required_itinerary_elements": [],
                "constraint_check": [],
            },
            {
                "hard_constraints": ["wrong"],
                "soft_constraints": ["wrong"],
                "required_itinerary_elements": ["wrong"],
                "constraint_check": [],
            },
        )
        support = invalid["gold_evaluable_metric_support"]
        self.assertTrue(support["constraint_recognition_accuracy"])
        self.assertTrue(support["constraint_check_coverage"])
        self.assertFalse(support["itinerary_element_f1"])
        self.assertNotIn(
            "required_itinerary_elements", invalid["multilabel_counts"]
        )

    def test_unsupported_product_collections_do_not_change_composite(self) -> None:
        annotation = {
            "business_category": "hotel",
            "price_range": "unknown",
            "style_tags": None,
            "visible_facilities": None,
        }
        outputs = (
            {
                "business_category": "hotel",
                "price_range": "unknown",
                "style_tags": [],
                "visible_facilities": [],
            },
            {
                "business_category": "hotel",
                "price_range": "premium",
                "style_tags": ["modern", "romantic"],
                "visible_facilities": ["pool", "parking"],
            },
        )
        composites = []
        supports = []
        for output in outputs:
            score = score_sample_with_gold_evaluable_support(
                result("image_product_search", output, structured_valid=True),
                annotation,
                self.aliases,
            )
            aggregate = aggregate_scenario_scores([score])
            composite, support = _weighted_metric(
                aggregate,
                {
                    "business_category_accuracy": 0.4,
                    "style_f1": 0.3,
                    "facility_f1": 0.3,
                },
            )
            composites.append(composite)
            supports.append(support)
            self.assertIsNone(aggregate["style_f1_macro"])
            self.assertIsNone(aggregate["facility_f1_macro"])
        self.assertEqual(composites, [1.0, 1.0])
        self.assertEqual(supports[0], supports[1])
        self.assertEqual(supports[0]["style_f1"], 0)
        self.assertEqual(supports[0]["facility_f1"], 0)


if __name__ == "__main__":
    unittest.main()

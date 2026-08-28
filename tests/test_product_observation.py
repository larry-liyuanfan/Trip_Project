import copy
import json
from pathlib import Path
import unittest
from src.inference.product_observation import map_observation, generate_observation
from src.inference.system_runtime import GenerationResult

ROOT = Path(__file__).resolve().parents[1]


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/week8/product_observation_v1.json").read_text(encoding="utf-8"))
        self.value = {"subject_kind": "dining_space", "subject_fact": "Tables and chairs in a dining room",
                      "style_evidence": [{"label": "modern", "fact": "Geometric furniture with plain walls"}],
                      "facility_evidence": [{"label": "seating", "fact": "Chairs at tables"}], "price_text": []}

    def test_mapping_preserves_positive_labels_and_unknown_price(self):
        target = map_observation(self.value, self.config)
        self.assertEqual(target["business_category"], "restaurant")
        self.assertEqual(target["style_tags"], ["modern"])
        self.assertEqual(target["visible_facilities"], ["seating"])
        self.assertEqual(target["unknown_fields"], ["price_range"])
        self.assertEqual(target["inferred_attributes"], [])

    def test_food_closeup_does_not_guess_venue_or_discard_positive_labels_silently(self):
        self.value["subject_kind"] = "food_closeup"
        with self.assertRaises(ValueError):
            map_observation(self.value, self.config)
        self.value.update(style_evidence=[], facility_evidence=[])
        result = map_observation(self.value, self.config)
        self.assertEqual(result["business_category"], "unknown")
        self.assertEqual(len(result["unknown_fields"]), 4)

    def test_negated_facility_fact_is_rejected(self):
        self.value["facility_evidence"] = [{"label": "parking", "fact": "Parking is not visible"}]
        with self.assertRaises(ValueError):
            map_observation(self.value, self.config)

    def test_duplicate_labels_are_not_silently_repaired(self):
        self.value["style_evidence"].append(copy.deepcopy(self.value["style_evidence"][0]))
        with self.assertRaises(ValueError):
            map_observation(self.value, self.config)

    def test_price_numbers_do_not_invent_price_tiers(self):
        self.value["price_text"] = ["$100"]
        self.assertEqual(map_observation(self.value, self.config)["price_range"], "unknown")

    def test_single_retry_preserves_invalid_raw_output(self):
        class Backend:
            def __init__(self, value):
                self.values = iter(["not-json", json.dumps(value)])
            def generate_with_usage(self, *args, **kwargs):
                return GenerationResult(content=next(self.values), input_tokens=10, output_tokens=20)
        result = generate_observation(Backend(self.value), "image.jpg", self.config)
        self.assertTrue(result["passed"])
        self.assertEqual(result["attempts"][0].raw_output, "not-json")
        self.assertIsNotNone(result["attempts"][0].error)
        self.assertIsNone(result["attempts"][1].error)


if __name__ == "__main__":
    unittest.main()

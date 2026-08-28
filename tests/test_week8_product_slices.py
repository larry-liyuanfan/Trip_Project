import unittest
from scripts.summarize_week8_product_slices import summarize


class ProductSliceTests(unittest.TestCase):
    def test_empty_style_references_still_count_false_positive(self):
        target = {"business_category": "unknown", "style_tags": [], "visible_facilities": [], "price_range": "unknown"}
        references = [{"sample_id": "s", "label_source": "model_generated_silver", "target": target,
                       "observation": {"subject_kind": "food_closeup"}}]
        baseline = {**target, "style_tags": ["modern"], "visible_facilities": ["parking"], "price_range": "budget"}
        result = summarize(references, {"baseline": {"s": baseline}, "candidate": {"s": target}})
        for key in ("food_closeup", "style_missing_or_extra", "facility_missing_or_extra", "price_without_evidence", "should_use_unknown"):
            self.assertEqual(result["error_slices"][key]["baseline"], {"support": 1, "errors": 1})
            self.assertEqual(result["error_slices"][key]["candidate"], {"support": 1, "errors": 0})
        self.assertFalse(result["human_accuracy_claim"])


if __name__ == "__main__":
    unittest.main()

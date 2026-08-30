import copy
import unittest

from scripts.summarize_week8_retry_semantics import semantic_counts


class RetrySemanticsTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{"case_id": "a", "sample_id": "s", "validation_error": "food error"},
                      {"case_id": "b", "sample_id": "s", "validation_error": "other error"}]
        self.target = {"business_category": "restaurant", "style_tags": ["modern"], "visible_facilities": ["seating"]}
        self.references = {"s": {"target": self.target}}
        self.records = [{"case_id": case["case_id"], "sample_id": "s", "passed": True,
                         "result": copy.deepcopy(self.target)} for case in self.cases]

    def test_repeated_image_cases_are_retained_and_reported_by_error(self):
        self.records[1]["result"]["visible_facilities"] = ["parking", "pool"]
        result = semantic_counts(self.cases, self.records, self.references)
        self.assertEqual(result["all_cases"]["count"], 2)
        self.assertEqual(result["all_cases"]["facility"], {"tp": 1, "fp": 2, "fn": 1})
        self.assertEqual(result["food error"]["facility"], {"tp": 1, "fp": 0, "fn": 0})

    def test_failure_gets_no_credit_even_with_a_saved_result(self):
        self.records[0]["passed"] = False
        result = semantic_counts(self.cases, self.records, self.references)["all_cases"]
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["category_errors"], 1)
        self.assertEqual(result["style"], {"tp": 1, "fp": 0, "fn": 1})

    def test_missing_reordered_or_reidentified_cases_are_rejected(self):
        for records in (self.records[:1], list(reversed(self.records)),
                        [{**self.records[0], "sample_id": "different"}, self.records[1]]):
            with self.assertRaises(ValueError):
                semantic_counts(self.cases, records, self.references)


if __name__ == "__main__":
    unittest.main()

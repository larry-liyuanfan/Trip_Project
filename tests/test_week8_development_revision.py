import copy
import unittest
from unittest.mock import patch

from scripts.compare_week8_development_revision import compare, SEMANTIC_FIELDS


class DevelopmentRevisionTests(unittest.TestCase):
    def comparison(self):
        return {"summaries": {"candidate": {"supports": {"samples": 60}, "reference_audit": {"reference": "fixed"},
                "metrics": {**dict.fromkeys(SEMANTIC_FIELDS, 0.7), "composite": 0.7},
                "latency_ms": {"mean": 1000, "p50": 1000, "p95": 1200}, "tokens": {"output_mean": 100}}}}

    @patch("scripts.compare_week8_development_revision.select_development_candidate", return_value={"selected_role": "candidate"})
    def test_unchanged_candidate_does_not_authorize_a_replacement_test(self, selection):
        original = self.comparison()
        self.assertFalse(compare(original, copy.deepcopy(original))["new_final_allowed"])

    @patch("scripts.compare_week8_development_revision.select_development_candidate", return_value={"selected_role": "candidate"})
    def test_actual_development_gain_can_form_a_new_candidate(self, selection):
        original = self.comparison()
        revised = copy.deepcopy(original)
        revised["summaries"]["candidate"]["metrics"]["composite"] = 0.8
        self.assertTrue(compare(original, revised)["semantic_gain"])

    @patch("scripts.compare_week8_development_revision.select_development_candidate", return_value={"selected_role": "candidate"})
    def test_speed_gain_requires_quality_and_token_evidence(self, selection):
        original = self.comparison()
        revised = copy.deepcopy(original)
        candidate = revised["summaries"]["candidate"]
        candidate["latency_ms"] = {"mean": 900, "p50": 900, "p95": 1100}
        self.assertFalse(compare(original, revised)["new_final_allowed"])
        candidate["tokens"]["output_mean"] = 80
        self.assertTrue(compare(original, revised)["speed_gain_without_semantic_regression"])
        candidate["metrics"]["style_f1"] = 0.6
        self.assertFalse(compare(original, revised)["new_final_allowed"])

    @patch("scripts.compare_week8_development_revision.select_development_candidate", return_value={"selected_role": "candidate"})
    def test_changed_support_or_reference_is_not_a_gain(self, selection):
        original = self.comparison()
        revised = copy.deepcopy(original)
        revised["summaries"]["candidate"]["supports"]["samples"] = 59
        with self.assertRaisesRegex(ValueError, "same fixed development"):
            compare(original, revised)

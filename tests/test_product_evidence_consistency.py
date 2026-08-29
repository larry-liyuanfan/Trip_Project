import unittest

from src.evaluation.product_evidence_consistency import (
    evidence_consistency_errors,
    summarize_evidence_consistency,
)


class ProductEvidenceConsistencyTests(unittest.TestCase):
    def test_cross_field_objects_do_not_support_seating_or_bar(self):
        observation = {"facility_evidence": [
            {"label": "seating", "fact": "Menu and napkins on table"},
            {"label": "bar", "fact": "Beer sign and bottles"},
            {"label": "parking", "fact": "Cars through windows"},
        ], "style_evidence": []}
        errors = evidence_consistency_errors(observation)
        self.assertEqual(len(errors), 3)
        self.assertTrue(all(error.endswith("object_mismatch") for error in errors))

    def test_direct_visible_objects_pass(self):
        observation = {"facility_evidence": [
            {"label": "seating", "fact": "Wooden chairs"},
            {"label": "bar", "fact": "Beer taps on bar"},
            {"label": "outdoor_seating", "fact": "Outdoor benches and tables"},
            {"label": "parking", "fact": "Visible parking spaces"},
        ], "style_evidence": [{"label": "natural", "fact": "Garden vegetation"}]}
        self.assertEqual(evidence_consistency_errors(observation), [])

    def test_depicted_nature_and_generic_frames_fail(self):
        observation = {"facility_evidence": [], "style_evidence": [
            {"label": "natural", "fact": "Desert mural with cacti"},
            {"label": "traditional", "fact": "Framed photos on walls"},
        ]}
        self.assertEqual(len(evidence_consistency_errors(observation)), 2)

    def test_summary_is_target_free_and_skips_failed_records(self):
        records = [{"observation": {"facility_evidence": [
            {"label": "seating", "fact": "Menu on table"},
            {"label": "bar", "fact": "Fixed bar with beer taps"},
        ], "style_evidence": []}}, {"passed": False}]
        summary = summarize_evidence_consistency(records)
        self.assertTrue(summary["target_free"])
        self.assertEqual(summary["selection_use"], "diagnostic_only")
        self.assertEqual(summary["records_read"], 2)
        self.assertEqual(summary["successful_observations"], 1)
        self.assertEqual(summary["positive_evidence_labels"], 2)
        self.assertEqual(summary["inconsistent_evidence_labels"], 1)
        self.assertEqual(summary["samples_with_errors"], 1)


if __name__ == "__main__":
    unittest.main()

import copy
import json
from pathlib import Path
import unittest
from src.evaluation.week8_visual_silver import score_paired, select_development_candidate, validate_locked_final, validate_incumbent_nonregression
from tests.test_system_runtime import PRODUCT_OUTPUT

ROOT = Path(__file__).resolve().parents[1]
AUDIT = {"protocol": "independent_image_model_observation_silver_v3", "metadata_supplied": False,
         "candidate_outputs_supplied": False, "test_rows_read": False, "model_independent": True,
         "reference_raw_sha256": "a" * 64}


def record(target, sample="s1"):
    return {"sample_id": sample, "passed": True, "result": target, "elapsed_ms": 10,
            "attempts": [{"raw_output": json.dumps(target), "error": None, "input_tokens": 10, "output_tokens": 20}]}


class VisualSilverTests(unittest.TestCase):
    def setUp(self):
        self.target = copy.deepcopy(PRODUCT_OUTPUT)
        self.references = [{"sample_id": "s1", "error": None, "label_source": "model_generated_silver", "target": self.target}]

    def score(self, prediction, audit=AUDIT):
        return score_paired(ROOT, self.references, [record(prediction)], reference_audit=audit)

    def test_missing_reference_audit_cannot_qualify_metadata_as_visual(self):
        for audit in (None, {**AUDIT, "metadata_supplied": True}, {**AUDIT, "model_independent": False}):
            with self.assertRaises(ValueError):
                self.score(self.target, audit)

    def test_false_positive_on_empty_reference_is_not_exempted(self):
        self.target["visible_facilities"] = []
        self.target["unknown_fields"].append("visible_facilities")
        prediction = copy.deepcopy(self.target)
        prediction["visible_facilities"] = ["parking"]
        prediction["unknown_fields"].remove("visible_facilities")
        result = self.score(prediction)
        self.assertEqual(result["multilabel_counts"]["facility"]["fp"], 1)
        self.assertEqual(result["metrics"]["facility_precision"], 0)
        self.assertEqual(result["metrics"]["unknown_accuracy"], 0.75)

    def test_raw_and_reported_result_must_agree(self):
        item = record(self.target)
        item["result"] = {**self.target, "business_category": "hotel"}
        with self.assertRaisesRegex(ValueError, "differs from raw"):
            score_paired(ROOT, self.references, [item], reference_audit=AUDIT)

    def test_missing_sample_is_rejected_not_removed(self):
        with self.assertRaises(ValueError):
            score_paired(ROOT, self.references, [], reference_audit=AUDIT)

    def test_perfect_prediction_is_silver_not_human_accuracy(self):
        result = self.score(self.target)
        self.assertFalse(result["human_visual_accuracy_claim"])
        self.assertEqual(result["metrics"]["composite"], 1)
        self.assertIsNone(result["metrics"]["price_range_accuracy"])

    def test_selection_is_not_promotion_and_rejects_nonfinite_metrics(self):
        good = self.score(self.target)
        wrong = copy.deepcopy(self.target)
        wrong["business_category"] = "other"
        baseline = self.score(wrong)
        result = select_development_candidate({"formal_adapter": baseline, "candidate": good})
        self.assertEqual(result["status"], "DEVELOPMENT_CANDIDATE")
        self.assertFalse(result["promotion_allowed"])
        good["metrics"]["composite"] = float("nan")
        with self.assertRaises(ValueError):
            select_development_candidate({"formal_adapter": baseline, "candidate": good})

    def test_label_support_cannot_be_reduced_to_qualify(self):
        good = self.score(self.target)
        wrong = copy.deepcopy(self.target)
        wrong["business_category"] = "other"
        baseline = self.score(wrong)
        good["supports"]["samples"] = 0
        result = select_development_candidate({"formal_adapter": baseline, "candidate": good})
        self.assertIsNone(result["selected_role"])
        self.assertIn("support_changed", result["failures"]["candidate"])

    def test_final_cannot_be_used_for_development_selection(self):
        audit = {**AUDIT, "test_rows_read": True, "candidate_lock_sha256": "b" * 64}
        good = score_paired(ROOT, self.references, [record(self.target)], reference_audit=audit, phase="final")
        wrong = copy.deepcopy(self.target)
        wrong["business_category"] = "other"
        baseline = score_paired(ROOT, self.references, [record(wrong)], reference_audit=audit, phase="final")
        self.assertEqual(validate_locked_final(baseline, good)["status"], "PASS")
        with self.assertRaises(ValueError):
            select_development_candidate({"formal_adapter": baseline, "candidate": good})
        with self.assertRaises(ValueError):
            score_paired(ROOT, self.references, [record(self.target)], reference_audit=audit)

    def test_final_without_prior_lock_fails_closed(self):
        with self.assertRaises(ValueError):
            score_paired(ROOT, self.references, [record(self.target)], reference_audit={**AUDIT, "test_rows_read": True}, phase="final")

    def test_additional_incumbent_check_does_not_replace_formal_improvement_requirement(self):
        audit = {**AUDIT, "test_rows_read": True, "candidate_lock_sha256": "b" * 64}
        good = score_paired(ROOT, self.references, [record(self.target)], reference_audit=audit, phase="final")
        self.assertEqual(validate_locked_final(good, good)["status"], "FAIL")
        self.assertEqual(validate_incumbent_nonregression(good, good)["status"], "PASS")
        wrong = copy.deepcopy(self.target)
        wrong["business_category"] = "other"
        worse = score_paired(ROOT, self.references, [record(wrong)], reference_audit=audit, phase="final")
        self.assertEqual(validate_incumbent_nonregression(good, worse)["status"], "FAIL")
        changed_support = copy.deepcopy(good)
        changed_support["supports"]["samples"] = 0
        self.assertEqual(validate_incumbent_nonregression(good, changed_support)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()

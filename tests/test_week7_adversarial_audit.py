import copy
import unittest
from pathlib import Path

from src.training.week7_adversarial_audit import (
    Week7AdversarialAuditError,
    audit_week7_repository,
    load_week7_evidence,
    validate_week7_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


class Week7AdversarialAuditTests(unittest.TestCase):
    def test_repository_evidence_passes_conservatively(self):
        result = audit_week7_repository(ROOT)
        self.assertEqual(result["status"], "PASS_WITH_KNOWN_IMMUTABLE_LIMITATION")
        self.assertEqual(result["counterfactual_probes"]["rejected"], 11)
        self.assertTrue(result["verdict"]["implementation_ready_for_dev_integration"])
        self.assertEqual(
            result["verdict"]["full_week7_claim_gate"],
            "FAIL_KNOWN_V3_TEST_DIALOGUE_INVALID",
        )
        self.assertFalse(result["verdict"]["stg_or_release_promotion_allowed"])

    def test_agent_filled_human_score_is_rejected(self):
        evidence, _ = load_week7_evidence(ROOT)
        mutated = copy.deepcopy(evidence)
        mutated["human"]["human_validation"]["agent_filled_score_count"] = 1
        with self.assertRaisesRegex(Week7AdversarialAuditError, "AGENT_IMPERSONATED_HUMAN"):
            validate_week7_evidence(ROOT, mutated)

    def test_failed_mdpo_cannot_be_selected(self):
        evidence, _ = load_week7_evidence(ROOT)
        mutated = copy.deepcopy(evidence)
        mutated["mdpo"]["result"]["selected_for_use"] = True
        with self.assertRaisesRegex(Week7AdversarialAuditError, "FAILED_MDPO_SELECTED"):
            validate_week7_evidence(ROOT, mutated)


if __name__ == "__main__":
    unittest.main()

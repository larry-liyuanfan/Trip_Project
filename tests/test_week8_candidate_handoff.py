import copy
import unittest

from scripts.verify_week8_candidate_handoff import bind_acceptance


class CandidateHandoffTests(unittest.TestCase):
    def setUp(self):
        self.release = {"release_id": "candidate", "config_sha256": "config", "adapter_model_sha256": "adapter"}
        self.acceptance = {"status": "PASS", "candidate_quality_accepted": True, "human_annotation_count": 0,
                           "human_visual_accuracy_claim": False, "label_source": "model_generated_silver",
                           "formal_release_replaced": False, "release_id": "candidate",
                           "release_config_sha256": "config", "adapter_model_sha256": "adapter"}

    def test_candidate_acceptance_must_match_every_release_identity(self):
        bind_acceptance(self.acceptance, self.release)
        for key in ("release_id", "release_config_sha256", "adapter_model_sha256"):
            value = copy.deepcopy(self.acceptance)
            value[key] = "different"
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                bind_acceptance(value, self.release)

    def test_failed_evidence_or_human_claim_cannot_be_promoted(self):
        for key, changed in (("status", "FAIL"), ("candidate_quality_accepted", False),
                             ("label_source", "human"), ("human_visual_accuracy_claim", True),
                             ("formal_release_replaced", True)):
            value = copy.deepcopy(self.acceptance)
            value[key] = changed
            with self.assertRaisesRegex(ValueError, "candidate quality evidence"):
                bind_acceptance(value, self.release)

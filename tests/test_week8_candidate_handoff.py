import copy
import unittest

from scripts.verify_week8_candidate_handoff import bind_acceptance, validate_inference_coverage


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

    def test_baseline_failures_are_retained_without_being_counted_as_candidate_failure(self):
        roles = {"formal_adapter", "incumbent", "locked_candidate"}
        records = {role: [{"passed": True}, {"passed": role == "locked_candidate"}] for role in roles}
        summary = {"roles": {role: {"count": 2, "failures": int(role != "locked_candidate")} for role in roles}}
        scores = {role: {"metrics": {"request_failure_rate": 0.5 if role != "locked_candidate" else 0.0}} for role in roles}
        validate_inference_coverage(summary, 2, roles, scores, records)
        records["locked_candidate"][1]["passed"] = False
        summary["roles"]["locked_candidate"]["failures"] = 1
        scores["locked_candidate"]["metrics"]["request_failure_rate"] = 0.5
        with self.assertRaisesRegex(ValueError, "candidate inference"):
            validate_inference_coverage(summary, 2, roles, scores, records)

    def test_packaged_baseline_failure_count_cannot_be_hidden(self):
        roles = {"formal_adapter", "locked_candidate"}
        summary = {"roles": {role: {"count": 1, "failures": 0} for role in roles}}
        scores = {role: {"metrics": {"request_failure_rate": 0}} for role in roles}
        with self.assertRaisesRegex(ValueError, "disclosure"):
            validate_inference_coverage(summary, 1, roles, scores,
                {"formal_adapter": [{"passed": False}], "locked_candidate": [{"passed": True}]})

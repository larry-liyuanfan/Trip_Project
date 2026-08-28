import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.validate_week8_correction_evidence import validate_nonregression, validate_correction_evidence


def semantics():
    group = {"count": 3, "failures": 0, "category_errors": 1,
             "style": {"tp": 2, "fp": 1, "fn": 1}, "facility": {"tp": 2, "fp": 1, "fn": 1}}
    return {"test_rows_read": False, "human_annotation_count": 0, "selection_allowed": False,
            "new_model_requests": 0, "label_source": "model_generated_silver", "case_count": 3,
            "unique_images": 2, "reference_audit": {"reference_raw_sha256": "reference"},
            "profiles": {role: {"all_cases": copy.deepcopy(group), "food error": copy.deepcopy(group)}
                         for role in ("legacy_correction", "subject_schema_correction")}}


class CorrectionAcceptanceTests(unittest.TestCase):
    def test_equal_or_improved_error_slices_are_accepted(self):
        value = semantics()
        validate_nonregression(value)
        for group in value["profiles"]["subject_schema_correction"].values():
            group["facility"] = {"tp": 3, "fp": 0, "fn": 0}
        validate_nonregression(value)

    def test_known_regressions_and_reduced_support_are_rejected(self):
        changes = ({"failures": 1}, {"category_errors": 2}, {"count": 2},
                   {"facility": {"tp": 1, "fp": 2, "fn": 2}},
                   {"style": {"tp": 2, "fp": 2, "fn": 1}},
                   {"style": {"tp": 2, "fp": 0, "fn": 0}})
        for change in changes:
            value = semantics()
            value["profiles"]["subject_schema_correction"]["food error"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_nonregression(value)

    def test_correction_subset_cannot_become_final_or_human_evidence(self):
        for change in ({"test_rows_read": True}, {"human_annotation_count": 1}, {"selection_allowed": True}):
            with self.assertRaises(ValueError):
                validate_nonregression({**semantics(), **change})

    def fixture(self, root):
        def put(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
        put("observation.json", {"correction_protocol": "food_conflict_schema_v1"})
        put("probe.json", {"profiles": {"subject_schema_correction": "observation.json"}, "output_root": "run"})
        put("run/identity.json", {"base_model": "m", "base_revision": "r", "adapter_sha256": "a"})
        put("candidate.json", {"model": {"base_model": "m", "base_revision": "r", "adapter_model_sha256": "a"}})
        receipt = {"status": "REPLAY_VERIFIED", "generation_root": "/original/project"}
        put("receipt.json", receipt)
        put("semantics.json", semantics())
        return {"candidate_observation": "observation.json", "candidate_release": "candidate.json",
                "correction_diagnostic_config": "probe.json", "correction_diagnostic_replay": "receipt.json",
                "correction_diagnostic_semantics": "semantics.json", "development_reference_revision": "reference.json"}, receipt

    def test_new_decoder_cannot_skip_the_required_correction_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self.fixture(root)
            del config["correction_diagnostic_replay"]
            with self.assertRaisesRegex(ValueError, "complete correction"):
                validate_correction_evidence(root, config)
            (root / "legacy.json").write_text("{}", encoding="utf-8")
            self.assertIsNone(validate_correction_evidence(root, {"candidate_observation": "legacy.json"}))

    def test_exact_execution_and_semantic_replay_are_bound_before_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, receipt = self.fixture(root)
            with patch("scripts.validate_week8_correction_evidence.verify_retry", return_value=receipt) as replay, patch(
                    "scripts.validate_week8_correction_evidence.build_summary", return_value=semantics()) as summary:
                result = validate_correction_evidence(root, config)
                self.assertEqual(result["status"], "PASS")
                self.assertIn("run/identity.json", result["artifact_sha256"])
                self.assertEqual(replay.call_args.args[-1], "/original/project")
                replay.return_value = {**receipt, "tampered": True}
                with self.assertRaisesRegex(ValueError, "execution receipt"):
                    validate_correction_evidence(root, config)
                replay.return_value = receipt
                summary.return_value = {**semantics(), "tampered": True}
                with self.assertRaisesRegex(ValueError, "semantic evidence"):
                    validate_correction_evidence(root, config)

    def test_other_observation_or_model_cannot_reuse_the_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self.fixture(root)
            (root / "other.json").write_text('{"correction_protocol":"food_conflict_schema_v1"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected observation"):
                validate_correction_evidence(root, {**config, "candidate_observation": "other.json"})
            (root / "candidate.json").write_text(json.dumps({"model": {"base_model": "different"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model identity"):
                validate_correction_evidence(root, config)


if __name__ == "__main__":
    unittest.main()

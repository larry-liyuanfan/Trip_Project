import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.review_week8_observation_retry import load_continuation, run
from src.training.week7_data import sha256_file


class ObservationRetryContinuationTests(unittest.TestCase):
    def fixture(self, root):
        def write(path, value):
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(value) + "\n", encoding="utf-8")
        original = {"output_root": "partial", "development_manifest": "manifest.jsonl",
                    "development_manifest_sha256": "m", "development_count": 2, "release_config": "release.json",
                    "profiles": {"candidate": "observation.json"}, "sources": [], "final_test_access": False, "human_annotation_count": 0}
        write("original.json", original)
        write("observation.json", {})
        cases = [{"case_id": "a"}, {"case_id": "b"}]
        (root / "partial").mkdir()
        (root / "partial/cases.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")
        write("partial/candidate.jsonl", {"case_id": "a", "passed": True})
        audit = {"case_count": 2}
        identity = {**audit, "config_sha256": sha256_file(root / "original.json"), "git_commit": "a" * 40,
                    "runner_sha256": hashlib.sha256(b"old runner").hexdigest(), "final_test_access": False,
                    "reference_targets_supplied": False, "human_annotation_count": 0,
                    "base_model": "model", "base_revision": "revision", "adapter_sha256": "adapter",
                    "profile_config_hashes": {"candidate": sha256_file(root / "observation.json")}}
        for key, path in (("correction_implementation_sha256", "src/inference/product_observation.py"),
                          ("decoder_implementation_sha256", "src/inference/observation_constraints.py"),
                          ("backend_implementation_sha256", "src/inference/system_runtime.py")):
            write(path, "unchanged implementation")
            identity[key] = sha256_file(root / path)
        write("partial/identity.json", identity)
        specification = {"output_root": "partial", "config_path": "original.json", "identity_sha256": sha256_file(root / "partial/identity.json"),
                         "prefixes": {"candidate": {"count": 1, "sha256": sha256_file(root / "partial/candidate.jsonl")}},
                         "interruption": {"job_id": "123", "state": "TIMEOUT"}}
        return {**original, "output_root": "continued", "continue_incomplete": specification}, cases, audit

    @patch("scripts.review_week8_observation_retry.subprocess.check_output", return_value=b"old runner")
    @patch("scripts.verify_week8_observation_retry.replay_records")
    def test_exact_prefix_is_replayed_and_interruption_cannot_be_hidden(self, replay, source):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, cases, audit = self.fixture(root)
            before = sha256_file(root / "partial/candidate.jsonl")
            prefixes, provenance = load_continuation(root, config, cases, audit, "/original/root")
            self.assertEqual(prefixes["candidate"], [{"case_id": "a", "passed": True}])
            self.assertEqual(replay.call_args.args[1], cases[:1])
            self.assertEqual(replay.call_args.args[-1], "/original/root")
            self.assertEqual(provenance["interruption"]["state"], "TIMEOUT")
            self.assertTrue(provenance["inflight_attempt_may_have_been_interrupted"])
            self.assertFalse(provenance["old_files_modified"])
            self.assertEqual(before, sha256_file(root / "partial/candidate.jsonl"))

    @patch("scripts.review_week8_observation_retry.subprocess.check_output", return_value=b"old runner")
    @patch("scripts.verify_week8_observation_retry.replay_records")
    def test_completed_execution_and_changed_source_are_not_resumable(self, replay, source):
        for mutation in ("completed", "runtime", "profile", "hash", "not_timeout", "output_collision"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, cases, audit = self.fixture(root)
                if mutation == "completed":
                    (root / "partial/summary.json").write_text("{}", encoding="utf-8")
                elif mutation == "runtime":
                    (root / "src/inference/system_runtime.py").write_text("changed", encoding="utf-8")
                elif mutation == "profile":
                    config["profiles"] = {"different": "observation.json"}
                elif mutation == "hash":
                    config["continue_incomplete"]["prefixes"]["candidate"]["sha256"] = "changed"
                elif mutation == "not_timeout":
                    config["continue_incomplete"]["interruption"]["state"] = "COMPLETED"
                else:
                    config["output_root"] = "partial"
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_continuation(root, config, cases, audit)

    @patch("scripts.review_week8_observation_retry.subprocess.check_output", return_value=b"old runner")
    @patch("scripts.verify_week8_observation_retry.replay_records")
    def test_skipping_an_earlier_case_cannot_be_a_valid_prefix(self, replay, source):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, cases, audit = self.fixture(root)
            path = root / "partial/candidate.jsonl"
            path.write_text(json.dumps({"case_id": "b", "passed": True}) + "\n", encoding="utf-8")
            config["continue_incomplete"]["prefixes"]["candidate"]["sha256"] = sha256_file(path)
            with self.assertRaisesRegex(ValueError, "skipped"):
                load_continuation(root, config, cases, audit)

    def test_normal_execution_has_no_continuation(self):
        self.assertEqual(load_continuation(Path("."), {}, [], {}), ({}, None))

    def test_audit_only_also_rejects_invalid_continuation_before_model_loading(self):
        with patch("scripts.review_week8_observation_retry.read_json", return_value={}), patch(
                "scripts.review_week8_observation_retry.load_cases", return_value=([], {})), patch(
                "scripts.review_week8_observation_retry.load_continuation", side_effect=ValueError("bad prefix")), patch(
                "scripts.review_week8_observation_retry.TransformersPeftBackend") as backend:
            with self.assertRaisesRegex(ValueError, "bad prefix"):
                run(Path("unused.json"), audit_only=True)
            backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()

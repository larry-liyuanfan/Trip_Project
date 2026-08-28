import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_week8_visual_final import final_context, source_hash, score, validate_development_identity, inference_roles
from src.inference.product_observation import canonical_config_sha256
from src.training.week7_data import sha256_file


class DevelopmentBindingTests(unittest.TestCase):
    def test_a_self_consistent_but_untested_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dev").mkdir()
            def put(name, value):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            put("observation.json", {"protocol": "tested"})
            specification = {"output_root": "dev", "final_test_access": False,
                             "reference_manifest_sha256": "manifest", "reference_raw_sha256": "reference",
                             "observation_profile_configs": {"observation_enhanced_base": "observation.json"}}
            put("development.json", specification)
            identity = {"config_sha256": sha256_file(root / "development.json"), "test_rows_read": False,
                        "development_sha256": "manifest", "base_model": "model", "base_revision": "revision",
                        "adapter_sha256": "adapter", "observation_profile_config_hashes": {
                            "observation_enhanced_base": sha256_file(root / "observation.json")}}
            put("dev/identity.json", identity)
            candidate = {"product_pipeline": {"config": "observation.json"}, "model": {"base_model": "model",
                         "base_revision": "revision", "adapter_model_sha256": "adapter",
                         "adapter_disabled_scenarios": ["image_product_search"]}}
            put("candidate.json", candidate)
            config = {"development_config": "development.json", "candidate_release": "candidate.json",
                      "candidate_observation": "observation.json"}
            comparison = {"generation_identity_sha256": sha256_file(root / "dev/identity.json"),
                          "reference_raw_sha256": "reference", "selection": {"selected_role": "observation_enhanced_base"}}
            self.assertEqual(validate_development_identity(root, config, comparison), sha256_file(root / "dev/identity.json"))
            config["development_reference_revision"] = "reference_revision.json"
            candidate["generation"] = {"visual_max_pixels": None}
            put("candidate.json", candidate)
            comparison["incumbent_comparison"] = {"status": "KEEP_V9_CANDIDATE", "selected_role": None}
            with patch("scripts.score_week8_reference_revision.build_comparison", return_value=comparison):
                with self.assertRaisesRegex(ValueError, "not improved"):
                    validate_development_identity(root, config, comparison)
            comparison["incumbent_comparison"] = {"status": "IMPROVED_DEVELOPMENT_CANDIDATE", "selected_role": "observation_enhanced_base"}
            with patch("scripts.score_week8_reference_revision.build_comparison", return_value=comparison):
                self.assertEqual(validate_development_identity(root, config, comparison), sha256_file(root / "dev/identity.json"))
                candidate["generation"]["visual_max_pixels"] = 131072
                put("candidate.json", candidate)
                with self.assertRaisesRegex(ValueError, "visual limits"):
                    validate_development_identity(root, config, comparison)
            with patch("scripts.score_week8_reference_revision.build_comparison", return_value={**comparison, "tampered": True}):
                with self.assertRaisesRegex(ValueError, "raw reference revision replay"):
                    validate_development_identity(root, config, comparison)
            del config["development_reference_revision"]
            put("untested.json", {"protocol": "untested"})
            candidate["product_pipeline"]["config"] = "untested.json"
            put("candidate.json", candidate)
            config["candidate_observation"] = "untested.json"
            with self.assertRaisesRegex(ValueError, "development-tested observation"):
                validate_development_identity(root, config, comparison)


class FinalExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {"output_root": "holdout"}
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps(self.config), encoding="utf-8")
        (self.root / "holdout").mkdir()
        lock = {"config_canonical_sha256": canonical_config_sha256(self.config),
                "source_files_lf_sha256": {"code.py": "a"}, "selection_only_used_development": True,
                "data_lock_sha256": "data", "final_roles": ["teacher", "inference"]}
        lock["lock_sha256"] = canonical_config_sha256(lock)
        (self.root / "holdout/candidate_lock.json").write_text(json.dumps(lock), encoding="utf-8")
        self.protocol = patch("scripts.run_week8_visual_final.protocol_files", return_value={"code.py": "a"})
        self.protocol.start()
        self.addCleanup(self.protocol.stop)
        self.data = patch("scripts.run_week8_visual_final.validate_holdout", return_value=([{"sample_id": "final-1"}], {"lock_sha256": "data", "manifest_sha256": "manifest"}))
        self.data.start()
        self.addCleanup(self.data.stop)

    def test_each_final_role_is_consumed_even_before_generation(self):
        final_context(self.root, self.path, "teacher")
        with self.assertRaises(FileExistsError):
            final_context(self.root, self.path, "teacher")
        final_context(self.root, self.path, "inference")
        with self.assertRaises(FileExistsError):
            final_context(self.root, self.path, "inference")

    def test_new_final_keeps_formal_and_incumbent_roles_without_ambiguous_config(self):
        self.assertEqual(inference_roles({}), ["formal_adapter", "locked_candidate"])
        self.assertEqual(inference_roles({"incumbent_release": "release.json", "incumbent_observation": "obs.json"}),
                         ["formal_adapter", "incumbent", "locked_candidate"])
        with self.assertRaisesRegex(ValueError, "both release"):
            inference_roles({"incumbent_release": "release.json"})

    def test_changed_code_or_unknown_role_cannot_start_final(self):
        with patch("scripts.run_week8_visual_final.protocol_files", return_value={"code.py": "changed"}):
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                final_context(self.root, self.path, "teacher")
        with self.assertRaisesRegex(ValueError, "unknown once-only"):
            final_context(self.root, self.path, "candidate_2")
        self.assertFalse((self.root / "holdout/teacher").exists())

    def test_source_identity_is_cross_platform_but_not_content_insensitive(self):
        a, b = self.root / "a.py", self.root / "b.py"
        a.write_bytes(b"one\ntwo\n")
        b.write_bytes(b"one\r\ntwo\r\n")
        self.assertEqual(source_hash(a), source_hash(b))
        b.write_bytes(b"one\nchanged\n")
        self.assertNotEqual(source_hash(a), source_hash(b))

    def test_scoring_revalidates_the_entire_candidate_lock(self):
        path = self.root / "holdout/candidate_lock.json"
        lock = json.loads(path.read_text(encoding="utf-8"))
        lock["selected_role"] = "tampered"
        path.write_text(json.dumps(lock), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unlocked implementation"):
            score(self.root, self.path)


if __name__ == "__main__":
    unittest.main()

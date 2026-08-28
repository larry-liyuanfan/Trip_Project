from pathlib import Path
import tempfile
import unittest

from scripts.build_release_bundle import ROOT, RUNTIME_PATHS, _archive, verify_runtime_archive


class RuntimeArchiveTests(unittest.TestCase):
    def test_exact_v9_candidate_is_self_contained(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "runtime.tar.gz"
            _archive(archive, [*((ROOT / path, Path(path)) for path in RUNTIME_PATHS),
                              (ROOT / "configs/releases/qwen3_vl_system_week8_v9.json", Path("release/release_config.json"))])
            result = verify_runtime_archive(archive)
            self.assertEqual(result["release_id"], "trip-qwen3-vl-8b-week8-visual-silver-v9")
            self.assertTrue(result["observation_loaded"])

    def test_observation_release_imports_without_source_worktree(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "runtime.tar.gz"
            _archive(archive, [*((ROOT / path, Path(path)) for path in RUNTIME_PATHS),
                              (ROOT / "configs/releases/qwen3_vl_week8_observation_probe_v2.json", Path("release/release_config.json"))])
            result = verify_runtime_archive(archive)
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["observation_loaded"])
            self.assertGreater(result["route_count"], 4)

    def test_missing_runtime_dependency_is_not_masked_by_current_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "runtime.tar.gz"
            paths = [path for path in RUNTIME_PATHS if path != "src/data/product_labels.py"]
            _archive(archive, [*((ROOT / path, Path(path)) for path in paths),
                              (ROOT / "configs/releases/qwen3_vl_week8_observation_probe_v2.json", Path("release/release_config.json"))])
            with self.assertRaisesRegex(ValueError, "isolated runtime import failed"):
                verify_runtime_archive(archive)


if __name__ == "__main__":
    unittest.main()

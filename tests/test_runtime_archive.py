from pathlib import Path
import json
import tempfile
import unittest

from scripts.build_release_bundle import ROOT, RUNTIME_PATHS, _archive, verify_runtime_archive, runtime_paths
from src.inference.product_observation import canonical_config_sha256


class RuntimeArchiveTests(unittest.TestCase):
    def test_exact_v9_candidate_is_self_contained(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "runtime.tar.gz"
            _archive(archive, [*((ROOT / path, Path(path)) for path in RUNTIME_PATHS),
                              (ROOT / "configs/releases/qwen3_vl_system_week8_v9.json", Path("release/release_config.json"))])
            result = verify_runtime_archive(archive)
            self.assertEqual(result["release_id"], "trip-qwen3-vl-8b-week8-visual-silver-v9")
            self.assertTrue(result["observation_loaded"])
            self.assertTrue(result["required_business_paths_present"])
            self.assertIn("/v1/tasks/image-product-search", result["registered_paths"])

    def test_importable_app_without_business_routes_is_not_a_valid_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            fake_app = temporary / "app.py"
            fake_app.write_text("from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8")
            sources = [(ROOT / path, Path(path)) for path in RUNTIME_PATHS if path != "src/api"]
            sources.extend((path, path.relative_to(ROOT)) for path in (ROOT / "src/api").glob("*.py") if path.name != "app.py")
            sources.extend([(fake_app, Path("src/api/app.py")),
                            (ROOT / "configs/releases/qwen3_vl_system_week8_v9.json", Path("release/release_config.json"))])
            archive = temporary / "runtime.tar.gz"
            _archive(archive, sources)
            with self.assertRaisesRegex(ValueError, "required business routes missing"):
                verify_runtime_archive(archive)

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

    def test_new_selected_observation_config_is_packaged_and_importable(self):
        release = json.loads((ROOT / "configs/releases/qwen3_vl_system_week8_v9.json").read_text(encoding="utf-8"))
        selected = "configs/week8/product_observation_scope_repair_v1.json"
        observation = json.loads((ROOT / selected).read_text(encoding="utf-8"))
        release["product_pipeline"].update(config=selected, config_canonical_sha256=canonical_config_sha256(observation))
        paths = runtime_paths(release)
        self.assertIn(selected, paths)
        self.assertEqual(len(paths), len(set(paths)))
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            config = temporary / "release_config.json"
            config.write_text(json.dumps(release), encoding="utf-8")
            archive = temporary / "runtime.tar.gz"
            _archive(archive, [*((ROOT / path, Path(path)) for path in paths),
                               (config, Path("release/release_config.json"))])
            result = verify_runtime_archive(archive)
            self.assertTrue(result["observation_loaded"])
            self.assertTrue(result["required_business_paths_present"])

    def test_runtime_config_cannot_package_secrets_or_escape_repository(self):
        for path in (".env", "secrets/key.json", "configs/../../secret.json", "C:/private.json", "/tmp/secret.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                runtime_paths({"product_pipeline": {"config": path}})
        old = {"product_pipeline": {"config": "configs/week8/product_observation_v3.json"}}
        self.assertEqual(runtime_paths(old), RUNTIME_PATHS)


if __name__ == "__main__":
    unittest.main()

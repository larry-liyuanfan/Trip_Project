import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_release_bundle import (
    ROOT,
    RUNTIME_PATHS,
    _archive,
    runtime_paths,
    verify_runtime_archive,
)


FINAL_CONFIG = ROOT / "configs/releases/qwen3_vl_system_final_v1.json"


class RuntimeArchiveTests(unittest.TestCase):
    def test_final_v1_runtime_is_self_contained(self):
        release = json.loads(FINAL_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "runtime.tar.gz"
            _archive(
                archive,
                [
                    *((ROOT / path, Path(path)) for path in runtime_paths(release)),
                    (FINAL_CONFIG, Path("release/release_config.json")),
                ],
            )
            result = verify_runtime_archive(archive)

        self.assertEqual(result["release_id"], "trip-qwen3-vl-8b-week8-final-v1")
        self.assertTrue(result["observation_loaded"])
        self.assertTrue(result["required_business_paths_present"])

    def test_importable_app_without_business_routes_is_rejected(self):
        release = json.loads(FINAL_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            fake_app = temporary / "app.py"
            fake_app.write_text("from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8")
            sources = [(ROOT / path, Path(path)) for path in runtime_paths(release) if path != "src/api"]
            sources.extend(
                (path, path.relative_to(ROOT))
                for path in (ROOT / "src/api").glob("*.py")
                if path.name != "app.py"
            )
            sources.extend(
                [(fake_app, Path("src/api/app.py")), (FINAL_CONFIG, Path("release/release_config.json"))]
            )
            archive = temporary / "runtime.tar.gz"
            _archive(archive, sources)
            with self.assertRaisesRegex(ValueError, "required business routes missing"):
                verify_runtime_archive(archive)

    def test_missing_runtime_dependency_is_not_masked_by_checkout(self):
        release = json.loads(FINAL_CONFIG.read_text(encoding="utf-8"))
        paths = [path for path in runtime_paths(release) if path != "src/data/product_labels.py"]
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "runtime.tar.gz"
            _archive(
                archive,
                [
                    *((ROOT / path, Path(path)) for path in paths),
                    (FINAL_CONFIG, Path("release/release_config.json")),
                ],
            )
            with self.assertRaisesRegex(ValueError, "isolated runtime import failed"):
                verify_runtime_archive(archive)

    def test_runtime_config_cannot_escape_or_reference_missing_config(self):
        for path in (".env", "secrets/key.json", "configs/../../secret.json", "C:/private.json", "/tmp/secret.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                runtime_paths({"product_pipeline": {"config": path}})
        with self.assertRaises(FileNotFoundError):
            runtime_paths({"product_pipeline": {"config": "configs/week8/missing.json"}})
        release = json.loads(FINAL_CONFIG.read_text(encoding="utf-8"))
        paths = runtime_paths(release)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertGreater(len(paths), len(RUNTIME_PATHS) - 1)


if __name__ == "__main__":
    unittest.main()

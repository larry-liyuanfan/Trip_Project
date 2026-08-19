from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.week7_data import canonical_sha256, sha256_file
from src.training.week7_inference import run_schema_experiment
from src.training.week7_qlora import Week7TrainingError


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v2.json"


class Week7SchemaEvidenceTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, str]:
        shutil.copytree(ROOT / "configs/evaluation", root / "configs/evaluation")
        config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        config["dataset"]["dataset_version"] = "week7_schema_evidence_unit"
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        lock_root = root / "outputs/week7/locked_data/week7_schema_evidence_unit"
        lock_root.mkdir(parents=True)
        rows = [
            {"sample_id": f"schema-{scenario}", "scenario": scenario}
            for scenario in ("image_product_search", "after_sales", "itinerary_planning")
        ]
        with (lock_root / "development.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        lock = {"config_sha256": sha256_file(config_path)}
        lock["lock_sha256"] = canonical_sha256(lock)
        (lock_root / "dataset_lock.json").write_text(json.dumps(lock), encoding="utf-8")
        return config_path, config["base_model"]

    def test_all_requests_failed_cannot_be_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, base_model = self._fixture(root)
            output = root / "schema-output"
            with (
                patch(
                    "src.training.week7_inference._request_model_registry",
                    return_value=[base_model],
                ),
                patch("src.training.week7_inference._openai_messages", return_value=[]),
                patch(
                    "src.training.week7_inference._request_completion",
                    return_value=("", 1.0, "HTTPError: unavailable", None),
                ),
            ):
                with self.assertRaisesRegex(Week7TrainingError, "not COMPLETED"):
                    run_schema_experiment(
                        root, config_path, output,
                        endpoint="http://127.0.0.1:8001",
                        served_model=base_model,
                    )
            comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["status"], "FAILED_REQUESTS")
            self.assertFalse(comparison["completion_eligibility"]["eligible"])
            self.assertEqual(comparison["modes"]["free"]["primary_failure_rate"], 1.0)
            self.assertEqual(comparison["modes"]["constrained"]["fallback_failure_rate"], 1.0)
            for mode in ("free", "constrained"):
                spec = comparison["raw_artifacts"][mode]
                self.assertEqual(spec["count"], 3)
                self.assertEqual(sha256_file(Path(spec["path"])), spec["sha256"])

    def test_served_model_must_equal_locked_qwen_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, _ = self._fixture(root)
            output = root / "schema-output"
            with self.assertRaisesRegex(Week7TrainingError, "exactly match"):
                run_schema_experiment(
                    root, config_path, output,
                    endpoint="http://127.0.0.1:8001",
                    served_model="not-the-locked-model",
                )
            self.assertFalse(output.exists())

    def test_successful_response_model_mismatch_cannot_be_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, base_model = self._fixture(root)
            output = root / "schema-output"
            with (
                patch(
                    "src.training.week7_inference._request_model_registry",
                    return_value=[base_model],
                ),
                patch("src.training.week7_inference._openai_messages", return_value=[]),
                patch(
                    "src.training.week7_inference._request_completion",
                    return_value=("{}", 1.0, None, "unexpected-served-model"),
                ),
            ):
                with self.assertRaisesRegex(Week7TrainingError, "not COMPLETED"):
                    run_schema_experiment(
                        root, config_path, output,
                        endpoint="http://127.0.0.1:8001",
                        served_model=base_model,
                    )
            comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["status"], "FAILED_REQUESTS")
            self.assertFalse(comparison["model_identity"]["verified"])


if __name__ == "__main__":
    unittest.main()

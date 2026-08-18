import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.training.week6_evaluation import (
    compare_itinerary_evaluations,
    run_itinerary_adapter_evaluation,
    summarize_itinerary_predictions,
)
from src.training.week6_qlora import Week6TrainingError, load_training_config


ROOT = Path(__file__).resolve().parents[1]


def _evaluation_row(dataset_version: str) -> dict:
    return {
        "sample_id": "evaluation-sample",
        "scenario": "itinerary_planning",
        "label_source": "model_preannotation",
        "sample_weight": 0.5,
        "dataset_lock": {
            "dataset_version": dataset_version,
            "manifest_sha256": "a" * 64,
            "split_sha256": "b" * 64,
        },
        "messages": [
            {"role": "user", "content": "原始文字约束：计划2天前往Melbourne"},
            {"role": "assistant", "content": "{}"},
        ],
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class Week6EvaluationTests(unittest.TestCase):
    def _summary(self, *, run_id: str, passed: int, day_count: int = 1) -> dict:
        counts = {
            "rows": 2,
            "passed": passed,
            "json_valid": 2,
            "schema_valid": 2,
            "expected_days_parsed": 2,
            "day_count_match": day_count,
            "day_indices_sequential": 2,
            "constraint_text_exact_match": 2,
            "constraint_check_exact_coverage": 2,
            "required_elements_complete": 2,
        }
        return {
            "status": "completed",
            "run_id": run_id,
            "scenario": "itinerary_planning",
            "base_model": "Qwen/Qwen3-VL-8B-Instruct",
            "evaluation_input_sha256": "a" * 64,
            "dataset_lock": {"dataset_version": "repair-v1"},
            "selected_sample_ids_sha256": "b" * 64,
            "selected_samples": 2,
            "generation": {"do_sample": False, "max_new_tokens": 2048},
            "counts": counts,
        }

    def test_comparison_passes_only_identity_matched_non_regressing_gain(self):
        baseline = self._summary(run_id="baseline", passed=0)
        candidate = self._summary(run_id="candidate", passed=1)
        result = compare_itinerary_evaluations(baseline, candidate)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["count_deltas"]["passed"], 1)

    def test_comparison_rejects_metric_regression(self):
        baseline = self._summary(run_id="baseline", passed=0, day_count=2)
        candidate = self._summary(run_id="candidate", passed=1, day_count=1)
        result = compare_itinerary_evaluations(baseline, candidate)
        self.assertEqual(result["status"], "failed")
        self.assertIn("candidate regressed at day_count_match", result["reasons"])

    def test_comparison_rejects_identity_mismatch(self):
        baseline = self._summary(run_id="baseline", passed=0)
        candidate = self._summary(run_id="candidate", passed=1)
        candidate["generation"]["max_new_tokens"] = 1024
        result = compare_itinerary_evaluations(baseline, candidate)
        self.assertEqual(result["status"], "failed")
        self.assertIn("evaluation identity differs at generation", result["reasons"])

    def test_summary_uses_prediction_audits(self):
        summary = summarize_itinerary_predictions(
            [
                {
                    "audit": {
                        "passed": True,
                        "checks": {"schema_valid": True, "day_count_match": True},
                    }
                },
                {
                    "audit": {
                        "passed": False,
                        "checks": {"schema_valid": True, "day_count_match": False},
                    }
                },
            ]
        )
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["schema_valid"], 2)
        self.assertEqual(summary["day_count_match"], 1)

    def test_summary_rejects_missing_audit(self):
        with self.assertRaisesRegex(Week6TrainingError, "missing"):
            summarize_itinerary_predictions([{}])

    def test_evaluation_rejects_adapter_hash_before_loading_gpu(self):
        config = load_training_config(
            ROOT / "configs/week6/qwen3_vl_8b_qlora_itinerary_refinement_v1.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            adapter = temp / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_bytes(b"config")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            evaluation = temp / "validation.jsonl"
            evaluation.write_text(
                json.dumps(_evaluation_row(config["dataset"]["dataset_version"])) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Week6TrainingError, "hashes"):
                run_itinerary_adapter_evaluation(
                    ROOT,
                    config,
                    eval_path=evaluation,
                    adapter_dir=adapter,
                    output_dir=temp / "output",
                )

    def test_evaluation_rejects_wrong_dataset_before_loading_gpu(self):
        config = copy.deepcopy(
            load_training_config(
                ROOT / "configs/week6/qwen3_vl_8b_qlora_itinerary_refinement_v1.json"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            adapter = temp / "adapter"
            adapter.mkdir()
            config_bytes = b"config"
            model_bytes = b"weights"
            (adapter / "adapter_config.json").write_bytes(config_bytes)
            (adapter / "adapter_model.safetensors").write_bytes(model_bytes)
            config["refinement"]["expected_initial_adapter_file_sha256"] = {
                "adapter_config.json": _sha256(config_bytes),
                "adapter_model.safetensors": _sha256(model_bytes),
            }
            evaluation = temp / "validation.jsonl"
            evaluation.write_text(
                json.dumps(_evaluation_row("wrong-dataset")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Week6TrainingError, "configured dataset"):
                run_itinerary_adapter_evaluation(
                    ROOT,
                    config,
                    eval_path=evaluation,
                    adapter_dir=adapter,
                    output_dir=temp / "output",
                )


if __name__ == "__main__":
    unittest.main()

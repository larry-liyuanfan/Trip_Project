from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.training.week6_qlora import (
    Week6TrainingError,
    iter_training_rows,
    load_training_config,
    validate_training_row,
)


ROOT = Path(__file__).resolve().parents[1]


class Week6QLoRATests(unittest.TestCase):
    def test_config_locks_8b_nf4_and_effective_batch(self) -> None:
        config = load_training_config(ROOT / "configs/week6/qwen3_vl_8b_qlora.json")
        self.assertEqual(config["base_model"], "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(config["training"]["effective_global_batch_size_one_gpu"], 16)

    def test_model_preannotation_weight_is_bounded(self) -> None:
        row = {
            "sample_id": "sample-1",
            "scenario": "after_sales",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "identify"}]},
                {"role": "assistant", "content": "{}"},
            ],
            "label_source": "model_preannotation",
            "sample_weight": 0.75,
            "dataset_lock": {
                "dataset_version": "v1",
                "manifest_sha256": "a" * 64,
                "split_sha256": "b" * 64,
            },
        }
        with self.assertRaises(Week6TrainingError):
            validate_training_row(row)
        row["sample_weight"] = 0.5
        validate_training_row(row)

    def test_jsonl_requires_dataset_lock_and_assistant_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps({
                "sample_id": "sample-1",
                "scenario": "itinerary_planning",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "plan"}]},
                    {"role": "assistant", "content": "{}"},
                ],
                "label_source": "human_revised",
                "sample_weight": 1.0,
                "dataset_lock": {
                    "dataset_version": "v1",
                    "manifest_sha256": "a" * 64,
                    "split_sha256": "b" * 64,
                },
            }) + "\n", encoding="utf-8")
            self.assertEqual(len(list(iter_training_rows(path, scenario="itinerary_planning"))), 1)

    def test_pilot_reads_only_configured_sample_cap(self) -> None:
        from src.training.week6_qlora import run_small_sample_training

        config = load_training_config(ROOT / "configs/week6/qwen3_vl_8b_qlora.json")
        config["pilot"]["max_samples"] = 2
        rows_read = []

        def rows(*args, **kwargs):
            for index in range(100):
                rows_read.append(index)
                yield {
                    "sample_id": str(index),
                    "dataset_lock": {
                        "dataset_version": config["dataset"]["dataset_version"],
                        "manifest_sha256": "a" * 64,
                        "split_sha256": "b" * 64,
                    },
                }

        with tempfile.TemporaryDirectory() as directory, patch(
            "src.training.week6_qlora.iter_training_rows", side_effect=rows
        ), patch(
            "src.training.week6_qlora.environment_report",
            return_value={"status": "cuda_unavailable"},
        ):
            with self.assertRaisesRegex(Week6TrainingError, "cuda_unavailable"):
                run_small_sample_training(
                    config,
                    scenario="after_sales",
                    train_path=Path(directory) / "train.jsonl",
                    eval_path=Path(directory) / "eval.jsonl",
                    output_dir=Path(directory) / "output",
                    dataset_lock_confirmed=True,
                )
        self.assertEqual(len(rows_read), 4)

    def test_pilot_rejects_mixed_dataset_locks(self) -> None:
        from src.training.week6_qlora import run_small_sample_training

        config = load_training_config(ROOT / "configs/week6/qwen3_vl_8b_qlora.json")
        config["pilot"]["max_samples"] = 1

        def rows(path, **kwargs):
            marker = "a" if Path(path).name == "train.jsonl" else "c"
            yield {
                "sample_id": marker,
                "dataset_lock": {
                    "dataset_version": config["dataset"]["dataset_version"],
                    "manifest_sha256": marker * 64,
                    "split_sha256": "b" * 64,
                },
            }

        with tempfile.TemporaryDirectory() as directory, patch(
            "src.training.week6_qlora.iter_training_rows", side_effect=rows
        ):
            with self.assertRaisesRegex(Week6TrainingError, "different dataset locks"):
                run_small_sample_training(
                    config,
                    scenario="after_sales",
                    train_path=Path(directory) / "train.jsonl",
                    eval_path=Path(directory) / "eval.jsonl",
                    output_dir=Path(directory) / "output",
                    dataset_lock_confirmed=True,
                )


if __name__ == "__main__":
    unittest.main()

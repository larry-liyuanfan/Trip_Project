from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.training.week6_qlora import (
    Week6TrainingError,
    _trainable_parameter_report,
    iter_training_rows,
    load_training_config,
    resolve_lora_targets,
    validate_training_row,
)


ROOT = Path(__file__).resolve().parents[1]


class Week6QLoRATests(unittest.TestCase):
    def test_config_locks_8b_nf4_and_effective_batch(self) -> None:
        config = load_training_config(ROOT / "configs/week6/qwen3_vl_8b_qlora.json")
        self.assertEqual(config["base_model"], "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(config["training"]["effective_global_batch_size_one_gpu"], 16)

    def test_final_config_locks_approved_training_hyperparameters(self) -> None:
        config = load_training_config(
            ROOT / "configs/week6/qwen3_vl_8b_qlora_final300_v4.json"
        )
        training = config["training"]
        self.assertEqual(training["optimizer"], "adamw_torch")
        self.assertEqual(training["learning_rate"], 0.0002)
        self.assertEqual(training["lr_scheduler_type"], "cosine")
        self.assertEqual(training["warmup_ratio"], 0.03)
        self.assertEqual(training["weight_decay"], 0.01)
        self.assertEqual(training["attn_implementation"], "sdpa")

    def test_runtime_targets_include_language_and_visual_merger(self) -> None:
        class Model:
            def named_modules(self):
                names = [
                    "model.language.layers.0.self_attn.q_proj",
                    "model.language.layers.0.self_attn.k_proj",
                    "model.language.layers.0.self_attn.v_proj",
                    "model.language.layers.0.self_attn.o_proj",
                    "model.visual.merger.linear_fc1",
                    "model.visual.merger.linear_fc2",
                ]
                return [(name, object()) for name in names]

        config = load_training_config(
            ROOT / "configs/week6/qwen3_vl_8b_qlora_final300_v4.json"
        )
        self.assertEqual(
            resolve_lora_targets(Model(), config),
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "visual.merger.linear_fc1",
                "visual.merger.linear_fc2",
            ],
        )

    def test_freeze_report_rejects_non_lora_trainable_parameters(self) -> None:
        class Parameter:
            def __init__(self, count: int, requires_grad: bool) -> None:
                self.count = count
                self.requires_grad = requires_grad

            def numel(self) -> int:
                return self.count

        class Model:
            def __init__(self, base_trainable: bool) -> None:
                self.base_trainable = base_trainable

            def named_parameters(self):
                return [
                    ("base.weight", Parameter(100, self.base_trainable)),
                    ("adapter.lora_A.weight", Parameter(10, True)),
                ]

        report = _trainable_parameter_report(Model(base_trainable=False))
        self.assertEqual(report["trainable_parameters"], 10)
        self.assertEqual(report["total_parameters"], 110)
        with self.assertRaisesRegex(Week6TrainingError, "freeze check failed"):
            _trainable_parameter_report(Model(base_trainable=True))

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

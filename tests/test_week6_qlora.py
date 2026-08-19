from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.training.week6_qlora import (
    IndexedMessageDataset,
    Week6TrainingError,
    _normalize_processor_messages,
    _trainable_parameter_report,
    evaluate_pilot_gate,
    iter_training_rows,
    load_training_config,
    resolve_lora_targets,
    run_small_sample_training,
    validate_training_row,
)


ROOT = Path(__file__).resolve().parents[1]


class Week6QLoRATests(unittest.TestCase):
    def test_processor_messages_wrap_plain_text_without_mutating_images(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": "data/example.jpg"},
                    {"type": "text", "text": "identify"},
                ],
            },
            {"role": "assistant", "content": '{"severity":"high"}'},
        ]

        normalized = _normalize_processor_messages(messages)

        self.assertEqual(normalized[0]["content"], messages[0]["content"])
        self.assertIsNot(normalized[0]["content"], messages[0]["content"])
        self.assertEqual(
            normalized[1]["content"],
            [{"type": "text", "text": '{"severity":"high"}'}],
        )
        self.assertIsInstance(messages[1]["content"], str)

    def test_processor_messages_reject_invalid_content_shape(self) -> None:
        with self.assertRaisesRegex(Week6TrainingError, "multimodal content list"):
            _normalize_processor_messages([{"role": "assistant", "content": []}])

    def test_spartan_environment_is_pinned_to_cuda_128_stack(self) -> None:
        requirements = (
            ROOT / "requirements-training-spartan-cu128.txt"
        ).read_text(encoding="utf-8")
        setup = (
            ROOT / "scripts/spartan/setup_week6_cuda128_venv.sbatch"
        ).read_text(encoding="utf-8")
        repair = (
            ROOT / "scripts/spartan/repair_week6_torchvision.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("transformers==4.57.1", requirements)
        self.assertIn("bitsandbytes==0.47.0", requirements)
        self.assertIn("kernels==0.11.7", requirements)
        self.assertIn("torch==2.8.0+cu128", setup)
        self.assertIn("torchvision==0.23.0+cu128", setup)
        self.assertIn("refusing to mutate an existing environment", setup)
        self.assertIn("PIP_NO_CACHE_DIR=1", setup)
        self.assertNotIn(
            '-r "${TRIP_PROJECT_ROOT}/requirements.txt"',
            setup,
        )
        self.assertIn("torchvision==0.23.0+cu128", repair)
        self.assertIn("refusing to repair an unexpected torch/CUDA environment", repair)
        supervised = (
            ROOT / "scripts/spartan/week6_qlora_supervised.sbatch"
        ).read_text(encoding="utf-8")
        evaluation = (
            ROOT / "scripts/spartan/week6_adapter_evaluation.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --time=08:00:00", supervised)
        self.assertIn("waiting_for_versioned_resume", supervised)
        self.assertIn("--init-adapter", supervised)
        self.assertIn('export HOME="${runtime_cache}/home"', supervised)
        self.assertIn('export XDG_CACHE_HOME="${runtime_cache}/xdg"', supervised)
        self.assertIn(
            'export PYTORCH_KERNEL_CACHE_PATH="${runtime_cache}/torch/kernels"',
            supervised,
        )
        self.assertIn("#SBATCH --time=02:00:00", evaluation)
        self.assertIn("--resume", evaluation)
        self.assertIn('export HOME="${runtime_cache}/home"', evaluation)
        self.assertIn(
            'export PYTORCH_KERNEL_CACHE_PATH="${runtime_cache}/torch/kernels"',
            evaluation,
        )

    def _pilot_summary(self, config: dict) -> dict:
        return {
            "status": "completed",
            "scenario": "after_sales",
            "git_commit": "a" * 40,
            "dataset_lock": {"dataset_version": config["dataset"]["dataset_version"]},
            "global_step": config["pilot"]["max_steps"],
            "checkpoints": ["checkpoint-10"],
            "adapter_only": True,
            "adapter_reload_verified": True,
            "adapter_file_sha256": {"adapter_model.safetensors": "b" * 64},
            "training_metrics": {"train_loss": 1.25},
            "log_history": [{"eval_loss": 1.5}],
            "peak_gpu_memory_reserved_bytes": 20 * 1024**3,
        }

    def test_pilot_gate_passes_complete_finite_run(self) -> None:
        config = load_training_config(
            ROOT / "configs/week6/qwen3_vl_8b_qlora_final300_v4.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "run_summary.json"
            summary_path.write_text(
                json.dumps(self._pilot_summary(config)), encoding="utf-8"
            )
            gate = evaluate_pilot_gate(
                config,
                summary_path=summary_path,
                expected_scenario="after_sales",
                expected_git_commit="a" * 40,
                gpu_total_memory_gb=48,
            )
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["reasons"], [])

    def test_pilot_gate_rejects_nonfinite_loss_and_missing_reload(self) -> None:
        config = load_training_config(
            ROOT / "configs/week6/qwen3_vl_8b_qlora_final300_v4.json"
        )
        summary = self._pilot_summary(config)
        summary["training_metrics"]["train_loss"] = float("nan")
        summary["adapter_reload_verified"] = False
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "run_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            gate = evaluate_pilot_gate(
                config,
                summary_path=summary_path,
                expected_scenario="after_sales",
                expected_git_commit="a" * 40,
                gpu_total_memory_gb=48,
            )
        self.assertEqual(gate["status"], "failed")
        self.assertIn("train_loss_not_finite", gate["reasons"])
        self.assertIn("adapter_reload_not_verified", gate["reasons"])

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
        self.assertEqual(
            config["scenarios"]["itinerary_planning"]["format_constraint_loss_weight"],
            0.1,
        )

    def test_refinement_config_binds_completed_itinerary_adapter(self) -> None:
        config = load_training_config(
            ROOT / "configs/week6/qwen3_vl_8b_qlora_itinerary_refinement_v1.json"
        )
        refinement = config["refinement"]
        self.assertTrue(refinement["initial_adapter_required"])
        self.assertEqual(
            refinement["expected_initial_adapter_file_sha256"][
                "adapter_model.safetensors"
            ],
            "18c5dfad0a423945f19b0d1ea863e82bda3934634aa4b5922023c3421ba114ac",
        )

    def test_refinement_training_requires_completed_adapter_or_checkpoint(self) -> None:
        config = load_training_config(
            ROOT / "configs/week6/qwen3_vl_8b_qlora_itinerary_refinement_v1.json"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(
                Week6TrainingError,
                "requires an initial adapter",
            ):
                run_small_sample_training(
                    config,
                    scenario="itinerary_planning",
                    train_path=root / "train.jsonl",
                    eval_path=root / "validation.jsonl",
                    output_dir=root / "output",
                    dataset_lock_confirmed=True,
                )

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

    def test_indexed_dataset_keeps_offsets_and_reads_rows_on_demand(self) -> None:
        lock = {
            "dataset_version": "v1",
            "manifest_sha256": "a" * 64,
            "split_sha256": "b" * 64,
        }
        rows = [
            {
                "sample_id": f"sample-{index}",
                "scenario": "after_sales",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "check"}]},
                    {"role": "assistant", "content": "{}"},
                ],
                "label_source": "human_revised",
                "sample_weight": 1.0,
                "dataset_lock": lock,
            }
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            dataset = IndexedMessageDataset(path, scenario="after_sales")
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.offsets[0], 0)
            self.assertEqual(dataset[2]["sample_id"], "sample-2")
            self.assertEqual(dataset.dataset_lock, lock)

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

    def test_initial_adapter_must_be_complete(self) -> None:
        from src.training.week6_qlora import run_small_sample_training

        config = load_training_config(ROOT / "configs/week6/qwen3_vl_8b_qlora.json")
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            adapter = temp / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(Week6TrainingError, "incomplete"):
                run_small_sample_training(
                    config,
                    scenario="after_sales",
                    train_path=temp / "train.jsonl",
                    eval_path=temp / "eval.jsonl",
                    output_dir=temp / "output",
                    dataset_lock_confirmed=True,
                    init_adapter=adapter,
                )

    def test_initial_adapter_and_resume_are_mutually_exclusive(self) -> None:
        from src.training.week6_qlora import run_small_sample_training

        config = load_training_config(ROOT / "configs/week6/qwen3_vl_8b_qlora.json")
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            adapter = temp / "adapter"
            checkpoint = temp / "output/checkpoint-1"
            adapter.mkdir()
            checkpoint.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            with self.assertRaisesRegex(Week6TrainingError, "mutually exclusive"):
                run_small_sample_training(
                    config,
                    scenario="after_sales",
                    train_path=temp / "train.jsonl",
                    eval_path=temp / "eval.jsonl",
                    output_dir=temp / "output",
                    dataset_lock_confirmed=True,
                    resume_from_checkpoint=checkpoint,
                    init_adapter=adapter,
                )


if __name__ == "__main__":
    unittest.main()

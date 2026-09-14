from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.week7_data import sha256_file
from src.training.week7_inference import combine_week6_development_baseline
from src.training.week7_qlora import Week7TrainingError
from src.training.week7_selection import select_development_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v3.json"
LOCK_SHA = "d" * 64
SCENARIOS = ("image_product_search", "after_sales", "itinerary_planning")


def _scenario(composite: float, latency: float = 100.0) -> dict:
    return {
        "composite": composite,
        "aggregate": {
            "json_compliance": 1.0,
            "schema_pass": 1.0,
            "latency_mean_ms": latency,
        },
        "metric_support": {"metric": 30},
    }


class Week7SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.config_sha = sha256_file(CONFIG)
        self.training = self.root / "training"
        self.training.mkdir()
        self.baseline_path = self._write_baseline()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_baseline(self) -> Path:
        inputs = {}
        scenarios = {name: _scenario(0.5) for name in SCENARIOS}
        for scenario in SCENARIOS:
            payload = {
                "status": "COMPLETED",
                "run_id": self.config["experiment_identity"]
                ["development_baseline_run_ids"][scenario],
                "model_role": "week6_single_task_adapter",
                "split": "development",
                "scenario_filter": scenario,
                "config_sha256": self.config_sha,
                "dataset_lock_sha256": LOCK_SHA,
                "sample_count": 30,
                "failure_count": 0,
                "latency_ms_mean": 100.0,
                "scenarios": {scenario: scenarios[scenario]},
                "adapter_hashes": {
                    "adapter_model.safetensors": self.config["evaluation"]
                    ["week6_adapter_sha256"][scenario]
                },
            }
            path = self.root / f"baseline-{scenario}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            inputs[scenario] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        dialogue = {
            "status": "COMPLETED",
            "run_id": self.config["experiment_identity"]
            ["week6_dialogue_development_run_id"],
            "model_role": "week6_single_task_adapters",
            "split": "development",
            "scenario_filter": "dialogue_routed",
            "config_sha256": self.config_sha,
            "dataset_lock_sha256": LOCK_SHA,
            "sample_count": 24,
            "failure_count": 0,
            "latency_ms_mean": 100.0,
            "scenarios": {},
            "dialogue": {"sample_count": 24},
            "adapter_hashes": {
                scenario: {
                    "adapter_model.safetensors": self.config["evaluation"]
                    ["week6_adapter_sha256"][scenario]
                }
                for scenario in SCENARIOS
            },
            "routing": {
                "method": "target_task_result_v1",
                "sample_counts": {scenario: 8 for scenario in SCENARIOS},
            },
        }
        dialogue_path = self.root / "baseline-dialogue.json"
        dialogue_path.write_text(json.dumps(dialogue), encoding="utf-8")
        inputs["dialogue"] = {
            "path": str(dialogue_path.resolve()),
            "sha256": sha256_file(dialogue_path),
        }
        baseline = {
            "status": "COMPLETED",
            "run_id": self.config["experiment_identity"]
            ["week6_combined_development_run_id"],
            "model_role": "week6_single_task_adapters",
            "split": "development",
            "config_sha256": self.config_sha,
            "dataset_lock_sha256": LOCK_SHA,
            "sample_count": 114,
            "failure_count": 0,
            "failure_rate": 0.0,
            "latency_ms_mean": 100.0,
            "latency_ms_median": None,
            "weighted_composite": 0.5,
            "scenarios": scenarios,
            "dialogue": dialogue["dialogue"],
            "inputs": inputs,
        }
        path = self.root / "baseline.json"
        path.write_text(json.dumps(baseline), encoding="utf-8")
        return path

    def _write_training(self, rows: tuple[tuple[int, tuple[float, float, float], float], ...]) -> Path:
        weights = self.config["evaluation"]["scenario_weights"]
        checkpoint_hashes = {}
        checkpoints = []
        development_evaluation_artifacts = {}
        for step, values, latency in rows:
            metrics_dir = self.training / "development_evaluations" / f"step-{step:06d}"
            checkpoint = self.training / f"checkpoint-{step}"
            metrics_dir.mkdir(parents=True)
            checkpoint.mkdir()
            adapter = checkpoint / "adapter_model.safetensors"
            adapter.write_bytes(f"adapter-{step}".encode())
            checkpoints.append(checkpoint.name)
            checkpoint_hashes[checkpoint.name] = sha256_file(adapter)
            scenarios = dict(zip(SCENARIOS, (_scenario(value) for value in values)))
            metrics = {
                "status": "COMPLETED",
                "model_role": "multitask_checkpoint",
                "split": "development",
                "run_id": (
                    f"{self.config['experiment_identity']['multitask_sft_run_id']}"
                    f"_development_step_{step:06d}"
                ),
                "config_sha256": self.config_sha,
                "dataset_lock_sha256": LOCK_SHA,
                "global_step": step,
                "sample_count": 114,
                "dialogue": {"sample_count": 24},
                "scenarios": scenarios,
                "failure_rate": 0.0,
                "latency_ms_mean": latency,
                "weighted_composite": sum(
                    weights[name] * scenarios[name]["composite"] for name in weights
                ),
            }
            raw_outputs = metrics_dir / "raw_outputs.jsonl"
            raw_outputs.write_text(
                "".join(
                    json.dumps({"sample_id": f"development-{index}"}) + "\n"
                    for index in range(114)
                ),
                encoding="utf-8",
            )
            metrics["raw_outputs"] = {
                "path": str(raw_outputs.resolve()),
                "sha256": sha256_file(raw_outputs),
                "count": 114,
            }
            metrics_path = metrics_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            development_evaluation_artifacts[str(step)] = {
                "raw_outputs_path": str(raw_outputs.resolve()),
                "raw_outputs_sha256": sha256_file(raw_outputs),
                "metrics_path": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
            }
        summary = {
            "status": "COMPLETED",
            "run_id": self.config["experiment_identity"]["multitask_sft_run_id"],
            "config_sha256": self.config_sha,
            "dataset_lock_sha256": LOCK_SHA,
            "development_samples": 114,
            "evaluation_steps": [row[0] for row in rows],
            "global_step": rows[-1][0],
            "checkpoints": checkpoints,
            "checkpoint_hashes": checkpoint_hashes,
            "development_evaluation_artifacts": development_evaluation_artifacts,
        }
        path = self.training / "run_summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        return path

    def test_selection_binds_completed_training_and_checkpoint_evidence(self) -> None:
        summary = self._write_training(
            ((38, (0.55, 0.55, 0.50), 100.0), (76, (0.9, 0.9, 0.2), 100.0))
        )
        result = select_development_checkpoint(
            CONFIG, self.training, summary, self.baseline_path,
            self.root / "selection.json",
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"]["step"], 38)
        self.assertEqual(result["training_summary"]["sha256"], sha256_file(summary))
        self.assertEqual(
            result["selected_evidence"]["checkpoint_adapter_sha256"],
            sha256_file(self.training / "checkpoint-38/adapter_model.safetensors"),
        )
        self.assertFalse(
            next(item for item in result["candidates"] if item["step"] == 76)["eligible"]
        )

    def test_global_full_development_latency_controls_eligibility(self) -> None:
        summary = self._write_training(
            ((38, (0.6, 0.6, 0.6), 124.0), (76, (0.6, 0.6, 0.6), 126.0))
        )
        result = select_development_checkpoint(
            CONFIG, self.training, summary, self.baseline_path,
            self.root / "selection.json",
        )
        self.assertEqual(result["selected"]["step"], 38)
        self.assertTrue(result["candidates"][0]["latency_gate"])
        self.assertFalse(result["candidates"][1]["latency_gate"])

    def test_protocol_selection_uses_complete_protocol_metrics_not_training_scores(self) -> None:
        summary = self._write_training(((38, (0.9, 0.9, 0.9), 100.0),))
        protocol_dir = self.root / "protocol"
        protocol_dir.mkdir()
        protocol_path = protocol_dir / "protocol_summary.json"
        protocol_path.write_text("{}", encoding="utf-8")
        baseline_metrics_path = protocol_dir / "week6.json"
        baseline_metrics_path.write_text(
            self.baseline_path.read_text(encoding="utf-8"), encoding="utf-8",
        )
        source_metrics_path = (
            self.training / "development_evaluations/step-000038/metrics.json"
        )
        protocol_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
        protocol_metrics["scenarios"] = {
            scenario: _scenario(0.1) for scenario in SCENARIOS
        }
        protocol_metrics["weighted_composite"] = 0.1
        protocol_metrics["latency_ms_mean"] = 126.0
        protocol_metrics_path = protocol_dir / "step-38.json"
        protocol_metrics_path.write_text(json.dumps(protocol_metrics), encoding="utf-8")
        protocol = {
            "schema_version": "week7_development_latency_protocol_v4",
            "run_id": "unit-protocol",
            "candidate_steps": [38],
            "latency_comparison": {
                "38": {
                    "candidate_latency_ms_mean": 126.0,
                    "baseline_latency_ms_mean": 100.0,
                    "latency_ratio": 1.26,
                },
            },
            "roles": {
                "week6_single_task_adapters": {
                    "metrics_path": str(baseline_metrics_path),
                    "metrics_sha256": sha256_file(baseline_metrics_path),
                },
                "multitask_step_000038": {
                    "metrics_path": str(protocol_metrics_path),
                    "metrics_sha256": sha256_file(protocol_metrics_path),
                },
            },
        }
        with patch(
            "src.training.week7_selection.validate_latency_protocol_v4",
            return_value=protocol,
        ):
            result = select_development_checkpoint(
                CONFIG, self.training, summary, self.baseline_path,
                self.root / "protocol-selection.json",
                latency_protocol_path=protocol_path,
            )
        self.assertEqual(result["status"], "BLOCKED_NO_ELIGIBLE_CHECKPOINT")
        self.assertEqual(result["candidates"][0]["weighted_composite"], 0.1)
        self.assertEqual(result["candidates"][0]["latency_ratio"], 1.26)
        self.assertEqual(
            result["candidates"][0]["source_training_metrics_path"],
            str(source_metrics_path.resolve()),
        )

    def test_rejects_forged_metrics_identity_and_checkpoint_hash(self) -> None:
        summary = self._write_training(((38, (0.6, 0.6, 0.6), 100.0),))
        metrics_path = self.training / "development_evaluations/step-000038/metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["run_id"] = "forged"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        with self.assertRaisesRegex(Week7TrainingError, "coverage mismatch"):
            select_development_checkpoint(
                CONFIG, self.training, summary, self.baseline_path,
                self.root / "bad-identity.json",
            )
        metrics["run_id"] = (
            f"{self.config['experiment_identity']['multitask_sft_run_id']}"
            "_development_step_000038"
        )
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        (self.training / "checkpoint-38/adapter_model.safetensors").write_bytes(b"forged")
        with self.assertRaisesRegex(Week7TrainingError, "checkpoint hash"):
            select_development_checkpoint(
                CONFIG, self.training, summary, self.baseline_path,
                self.root / "bad-hash.json",
            )

    def test_rejects_modified_development_raw_outputs(self) -> None:
        summary = self._write_training(((38, (0.6, 0.6, 0.6), 100.0),))
        raw_path = self.training / "development_evaluations/step-000038/raw_outputs.jsonl"
        raw_path.write_text(raw_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
        with self.assertRaisesRegex(Week7TrainingError, "raw-output artifact"):
            select_development_checkpoint(
                CONFIG, self.training, summary, self.baseline_path,
                self.root / "bad-raw.json",
            )

    def test_rejects_7_8_9_routed_dialogue_distribution(self) -> None:
        summary = self._write_training(((38, (0.6, 0.6, 0.6), 100.0),))
        baseline = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        dialogue_path = Path(baseline["inputs"]["dialogue"]["path"])
        dialogue = json.loads(dialogue_path.read_text(encoding="utf-8"))
        dialogue["routing"]["sample_counts"] = dict(zip(SCENARIOS, (7, 8, 9)))
        dialogue_path.write_text(json.dumps(dialogue), encoding="utf-8")
        baseline["inputs"]["dialogue"]["sha256"] = sha256_file(dialogue_path)
        self.baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        with self.assertRaisesRegex(Week7TrainingError, "dialogue baseline input"):
            select_development_checkpoint(
                CONFIG, self.training, summary, self.baseline_path,
                self.root / "bad-route-selection.json",
            )
        scenario_paths = {
            scenario: Path(baseline["inputs"][scenario]["path"])
            for scenario in SCENARIOS
        }
        with self.assertRaisesRegex(Week7TrainingError, "dialogue coverage"):
            combine_week6_development_baseline(
                CONFIG, scenario_paths, dialogue_path,
                self.root / "bad-route-combined.json",
            )


if __name__ == "__main__":
    unittest.main()

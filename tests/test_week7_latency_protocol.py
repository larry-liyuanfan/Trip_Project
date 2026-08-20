from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.evaluation.metrics import WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL
from src.training.week7_data import iter_jsonl, load_week7_config, sha256_file
from src.training.week7_evaluation import summarize_raw_records
from src.training.week7_latency_protocol import validate_latency_protocol_v4
from src.training.week7_qlora import Week7TrainingError
from src.training.week7_runtime import (
    LATENCY_PROTOCOL_VERSION,
    generate_record,
    inference_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v3.json"
DEVELOPMENT = (
    ROOT / "outputs/week7/locked_data/week7_fresh_multitask_context_20260820_v3"
    / "development.jsonl"
)


class _Flag:
    def __init__(self, use_cache: bool) -> None:
        self.use_cache = use_cache


class _Model:
    def __init__(self) -> None:
        self.training = True
        self.config = _Flag(False)
        self.generation_config = _Flag(False)
        self.parameter = SimpleNamespace(device="cpu")
        self.generate_kwargs = None

    def parameters(self):
        return iter((self.parameter,))

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return _Tensor(5)


class _Tensor:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)

    def to(self, device):
        return self

    def __getitem__(self, key):
        return _Tensor(2)


class _Processor:
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": _Tensor(3)}

    def batch_decode(self, suffix, **kwargs):
        return ["ok"]


class Week7LatencyProtocolTests(unittest.TestCase):
    def test_shared_runtime_enables_cache_and_records_synchronized_token_counts(self):
        model = _Model()
        with inference_runtime(model):
            self.assertFalse(model.training)
            self.assertTrue(model.config.use_cache)
            self.assertTrue(model.generation_config.use_cache)
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False),
                inference_mode=lambda: nullcontext(),
            )
            with patch.dict("sys.modules", {"torch": fake_torch}):
                record = generate_record(
                    model, _Processor(), [{"role": "user", "content": "x"}],
                    sample_id="development-1", run_id="protocol-v4",
                    model_name="multitask_step_000076", max_new_tokens=2048,
                )
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.generation_config.use_cache)
        self.assertTrue(model.generate_kwargs["use_cache"])
        self.assertEqual(record["input_token_count"], 3)
        self.assertEqual(record["generated_token_count"], 2)
        self.assertEqual(record["latency_protocol"], LATENCY_PROTOCOL_VERSION)

    def _write_protocol(self, root: Path) -> tuple[Path, Path, Path]:
        config = load_week7_config(CONFIG)
        rows = list(iter_jsonl(DEVELOPMENT))
        run_id = "unit_protocol_v4"
        training_dir = root / "training"
        checkpoint = training_dir / "checkpoint-76"
        evaluation_dir = training_dir / "development_evaluations/step-000076"
        checkpoint.mkdir(parents=True)
        evaluation_dir.mkdir(parents=True)
        adapter = checkpoint / "adapter_model.safetensors"
        adapter.write_bytes(b"adapter-76")
        source_raw = evaluation_dir / "raw_outputs.jsonl"
        source_raw.write_text("{}\n", encoding="utf-8")
        source_metrics = evaluation_dir / "metrics.json"
        source_metrics.write_text("{}", encoding="utf-8")
        training_summary = training_dir / "run_summary.json"
        training_summary.write_text(json.dumps({
            "evaluation_steps": [76], "global_step": 76,
            "checkpoint_hashes": {"checkpoint-76": sha256_file(adapter)},
            "development_evaluation_artifacts": {
                "76": {
                    "raw_outputs_path": str(source_raw.resolve()),
                    "raw_outputs_sha256": sha256_file(source_raw),
                    "metrics_path": str(source_metrics.resolve()),
                    "metrics_sha256": sha256_file(source_metrics),
                },
            },
        }), encoding="utf-8")
        baseline = root / "baseline.json"
        baseline.write_text("{}", encoding="utf-8")
        source_week6 = {}
        for role in (*config["evaluation"]["scenario_weights"], "dialogue"):
            path = root / f"source-week6-{role}.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            source_week6[role] = {"path": str(path), "sha256": sha256_file(path)}
        source_candidates = {
            "76": {
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_adapter_sha256": sha256_file(adapter),
                "training_raw_outputs_path": str(source_raw.resolve()),
                "training_raw_outputs_sha256": sha256_file(source_raw),
                "training_metrics_path": str(source_metrics.resolve()),
                "training_metrics_sha256": sha256_file(source_metrics),
            },
        }
        baseline_evidence = root / "baseline-evidence.json"
        baseline_evidence.write_text("{}", encoding="utf-8")
        dataset_lock_path = DEVELOPMENT.parent / "dataset_lock.json"
        dataset_lock = json.loads(dataset_lock_path.read_text(encoding="utf-8"))
        order = [
            "week6_image_product_search", "week6_after_sales",
            "week6_itinerary_planning", "multitask_step_000076", "zero_shot",
        ]
        protocol_config = root / "protocol_config.json"
        protocol_config.write_text(json.dumps({
            "schema_version": "week7_evaluation_protocol_v4",
            "run_id": run_id,
            "base_config": {
                "path": "configs/week7/qwen3_vl_8b_multitask_context_v3.json",
                "sha256": sha256_file(CONFIG),
            },
            "dataset": {
                "version": config["dataset"]["dataset_version"],
                "lock_sha256": dataset_lock["lock_sha256"],
                "development_sha256": sha256_file(DEVELOPMENT),
                "development_count": len(rows),
                "test_allowed": False,
            },
            "source_evidence": {
                "week6_combined_metrics_sha256": sha256_file(baseline),
                "week6_baseline_evidence_sha256": sha256_file(baseline_evidence),
                "training_summary_sha256": sha256_file(training_summary),
                "week6_raw_sha256": {
                    role: spec["sha256"] for role, spec in source_week6.items()
                },
                "candidates": {
                    step: {
                        "checkpoint_adapter_sha256": spec[
                            "checkpoint_adapter_sha256"
                        ],
                        "training_raw_outputs_sha256": spec[
                            "training_raw_outputs_sha256"
                        ],
                    }
                    for step, spec in source_candidates.items()
                },
            },
            "candidate_steps": [76],
            "timing": {
                "protocol": LATENCY_PROTOCOL_VERSION,
                "scope": "apply_chat_template+device_transfer+generate+decode",
                "cuda_synchronize_before": True, "cuda_synchronize_after": True,
                "model_loading_excluded": True, "warmup_excluded": True,
                "single_slurm_allocation": True, "sequential_model_order": order,
            },
            "generation": {
                "max_new_tokens": 2048, "warmup_max_new_tokens": 1,
                "do_sample": False, "use_cache": True,
                "max_input_length": 8192, "structure_aware_truncation": True,
            },
            "metric_support_protocol": WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL,
            "max_latency_ratio": 1.25,
        }), encoding="utf-8")
        roles = {}
        for role, latency, valid in (
            ("week6_single_task_adapters", 100.0, True),
            ("multitask_step_000076", 126.0, False),
            ("zero_shot", 100.0, True),
        ):
            role_dir = root / "roles" / role
            role_dir.mkdir(parents=True)
            records = []
            for row in rows:
                raw = json.dumps(row["target"], ensure_ascii=False) if valid else "{}"
                records.append({
                    "run_id": run_id, "sample_id": row["sample_id"],
                    "model_name": role, "raw_output": raw, "latency_ms": latency,
                    "failed": False, "error": None, "input_token_count": 10,
                    "generated_token_count": 10,
                    "generation_max_new_tokens": 2048,
                    "latency_protocol": LATENCY_PROTOCOL_VERSION, "warmup": False,
                })
            raw_path = role_dir / "raw_outputs.jsonl"
            raw_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            warmup_count = 3 if role == "week6_single_task_adapters" else 1
            warmups = [{
                "run_id": run_id, "sample_id": rows[0]["sample_id"],
                "model_name": role, "raw_output": "", "latency_ms": 1.0,
                "failed": False, "error": None, "input_token_count": 10,
                "generated_token_count": 1, "generation_max_new_tokens": 1,
                "latency_protocol": LATENCY_PROTOCOL_VERSION, "warmup": True,
            } for _ in range(warmup_count)]
            warmup_path = role_dir / "warmups.jsonl"
            warmup_path.write_text(
                "".join(json.dumps(record) + "\n" for record in warmups),
                encoding="utf-8",
            )
            metrics = summarize_raw_records(
                ROOT, config, rows, records,
                metric_support_protocol=WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL,
            )
            metrics.update({
                "status": "COMPLETED", "model_role": role,
                "split": "development", "run_id": run_id,
                "latency_protocol": LATENCY_PROTOCOL_VERSION,
                "raw_outputs": {"sha256": sha256_file(raw_path)},
                "warmups": {"sha256": sha256_file(warmup_path)},
            })
            if role == "week6_single_task_adapters":
                metrics["source_baseline_sha256"] = sha256_file(baseline)
            elif role.startswith("multitask"):
                metrics.update({
                    "global_step": 76,
                    "checkpoint_adapter_sha256": sha256_file(adapter),
                    "source_training_raw_outputs_sha256": sha256_file(source_raw),
                })
            metrics_path = role_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            roles[role] = {
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256_file(metrics_path),
                "raw_outputs_path": str(raw_path),
                "raw_outputs_sha256": sha256_file(raw_path),
                "warmups_path": str(warmup_path),
                "warmups_sha256": sha256_file(warmup_path),
                "sample_count": len(rows),
                "latency_ms_mean": metrics["latency_ms_mean"],
                "failure_rate": metrics["failure_rate"],
            }
        payload = {
            "schema_version": "week7_development_latency_protocol_v4",
            "status": "COMPLETED", "latency_protocol": LATENCY_PROTOCOL_VERSION,
            "run_id": run_id, "protocol_config_path": str(protocol_config),
            "protocol_config_sha256": sha256_file(protocol_config),
            "config_sha256": sha256_file(CONFIG),
            "dataset_lock_path": str(dataset_lock_path.resolve()),
            "dataset_lock_sha256": dataset_lock["lock_sha256"],
            "training_summary_path": str(training_summary.resolve()),
            "training_summary_sha256": sha256_file(training_summary),
            "week6_baseline_sha256": sha256_file(baseline),
            "week6_baseline_evidence_path": str(baseline_evidence.resolve()),
            "week6_baseline_evidence_sha256": sha256_file(baseline_evidence),
            "candidate_steps": [76],
            "execution_order": json.loads(protocol_config.read_text())
            ["timing"]["sequential_model_order"],
            "max_new_tokens": 2048, "warmup_max_new_tokens": 1,
            "development_path": str(DEVELOPMENT),
            "development_sha256": sha256_file(DEVELOPMENT),
            "development_count": len(rows),
            "source_candidates": source_candidates,
            "source_week6_raw_outputs": source_week6,
            "roles": roles,
            "latency_comparison": {
                "76": {
                    "candidate_latency_ms_mean": 126.0,
                    "baseline_latency_ms_mean": 100.0,
                    "latency_ratio": 1.26,
                },
            },
        }
        path = root / "protocol_summary.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path, training_summary, baseline

    def test_protocol_validator_recomputes_metrics_and_rejects_raw_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, training, baseline = self._write_protocol(root)
            result = validate_latency_protocol_v4(
                protocol, config_path=CONFIG,
                training_summary_path=training, week6_baseline_path=baseline,
            )
            self.assertEqual(result["latency_comparison"]["76"]["latency_ratio"], 1.26)
            raw_path = Path(result["roles"]["multitask_step_000076"]["raw_outputs_path"])
            raw_path.write_text(raw_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(Week7TrainingError, "artifact mismatch"):
                validate_latency_protocol_v4(
                    protocol, config_path=CONFIG,
                    training_summary_path=training, week6_baseline_path=baseline,
                )

    def test_protocol_validator_rejects_alternate_development_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, training, baseline = self._write_protocol(root)
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            alternate = root / "alternate-development.jsonl"
            alternate.write_bytes(DEVELOPMENT.read_bytes())
            payload["development_path"] = str(alternate.resolve())
            payload["development_sha256"] = sha256_file(alternate)
            protocol.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                Week7TrainingError, "canonical development binding",
            ):
                validate_latency_protocol_v4(
                    protocol, config_path=CONFIG,
                    training_summary_path=training, week6_baseline_path=baseline,
                )


if __name__ == "__main__":
    unittest.main()

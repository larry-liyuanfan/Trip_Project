import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.training.system_prompt_pilot import (
    PromptPilotError,
    _prompt_summary_config,
    _validate_resumable_prompt_records,
    load_completed_prompt_pilot,
    run_prompt_pilot,
)
from src.training.system_repair import (
    SystemRepairError,
    _validate_historical_failure_contract,
    evaluate_system_release_gates,
    load_repair_config,
    run_week5_repair_queue,
    select_system_repair_candidate,
)
from src.training.week7_data import _product_target, load_week7_config, sha256_file
from src.training.week7_inference import run_transformers_development


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_CONFIG = ROOT / "configs/system_repair/qwen3_vl_8b_system_repair_v1.json"
WEEK5_CONFIG = ROOT / "configs/system_repair/week5_preannotation_repair_v2.json"


def metrics(composite=0.8, dialogue=0.8):
    scenarios = {}
    for scenario in ("image_product_search", "after_sales", "itinerary_planning"):
        scenarios[scenario] = {
            "composite": composite,
            "aggregate": {
                "json_compliance": 1.0,
                "schema_pass": 1.0,
                "sample_count": 48,
            },
            "metric_support": {
                "style_f1": 5,
                "facility_f1": 5,
                "price_range_accuracy": 5,
            },
            "json_compliance": 1.0,
            "schema_pass": 1.0,
            "support_ratio": 1.0,
            "support": {
                "style_tags": 5,
                "visible_facilities": 5,
                "price_range": 5,
            },
        }
    return {
        "scenarios": scenarios,
        "failure_rate": 0.0,
        "mean_latency_ms": 1000.0,
        "latency_ms_mean": 1000.0,
        "dialogue": {
            "automatic_composite": dialogue,
            "format_compliance": 0.95,
            "context_recall": 0.75,
            "context_state_value_accuracy": 0.72,
            "task_result_key_coverage": 0.80,
            "task_result_value_accuracy": 0.70,
            "sequential_protocol_coverage": 0.95,
            "sequential_semantic_accuracy": 0.85,
            "tool_protocol_compliance": 0.70,
            "failure_rate": 1 / 24,
        },
    }


class SystemRepairTest(unittest.TestCase):
    def test_week5_in_process_infrastructure_failure_stops_immediately(self):
        class BrokenService:
            def run_task(self, scenario, request):
                raise OSError("model cache missing")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "repair.json"
            config.write_text("{}\n", encoding="utf-8")
            output = root / "repair"
            output.mkdir()
            candidate = {
                "sample_id": "sample-1",
                "input": {
                    "images": [{"path": "image.jpg"}],
                    "text_constraints": None,
                },
            }
            (output / "repair_queue.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "sample-1",
                        "scenario": "image_product_search",
                        "candidate": candidate,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            repair_config = {
                "output_dir": "repair",
                "repair_id": "repair-v1",
                "model": {
                    "base_model": "Qwen/Qwen3-VL-8B-Instruct",
                    "max_network_retries": 2,
                    "timeout_seconds": 1,
                },
            }
            with patch(
                "src.training.system_repair.load_repair_config",
                return_value=repair_config,
            ), patch(
                "src.training.system_repair._git_commit",
                return_value="a" * 40,
            ):
                with self.assertRaisesRegex(
                    SystemRepairError,
                    "in-process model backend failed",
                ):
                    run_week5_repair_queue(
                        root,
                        config,
                        run_id="run-v1",
                        service=BrokenService(),
                    )

    def test_system_config_locks_1980_continuation_examples(self):
        config = load_week7_config(SYSTEM_CONFIG)

        self.assertTrue(config["system_repair"]["enabled"])
        self.assertEqual(config["dataset"]["train_total"], 1980)
        self.assertEqual(config["training"]["learning_rate"], 5e-5)
        self.assertEqual(config["training"]["epochs"], 1)
        self.assertEqual(config["dataset"]["silver_weight"], 0.5)
        self.assertFalse(config["continuation"]["overwrite_initial_adapter"])

    def test_week5_repair_config_locks_observed_failure_breakdown(self):
        config = load_repair_config(WEEK5_CONFIG)

        self.assertEqual(config["expected"]["input_replacements"], 44)
        self.assertEqual(config["expected"]["schema_retries"], 19)
        self.assertEqual(config["expected"]["json_retries"], 1)
        self.assertEqual(config["expected"]["repair_queue"], 64)

    def test_historical_failure_contract_accepts_only_44_19_1(self):
        failures = (
            [{"sample_id": f"input-{index}", "error_type": "input_error"} for index in range(44)]
            + [{"sample_id": f"schema-{index}", "error_type": "schema_error"} for index in range(19)]
            + [{"sample_id": "json-0", "error_type": "json_parse_error"}]
        )
        expected = load_repair_config(WEEK5_CONFIG)["expected"]

        _validate_historical_failure_contract(failures, [{}] * 79936, expected)
        failures[0]["error_type"] = "schema_error"
        with self.assertRaises(SystemRepairError):
            _validate_historical_failure_contract(failures, [{}] * 79936, expected)

    def test_repair_product_target_adds_price_only_as_inferred_metadata(self):
        target = _product_target(
            {
                "caption": "modern cafe with an outdoor patio",
                "business_description": "Cafe | RestaurantsPriceRange2: '2'",
                "repair_mode": True,
            }
        )

        self.assertEqual(target["price_range"], "mid_range")
        self.assertIn("modern", target["style_tags"])
        self.assertIn("outdoor_seating", target["visible_facilities"])
        self.assertTrue(target["inferred_attributes"])
        self.assertNotIn("price_range", target["unknown_fields"])

    def test_historical_product_target_does_not_consume_repair_metadata(self):
        target = _product_target(
            {
                "caption": "cafe counter",
                "business_description": "Cafe | RestaurantsPriceRange2: '4'",
            }
        )

        self.assertEqual(target["price_range"], "unknown")
        self.assertEqual(target["inferred_attributes"], [])

    def test_release_gate_passes_beta_thresholds_and_better_baselines(self):
        config = load_week7_config(SYSTEM_CONFIG)
        candidate = metrics(composite=0.8, dialogue=0.8)
        existing = metrics(composite=0.7, dialogue=0.6)
        zero = metrics(composite=0.5, dialogue=0.4)

        result = evaluate_system_release_gates(config, candidate, existing, zero)

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["test_consumption_allowed"])

    def test_release_gate_blocks_sparse_product_support(self):
        config = load_week7_config(SYSTEM_CONFIG)
        candidate = metrics(composite=0.8, dialogue=0.8)
        candidate["scenarios"]["image_product_search"]["metric_support"][
            "price_range_accuracy"
        ] = 0

        result = evaluate_system_release_gates(
            config,
            candidate,
            metrics(composite=0.7, dialogue=0.6),
            metrics(composite=0.5, dialogue=0.4),
        )

        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["test_consumption_allowed"])
        self.assertTrue(any("price_range_support" in item for item in result["failures"]))

    def test_prompt_candidates_are_exactly_the_three_approved_versions(self):
        payload = json.loads(
            (ROOT / "configs/system_repair/prompt_candidates_v1.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            set(payload["versions"]),
            {"current_week7", "compact_schema_v1", "evidence_state_v1"},
        )
        self.assertEqual(payload["generation"]["max_new_tokens"], 512)

    def test_completed_prompt_pilot_resume_is_hash_bound(self):
        versions = {
            "current_week7",
            "compact_schema_v1",
            "evidence_state_v1",
        }
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.json"
            prompts = root / "prompts.json"
            output = root / "pilot"
            output.mkdir()
            config.write_text(
                SYSTEM_CONFIG.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            prompts.write_text(
                json.dumps({"generation": {"max_new_tokens": 512}}) + "\n",
                encoding="utf-8",
            )
            identity = {
                "split": "development",
                "test_consumed": False,
                "config_sha256": sha256_file(config),
                "prompt_candidates_sha256": sha256_file(prompts),
                "counts": {
                    "image_product_search": 48,
                    "after_sales": 48,
                    "itinerary_planning": 48,
                },
                "endpoint": "in-process://unit-test",
                "served_model": "unit-test-adapter",
                "max_new_tokens": 512,
            }
            (output / "pilot_identity.json").write_text(
                json.dumps(identity), encoding="utf-8"
            )
            summaries = {}
            for version in versions:
                raw = output / f"{version}_raw.jsonl"
                raw.write_text(
                    "".join(
                        json.dumps({"sample_id": f"sample-{index}"}) + "\n"
                        for index in range(144)
                    ),
                    encoding="utf-8",
                )
                summaries[version] = {"raw_sha256": sha256_file(raw)}
            selection = {
                "status": "COMPLETED",
                "split": "development",
                "test_consumed": False,
                "config_sha256": sha256_file(config),
                "prompt_candidates_sha256": sha256_file(prompts),
                "counts": {
                    "image_product_search": 48,
                    "after_sales": 48,
                    "itinerary_planning": 48,
                },
                "summaries": summaries,
            }
            (output / "selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )

            loaded = load_completed_prompt_pilot(config, prompts, output)
            self.assertEqual(loaded["status"], "COMPLETED")

            with (output / "current_week7_raw.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("{}\n")
            with self.assertRaises(PromptPilotError):
                load_completed_prompt_pilot(config, prompts, output)

    def test_prompt_pilot_core_scoring_does_not_disable_release_dialogue_gate(self):
        config = load_week7_config(SYSTEM_CONFIG)

        pilot_config = _prompt_summary_config(config)

        self.assertFalse(
            pilot_config["evaluation"]["dialogue_automatic_gate"]["enabled"]
        )
        self.assertTrue(
            config["evaluation"]["dialogue_automatic_gate"]["enabled"]
        )

    def test_prompt_pilot_stops_after_three_consecutive_model_failures(self):
        source = inspect.getsource(run_prompt_pilot)

        self.assertIn("consecutive_failures >= 3", source)

    def test_prompt_pilot_resume_requires_an_identity_bound_prefix(self):
        rows = [{"sample_id": "a"}, {"sample_id": "b"}]
        records = [
            {
                "sample_id": "a",
                "run_id": "system_repair_prompt_pilot_compact_schema_v1",
                "model_name": "adapter",
                "generation_max_new_tokens": 512,
            }
        ]

        _validate_resumable_prompt_records(
            records,
            rows,
            version="compact_schema_v1",
            served_model="adapter",
            max_new_tokens=512,
        )
        records[0]["sample_id"] = "b"
        with self.assertRaises(PromptPilotError):
            _validate_resumable_prompt_records(
                records,
                rows,
                version="compact_schema_v1",
                served_model="adapter",
                max_new_tokens=512,
            )

    def test_candidate_selection_binds_best_step_adapter_and_raw_evidence(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_payload = json.loads(SYSTEM_CONFIG.read_text(encoding="utf-8"))
            config_payload["dataset"]["output_root"] = "locks"
            config_payload["dataset"]["dataset_version"] = "repair-lock"
            config = root / "config.json"
            config.write_text(json.dumps(config_payload), encoding="utf-8")
            config_hash = sha256_file(config)
            lock_dir = root / "locks/repair-lock"
            lock_dir.mkdir(parents=True)
            (lock_dir / "dataset_lock.json").write_text(
                json.dumps({"lock_sha256": "lock-sha"}), encoding="utf-8"
            )
            training = root / "training"
            adapter = training / "adapter"
            adapter.mkdir(parents=True)
            adapter_model = adapter / "adapter_model.safetensors"
            adapter_model.write_bytes(b"candidate")
            adapter_hash = sha256_file(adapter_model)
            evidence = training / "development_evaluations/step-000012"
            evidence.mkdir(parents=True)
            raw = evidence / "raw_outputs.jsonl"
            raw.write_text("{}\n", encoding="utf-8")
            metrics = {
                "status": "COMPLETED",
                "global_step": 12,
                "sample_count": 168,
                "config_sha256": config_hash,
                "dataset_lock_sha256": "lock-sha",
                "scenarios": {
                    scenario: {}
                    for scenario in (
                        "image_product_search",
                        "after_sales",
                        "itinerary_planning",
                    )
                },
                "dialogue": {},
                "raw_outputs": {"sha256": sha256_file(raw)},
            }
            (evidence / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            summary = {
                "status": "COMPLETED",
                "run_id": config_payload["experiment_identity"][
                    "multitask_sft_run_id"
                ],
                "config_sha256": config_hash,
                "dataset_lock_sha256": "lock-sha",
                "best_checkpoint": str(training / "checkpoint-12"),
                "best_metric": 0.8,
                "adapter_only": True,
                "adapter_reload_verified": True,
                "continued_from_adapter": {
                    "adapter_model_sha256": config_payload["continuation"][
                        "adapter_model_sha256"
                    ]
                },
                "adapter_hashes": {
                    "adapter_model.safetensors": adapter_hash
                },
            }
            (training / "run_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

            result = select_system_repair_candidate(
                root,
                config,
                training,
                root / "selection.json",
            )

            self.assertEqual(result["best_checkpoint"], "checkpoint-12")
            self.assertEqual(result["adapter_model_sha256"], adapter_hash)
            self.assertFalse(result["test_consumed"])

    def test_spartan_job_requests_one_gpu_and_six_hours(self):
        text = (ROOT / "scripts/spartan/system_repair_train.sbatch").read_text(
            encoding="utf-8"
        )

        self.assertIn("#SBATCH --gpus=1", text)
        self.assertIn("#SBATCH --time=06:00:00", text)
        self.assertNotIn("--gpus=2", text)

    def test_week5_resume_identity_binds_git_commit(self):
        source = inspect.getsource(run_week5_repair_queue)

        self.assertIn('"git_commit": _git_commit(root)', source)

    def test_spartan_inference_job_is_resumable_and_reuses_one_adapter(self):
        text = (
            ROOT / "scripts/spartan/system_repair_inference.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gpus=1", text)
        self.assertIn("#SBATCH --time=03:00:00", text)
        self.assertIn("run-inference-repair", text)
        self.assertIn("TRIP_ADAPTER_DIR", text)
        self.assertNotIn("uvicorn", text)
        self.assertNotIn("#SBATCH --array", text)

    def test_spartan_development_job_compares_existing_and_zero_shot(self):
        text = (
            ROOT / "scripts/spartan/system_repair_development_eval.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gpus=1", text)
        self.assertIn("#SBATCH --time=06:00:00", text)
        self.assertIn("--model-role multitask_existing", text)
        self.assertIn("--model-role zero_shot", text)
        self.assertNotIn("final-test", text)

    def test_development_baselines_use_candidate_metric_support_protocol(self):
        source = inspect.getsource(run_transformers_development)

        self.assertIn(
            'metric_support_protocol=config["evaluation"].get(',
            source,
        )

    def test_spartan_final_test_is_one_gpu_and_gate_bound(self):
        text = (
            ROOT / "scripts/spartan/system_repair_final_test.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gpus=1", text)
        self.assertIn("#SBATCH --time=04:00:00", text)
        self.assertIn("run-final-test", text)
        self.assertIn("TRIP_SELECTION", text)
        self.assertIn("TRIP_GATE", text)
        self.assertNotIn("evaluate-development", text)


if __name__ == "__main__":
    unittest.main()

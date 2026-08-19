import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.training.week7_data import canonical_sha256, sha256_file, validate_week7_lock
from src.training.week7_evaluation import Week7EvaluationError
from src.training.week7_final_evaluation import create_parameter_lock, run_final_test_suite
from src.training.week7_selection import select_development_checkpoint


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v3.json"


TARGETS = {
    "image_product_search": {
        "business_category": "other", "style_tags": [], "visible_facilities": [],
        "price_range": "unknown", "observed_evidence": [], "inferred_attributes": [],
        "unknown_fields": ["price_range"], "confidence": None,
    },
    "after_sales": {
        "issue_type": "other", "severity": "low", "issue_location": None,
        "key_information": [], "ocr_text": None, "observed_evidence": [],
        "unknown_fields": [], "confidence": None,
    },
    "itinerary_planning": {
        "style_preferences": [], "hard_constraints": [], "soft_constraints": [],
        "required_itinerary_elements": [],
        "itinerary": [{
            "day_index": 1, "date": None, "summary": "day",
            "activities": [{
                "start_time": None, "end_time": None, "place_name": None,
                "activity": "visit", "transport": None, "source_evidence": [],
            }],
        }],
        "constraint_check": [], "observed_evidence": [], "unknown_fields": [],
        "confidence": None,
    },
}


class Week7FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.selected = self._adapter("checkpoint-10", b"multitask")
        self.week6 = {
            scenario: self._adapter(f"week6-{scenario}", scenario.encode())
            for scenario in TARGETS
        }
        config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        config["dataset"]["dataset_version"] = "week7_unit_test_lock_v1"
        config["experiment_identity"]["test_run_id"] = "week7_unit_test_once_v3"
        config["evaluation"]["week6_adapter_sha256"] = {
            scenario: sha256_file(path / "adapter_model.safetensors")
            for scenario, path in self.week6.items()
        }
        self.config = config
        self.config_path = self.root / "configs/week7/config.json"
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        shutil.copytree(ROOT / "configs/evaluation", self.root / "configs/evaluation")

        self.lock_root = self.root / "outputs/week7/locked_data/week7_unit_test_lock_v1"
        self.lock_root.mkdir(parents=True)
        rows = []
        for scenario, target in TARGETS.items():
            rows.append({
                "sample_id": f"test-{scenario}", "scenario": scenario, "split": "test",
                "target": target, "messages": [{"role": "user", "content": "test"}],
            })
        rows.append({
            "sample_id": "test-dialogue", "scenario": "dialogue", "split": "test",
            "target": TARGETS["image_product_search"],
            "messages": [{"role": "user", "content": "test"}],
            "context_expectations": {
                "historical_image_reference": [],
                "updated_requirement": "预算优先",
                "retained_hard_constraints": [],
                "evidence_policy": "visible only",
            },
        })
        test_path = self.lock_root / "test.jsonl"
        with test_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        dataset_lock = {
            "config_sha256": sha256_file(self.config_path),
            "dataset_version": "week7_unit_test_lock_v1",
            "files": {"test.jsonl": {"count": len(rows), "sha256": sha256_file(test_path)}},
            "test_policy": {"status": "LOCKED_UNCONSUMED"},
        }
        dataset_lock["lock_sha256"] = canonical_sha256(dataset_lock)
        (self.lock_root / "dataset_lock.json").write_text(json.dumps(dataset_lock), encoding="utf-8")
        self.dataset_lock = dataset_lock

        summary = {
            "status": "COMPLETED",
            "run_id": self.config["experiment_identity"]["multitask_sft_run_id"],
            "config_sha256": sha256_file(self.config_path),
            "dataset_lock_sha256": dataset_lock["lock_sha256"],
            "development_samples": 114,
            "evaluation_steps": [10],
            "global_step": 10,
            "checkpoints": [self.selected.name],
            "checkpoint_hashes": {self.selected.name: sha256_file(self.selected / "adapter_model.safetensors")},
        }
        self.training_summary = self.selected.parent / "run_summary.json"
        self.training_summary.write_text(json.dumps(summary), encoding="utf-8")
        self.evidence = self._development_evidence()
        self.selection = self.root / "selection.json"
        select_development_checkpoint(
            self.config_path, self.selected.parent, self.training_summary,
            self.evidence["week6_development_baseline"], self.selection,
        )
        self.parameter_lock = self.root / "parameter-lock.json"
        create_parameter_lock(
            self.root, self.config_path, self.parameter_lock,
            training_summary_path=self.training_summary,
            selection_path=self.selection,
            selected_checkpoint=self.selected,
            week6_adapters={
                scenario: (path, sha256_file(path / "adapter_model.safetensors"))
                for scenario, path in self.week6.items()
            },
            development_evidence=self.evidence,
            schema_decoding_mode="free",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _adapter(self, name, data):
        path = self.root / "adapters" / name
        path.mkdir(parents=True)
        (path / "adapter_model.safetensors").write_bytes(data)
        (path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct"}),
            encoding="utf-8",
        )
        return path

    def _development_evidence(self):
        config_hash = sha256_file(self.config_path)
        dataset_hash = self.dataset_lock["lock_sha256"]
        scenario_summaries = {
            scenario: {
                "composite": 0.5,
                "aggregate": {
                    "json_compliance": 1.0, "schema_pass": 1.0,
                    "latency_mean_ms": 10.0,
                },
                "metric_support": {"metric": 30},
            }
            for scenario in TARGETS
        }
        inputs = {}
        for scenario in TARGETS:
            payload = {
                "status": "COMPLETED",
                "run_id": self.config["experiment_identity"]["development_baseline_run_ids"][scenario],
                "model_role": "week6_single_task_adapter",
                "split": "development",
                "scenario_filter": scenario,
                "config_sha256": config_hash,
                "dataset_lock_sha256": dataset_hash,
                "sample_count": 30,
                "failure_count": 0, "latency_ms_mean": 10.0,
                "scenarios": {scenario: scenario_summaries[scenario]},
                "adapter_hashes": {
                    "adapter_model.safetensors": self.config["evaluation"]["week6_adapter_sha256"][scenario],
                },
            }
            path = self.root / f"week6-{scenario}-development.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            inputs[scenario] = {"path": str(path), "sha256": sha256_file(path)}

        dialogue_payload = {
            "status": "COMPLETED",
            "run_id": self.config["experiment_identity"]
            ["week6_dialogue_development_run_id"],
            "model_role": "week6_single_task_adapters",
            "split": "development", "scenario_filter": "dialogue_routed",
            "config_sha256": config_hash, "dataset_lock_sha256": dataset_hash,
            "sample_count": 24, "scenarios": {}, "dialogue": {"sample_count": 24},
            "failure_count": 0, "latency_ms_mean": 10.0,
            "adapter_hashes": {
                scenario: {
                    "adapter_model.safetensors": self.config["evaluation"]["week6_adapter_sha256"][scenario],
                }
                for scenario in TARGETS
            },
            "routing": {
                "method": "target_task_result_v1",
                "sample_counts": {scenario: 8 for scenario in TARGETS},
            },
        }
        dialogue_path = self.root / "week6-dialogue-development.json"
        dialogue_path.write_text(json.dumps(dialogue_payload), encoding="utf-8")
        inputs["dialogue"] = {"path": str(dialogue_path), "sha256": sha256_file(dialogue_path)}

        week6 = {
            "status": "COMPLETED",
            "run_id": self.config["experiment_identity"]
            ["week6_combined_development_run_id"],
            "model_role": "week6_single_task_adapters", "split": "development",
            "config_sha256": config_hash, "dataset_lock_sha256": dataset_hash,
            "sample_count": 114, "scenarios": scenario_summaries,
            "dialogue": {"sample_count": 24}, "inputs": inputs,
            "failure_count": 0, "failure_rate": 0.0,
            "latency_ms_mean": 10.0, "latency_ms_median": None,
            "weighted_composite": 0.5,
        }
        zero = {
            "status": "COMPLETED",
            "run_id": self.config["experiment_identity"]["zero_shot_development_run_id"],
            "model_role": "zero_shot", "split": "development", "scenario_filter": None,
            "config_sha256": config_hash, "dataset_lock_sha256": dataset_hash,
            "sample_count": 114, "scenarios": scenario_summaries,
            "dialogue": {"sample_count": 24}, "adapter_hashes": None,
        }
        multitask = {
            "status": "COMPLETED",
            "run_id": (
                f"{self.config['experiment_identity']['multitask_sft_run_id']}"
                "_development_step_000010"
            ),
            "model_role": "multitask_checkpoint", "split": "development",
            "scenario_filter": None, "global_step": 10,
            "config_sha256": config_hash, "dataset_lock_sha256": dataset_hash,
            "sample_count": 114, "scenarios": scenario_summaries,
            "dialogue": {"sample_count": 24},
            "failure_rate": 0.0, "latency_ms_mean": 10.0,
            "weighted_composite": 0.5,
        }
        schema = {
            "status": "COMPLETED", "model_role": "schema_format_only_experiment",
            "split": "development", "config_sha256": config_hash,
            "dataset_lock_sha256": dataset_hash, "scope": "format_only",
            "semantic_claims": "FORBIDDEN", "sample_count": 90,
            "run_ids": {
                "free": self.config["experiment_identity"]["schema_free_run_id"],
                "constrained": self.config["experiment_identity"]["schema_constrained_run_id"],
            },
            "gate": {"latency": True, "fallback": True},
            "modes": {
                mode: {
                    "json_compliance": 1.0, "schema_coverage": 1.0,
                    "fallback_failure_rate": 0.0, "latency_ms_mean": 10.0,
                }
                for mode in ("free", "constrained")
            },
        }
        payloads = {
            "week6_development_baseline": week6,
            "zero_shot_development": zero,
            "multitask_development": multitask,
            "schema_decoding": schema,
        }
        result = {}
        for name, payload in payloads.items():
            path = (
                self.selected.parent
                / "development_evaluations/step-000010/metrics.json"
                if name == "multitask_development"
                else self.root / f"{name}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            result[name] = path
        return result

    @staticmethod
    def _successful_runner(role, rows, adapter, record_sink):
        records = [{
            "sample_id": row["sample_id"], "model_name": role,
            "raw_output": json.dumps(row["target"]), "latency_ms": 10.0,
            "failed": False, "error": None,
        } for row in rows]
        for record in records:
            record_sink(record)
        return records

    def test_include_test_validator_cannot_bypass_parameter_lock(self):
        with self.assertRaisesRegex(ValueError, "one-shot final suite"):
            validate_week7_lock(self.root, self.config_path, include_test=True)

    def test_parameter_lock_binds_checkpoint_and_development_evidence(self):
        payload = json.loads(self.parameter_lock.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "LOCKED")
        self.assertEqual(payload["selected_checkpoint_sha256"], sha256_file(self.selected / "adapter_model.safetensors"))
        self.assertEqual(payload["selection"]["sha256"], sha256_file(self.selection))
        self.assertEqual(set(payload["week6_adapters"]), set(TARGETS))
        changed = copy.deepcopy(payload)
        changed["dataset_lock_sha256"] = "0" * 64
        changed["lock_sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "lock_sha256"})
        bad = self.root / "bad-lock.json"
        bad.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(Week7EvaluationError, "dataset identity"):
            run_final_test_suite(self.root, self.config_path, bad, self.root / "bad-output", inference_runner=self._successful_runner)
        selection = json.loads(self.selection.read_text(encoding="utf-8"))
        selection["selected"]["eligible"] = False
        self.selection.write_text(json.dumps(selection), encoding="utf-8")
        with self.assertRaisesRegex(Week7EvaluationError, "selection artifact changed"):
            run_final_test_suite(
                self.root, self.config_path, self.parameter_lock,
                self.root / "changed-selection-output",
                inference_runner=self._successful_runner,
            )

    def test_parameter_lock_rejects_weak_evidence_and_constrained_mode(self):
        bad_evidence = dict(self.evidence)
        invalid = self.root / "invalid-zero.json"
        payload = json.loads(self.evidence["zero_shot_development"].read_text(encoding="utf-8"))
        payload["run_id"] = "wrong-run"
        invalid.write_text(json.dumps(payload), encoding="utf-8")
        bad_evidence["zero_shot_development"] = invalid
        with self.assertRaisesRegex(Week7EvaluationError, "coverage mismatch"):
            create_parameter_lock(
                self.root, self.config_path, self.root / "bad-evidence-lock.json",
                training_summary_path=self.training_summary,
                selection_path=self.selection,
                selected_checkpoint=self.selected,
                week6_adapters={
                    scenario: (path, sha256_file(path / "adapter_model.safetensors"))
                    for scenario, path in self.week6.items()
                },
                development_evidence=bad_evidence,
                schema_decoding_mode="free",
            )
        bad_schema_evidence = dict(self.evidence)
        invalid_schema = self.root / "invalid-schema.json"
        schema_payload = json.loads(self.evidence["schema_decoding"].read_text(encoding="utf-8"))
        schema_payload["gate"]["latency"] = False
        invalid_schema.write_text(json.dumps(schema_payload), encoding="utf-8")
        bad_schema_evidence["schema_decoding"] = invalid_schema
        with self.assertRaisesRegex(Week7EvaluationError, "gate is inconsistent"):
            create_parameter_lock(
                self.root, self.config_path, self.root / "bad-schema-lock.json",
                training_summary_path=self.training_summary,
                selection_path=self.selection,
                selected_checkpoint=self.selected,
                week6_adapters={
                    scenario: (path, sha256_file(path / "adapter_model.safetensors"))
                    for scenario, path in self.week6.items()
                },
                development_evidence=bad_schema_evidence,
                schema_decoding_mode="free",
            )
        with self.assertRaisesRegex(Week7EvaluationError, "only.*free"):
            create_parameter_lock(
                self.root, self.config_path, self.root / "constrained-lock.json",
                training_summary_path=self.training_summary,
                selection_path=self.selection,
                selected_checkpoint=self.selected,
                week6_adapters={
                    scenario: (path, sha256_file(path / "adapter_model.safetensors"))
                    for scenario, path in self.week6.items()
                },
                development_evidence=self.evidence,
                schema_decoding_mode="constrained",
            )

    def test_parameter_lock_rejects_forged_dialogue_routing(self):
        combined = json.loads(
            self.evidence["week6_development_baseline"].read_text(encoding="utf-8")
        )
        original_dialogue = Path(combined["inputs"]["dialogue"]["path"])
        dialogue = json.loads(original_dialogue.read_text(encoding="utf-8"))
        dialogue["routing"]["sample_counts"] = dict(zip(TARGETS, (7, 8, 9)))
        bad_dialogue = self.root / "bad-dialogue-route.json"
        bad_dialogue.write_text(json.dumps(dialogue), encoding="utf-8")
        combined["inputs"]["dialogue"] = {
            "path": str(bad_dialogue), "sha256": sha256_file(bad_dialogue),
        }
        bad_combined = self.root / "bad-week6-combined.json"
        bad_combined.write_text(json.dumps(combined), encoding="utf-8")
        selection = json.loads(self.selection.read_text(encoding="utf-8"))
        selection["week6_baseline"] = {
            "path": str(bad_combined.resolve()), "sha256": sha256_file(bad_combined),
        }
        bad_selection = self.root / "bad-route-selection.json"
        bad_selection.write_text(json.dumps(selection), encoding="utf-8")
        evidence = dict(self.evidence)
        evidence["week6_development_baseline"] = bad_combined
        with self.assertRaisesRegex(Week7EvaluationError, "dialogue evidence mismatch"):
            create_parameter_lock(
                self.root, self.config_path, self.root / "bad-route-lock.json",
                training_summary_path=self.training_summary,
                selection_path=bad_selection,
                selected_checkpoint=self.selected,
                week6_adapters={
                    scenario: (path, sha256_file(path / "adapter_model.safetensors"))
                    for scenario, path in self.week6.items()
                },
                development_evidence=evidence,
                schema_decoding_mode="free",
            )

    def test_parameter_lock_recomputes_selection_gates(self):
        selection = json.loads(self.selection.read_text(encoding="utf-8"))
        selection["selected"]["failure_gate"] = False
        selection["candidates"][0]["failure_gate"] = False
        forged = self.root / "forged-selection-gates.json"
        forged.write_text(json.dumps(selection), encoding="utf-8")
        with self.assertRaisesRegex(Week7EvaluationError, "gates were not reproducible"):
            create_parameter_lock(
                self.root, self.config_path, self.root / "forged-gate-lock.json",
                training_summary_path=self.training_summary,
                selection_path=forged,
                selected_checkpoint=self.selected,
                week6_adapters={
                    scenario: (path, sha256_file(path / "adapter_model.safetensors"))
                    for scenario, path in self.week6.items()
                },
                development_evidence=self.evidence,
                schema_decoding_mode="free",
            )

    def test_same_run_resumes_but_second_output_is_rejected(self):
        output = self.root / "final-output"
        calls = []

        def interrupted(role, rows, adapter, record_sink):
            calls.append(role)
            if role == "multitask":
                raise RuntimeError("simulated interruption")
            return self._successful_runner(role, rows, adapter, record_sink)

        with self.assertRaisesRegex(RuntimeError, "simulated"):
            run_final_test_suite(
                self.root, self.config_path, self.parameter_lock, output,
                inference_runner=interrupted,
            )
        self.assertEqual(calls.count("week6_single_task_adapters"), 3)
        marker = self.root / "outputs/week7/test_consumption/week7_unit_test_lock_v1.json"
        self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["status"], "FAILED")
        result = run_final_test_suite(
            self.root, self.config_path, self.parameter_lock, output,
            resume=True, inference_runner=self._successful_runner,
        )
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(set(result["models"]), {
            "week6_single_task_adapters", "multitask", "zero_shot",
        })
        completed = run_final_test_suite(
            self.root, self.config_path, self.parameter_lock, output,
            resume=True, inference_runner=lambda *_: self.fail("completed run must not infer again"),
        )
        self.assertEqual(completed["run_id"], "week7_unit_test_once_v3")
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual(marker_payload["status"], "COMPLETED")
        self.assertEqual(len(marker_payload["artifact_hashes"]), 7)
        with self.assertRaisesRegex(Week7EvaluationError, "another run identity"):
            run_final_test_suite(
                self.root, self.config_path, self.parameter_lock, self.root / "second-output",
                resume=True, inference_runner=self._successful_runner,
            )
        raw = output / "raw_outputs/multitask.jsonl"
        raw.write_text(raw.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(Week7EvaluationError, "artifact hash mismatch"):
            run_final_test_suite(
                self.root, self.config_path, self.parameter_lock, output,
                resume=True, inference_runner=self._successful_runner,
            )

    def test_concurrent_resume_is_rejected_while_owner_is_in_progress(self):
        output = self.root / "concurrent-output"
        observed = []

        def runner(role, rows, adapter, record_sink):
            if not observed:
                with self.assertRaisesRegex(Week7EvaluationError, "IN_PROGRESS"):
                    run_final_test_suite(
                        self.root, self.config_path, self.parameter_lock, output,
                        resume=True, inference_runner=self._successful_runner,
                    )
                observed.append("rejected")
            return self._successful_runner(role, rows, adapter, record_sink)

        result = run_final_test_suite(
            self.root, self.config_path, self.parameter_lock, output,
            inference_runner=runner,
        )
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(observed, ["rejected"])


if __name__ == "__main__":
    unittest.main()

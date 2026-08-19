import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.training.week7_data import canonical_sha256, sha256_file, validate_week7_lock
from src.training.week7_evaluation import Week7EvaluationError
from src.training.week7_final_evaluation import create_parameter_lock, run_final_test_suite


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v1.json"


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
        config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        config["dataset"]["dataset_version"] = "week7_unit_test_lock_v1"
        config["experiment_identity"]["test_run_id"] = "week7-unit-test-once"
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

        self.selected = self._adapter("checkpoint-10", b"multitask")
        self.week6 = {
            scenario: self._adapter(f"week6-{scenario}", scenario.encode())
            for scenario in TARGETS
        }
        summary = {
            "status": "COMPLETED", "config_sha256": sha256_file(self.config_path),
            "dataset_lock_sha256": dataset_lock["lock_sha256"],
            "best_checkpoint": str(self.selected),
            "checkpoint_hashes": {self.selected.name: sha256_file(self.selected / "adapter_model.safetensors")},
        }
        self.training_summary = self.root / "training-summary.json"
        self.training_summary.write_text(json.dumps(summary), encoding="utf-8")
        self.evidence = {}
        for name in (
            "week6_development_baseline", "zero_shot_development",
            "multitask_development", "schema_decoding",
        ):
            path = self.root / f"{name}.json"
            path.write_text(json.dumps({"status": "COMPLETED", "name": name}), encoding="utf-8")
            self.evidence[name] = path
        self.parameter_lock = self.root / "parameter-lock.json"
        create_parameter_lock(
            self.root, self.config_path, self.parameter_lock,
            training_summary_path=self.training_summary,
            selected_checkpoint=self.selected,
            week6_adapters={
                scenario: (path, sha256_file(path / "adapter_model.safetensors"))
                for scenario, path in self.week6.items()
            },
            development_evidence=self.evidence,
            schema_decoding_mode="constrained",
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
        self.assertEqual(set(payload["week6_adapters"]), set(TARGETS))
        changed = copy.deepcopy(payload)
        changed["dataset_lock_sha256"] = "0" * 64
        changed["lock_sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "lock_sha256"})
        bad = self.root / "bad-lock.json"
        bad.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(Week7EvaluationError, "dataset identity"):
            run_final_test_suite(self.root, self.config_path, bad, self.root / "bad-output", inference_runner=self._successful_runner)

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
        self.assertEqual(completed["run_id"], "week7-unit-test-once")
        with self.assertRaisesRegex(Week7EvaluationError, "another run identity"):
            run_final_test_suite(
                self.root, self.config_path, self.parameter_lock, self.root / "second-output",
                resume=True, inference_runner=self._successful_runner,
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.week7_data import CORE_SCENARIOS, sha256_file
from src.training.week7_evaluation import summarize_dialogue_raw_records
from src.training.week7_inference import (
    WEEK6_DIALOGUE_DEVELOPMENT_RUN_ID,
    _dialogue_task,
    combine_week6_development_baseline,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v2.json"


class Week7DialogueBaselineTests(unittest.TestCase):
    def test_dialogue_route_uses_nested_task_result(self) -> None:
        targets = {
            "image_product_search": {"business_category": "hotel"},
            "after_sales": {"issue_type": "damage"},
            "itinerary_planning": {"hard_constraints": ["budget"]},
        }
        for expected, target in targets.items():
            self.assertEqual(_dialogue_task({
                "sample_id": expected,
                "target": {"task_result": target},
            }), expected)

    def test_dialogue_only_summary_preserves_pending_human_status(self) -> None:
        row = {
            "sample_id": "dialogue-1", "scenario": "dialogue",
            "context_expectations": {
                "historical_image_reference": ["红色招牌"],
                "updated_requirement": "预算优先",
                "retained_hard_constraints": [],
            },
        }
        records = [{
            "sample_id": "dialogue-1",
            "raw_output": json.dumps({"note": "红色招牌，预算优先"}, ensure_ascii=False),
            "latency_ms": 25.0, "failed": False,
        }]
        result = summarize_dialogue_raw_records([row], records)
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["scenarios"], {})
        self.assertIsNone(result["weighted_composite"])
        self.assertEqual(
            result["dialogue"]["human_dimensions_status"],
            "PENDING_REAL_HUMAN_INPUT",
        )

    def test_combine_requires_and_includes_routed_dialogue_evidence(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config_hash = sha256_file(CONFIG)
        lock_hash = "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario_paths = {}
            for index, scenario in enumerate(CORE_SCENARIOS):
                payload = {
                    "status": "COMPLETED",
                    "run_id": config["experiment_identity"]["development_baseline_run_ids"][scenario],
                    "model_role": "week6_single_task_adapter",
                    "split": "development", "scenario_filter": scenario,
                    "config_sha256": config_hash, "dataset_lock_sha256": lock_hash,
                    "sample_count": 30, "failure_count": 0,
                    "latency_ms_mean": 100.0 + index,
                    "scenarios": {scenario: {"composite": 0.5}},
                    "adapter_hashes": {
                        "adapter_model.safetensors": config["evaluation"]["week6_adapter_sha256"][scenario],
                    },
                }
                path = root / f"{scenario}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                scenario_paths[scenario] = path
            dialogue = {
                "status": "COMPLETED", "run_id": WEEK6_DIALOGUE_DEVELOPMENT_RUN_ID,
                "model_role": "week6_single_task_adapters", "split": "development",
                "scenario_filter": "dialogue_routed", "config_sha256": config_hash,
                "dataset_lock_sha256": lock_hash, "sample_count": 24,
                "failure_count": 0, "latency_ms_mean": 200.0,
                "scenarios": {}, "dialogue": {"sample_count": 24},
                "adapter_hashes": {
                    scenario: {
                        "adapter_model.safetensors": config["evaluation"]["week6_adapter_sha256"][scenario],
                    }
                    for scenario in CORE_SCENARIOS
                },
                "routing": {
                    "method": "target_task_result_v1",
                    "sample_counts": {
                        "image_product_search": 8, "after_sales": 8,
                        "itinerary_planning": 8,
                    },
                },
            }
            dialogue_path = root / "dialogue.json"
            dialogue_path.write_text(json.dumps(dialogue), encoding="utf-8")
            result = combine_week6_development_baseline(
                CONFIG, scenario_paths, dialogue_path, root / "combined.json",
            )
            self.assertEqual(result["sample_count"], 114)
            self.assertEqual(result["dialogue"]["sample_count"], 24)
            self.assertEqual(set(result["inputs"]), set(CORE_SCENARIOS) | {"dialogue"})
            expected_latency = (30 * (100 + 101 + 102) + 24 * 200) / 114
            self.assertAlmostEqual(result["latency_ms_mean"], expected_latency)


if __name__ == "__main__":
    unittest.main()

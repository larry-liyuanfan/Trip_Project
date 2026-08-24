import json
import unittest
from pathlib import Path

from src.training.system_repair import (
    SystemRepairError,
    _validate_historical_failure_contract,
    evaluate_system_release_gates,
    load_repair_config,
)
from src.training.week7_data import _product_target, load_week7_config


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_CONFIG = ROOT / "configs/system_repair/qwen3_vl_8b_system_repair_v1.json"
WEEK5_CONFIG = ROOT / "configs/system_repair/week5_preannotation_repair_v2.json"


def metrics(composite=0.8, dialogue=0.8):
    scenarios = {}
    for scenario in ("image_product_search", "after_sales", "itinerary_planning"):
        scenarios[scenario] = {
            "composite": composite,
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
        candidate["scenarios"]["image_product_search"]["support"]["price_range"] = 0

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

    def test_spartan_job_requests_one_gpu_and_six_hours(self):
        text = (ROOT / "scripts/spartan/system_repair_train.sbatch").read_text(
            encoding="utf-8"
        )

        self.assertIn("#SBATCH --gpus=1", text)
        self.assertIn("#SBATCH --time=06:00:00", text)
        self.assertNotIn("--gpus=2", text)


if __name__ == "__main__":
    unittest.main()

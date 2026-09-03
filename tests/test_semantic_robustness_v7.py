"""Tests for the development-only v7 semantic robustness cycle."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_semantic_robustness_pool_v7 import build_pool as build_v7_pool
from src.evaluation.semantic_robustness_v7 import apply_semantic_robustness_v7_gates


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "evaluation" / "evidence_enhancement" / "semantic_robustness_pool_lock_v7.json"
V5_LOCK = ROOT / "configs" / "evaluation" / "evidence_enhancement" / "context_focus_pool_lock_v5.json"


class SemanticRobustnessV7Tests(unittest.TestCase):
    def test_pool_has_no_final_and_is_isolated_from_v5(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v7 = root / "v7"
            lock = build_v7_pool(v7, V5_LOCK)
            self.assertEqual(lock, json.loads(LOCK.read_text(encoding="utf-8")))
            self.assertEqual(lock["splits"], ["training", "development"])
            self.assertEqual(lock["vlm"]["training"]["sample_support"], 512)
            self.assertEqual(lock["vlm"]["development"]["sample_support"], 96)
            self.assertEqual(lock["prior_v5_training_isolation"]["status"], "PASS")
            self.assertFalse((v7 / "vlm_final_manifest.jsonl").exists())
            self.assertEqual(lock["vlm"]["development"]["slice_support"]["known_visible_price"], 16)
            self.assertEqual(lock["vlm"]["development"]["slice_support"]["multi_subject_conflict"], 8)

    def test_gate_requires_real_improvement_and_keeps_failures_negative(self) -> None:
        baseline = _variant(0.80)
        candidate = _variant(0.83)
        report = {"variants": {"v5": baseline, "v7": candidate}}
        gates = _gates()
        objective = _objective()
        passed = apply_semantic_robustness_v7_gates(
            report, gates, objective, candidate="v7", baseline="v5"
        )
        self.assertEqual(passed["status"], "PASS")
        candidate["dialogue_metrics"]["context_recall"] = 0.70
        failed = apply_semantic_robustness_v7_gates(
            report, gates, objective, candidate="v7", baseline="v5"
        )
        self.assertEqual(failed["status"], "FAIL")
        self.assertFalse(failed["checks"]["dialogue_context_recall"])


def _variant(value: float) -> dict[str, object]:
    return {
        "support": 96,
        "product_support": 48,
        "dialogue_support": 48,
        "field_metrics": {
            field: {"f1": value}
            for field in ("business_category", "style_tags", "visible_facilities", "price_range")
        },
        "supported_field_exact_match": value,
        "unsupported_hallucination_rate": 1 - value,
        "unknown_field_opportunity_support": 96,
        "unknown_field_abstention_accuracy": value,
        "first_attempt_json_compliance": 1.0,
        "dialogue_metrics": {
            "context_recall": value,
            "state_value_correct": value,
            "task_key_correct": value,
            "value_correct": value,
            "first_turn_routing_correct": 1.0,
        },
        "slice_metrics": {
            "multi_subject_conflict": {"support": 8, "all_unknown_fields_abstained_accuracy": value},
            "insufficient_visual_evidence": {"support": 16, "all_unknown_fields_abstained_accuracy": value},
        },
    }


def _gates() -> dict[str, float | int]:
    return {
        "min_product_support": 48,
        "min_dialogue_support": 48,
        "min_field_f1": 0.65,
        "max_unsupported_hallucination_rate": 0.20,
        "min_unknown_abstention_accuracy": 0.80,
        "min_first_attempt_json_compliance": 0.90,
        "min_multi_subject_abstention_accuracy": 0.75,
        "min_insufficient_evidence_abstention_accuracy": 0.75,
        "min_dialogue_metric": 0.75,
        "max_per_metric_regression": 0.05,
    }


def _objective() -> dict[str, float]:
    return {
        "field_f1_mean_weight": 0.35,
        "unknown_abstention_weight": 0.15,
        "dialogue_mean_weight": 0.30,
        "first_json_weight": 0.10,
        "supported_exact_match_weight": 0.10,
        "min_improvement_over_baseline": 0.01,
    }


if __name__ == "__main__":
    unittest.main()

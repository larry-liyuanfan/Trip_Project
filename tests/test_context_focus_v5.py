"""Unit tests for the locked synthetic context-focus v5 cycle."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_context_focus_pool_v5 import build_pool
from src.evaluation.context_focus_v5 import apply_context_focus_v5_development_gates
from src.evaluation.relevance_evidence import load_jsonl


ROOT = Path(__file__).resolve().parents[1]


class ContextFocusV5Tests(unittest.TestCase):
    def test_generator_matches_lock_and_all_identities_are_isolated(self) -> None:
        expected = json.loads((
            ROOT / "configs/evaluation/evidence_enhancement/context_focus_pool_lock_v5.json"
        ).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pool"
            actual = build_pool(output)
            self.assertEqual(actual, expected)
            self.assertEqual(actual["isolation"]["status"], "PASS")
            rows = {
                split: load_jsonl(output / f"vlm_{split}_manifest.jsonl")
                for split in ("training", "development", "final")
            }
            self.assertEqual([len(rows[name]) for name in rows], [528, 48, 48])
            self.assertTrue(all(
                row["prompt"] == rows["training"][0]["prompt"]
                for row in rows["training"]
                if row["scenario"] == "product"
            ))

    def test_development_gate_requires_context_gain_and_retention(self) -> None:
        def variant(context: float, state: float = 1.0) -> dict:
            fields = {
                field: {"f1": 1.0}
                for field in (
                    "business_category", "style_tags", "visible_facilities", "price_range"
                )
            }
            return {
                "support": 48,
                "product_support": 24,
                "dialogue_support": 24,
                "field_metrics": fields,
                "unsupported_hallucination_rate": 0.0,
                "unknown_field_abstention_accuracy": 1.0,
                "unknown_field_opportunity_support": 36,
                "first_attempt_json_compliance": 1.0,
                "slice_metrics": {
                    "multi_subject_conflict": {
                        "support": 6,
                        "all_unknown_fields_abstained_accuracy": 1.0,
                    }
                },
                "dialogue_metrics": {
                    "context_recall": context,
                    "state_value_correct": state,
                    "task_key_correct": 1.0,
                    "value_correct": 1.0,
                    "first_turn_routing_correct": 1.0,
                },
            }

        gates = {
            "min_product_support": 24,
            "min_dialogue_support": 24,
            "min_field_f1": 0.45,
            "max_unsupported_hallucination_rate": 0.35,
            "min_unknown_abstention_accuracy": 0.65,
            "min_first_attempt_json_compliance": 0.8,
            "min_multi_subject_abstention_accuracy": 0.5,
            "min_dialogue_metric": 0.6,
            "min_context_recall_improvement": 0.1,
            "max_other_dialogue_regression": 0.1,
        }
        report = {"variants": {"baseline": variant(0.5), "candidate": variant(0.75)}}
        result = apply_context_focus_v5_development_gates(
            report, gates, candidate="candidate", baseline="baseline"
        )
        self.assertEqual(result["status"], "PASS")
        report["variants"]["candidate"] = variant(0.55)
        self.assertEqual(apply_context_focus_v5_development_gates(
            report, gates, candidate="candidate", baseline="baseline"
        )["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()

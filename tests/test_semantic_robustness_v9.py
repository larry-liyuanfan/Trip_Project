"""Tests for the development-only v9 multi-subject robustness cycle."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_semantic_robustness_pool_v7 import PRODUCT_PROMPT
from scripts.build_semantic_robustness_pool_v9 import build_pool
from src.evaluation.relevance_evidence import load_jsonl


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "evaluation" / "evidence_enhancement" / "semantic_robustness_pool_lock_v9.json"
V5_LOCK = ROOT / "configs" / "evaluation" / "evidence_enhancement" / "context_focus_pool_lock_v5.json"
V7_LOCK = ROOT / "configs" / "evaluation" / "evidence_enhancement" / "semantic_robustness_pool_lock_v7.json"
V7_CONFIG = ROOT / "configs" / "evaluation" / "automated_evidence_v7.json"
V9_CONFIG = ROOT / "configs" / "evaluation" / "automated_evidence_v9.json"


class SemanticRobustnessV9Tests(unittest.TestCase):
    def test_config_changes_only_the_preregistered_data_factor(self) -> None:
        v7 = json.loads(V7_CONFIG.read_text(encoding="utf-8"))
        v9 = json.loads(V9_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(v9["vlm"]["base_model"], v7["vlm"]["base_model"])
        self.assertEqual(v9["vlm"]["base_revision"], v7["vlm"]["base_revision"])
        for key in (
            "optimizer",
            "learning_rate",
            "lr_scheduler_type",
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "gradient_accumulation_steps",
            "epochs",
            "max_length",
            "logging_steps",
            "attn_implementation",
            "seed",
        ):
            self.assertEqual(v9["training"][key], v7["training"][key])
        v7_gates = dict(v7["vlm"]["exploration_gates"])
        v9_gates = dict(v9["vlm"]["exploration_gates"])
        self.assertGreater(v9_gates.pop("min_product_support"), v7_gates.pop("min_product_support"))
        self.assertEqual(v9_gates, v7_gates)
        self.assertEqual(v9["vlm"]["selection_objective"], v7["vlm"]["selection_objective"])
        self.assertEqual(v9["training"]["initial_adapter_role"], "semantic_robustness_adapter_v7")

    def test_pool_is_locked_and_isolated_from_v5_and_v7(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pool = Path(temp_dir) / "v9"
            lock = build_pool(pool, V5_LOCK, V7_LOCK)
            self.assertEqual(lock, json.loads(LOCK.read_text(encoding="utf-8")))
            self.assertEqual(lock["vlm"]["training"]["sample_support"], 640)
            self.assertEqual(lock["vlm"]["development"]["sample_support"], 132)
            self.assertEqual(lock["vlm"]["development"]["slice_support"]["multi_subject_conflict"], 24)
            self.assertEqual(lock["prior_v5_training_isolation"]["status"], "PASS")
            self.assertEqual(lock["prior_v7_training_and_development_isolation"]["status"], "PASS")
            self.assertFalse((pool / "vlm_final_manifest.jsonl").exists())

    def test_multi_subject_cases_keep_prompt_fixed_and_require_full_abstention(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pool = Path(temp_dir) / "v9"
            build_pool(pool, V5_LOCK, V7_LOCK)
            rows = load_jsonl(pool / "vlm_development_manifest.jsonl")
            cases = [row for row in rows if "multi_subject_conflict" in row["slices"]]
            self.assertEqual(len(cases), 24)
            for row in cases:
                self.assertEqual(row["prompt"], PRODUCT_PROMPT)
                self.assertEqual(row["gold"]["business_category"], "unknown")
                self.assertEqual(row["gold"]["style_tags"], [])
                self.assertEqual(row["gold"]["visible_facilities"], [])
                self.assertEqual(row["gold"]["price_range"], "unknown")
                self.assertEqual(len(row["gold"]["unknown_fields"]), 4)
                self.assertEqual(
                    row["label_provenance"],
                    "synthetic_programmatic_card_v9_no_human_annotation",
                )


if __name__ == "__main__":
    unittest.main()

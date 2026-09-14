"""Tests for the synthetic no-result calibration/validation cycle."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_no_result_stress_pool_v8 import build_pool
from scripts.run_no_result_stress_v8 import _evaluate, load_validation_after_marker
from src.evaluation.no_result_stress_v8 import apply_no_result_stress_v8_gate
from src.evaluation.relevance_evidence import load_jsonl, validate_annotation_protocol, validate_query_manifest


ROOT = Path(__file__).resolve().parents[1]
V4_LOCK = ROOT / "configs" / "evaluation" / "evidence_enhancement" / "exploration_pool_lock_v4.json"
V8_LOCK = ROOT / "configs" / "evaluation" / "evidence_enhancement" / "no_result_stress_pool_lock_v8.json"


class NoResultStressV8Tests(unittest.TestCase):
    def test_pool_has_no_final_and_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "v8"
            lock = build_pool(output, V4_LOCK)
            self.assertEqual(lock, json.loads(V8_LOCK.read_text(encoding="utf-8")))
            self.assertEqual(lock["search"]["calibration"]["query_support"], 40)
            self.assertEqual(lock["search"]["validation"]["no_result_support"], 20)
            self.assertEqual(lock["prior_v4_training_isolation"]["status"], "PASS")
            self.assertFalse((output / "search_final_manifest.jsonl").exists())
            for split in ("calibration", "validation"):
                queries = load_jsonl(output / f"search_{split}_manifest.jsonl")
                annotations = load_jsonl(output / f"search_{split}_annotations.jsonl")
                self.assertEqual(validate_query_manifest(queries, output)["status"], "PASS")
                self.assertEqual(validate_annotation_protocol(queries, annotations)["status"], "PASS")

    def test_gate_preserves_failed_candidate_as_negative(self) -> None:
        report = {"methods": {
            "v4": _method(0.80, 0.95, 0.85),
            "v8": _method(0.85, 0.95, 0.88),
        }}
        gate = apply_no_result_stress_v8_gate(report, _gates(), candidate="v8", baseline="v4")
        self.assertEqual(gate["status"], "PASS")
        report["methods"]["v8"]["slices"]["no_result"]["no_result_accuracy"] = 0.70
        gate = apply_no_result_stress_v8_gate(report, _gates(), candidate="v8", baseline="v4")
        self.assertEqual(gate["status"], "FAIL")
        self.assertFalse(gate["checks"]["no_result_accuracy"])

    def test_validation_marker_is_written_before_evidence_is_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            marker = root / "marker.json"
            calls = []

            def loader(path: Path) -> list[dict[str, object]]:
                self.assertTrue(marker.is_file())
                calls.append(path.name)
                return []

            load_validation_after_marker(root, marker, {"status": "locked"}, loader=loader)
            self.assertEqual(
                calls,
                ["search_validation_manifest.jsonl", "search_validation_annotations.jsonl"],
            )

    def test_dual_guard_adds_absolute_business_similarity_abstention(self) -> None:
        prepared = [{
            "query": {"query_id": "q1", "requested_filters": {}, "unsupported_constraints": []},
            "query_vector": np.array([1.0, 0.0]),
            "similarities": np.array([0.9]),
            "metadata": [{"image_id": 1, "business_category": "restaurant"}],
            "eligible": [0],
            "relevant_total": 1,
        }]
        rows = _evaluate(
            prepared,
            {"business": np.array([0.2, 0.0]), "non_business": np.array([0.0, 0.0])},
            {"margin_threshold": 0.0, "business_similarity_threshold": 0.5},
            {"search": {"top_k": 10, "prior_v4_business_guard": {"margin_threshold": 0.0}}},
        )
        methods = rows[0]["methods"]
        self.assertFalse(methods["v4_margin_guard"]["no_result"])
        self.assertTrue(methods["dual_centroid_guard"]["no_result"])


def _method(no_result: float, positive: float, ndcg: float) -> dict[str, object]:
    return {
        "support": 40,
        "ranking_support": 20,
        "ndcg_at_10": ndcg,
        "filter_correctness": 1.0,
        "failure_rate": 0.0,
        "slices": {
            "no_result": {"support": 20, "no_result_accuracy": no_result},
            "business_positive": {"support": 20, "no_result_accuracy": positive},
        },
    }


def _gates() -> dict[str, float | int]:
    return {
        "min_query_support": 40,
        "min_ranking_support": 20,
        "min_no_result_support": 20,
        "min_no_result_accuracy": 0.75,
        "min_business_positive_acceptance": 0.90,
        "min_ndcg_at_10": 0.75,
        "min_filter_correctness": 1.0,
        "max_failure_rate": 0.0,
        "max_metric_regression": 0.05,
    }


if __name__ == "__main__":
    unittest.main()

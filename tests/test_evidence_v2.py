import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_retrieval_query_leakage_v2 import audit
from scripts.build_automated_evidence_pool_v2 import build_pool
from src.evaluation.evidence_v2 import (
    score_vlm_v3_comparison,
    select_calibration_configuration,
    summarize_performance_matrix,
    validate_calibration_holdout_isolation,
)
from src.evaluation.relevance_evidence import file_sha256, load_jsonl


class AutomatedPoolV2Tests(unittest.TestCase):
    def test_generated_pool_matches_committed_lock_and_is_disjoint(self):
        expected = json.loads(
            Path("configs/evaluation/evidence_enhancement/automated_pool_lock_v2.json")
            .read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pool"
            actual = build_pool(root)
            calibration = load_jsonl(root / "search_calibration_manifest.jsonl")
            holdout = load_jsonl(root / "search_holdout_manifest.jsonl")
            isolation = validate_calibration_holdout_isolation(calibration, holdout)
        self.assertEqual(actual, expected)
        self.assertEqual(isolation["status"], "PASS")
        self.assertEqual(isolation["calibration_support"], 16)
        self.assertEqual(isolation["holdout_support"], 16)

    def test_calibration_selection_uses_fixed_tie_break(self):
        candidates = [
            {"objective": 0.8, "configuration": {"star_rating_weight": 0.02, "no_result_similarity_threshold": 0.18}},
            {"objective": 0.8, "configuration": {"star_rating_weight": 0.0, "no_result_similarity_threshold": 0.24}},
            {"objective": 0.8, "configuration": {"star_rating_weight": 0.0, "no_result_similarity_threshold": 0.12}},
        ]
        selected = select_calibration_configuration(candidates)
        self.assertEqual(selected["configuration"]["star_rating_weight"], 0.0)
        self.assertEqual(selected["configuration"]["no_result_similarity_threshold"], 0.12)


class LeakageAuditV2Tests(unittest.TestCase):
    def test_complete_primary_can_pass_with_explicit_incomplete_replica(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            replica = root / "replica"
            (primary / "data/yelp/raw/photos").mkdir(parents=True)
            replica.mkdir()
            (primary / "data/yelp/raw/photos/a.jpg").write_bytes(b"a")
            (primary / "data/yelp/raw/photos/b.jpg").write_bytes(b"b")
            (replica / "a.jpg").write_bytes(b"a")
            query = root / "queries.jsonl"
            query.write_text(
                json.dumps({"query_id": "q", "source": {"source_id": "synthetic:q"}, "image": {"sha256": "0" * 64}}) + "\n",
                encoding="utf-8",
            )
            metadata = [
                {"image_id": "a", "business_id": "x", "source_image_path": "data/yelp/raw/photos/a.jpg"},
                {"image_id": "b", "business_id": "y", "source_image_path": "data/yelp/raw/photos/b.jpg"},
            ]
            report, registry = audit(metadata, primary, replica, [query], 2)
        self.assertEqual(report["status"], "PASS_COMPLETE_NO_QUERY_INDEX_COLLISION")
        self.assertEqual(report["comparison_replica_coverage"]["status"], "UNKNOWN_INCOMPLETE")
        self.assertEqual(len(registry), 2)

    def test_byte_collision_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            replica = root / "replica"
            (primary / "data/yelp/raw/photos").mkdir(parents=True)
            replica.mkdir()
            image = primary / "data/yelp/raw/photos/a.jpg"
            image.write_bytes(b"same")
            (replica / "a.jpg").write_bytes(b"same")
            query = root / "queries.jsonl"
            query.write_text(
                json.dumps({"query_id": "q", "source": {"source_id": "synthetic:q"}, "image": {"sha256": file_sha256(image)}}) + "\n",
                encoding="utf-8",
            )
            report, _ = audit(
                [{"image_id": "a", "business_id": "x", "source_image_path": "data/yelp/raw/photos/a.jpg"}],
                primary,
                replica,
                [query],
                1,
            )
        self.assertEqual(report["collision_check"]["status"], "FAIL")


class VlmAndPerformanceV2Tests(unittest.TestCase):
    def _rows(self, variant):
        common = {
            "variant": variant,
            "data_lock_sha256": "a" * 64,
            "base_model": "Qwen/Qwen3-VL-8B-Instruct",
            "base_revision": "rev",
            "prompt_sha256": "b" * 64,
            "generation_config_sha256": "c" * 64,
            "first_attempt_json_valid": True,
            "correction_triggered": False,
        }
        return [
            {**common, "sample_id": "price", "scenario": "product", "slices": ["known_visible_price"],
             "gold": {"business_category": "restaurant", "style_tags": [], "visible_facilities": [], "price_range": "budget", "unknown_fields": []},
             "prediction": {"business_category": "restaurant", "style_tags": [], "visible_facilities": [], "price_range": "budget"}},
            {**common, "sample_id": "multi", "scenario": "product", "slices": ["multi_subject_conflict"],
             "gold": {"business_category": "other", "style_tags": [], "visible_facilities": [], "price_range": "unknown", "unknown_fields": ["style_tags", "visible_facilities", "price_range"]},
             "prediction": {"business_category": "other", "style_tags": [], "visible_facilities": [], "price_range": "unknown"}},
            {**common, "sample_id": "dialogue", "scenario": "dialogue", "slices": ["dialogue_state"],
             "gold": {}, "prediction": {}, "context_recall": True, "state_value_correct": True,
             "task_key_correct": True, "value_correct": True, "first_turn_routing_correct": True},
        ]

    def test_vlm_v3_reports_slice_denominators(self):
        rows = self._rows("zero_shot") + self._rows("old_unified_adapter")
        report = score_vlm_v3_comparison(rows)
        metrics = report["variants"]["zero_shot"]["slice_metrics"]
        self.assertEqual(metrics["known_visible_price"]["support"], 1)
        self.assertEqual(metrics["known_visible_price"]["exact_accuracy"], 1.0)
        self.assertEqual(
            metrics["multi_subject_conflict"]["all_declared_unknown_fields_abstained_accuracy"], 1.0
        )

    def test_performance_matrix_retains_component_scope(self):
        rows = []
        for role in ("old", "current"):
            for phase, count in (("cold", 1), ("steady", 2)):
                for _ in range(count):
                    rows.append({
                        "role": role,
                        "profile_id": "short",
                        "concurrency": 1,
                        "transport": "component_milvus_lite",
                        "phase": phase,
                        "status": "MEASURED",
                        "clip_encode_ms": 1.0,
                        "milvus_ms": 1.0,
                        "rerank_ms": 1.0,
                        "vlm_ms": 10.0,
                        "end_to_end_ms": 14.0,
                        "peak_vram_mib": 100.0,
                        "failed": False,
                        "hardware": "gpu",
                    })
        report = summarize_performance_matrix(rows, {
            "min_cold_repetitions": 1,
            "min_steady_repetitions": 2,
            "max_failure_rate": 0,
            "max_peak_vram_mib": 200,
            "profiles": {"short": {"max_steady_end_to_end_p95_ms": 20}},
        })
        self.assertEqual(report["measured_gate_status"], "PASS")
        self.assertFalse("production_sla" in report["scope"] and report["overall_status"] == "PASS")


if __name__ == "__main__":
    unittest.main()

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.relevance_evidence import (
    canonical_json_sha256,
    compare_performance,
    score_ann_fidelity,
    score_search_results,
    score_vlm_comparison,
    summarize_performance,
    validate_annotation_protocol,
    validate_asset_source_registry,
    validate_query_manifest,
)


def _query(query_id="q1", slices=None):
    source = {
        "source_id": "commons:test",
        "page_url": "https://commons.wikimedia.org/wiki/File:Test.jpg",
        "download_url": "https://upload.wikimedia.org/test.jpg",
        "license": "CC0",
        "author": "Example",
        "dataset_relation": "independent_public_source_not_yelp",
    }
    record = {
        "query_id": query_id,
        "slices": slices or [
            "image_similar",
            "city_business_facility_price",
            "visual_similar_business_irrelevant",
            "no_result",
            "filter_conflict",
        ],
        "image": {"relative_path": "test.jpg", "sha256": "0" * 64},
        "source": source,
        "source_record_sha256": canonical_json_sha256(source),
        "requested_filters": {"city": "Tampa", "business_category": "hotel"},
        "unsupported_constraints": ["facility"],
    }
    record["query_sha256"] = canonical_json_sha256(record)
    return record


def _annotation(query_id="q1", provenance="weak_programmatic_metadata"):
    return {
        "query_id": query_id,
        "label_provenance": provenance,
        "annotators": ["programmatic_metadata_rule_v1"],
        "conflict_resolution": "not_applicable_single_programmatic",
        "grade_rules": {
            "target_business_category": "hotel",
            "required_metadata": {"city": "Tampa"},
            "excluded_business_categories": ["restaurant"],
            "grades": {"3": "target and filters", "2": "target", "1": "other", "0": "excluded"},
        },
    }


class QueryManifestTests(unittest.TestCase):
    def test_manifest_validates_asset_hash_and_declares_collision_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "test.jpg"
            asset.write_bytes(b"independent")
            record = _query()
            record["image"]["sha256"] = hashlib.sha256(b"independent").hexdigest()
            record["query_sha256"] = canonical_json_sha256(
                {key: value for key, value in record.items() if key != "query_sha256"}
            )
            result = validate_query_manifest([record], Path(directory))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["byte_level_index_query_collision_check"],
            "NOT_RUN_MISSING_INDEX_IMAGE_SHA",
        )
        self.assertFalse(result["promotion_eligible_as_human_ground_truth"])

    def test_manifest_fails_on_query_hash_mismatch(self):
        record = _query()
        record["requested_filters"]["city"] = "Nashville"
        with self.assertRaisesRegex(ValueError, "query_sha256 mismatch"):
            validate_query_manifest([record])

    def test_weak_labels_cannot_claim_human_annotator(self):
        annotation = _annotation()
        annotation["annotators"] = ["Alice"]
        with self.assertRaisesRegex(ValueError, "cannot name a human"):
            validate_annotation_protocol([_query()], [annotation])

    def test_human_labels_require_two_annotators(self):
        annotation = _annotation(provenance="human")
        annotation["annotators"] = ["Alice"]
        annotation["conflict_resolution"] = "none"
        with self.assertRaisesRegex(ValueError, "two annotators"):
            validate_annotation_protocol([_query()], [annotation])

    def test_asset_registry_must_match_query_bytes_and_identity(self):
        query = _query()
        registry = [{
            "source_id": query["source"]["source_id"],
            "relative_path": "test.jpg",
            "exact_asset_url": "https://upload.wikimedia.org/test.jpg",
            "sha256": "0" * 64,
            "width": 1,
            "height": 1,
        }]
        query["image"].update({"width": 1, "height": 1})
        report = validate_asset_source_registry([query], registry)
        self.assertEqual(report["status"], "PASS")
        broken = copy.deepcopy(registry)
        broken[0]["sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            validate_asset_source_registry([query], broken)


class SearchScoringTests(unittest.TestCase):
    def test_ann_recall_is_labeled_as_fidelity_only(self):
        report = score_ann_fidelity(
            [{"exact_ids": ["a", "b"], "ann_ids": ["a", "x"]}], top_k=2
        )
        self.assertEqual(report["value"], 0.5)
        self.assertEqual(report["scope"], "ann_vs_exact_only")
        self.assertFalse(report["business_semantic_relevance_supported"])

    def test_search_reports_support_and_filter_correctness(self):
        query = _query(slices=["image_similar"])
        annotation = _annotation()
        hit = {
            "image_id": "i1",
            "business_category": "hotel",
            "city": "Tampa",
            "price_range": "mid_range",
        }
        methods = {
            name: {
                "hits": [hit],
                "relevant_total": 1,
                "no_result": False,
                "unsupported_constraints_unapplied": ["facility"],
            }
            for name in ("clip_exact", "clip_milvus", "structured_filter_clip", "lightweight_rerank")
        }
        report = score_search_results(
            [query], [annotation], [{"query_id": "q1", "methods": methods}]
        )
        exact = report["methods"]["clip_exact"]
        self.assertEqual(exact["ranking_support"], 1)
        self.assertEqual(exact["ndcg_at_10"], 1.0)
        self.assertEqual(exact["filter_correctness"], 1.0)

    def test_no_result_has_no_ranking_denominator(self):
        query = _query(slices=["no_result"])
        methods = {
            name: {
                "hits": [],
                "relevant_total": 0,
                "no_result": True,
                "unsupported_constraints_unapplied": ["facility"],
            }
            for name in ("clip_exact", "clip_milvus", "structured_filter_clip", "lightweight_rerank")
        }
        report = score_search_results(
            [query], [_annotation()], [{"query_id": "q1", "methods": methods}]
        )
        self.assertEqual(report["methods"]["clip_exact"]["ranking_support"], 0)
        self.assertEqual(report["methods"]["clip_exact"]["no_result_accuracy"], 1.0)


class VlmAndPerformanceTests(unittest.TestCase):
    def _vlm_row(self, variant):
        return {
            "variant": variant,
            "sample_id": "p1",
            "scenario": "product",
            "data_lock_sha256": "a" * 64,
            "base_model": "Qwen/Qwen3-VL-8B-Instruct",
            "base_revision": "revision-1",
            "prompt_sha256": "b" * 64,
            "generation_config_sha256": "c" * 64,
            "gold": {
                "business_category": "hotel",
                "style_tags": ["modern"],
                "visible_facilities": [],
                "price_range": "unknown",
                "unknown_fields": ["price_range"],
            },
            "prediction": {
                "business_category": "hotel",
                "style_tags": ["modern"],
                "visible_facilities": [],
                "price_range": "luxury",
            },
            "first_attempt_json_valid": False,
            "correction_triggered": True,
        }

    def test_vlm_keeps_first_attempt_and_correction_separate(self):
        report = score_vlm_comparison(
            [self._vlm_row("zero_shot"), self._vlm_row("current_checkpoint_87")]
        )
        current = report["variants"]["current_checkpoint_87"]
        self.assertEqual(current["first_attempt_json_compliance"], 0.0)
        self.assertEqual(current["correction_trigger_rate"], 1.0)
        self.assertGreater(current["unsupported_hallucination_rate"], 0.0)
        self.assertEqual(
            current["field_metrics"]["price_range"]["status"],
            "NOT_APPLICABLE_NO_SUPPORTED_REFERENCE",
        )

    def test_vlm_fails_when_prompt_changes(self):
        rows = [self._vlm_row("zero_shot"), self._vlm_row("current_checkpoint_87")]
        rows[1]["prompt_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "more than the adapter factor"):
            score_vlm_comparison(rows)

    def test_performance_gate_requires_all_stages_and_support(self):
        rows = []
        for phase, count in (("cold", 1), ("steady", 3)):
            for _ in range(count):
                rows.append(
                    {
                        "phase": phase,
                        "clip_encode_ms": 10.0,
                        "milvus_ms": 2.4,
                        "rerank_ms": 1.0,
                        "vlm_ms": 100.0,
                        "end_to_end_ms": 120.0,
                        "peak_vram_mib": 7000,
                        "failed": False,
                        "hardware": "test-gpu",
                    }
                )
        report = summarize_performance(
            rows,
            {
                "max_steady_end_to_end_p95_ms": 150,
                "max_failure_rate": 0,
                "max_peak_vram_mib": 8000,
                "min_steady_repetitions": 3,
                "min_cold_repetitions": 1,
            },
        )
        self.assertEqual(report["fixed_gates"]["status"], "PASS")
        self.assertEqual(report["stages"]["steady"]["milvus_ms"]["p95"], 2.4)

    def test_candidate_comparison_fails_closed_on_hardware_change(self):
        rows = []
        for phase, count in (("cold", 1), ("steady", 3)):
            for _ in range(count):
                rows.append({
                    "phase": phase,
                    "clip_encode_ms": 10.0,
                    "milvus_ms": 2.0,
                    "rerank_ms": 1.0,
                    "vlm_ms": 100.0,
                    "end_to_end_ms": 120.0,
                    "peak_vram_mib": 7000,
                    "failed": False,
                    "hardware": "gpu-a",
                })
        baseline = copy.deepcopy(rows)
        baseline[0]["hardware"] = "gpu-b"
        for row in baseline[1:]:
            row["hardware"] = "gpu-b"
        gates = {
            "max_steady_end_to_end_p95_ms": 150,
            "max_failure_rate": 0,
            "max_peak_vram_mib": 8000,
            "max_candidate_to_baseline_p95_ratio": 1.25,
            "min_steady_repetitions": 3,
            "min_cold_repetitions": 1,
        }
        report = compare_performance(rows, baseline, gates)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["checks"]["same_hardware"])


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_exploration_pool_v4 import build_pool
from scripts.run_search_relevance_v4 import load_final_after_marker as load_search_final
from scripts.run_vlm_semantic_evidence_v4 import load_final_after_marker as load_vlm_final
from src.evaluation.exploration_v4 import (
    apply_search_v4_gates,
    validate_three_way_isolation,
)
from src.evaluation.relevance_evidence import load_jsonl


ROOT = Path(__file__).resolve().parents[1]


class ExplorationV4Test(unittest.TestCase):
    def test_generator_matches_committed_lock_and_splits_are_isolated(self) -> None:
        expected = json.loads((
            ROOT / "configs/evaluation/evidence_enhancement/exploration_pool_lock_v4.json"
        ).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pool"
            self.assertEqual(build_pool(output), expected)
            search = {
                split: load_jsonl(output / f"search_{split}_manifest.jsonl")
                for split in ("training", "development", "final")
            }
            vlm = {
                split: load_jsonl(output / f"vlm_{split}_manifest.jsonl")
                for split in ("training", "development", "final")
            }
            self.assertEqual(validate_three_way_isolation(search, record_kind="search")["status"], "PASS")
            self.assertEqual(validate_three_way_isolation(vlm, record_kind="vlm")["status"], "PASS")

    def test_three_way_isolation_fails_closed_on_image_overlap(self) -> None:
        splits = {
            name: [{"sample_id": name, "source_id": name, "image_sha256": name}]
            for name in ("training", "development", "final")
        }
        splits["final"][0]["image_sha256"] = "development"
        with self.assertRaisesRegex(ValueError, "identity leakage"):
            validate_three_way_isolation(splits, record_kind="vlm")

    def test_search_gate_uses_no_result_slice_not_aggregate(self) -> None:
        candidate = {
            "support": 24,
            "ranking_support": 12,
            "failure_rate": 0.0,
            "slices": {
                "no_result": {"support": 12, "no_result_accuracy": 0.75},
                "hard_filter_before_rerank": {
                    "support": 16,
                    "filter_correctness": 1.0,
                    "ndcg_at_10": 0.9,
                },
            },
        }
        gates = {
            "min_query_support": 24,
            "max_failure_rate": 0.0,
            "min_no_result_accuracy": 0.7,
            "min_filter_correctness": 0.95,
            "min_ndcg_at_10": 0.3,
            "min_ann_recall_at_10": 0.95,
        }
        report = {"methods": {"hard_filter_clip_business_guard": candidate}}
        result = apply_search_v4_gates(report, {"value": 1.0}, gates)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["denominators"]["no_result_support"], 12)

    def test_final_markers_are_written_before_loaders_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seen = []

            def search_loader(path: Path):
                seen.append((path.name, (root / "search-marker.json").is_file()))
                return []

            load_search_final(
                root, root / "search-marker.json", {"selection": "locked"}, loader=search_loader
            )
            self.assertEqual(seen, [
                ("search_final_manifest.jsonl", True),
                ("search_final_annotations.jsonl", True),
            ])

            def vlm_loader(path: Path):
                self.assertTrue((root / "vlm-marker.json").is_file())
                return []

            load_vlm_final(
                root / "vlm_final_manifest.jsonl",
                root / "vlm-marker.json",
                {"selection": "locked"},
                loader=vlm_loader,
            )


if __name__ == "__main__":
    unittest.main()

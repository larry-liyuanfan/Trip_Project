import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import build_release_bundle, tripctl
from scripts.release_manifest import ReleaseVerificationError
from scripts.verify_final_delivery import validate_release_lineage
from src.inference.release_config import DEFAULT_RELEASE_CONFIG
from src.inference.system_runtime import DEFAULT_RELEASE_CONFIG as RUNTIME_DEFAULT_RELEASE_CONFIG


ROOT = Path(__file__).resolve().parents[1]


class FinalDeliveryTests(unittest.TestCase):
    def test_formal_v1_package_is_verify_only(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "formal v1 is immutable"):
                build_release_bundle.build_bundle(
                    root / "release",
                    adapter_dir=root / "adapter",
                    retrieval_dir=root / "retrieval",
                    evidence_paths=[],
                )

    def load(self, name):
        return json.loads((ROOT / "configs/releases" / name).read_text(encoding="utf-8"))

    def test_final_release_composes_v12_product_and_v13_itinerary(self):
        final = self.load("qwen3_vl_system_final_v1.json")
        v12 = self.load("qwen3_vl_system_week8_v12.json")
        v13 = self.load("qwen3_vl_system_week8_v13.json")
        acceptance = {
            "status": "PASS",
            "candidate_quality_accepted": True,
            "release_id": v12["release_id"],
            "human_annotation_count": 0,
            "human_visual_accuracy_claim": False,
            "label_source": "model_generated_silver",
            "formal_release_replaced": False,
        }
        itinerary = {
            "status": "PASS",
            "selected_release": v13["release_id"],
            "release_change_scope": ["release_id", "prompts.itinerary_planning"],
            "direct_itinerary_nonregression": True,
            "dialogue_first_attempt_improved": True,
            "final_test_rows_read": False,
        }
        validate_release_lineage(final, v12, v13, acceptance, itinerary)
        invalid = copy.deepcopy(final)
        invalid["quality"]["human_visual_accuracy_claim"] = True
        with self.assertRaises(ReleaseVerificationError):
            validate_release_lineage(invalid, v12, v13, acceptance, itinerary)

    def test_all_runtime_defaults_select_final_release(self):
        expected = "configs/releases/qwen3_vl_system_final_v1.json"
        self.assertEqual(DEFAULT_RELEASE_CONFIG, expected)
        self.assertEqual(RUNTIME_DEFAULT_RELEASE_CONFIG, expected)
        self.assertEqual(tripctl.DEFAULT_RELEASE, ROOT / expected)
        self.assertEqual(build_release_bundle.DEFAULT_RELEASE_CONFIG, ROOT / expected)

    def test_runtime_bundle_includes_only_selected_release_config(self):
        self.assertNotIn("configs/releases", build_release_bundle.RUNTIME_PATHS)
        final = self.load("qwen3_vl_system_final_v1.json")
        self.assertIn(
            "configs/week8/product_observation_subject_review_v2.json",
            build_release_bundle.runtime_paths(final),
        )


if __name__ == "__main__":
    unittest.main()

import copy
import json
import unittest
from pathlib import Path

from src.training.week8_product_two_stage import (
    Week8TwoStageError,
    _hard_slice_row,
    caption_to_silver_evidence,
    load_two_stage_config,
    map_evidence_to_product,
    validate_observable_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/week8/product_two_stage_v1.json"


def evidence(**changes):
    value = {
        "subject_category": "hotel",
        "subject_clarity": "clear",
        "style_cues": ["modern"],
        "facility_cues": ["front_desk"],
        "price_text": [],
        "observable_facts": ["front desk is visible"],
        "uncertainty_reasons": ["no_price_evidence"],
    }
    value.update(changes)
    return value


class Week8ProductTwoStageTests(unittest.TestCase):
    @unittest.skipUnless(
        (ROOT / "scripts/spartan/week8_product_two_stage_development.sbatch").is_file(),
        "Spartan launch scripts are intentionally excluded from the local handoff",
    )
    def test_gpu_jobs_use_available_public_gpu_qos(self):
        for name in (
            "week8_product_two_stage_development.sbatch",
            "week8_product_two_stage_sft.sbatch",
        ):
            script = (ROOT / "scripts" / "spartan" / name).read_text(
                encoding="utf-8"
            )
            self.assertIn("#SBATCH --partition=gpu-l40s,gpu-a100", script)
            self.assertIn("#SBATCH --qos=publicgpu", script)
            self.assertIn("#SBATCH --gres=gpu:1", script)
        training = (
            ROOT / "scripts" / "spartan" / "week8_product_two_stage_sft.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --time=02:00:00", training)

    def test_config_forbids_all_human_states_and_final_test_access(self):
        config = load_two_stage_config(CONFIG)
        self.assertFalse(config["policy"]["human_annotation"])
        self.assertFalse(config["policy"]["human_review"])
        self.assertFalse(config["policy"]["human_acceptance"])
        self.assertEqual(config["policy"]["new_label_provenance"], "programmatic_silver")
        self.assertEqual(config["policy"]["final_test_access"], "forbidden")
        self.assertEqual(config["hard_slice_data"]["source_splits"], ["train", "development"])

    def test_evidence_contract_rejects_extra_keys_and_open_vocabulary(self):
        config = load_two_stage_config(CONFIG)
        invalid = evidence(explanation="hidden reasoning")
        with self.assertRaisesRegex(Week8TwoStageError, "keys changed"):
            validate_observable_evidence(invalid, config)
        invalid = evidence(style_cues=["palatial_guess"])
        with self.assertRaisesRegex(Week8TwoStageError, "style_cues"):
            validate_observable_evidence(invalid, config)

    def test_mapping_calibrates_ambiguous_and_numeric_price_to_unknown(self):
        config = load_two_stage_config(CONFIG)
        mapped = map_evidence_to_product(
            evidence(
                subject_clarity="multiple",
                price_text=["$25"],
                uncertainty_reasons=["multiple_subjects"],
            ),
            config,
        )
        self.assertEqual(mapped["business_category"], "unknown")
        self.assertEqual(mapped["style_tags"], [])
        self.assertEqual(mapped["visible_facilities"], [])
        self.assertEqual(mapped["price_range"], "unknown")
        self.assertIn("business_category", mapped["unknown_fields"])
        self.assertEqual(mapped["inferred_attributes"], [])

    def test_mapping_accepts_only_an_explicit_visible_tier_word(self):
        config = load_two_stage_config(CONFIG)
        mapped = map_evidence_to_product(
            evidence(price_text=["Premium menu"]), config
        )
        self.assertEqual(mapped["price_range"], "premium")
        self.assertNotIn("price_range", mapped["unknown_fields"])

    def test_caption_proxy_is_explicitly_lexical_and_conservative(self):
        proxy = caption_to_silver_evidence(
            "Several customers in a cozy modern hotel lobby with reception and pool",
            protocol="legacy_caption_v1",
        )
        self.assertEqual(proxy["subject_category"], "hotel")
        self.assertEqual(proxy["subject_clarity"], "multiple")
        self.assertEqual(proxy["style_cues"], ["cozy", "modern"])
        self.assertIn("front_desk", proxy["facility_cues"])
        self.assertIn("multiple_subjects", proxy["uncertainty_reasons"])

    def test_hard_slice_row_never_promotes_silver_to_human(self):
        config = load_two_stage_config(CONFIG)
        source = {
            "sample_id": "source-1",
            "split": "train",
            "scenario": "image_product_search",
            "source_id": "yelp-photo:p1",
            "group_id": "yelp-business:b1",
            "image_sha256": "a" * 64,
            "image_path": "outputs/images/a.jpg",
            "target": {"business_category": "hotel"},
            "label_source": "programmatic_silver",
            "sample_weight": 0.5,
        }
        row = _hard_slice_row(source, "unclear room photo", config)
        self.assertEqual(row["label_source"], "programmatic_silver")
        self.assertLessEqual(row["sample_weight"], 0.5)
        self.assertFalse(row["target_provenance"]["human_annotation"])
        self.assertFalse(row["target_provenance"]["human_review"])
        self.assertFalse(row["target_provenance"]["human_acceptance"])
        self.assertIn("unknown_price_negative", row["hard_slices"])
        self.assertEqual(row["source_sample_id"], "source-1")

    def test_config_rejects_any_future_human_review_toggle(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["policy"]["human_review"] = True
        temporary = ROOT / "tests" / ".tmp_two_stage_invalid.json"
        try:
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(Week8TwoStageError, "automatic silver"):
                load_two_stage_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

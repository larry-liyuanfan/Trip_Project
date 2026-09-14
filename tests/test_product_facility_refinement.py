import copy
import json
from pathlib import Path
import unittest

from src.evaluation.week8_visual_silver import replay_record
from src.inference.product_facility_refinement import (
    apply_facility_review, facility_review_messages, facility_review_source_hashes,
    validate_facility_refinement, validate_facility_review_identity,
)
from src.inference.product_observation import (
    canonical_config_sha256, generate_observation, load_observation_config, map_observation,
)
from src.inference.system_runtime import ModelGenerationError
from tests.test_product_style_refinement import Backend


ROOT = Path(__file__).resolve().parents[1]


class ProductFacilityRefinementTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "configs/week8/product_observation_facility_review_v1.json"
        self.config = json.loads(self.path.read_text(encoding="utf-8"))
        self.primary = {
            "subject_kind": "dining_space", "subject_fact": "Dining room",
            "style_evidence": [{"label": "modern", "fact": "Geometric pendant lights"}],
            "facility_evidence": [{"label": "parking", "fact": "Visible marked spaces"}],
            "price_text": [],
        }
        self.review = {"facility_evidence": [
            {"label": "seating", "fact": "Wooden chairs"},
            {"label": "dining_tables", "fact": "Dining tables"},
        ]}

    @staticmethod
    def record(generated):
        return {"passed": generated["passed"], "result": generated["result"],
                "attempts": [attempt.model_dump() for attempt in generated["attempts"]]}

    def test_config_extends_incumbent_only_with_facility_stage(self):
        incumbent = json.loads((ROOT / "configs/week8/product_observation_subject_review_v2.json").read_text(encoding="utf-8"))
        self.assertEqual({key: value for key, value in self.config.items() if key != "facility_refinement"}, incumbent)
        self.assertEqual(load_observation_config(self.path, canonical_config_sha256(self.config)), self.config)

    def test_replace_changes_only_facilities_and_raw_replay_matches(self):
        original = copy.deepcopy(self.primary)
        generated = generate_observation(Backend([self.primary, self.review]), "image.jpg", self.config)
        self.assertTrue(generated["passed"])
        self.assertEqual(self.primary, original)
        self.assertTrue(generated["facility_refinement_applied"])
        self.assertEqual(generated["result"]["visible_facilities"], ["dining_tables", "seating"])
        base = map_observation(self.primary, self.config)
        for field in ("business_category", "style_tags", "price_range", "inferred_attributes"):
            self.assertEqual(generated["result"][field], base[field])
        self.assertEqual(replay_record(ROOT, self.record(generated), self.config), generated["result"])

    def test_prompt_is_independent_of_primary_and_reference_labels(self):
        text = json.dumps(facility_review_messages("image.jpg", self.config), ensure_ascii=False)
        for value in ("Visible marked spaces", "sample_id", "source_id", "target", "reference"):
            self.assertNotIn(value, text)

    def test_food_closeup_skips_review(self):
        self.primary.update(subject_kind="food_closeup", subject_fact="Bowl of soup",
                            style_evidence=[], facility_evidence=[])
        backend = Backend([self.primary])
        generated = generate_observation(backend, "image.jpg", self.config)
        self.assertTrue(generated["passed"])
        self.assertFalse(generated["facility_refinement_applied"])
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(replay_record(ROOT, self.record(generated), self.config), generated["result"])

    def test_extra_fields_duplicate_keys_negation_and_inference_fail(self):
        invalid = (
            '{"facility_evidence":[],"facility_evidence":[]}',
            json.dumps({**self.review, "style_evidence": []}),
            json.dumps({"facility_evidence": [{"label": "parking", "fact": "No parking"}]}),
            json.dumps({"facility_evidence": [{"label": "spa", "fact": "Lobby implies spa"}]}),
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                apply_facility_review(self.primary, raw, self.config)

    def test_failed_review_is_not_masked_and_retry_replays(self):
        generated = generate_observation(Backend([self.primary, "bad", self.review]), "image.jpg", self.config)
        self.assertTrue(generated["passed"])
        self.assertEqual(len(generated["attempts"]), 3)
        self.assertEqual(replay_record(ROOT, self.record(generated), self.config), generated["result"])
        failed = generate_observation(Backend([self.primary, "bad", "still bad"]), "image.jpg", self.config)
        self.assertFalse(failed["passed"])
        self.assertIsNone(failed["result"])

    def test_transport_failure_is_not_masked(self):
        failed = generate_observation(
            Backend([self.primary, ModelGenerationError("offline", attempts=[])]), "image.jpg", self.config)
        self.assertFalse(failed["passed"])
        self.assertEqual(len(failed["attempts"]), 2)

    def test_nested_style_subject_and_facility_stages_replay(self):
        primary = {**self.primary, "subject_kind": "retail_space", "subject_fact": "Pizza storefront sign",
                   "style_evidence": [{"label": "casual", "fact": "Casual cotton shirt"}]}
        style_review = {"style_evidence": []}
        subject_review = {"subject_kind": "dining_space", "subject_fact": "Restaurant entrance sign"}
        generated = generate_observation(
            Backend([primary, style_review, subject_review, self.review]), "image.jpg", self.config)
        self.assertTrue(generated["passed"])
        self.assertEqual(len(generated["attempts"]), 4)
        self.assertEqual(generated["result"]["business_category"], "restaurant")
        self.assertEqual(generated["result"]["style_tags"], [])
        self.assertEqual(replay_record(ROOT, self.record(generated), self.config), generated["result"])

    def test_replay_rejects_missing_extra_and_tampered_stage(self):
        generated = generate_observation(Backend([self.primary, self.review]), "image.jpg", self.config)
        record = self.record(generated)
        for attempts in (record["attempts"][:-1], record["attempts"] + [record["attempts"][-1]]):
            with self.subTest(count=len(attempts)), self.assertRaises(ValueError):
                replay_record(ROOT, {**record, "attempts": attempts}, self.config)
        record["result"]["style_tags"] = []
        with self.assertRaisesRegex(ValueError, "differs from raw"):
            replay_record(ROOT, record, self.config)

    def test_identity_binding_and_invalid_config(self):
        generation = json.loads((ROOT / "configs/week8/contract_ablation_v18.json").read_text(encoding="utf-8"))
        hashes = facility_review_source_hashes(ROOT, generation)
        self.assertIn("src/inference/product_facility_refinement.py", hashes)
        validate_facility_review_identity(ROOT, generation, {"facility_review_source_lf_sha256": hashes})
        with self.assertRaises(ValueError):
            validate_facility_review_identity(ROOT, generation, {})
        for change in ({"max_attempts": 3}, {"mode": "add_only"}, {"eligibility": "all"},
                       {"max_new_tokens": True}):
            invalid = copy.deepcopy(self.config)
            invalid["facility_refinement"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_facility_refinement(invalid)


if __name__ == "__main__":
    unittest.main()

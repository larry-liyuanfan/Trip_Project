import copy
import json
from pathlib import Path
import unittest

from src.evaluation.week8_visual_silver import replay_record
from src.inference.product_category_refinement import (
    apply_subject_review, subject_review_messages, validate_category_refinement,
    category_review_source_hashes, validate_category_review_identity,
)
from src.inference.product_observation import generate_observation, map_observation, observation_messages
from src.inference.system_runtime import ModelGenerationError
from tests.test_product_style_refinement import Backend


ROOT = Path(__file__).resolve().parents[1]


class ProductCategoryRefinementTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/week8/product_observation_subject_review_v1.json").read_text(encoding="utf-8"))
        self.primary = {"subject_kind": "retail_space", "subject_fact": "Pizza restaurant storefront",
                        "style_evidence": [{"label": "modern", "fact": "Geometric facade panels"}],
                        "facility_evidence": [{"label": "parking", "fact": "Visible marked spaces"}], "price_text": []}
        self.review = {"subject_kind": "dining_space", "subject_fact": "Restaurant entrance sign"}

    def record(self, generated):
        return {"passed": generated["passed"], "result": generated["result"],
                "attempts": [attempt.model_dump() for attempt in generated["attempts"]]}

    def test_first_observation_and_existing_style_correction_remain_unchanged(self):
        previous = json.loads((ROOT / "configs/week8/product_observation_food_retry_v1.json").read_text(encoding="utf-8"))
        self.assertEqual({key: value for key, value in self.config.items() if key != "category_refinement"}, previous)
        self.assertEqual(observation_messages("image.jpg", self.config), observation_messages("image.jpg", previous))

    def test_only_subject_fields_change_and_raw_replay_matches(self):
        original = copy.deepcopy(self.primary)
        generated = generate_observation(Backend([self.primary, self.review]), "image.jpg", self.config)
        self.assertTrue(generated["passed"])
        self.assertEqual(self.primary, original)
        self.assertEqual(generated["result"]["business_category"], "restaurant")
        for field in ("style_evidence", "facility_evidence", "price_text"):
            self.assertEqual(generated["observation"][field], self.primary[field])
        for field in ("style_tags", "visible_facilities", "price_range"):
            self.assertEqual(generated["result"][field], map_observation(self.primary, self.config)[field])
        self.assertEqual(replay_record(ROOT, self.record(generated), self.config), generated["result"])

    def test_subject_messages_do_not_receive_primary_labels_or_reference(self):
        messages = json.dumps(subject_review_messages("image.jpg", self.config))
        for value in ("Geometric facade panels", "Visible marked spaces", "sample_id", "source_id", "target"):
            self.assertNotIn(value, messages)

    def test_ineligible_scene_makes_no_extra_call(self):
        self.primary["subject_kind"] = "dining_space"
        backend = Backend([self.primary])
        generated = generate_observation(backend, "image.jpg", self.config)
        self.assertEqual(len(backend.calls), 1)
        self.assertFalse(generated["category_refinement_applied"])
        self.assertEqual(replay_record(ROOT, self.record(generated), self.config), generated["result"])

    def test_subject_unknown_preserves_visible_field_evidence(self):
        self.review.update(subject_kind="unidentified_space", subject_fact="Unidentified building entrance")
        generated = generate_observation(Backend([self.primary, self.review]), "image.jpg", self.config)
        self.assertEqual(generated["result"]["business_category"], "unknown")
        self.assertEqual(generated["result"]["visible_facilities"], ["parking"])
        self.assertIn("business_category", generated["result"]["unknown_fields"])

    def test_new_food_subject_cannot_silently_delete_original_facilities(self):
        self.review.update(subject_kind="food_closeup", subject_fact="Bowl of noodles")
        generated = generate_observation(Backend([self.primary, self.review, self.review]), "image.jpg", self.config)
        self.assertFalse(generated["passed"])
        self.assertIsNone(generated["result"])
        self.assertEqual(len(generated["attempts"]), 3)
        self.assertIn("food closeup", generated["attempts"][-1].error)

    def test_duplicate_or_extra_fields_and_inferred_facts_are_rejected(self):
        for raw in ('{"subject_kind":"dining_space","subject_kind":"retail_space","subject_fact":"Sign"}',
                    json.dumps({**self.review, "facility_evidence": []}),
                    json.dumps({**self.review, "subject_fact": "Likely a restaurant"})):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                apply_subject_review(self.primary, raw, self.config)

    def test_correction_retains_invalid_raw_and_all_tokens(self):
        backend = Backend([self.primary, "invalid subject", self.review])
        generated = generate_observation(backend, "image.jpg", self.config)
        self.assertTrue(generated["passed"])
        self.assertEqual([attempt.attempt for attempt in generated["attempts"]], [1, 2, 3])
        self.assertEqual(sum(attempt.output_tokens for attempt in generated["attempts"]), 60)
        self.assertEqual(backend.calls[-1][-2]["content"], "invalid subject")
        self.assertEqual(replay_record(ROOT, self.record(generated), self.config), generated["result"])

    def test_style_then_subject_review_replays_without_reported_stage_markers(self):
        self.primary["style_evidence"] = [{"label": "casual", "fact": "Casual shirts and jeans"}]
        backend = Backend([self.primary, {"style_evidence": []}, self.review])
        generated = generate_observation(backend, "image.jpg", self.config)
        self.assertTrue(generated["passed"])
        self.assertEqual(generated["result"]["style_tags"], [])
        self.assertEqual(len(generated["attempts"]), 3)
        self.assertEqual(replay_record(ROOT, self.record(generated), self.config), generated["result"])

    def test_failed_primary_or_transport_is_not_masked(self):
        failed = generate_observation(Backend(["invalid", "invalid"]), "image.jpg", self.config)
        self.assertFalse(failed["passed"])
        self.assertEqual(len(failed["attempts"]), 2)
        failed = generate_observation(Backend([self.primary, ModelGenerationError("offline", attempts=[])]), "image.jpg", self.config)
        self.assertFalse(failed["passed"])
        self.assertEqual(len(failed["attempts"]), 2)

    def test_replay_rejects_missing_extra_or_changed_subject_stage(self):
        generated = generate_observation(Backend([self.primary, self.review]), "image.jpg", self.config)
        record = self.record(generated)
        for attempts in (record["attempts"][:-1], record["attempts"] + [record["attempts"][-1]]):
            with self.subTest(count=len(attempts)), self.assertRaises(ValueError):
                replay_record(ROOT, {**record, "attempts": attempts}, self.config)
        record["result"]["business_category"] = "hotel"
        with self.assertRaises(ValueError):
            replay_record(ROOT, record, self.config)

    def test_config_rejects_unbounded_attempts_or_unknown_eligibility(self):
        for change in ({"max_attempts": 3}, {"max_new_tokens": True}, {"eligible_subject_kinds": ["unknown_kind"]},
                       {"eligible_subject_kinds": []}, {"eligible_subject_kinds": ["retail_space", "retail_space"]}):
            invalid = copy.deepcopy(self.config)
            invalid["category_refinement"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_category_refinement(invalid)

    def test_new_generation_requires_exact_review_source_identity(self):
        generation = json.loads((ROOT / "configs/week8/contract_ablation_v15.json").read_text(encoding="utf-8"))
        hashes = category_review_source_hashes(ROOT, generation)
        self.assertIn("src/inference/product_category_refinement.py", hashes)
        validate_category_review_identity(ROOT, generation, {"category_review_source_lf_sha256": hashes})
        for value in ({}, {**hashes, "src/inference/product_category_refinement.py": "0" * 64}):
            with self.assertRaises(ValueError):
                validate_category_review_identity(ROOT, generation, {"category_review_source_lf_sha256": value})

    def test_existing_generation_does_not_invent_new_source_binding(self):
        generation = json.loads((ROOT / "configs/week8/contract_ablation_v14.json").read_text(encoding="utf-8"))
        self.assertEqual(category_review_source_hashes(ROOT, generation), {})
        validate_category_review_identity(ROOT, generation, {})

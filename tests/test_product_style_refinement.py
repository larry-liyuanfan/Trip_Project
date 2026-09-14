import copy
import json
from pathlib import Path
import unittest

from src.evaluation.week8_visual_silver import replay_record
from src.inference.product_observation import (
    canonical_config_sha256, generate_observation, load_observation_config, map_observation, observation_messages,
)
from src.inference.product_style_refinement import apply_refinement, refinement_messages
from src.inference.system_runtime import GenerationResult, ModelGenerationError


ROOT = Path(__file__).resolve().parents[1]


class Backend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate_with_usage(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return GenerationResult(content=value if isinstance(value, str) else json.dumps(value),
                                input_tokens=10, output_tokens=20)


class ProductStyleRefinementTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/week8/product_observation_v7.json").read_text(encoding="utf-8"))
        self.primary = {"subject_kind": "dining_space", "subject_fact": "Tables and chairs",
                        "style_evidence": [{"label": "modern", "fact": "Geometric pendant lights"}],
                        "facility_evidence": [{"label": "seating", "fact": "Visible chairs"}], "price_text": []}
        self.refined = {"style_evidence": [{"label": "industrial", "fact": "Exposed metal pipes"}]}

    def record(self, result):
        return {"passed": result["passed"], "result": result["result"],
                "attempts": [attempt.model_dump() for attempt in result["attempts"]]}

    def test_first_stage_prompt_is_byte_equal_to_v9(self):
        old = json.loads((ROOT / "configs/week8/product_observation_v3.json").read_text(encoding="utf-8"))
        self.assertEqual(observation_messages("fixture.jpg", old), observation_messages("fixture.jpg", self.config))

    def test_replace_changes_only_style_and_derived_evidence_unknowns(self):
        before = copy.deepcopy(self.primary)
        backend = Backend([self.primary, self.refined])
        result = generate_observation(backend, "fixture.jpg", self.config)
        self.assertTrue(result["passed"])
        self.assertEqual(self.primary, before)
        self.assertEqual(result["result"]["style_tags"], ["industrial"])
        base = map_observation(self.primary, self.config)
        for field in ("business_category", "visible_facilities", "price_range", "inferred_attributes"):
            self.assertEqual(result["result"][field], base[field])
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(sum(attempt.output_tokens for attempt in result["attempts"]), 40)
        self.assertEqual(replay_record(ROOT, self.record(result), self.config), result["result"])

    def test_replace_prompt_is_independent_of_primary_labels(self):
        messages = refinement_messages("fixture.jpg", self.primary, self.config)
        self.assertNotIn("Geometric pendant lights", json.dumps(messages))
        self.assertNotIn("seating", messages[-1]["content"][-1]["text"])

    def test_add_only_preserves_original_positive_labels_and_facts(self):
        self.config["style_refinement"].update(mode="add_only", eligibility="nonempty_style")
        result = generate_observation(Backend([self.primary, self.refined]), "fixture.jpg", self.config)
        self.assertEqual(result["result"]["style_tags"], ["industrial", "modern"])
        self.assertEqual(result["observation"]["style_evidence"][0], self.primary["style_evidence"][0])
        self.assertEqual(replay_record(ROOT, self.record(result), self.config), result["result"])

    def test_add_only_validates_repeated_existing_label_before_ignoring_it(self):
        self.config["style_refinement"]["mode"] = "add_only"
        duplicate = {"style_evidence": [self.primary["style_evidence"][0]] * 2}
        with self.assertRaisesRegex(ValueError, "duplicate labels"):
            apply_refinement(self.primary, json.dumps(duplicate), self.config)

    def test_food_closeup_skips_refinement_without_fabricated_attempt(self):
        self.primary.update(subject_kind="food_closeup", style_evidence=[], facility_evidence=[])
        result = generate_observation(Backend([self.primary]), "fixture.jpg", self.config)
        self.assertTrue(result["passed"])
        self.assertFalse(result["refinement_applied"])
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(replay_record(ROOT, self.record(result), self.config), result["result"])

    def test_nonempty_policy_skips_unknown_style(self):
        self.primary["style_evidence"] = []
        self.config["style_refinement"]["eligibility"] = "nonempty_style"
        result = generate_observation(Backend([self.primary]), "fixture.jpg", self.config)
        self.assertFalse(result["refinement_applied"])
        self.assertEqual(result["result"]["style_tags"], [])

    def test_empty_review_is_valid_unknown_without_removing_other_fields(self):
        result = generate_observation(Backend([self.primary, {"style_evidence": []}]), "fixture.jpg", self.config)
        self.assertEqual(result["result"]["style_tags"], [])
        self.assertIn("style_tags", result["result"]["unknown_fields"])
        self.assertEqual(result["result"]["visible_facilities"], ["seating"])

    def test_bounded_retries_keep_all_raw_outputs_and_tokens(self):
        backend = Backend(["bad primary", self.primary, "bad review", self.refined])
        result = generate_observation(backend, "fixture.jpg", self.config)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["attempts"]), 4)
        self.assertEqual([a.attempt for a in result["attempts"]], [1, 2, 3, 4])
        self.assertEqual(sum(a.output_tokens for a in result["attempts"]), 80)
        self.assertEqual(backend.calls[-1][-2]["content"], "bad review")
        self.assertEqual(replay_record(ROOT, self.record(result), self.config), result["result"])

    def test_refinement_failure_is_not_masked_by_valid_primary(self):
        result = generate_observation(Backend([self.primary, "bad", "still bad"]), "fixture.jpg", self.config)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["result"])
        self.assertEqual(len(result["attempts"]), 3)

    def test_backend_failure_preserves_completed_primary_usage(self):
        result = generate_observation(Backend([self.primary, ModelGenerationError("backend unavailable")]), "fixture.jpg", self.config)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["attempts"][0].output_tokens, 20)
        self.assertIn("backend unavailable", result["attempts"][1].error)

    def test_invalid_primary_never_calls_style_stage(self):
        backend = Backend(["bad", "bad"])
        result = generate_observation(backend, "fixture.jpg", self.config)
        self.assertFalse(result["passed"])
        self.assertFalse(result["refinement_applied"])
        self.assertEqual(len(backend.calls), 2)

    def test_review_cannot_overwrite_category_or_facilities(self):
        with self.assertRaises(ValueError):
            apply_refinement(self.primary, json.dumps({**self.refined, "business_category": "hotel"}), self.config)

    def test_duplicate_json_keys_and_inferred_facts_are_rejected(self):
        for raw in ('{"style_evidence":[],"style_evidence":[]}',
                    json.dumps({"style_evidence": [{"label": "cozy", "fact": "Dining suggests comfort"}]})):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                apply_refinement(self.primary, raw, self.config)

    def test_fact_bound_does_not_silently_truncate_supported_labels(self):
        self.config["style_refinement"]["mode"] = "add_only"
        proposed = {"style_evidence": [{"label": label, "fact": f"Distinct visual material {i}"}
                                      for i, label in enumerate(self.config["style_vocabulary"][:11])]}
        with self.assertRaisesRegex(ValueError, "ten distinct facts"):
            apply_refinement(self.primary, json.dumps(proposed), self.config)

    def test_replay_requires_complete_stages_and_rejects_tampered_public_fields(self):
        result = generate_observation(Backend([self.primary, self.refined]), "fixture.jpg", self.config)
        record = self.record(result)
        record["attempts"] = record["attempts"][:1]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            replay_record(ROOT, record, self.config)
        record = self.record(result)
        record["result"]["visible_facilities"] = ["parking"]
        with self.assertRaisesRegex(ValueError, "differs from raw"):
            replay_record(ROOT, record, self.config)

    def test_new_configs_are_identity_bound_and_bounded(self):
        for version in (7, 8):
            path = ROOT / f"configs/week8/product_observation_v{version}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(load_observation_config(path, canonical_config_sha256(value)), value)
        from src.inference.product_style_refinement import validate_refinement_config
        self.config["style_refinement"]["max_attempts"] = 3
        with self.assertRaises(ValueError):
            validate_refinement_config(self.config)


if __name__ == "__main__":
    unittest.main()

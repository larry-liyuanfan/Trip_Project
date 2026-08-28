import copy
import json
from pathlib import Path
import unittest

from src.evaluation.week8_visual_silver import replay_record
from src.inference.product_observation import (
    canonical_config_sha256, generate_observation, load_observation_config, map_observation, observation_messages,
)
from src.inference.product_style_scope import venue_style_evidence, validate_style_scope
from tests.test_product_style_refinement import Backend


ROOT = Path(__file__).resolve().parents[1]


class VenueStyleScopeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/week8/product_observation_scope_v1.json").read_text(encoding="utf-8"))
        self.observation = {"subject_kind": "dining_space", "subject_fact": "Visible tables and chairs",
            "style_evidence": [{"label": "casual", "fact": "People in cotton shirts"},
                               {"label": "modern", "fact": "Geometric pendant lighting"}],
            "facility_evidence": [{"label": "seating", "fact": "Visible chairs"}], "price_text": []}

    def test_people_style_does_not_become_venue_style(self):
        original = copy.deepcopy(self.observation)
        product = map_observation(self.observation, self.config)
        self.assertEqual(product["style_tags"], ["modern"])
        self.assertEqual(self.observation, original)
        self.assertIn("People in cotton shirts", product["observed_evidence"])
        self.assertEqual(product["visible_facilities"], ["seating"])
        kept, rejected = venue_style_evidence(self.observation, self.config)
        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected[0]["label"], "casual")
        self.assertEqual(rejected[0]["reason"], "nonvenue_object_fact")

    def test_drink_object_alone_cannot_establish_venue_decor(self):
        self.observation["style_evidence"] = [{"label": "upscale", "fact": "Decorated cocktail glass"}]
        product = map_observation(self.observation, self.config)
        self.assertEqual(product["style_tags"], [])
        self.assertIn("style_tags", product["unknown_fields"])

    def test_mixed_venue_context_is_not_removed_by_object_word(self):
        for fact in ("Cocktail bar with brass shelves", "Framed shirts decorate the wall", "Drinks on a geometric table"):
            with self.subTest(fact=fact):
                self.observation["style_evidence"] = [{"label": "modern", "fact": fact}]
                self.assertEqual(map_observation(self.observation, self.config)["style_tags"], ["modern"])

    def test_full_words_avoid_dish_in_dishwasher_or_dress_in_dresser(self):
        self.observation["style_evidence"] = [{"label": "vintage", "fact": "Antique wooden dresser"}]
        self.assertEqual(map_observation(self.observation, self.config)["style_tags"], ["vintage"])

    def test_negated_object_is_not_positive_scope_evidence(self):
        self.observation["style_evidence"] = [{"label": "casual", "fact": "No uniforms; simple setting"}]
        self.assertEqual(venue_style_evidence(self.observation, self.config)[1], [])
        self.observation["style_evidence"] = [{"label": "casual", "fact": "People in shirts; no furniture"}]
        self.assertEqual(map_observation(self.observation, self.config)["style_tags"], [])

    def test_no_reference_or_sample_identity_is_used(self):
        value = {**self.observation, "sample_id": "arbitrary", "target": {"style_tags": ["casual"]}}
        self.assertEqual(venue_style_evidence(value, self.config), venue_style_evidence(self.observation, self.config))

    def test_original_protocol_and_model_prompt_are_unchanged_without_policy(self):
        original = {k: v for k, v in self.config.items() if k != "style_scope_policy"}
        self.assertEqual(map_observation(self.observation, original)["style_tags"], ["casual", "modern"])
        self.assertEqual(observation_messages("x.jpg", original), observation_messages("x.jpg", self.config))

    def test_generation_keeps_raw_and_explicit_exclusion_audit_without_extra_call(self):
        backend = Backend([self.observation])
        result = generate_observation(backend, "x.jpg", self.config)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(result["observation"], self.observation)
        self.assertEqual(len(result["style_scope_exclusions"]), 1)
        record = {**result, "attempts": [a.model_dump() for a in result["attempts"]]}
        self.assertEqual(replay_record(ROOT, record, self.config), result["result"])

    def test_scope_filter_cannot_make_invalid_raw_duplicate_labels_pass(self):
        self.observation["style_evidence"] = [self.observation["style_evidence"][0]] * 2
        with self.assertRaisesRegex(ValueError, "duplicate labels"):
            map_observation(self.observation, self.config)

    def test_original_fact_bound_still_counts_excluded_cues(self):
        self.observation["style_evidence"] = [{"label": label, "fact": f"People wearing shirts number {i}"}
            for i, label in enumerate(self.config["style_vocabulary"][:11])]
        with self.assertRaisesRegex(ValueError, "ten distinct facts"):
            map_observation(self.observation, self.config)

    def test_config_is_hash_bound_and_rejects_nonliteral_policy(self):
        path = ROOT / "configs/week8/product_observation_scope_v1.json"
        self.assertEqual(load_observation_config(path, canonical_config_sha256(self.config)), self.config)
        self.config["style_scope_policy"]["nonvenue_terms"] = [{}]
        with self.assertRaises(ValueError):
            validate_style_scope(self.config)

    def review_config(self):
        refinement = json.loads((ROOT / "configs/week8/product_observation_v7.json").read_text(encoding="utf-8"))["style_refinement"]
        self.config["style_refinement"] = {**refinement, "eligibility": "out_of_scope_fact"}
        return self.config

    def test_scope_error_triggers_real_visual_review_not_only_label_deletion(self):
        config = self.review_config()
        reviewed = {"style_evidence": [{"label": "casual", "fact": "Simple plastic chairs"}]}
        backend = Backend([self.observation, reviewed])
        result = generate_observation(backend, "x.jpg", config)
        self.assertTrue(result["passed"])
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(result["result"]["style_tags"], ["casual"])
        self.assertEqual(len(result["primary_scope_exclusions"]), 1)
        record = {**result, "attempts": [a.model_dump() for a in result["attempts"]]}
        self.assertEqual(replay_record(ROOT, record, config), result["result"])
        record["attempts"] = record["attempts"][:1]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            replay_record(ROOT, record, config)

    def test_no_scope_error_has_no_extra_generation_or_label_change(self):
        config = self.review_config()
        self.observation["style_evidence"] = self.observation["style_evidence"][1:]
        backend = Backend([self.observation])
        result = generate_observation(backend, "x.jpg", config)
        self.assertFalse(result["refinement_applied"])
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(result["result"]["style_tags"], ["modern"])

    def test_review_cannot_pass_by_silently_dropping_new_nonvenue_cues(self):
        config = self.review_config()
        invalid = {"style_evidence": [{"label": "casual", "fact": "Cotton shirts"}]}
        result = generate_observation(Backend([self.observation, invalid, invalid]), "x.jpg", config)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["attempts"]), 3)
        self.assertIn("nonvenue object", result["attempts"][-1].error)

    def test_scope_trigger_without_explicit_policy_is_rejected(self):
        config = self.review_config()
        del config["style_scope_policy"]
        from src.inference.product_style_refinement import validate_refinement_config
        with self.assertRaisesRegex(ValueError, "explicit scope policy"):
            validate_refinement_config(config)

    def targeted_config(self):
        return json.loads((ROOT / "configs/week8/product_observation_scope_repair_v1.json").read_text(encoding="utf-8"))

    def test_targeted_repair_keeps_valid_style_and_rechecks_only_invalid_hypothesis(self):
        config = self.targeted_config()
        backend = Backend([self.observation, {"style_evidence": [{"label": "casual", "fact": "Simple plastic chairs"}]}])
        result = generate_observation(backend, "x.jpg", config)
        self.assertTrue(result["passed"])
        self.assertEqual(set(result["result"]["style_tags"]), {"casual", "modern"})
        self.assertEqual(result["observation"]["style_evidence"][0], self.observation["style_evidence"][1])
        record = {**result, "attempts": [item.model_dump() for item in result["attempts"]]}
        self.assertEqual(replay_record(ROOT, record, config), result["result"])

    def test_targeted_repair_can_reject_invalid_hypothesis_without_dropping_valid_style(self):
        result = generate_observation(Backend([self.observation, {"style_evidence": []}]), "x.jpg", self.targeted_config())
        self.assertTrue(result["passed"])
        self.assertEqual(result["result"]["style_tags"], ["modern"])

    def test_targeted_repair_cannot_expand_unrequested_styles(self):
        proposal = {"style_evidence": [{"label": "cozy", "fact": "Warm soft sofa"}]}
        result = generate_observation(Backend([self.observation, proposal, proposal]), "x.jpg", self.targeted_config())
        self.assertFalse(result["passed"])
        self.assertIn("unrequested", result["attempts"][-1].error)

    def abstention_config(self):
        return json.loads((ROOT / "configs/week8/product_observation_scope_repair_v2.json").read_text(encoding="utf-8"))

    def test_observable_tableware_is_not_venue_style_and_is_explicitly_unknown(self):
        config = self.abstention_config()
        self.observation["style_evidence"] = [{"label": "classy", "fact": "Sugared cocktail glass"}]
        proposal = {"style_evidence": [{"label": "classy", "fact": "Glass with twisted stem"}]}
        result = generate_observation(Backend([self.observation, proposal]), "x.jpg", config)
        self.assertTrue(result["passed"])
        self.assertEqual(result["result"]["style_tags"], [])
        self.assertIn("style_tags", result["result"]["unknown_fields"])
        self.assertEqual(result["style_evidence_abstentions"][0]["label"], "classy")
        self.assertIn("Glass with twisted stem", result["attempts"][-1].raw_output)
        record = {**result, "attempts": [item.model_dump() for item in result["attempts"]]}
        self.assertEqual(replay_record(ROOT, record, config), result["result"])

    def test_glass_architecture_and_lighting_are_not_mistaken_for_tableware(self):
        for fact in ("Glass curtain wall", "Colored glass chandelier", "Etched glass doors", "Bottle display shelves"):
            with self.subTest(fact=fact):
                self.observation["style_evidence"] = [{"label": "classy", "fact": fact}]
                self.assertEqual(venue_style_evidence(self.observation, self.abstention_config())[1], [])

    def test_abstention_never_hides_invalid_structure_or_unrequested_hypotheses(self):
        config = self.abstention_config()
        for proposal in ({"style_evidence": [{"label": "casual", "fact": "Cotton shirts"}] * 2},
                         {"style_evidence": [{"label": "classy", "fact": "Cocktail glass"}]},
                         {"style_evidence": [{"label": "casual", "fact": "Shirts imply informal style"}]}):
            with self.subTest(proposal=proposal):
                result = generate_observation(Backend([self.observation, proposal, proposal]), "x.jpg", config)
                self.assertFalse(result["passed"])

    def test_abstention_cannot_be_enabled_for_independent_reference_replacement(self):
        from src.inference.product_style_refinement import validate_refinement_config
        config = self.abstention_config()
        config["style_refinement"]["mode"] = "replace"
        with self.assertRaisesRegex(ValueError, "only allowed"):
            validate_refinement_config(config)


if __name__ == "__main__":
    unittest.main()

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from src.inference.product_observation import (
    FOOD_SUBJECT_CONFLICT, canonical_config_sha256, generate_observation,
    load_observation_config, map_observation, observation_correction_messages,
    observation_correction_response_format, observation_messages, observation_schema,
)
from src.inference.observation_constraints import SHARED_SERIALIZATION_PROTOCOL
from src.inference.system_runtime import GenerationResult
from scripts.verify_week8_observation_retry import replay_records


ROOT = Path(__file__).resolve().parents[1]


class FoodCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "configs/week8/product_observation_food_retry_v1.json"
        self.config = json.loads(self.path.read_text(encoding="utf-8"))
        self.base = {key: value for key, value in self.config.items() if key != "style_refinement"}
        self.food = {"subject_kind": "food_closeup", "subject_fact": "Bowl of noodles",
                     "style_evidence": [], "facility_evidence": [], "price_text": []}

    def test_new_version_preserves_first_prompt_schema_and_all_other_settings(self):
        legacy = json.loads((ROOT / "configs/week8/product_observation_scope_repair_v2.json").read_text(encoding="utf-8"))
        self.assertEqual({k: v for k, v in self.config.items() if k != "correction_protocol"}, legacy)
        self.assertEqual(load_observation_config(self.path, canonical_config_sha256(self.config)), self.config)
        self.assertEqual(observation_messages("image.jpg", self.config), observation_messages("image.jpg", legacy))
        self.assertEqual(observation_schema(self.config), observation_schema(legacy))

    def test_only_exact_food_contract_error_enables_shared_decoder(self):
        result = observation_correction_response_format(self.config, FOOD_SUBJECT_CONFLICT)
        self.assertEqual(result["constraint_protocol"], SHARED_SERIALIZATION_PROTOCOL)
        for error in ("negated positive label evidence in facility_evidence", "duplicate labels in style_evidence",
                      "$ missing required properties: ['price_text']", "prefix " + FOOD_SUBJECT_CONFLICT):
            with self.subTest(error=error):
                self.assertIsNone(observation_correction_response_format(self.config, error))
        for missing in (None, "", 1):
            with self.assertRaisesRegex(ValueError, "previous validation error"):
                observation_correction_response_format(self.config, missing)

    def test_model_retains_subject_choice_in_the_single_food_correction(self):
        invalid = {**self.food, "facility_evidence": [{"label": "seating", "fact": "Wooden chairs"}]}
        corrected = {**copy.deepcopy(invalid), "subject_kind": "dining_space", "subject_fact": "Dining room"}
        backend = Mock()
        backend.generate_with_usage.side_effect = [GenerationResult(content=json.dumps(value), input_tokens=10, output_tokens=20)
                                                   for value in (invalid, corrected)]
        result = generate_observation(backend, "image.jpg", self.base)
        self.assertTrue(result["passed"])
        self.assertEqual(result["result"]["business_category"], "restaurant")
        calls = backend.generate_with_usage.call_args_list
        self.assertIsNone(calls[0].kwargs["response_format"])
        self.assertEqual(calls[1].kwargs["response_format"]["constraint_protocol"], SHARED_SERIALIZATION_PROTOCOL)
        self.assertEqual(json.loads(result["attempts"][0].raw_output), invalid)
        self.assertEqual(len(calls), 2)

    def test_other_errors_keep_legacy_correction_without_suppressing_validation(self):
        invalids = [
            {**self.food, "subject_kind": "dining_space", "facility_evidence": [{"label": "parking", "fact": "No parking"}]},
            {key: value for key, value in self.food.items() if key != "price_text"},
        ]
        for invalid in invalids:
            backend = Mock()
            backend.generate_with_usage.side_effect = [GenerationResult(content=json.dumps(value), input_tokens=10, output_tokens=20) for value in (invalid, self.food)]
            result = generate_observation(backend, "image.jpg", self.base)
            self.assertTrue(result["passed"])
            self.assertIsNotNone(result["attempts"][0].error)
            self.assertTrue(all(call.kwargs["response_format"] is None for call in backend.generate_with_usage.call_args_list))
            legacy = {k: v for k, v in self.base.items() if k != "correction_protocol"}
            messages = observation_messages("image.jpg", legacy)
            self.assertEqual(backend.generate_with_usage.call_args_list[1].args[0],
                observation_correction_messages(messages, json.dumps(invalid), result["attempts"][0].error, legacy))

    def test_repeated_food_conflict_is_not_silently_repaired_or_given_more_attempts(self):
        invalid = {**self.food, "style_evidence": [{"label": "classy", "fact": "Porcelain bowl"}]}
        backend = Mock()
        backend.generate_with_usage.return_value = GenerationResult(content=json.dumps(invalid), input_tokens=10, output_tokens=20)
        result = generate_observation(backend, "image.jpg", self.base)
        self.assertFalse(result["passed"])
        self.assertEqual(backend.generate_with_usage.call_count, 2)
        self.assertTrue(all(row.error == FOOD_SUBJECT_CONFLICT for row in result["attempts"]))

    def test_diagnostic_replay_binds_error_specific_decoder_choice(self):
        for error in (FOOD_SUBJECT_CONFLICT, "$ missing required properties: ['price_text']"):
            case = {"case_id": "case", "sample_id": "dev", "image_path": "image.jpg", "previous_raw": "{}", "validation_error": error}
            messages = observation_correction_messages(observation_messages(str(ROOT / "image.jpg"), self.config), "{}", error, self.config)
            record = {"case_id": "case", "sample_id": "dev", "passed": True, "error": None,
                      "raw_output": json.dumps(self.food), "result": map_observation(self.food, self.config), "elapsed_ms": 1,
                      "input_messages_sha256": canonical_config_sha256(messages),
                      "response_format_sha256": canonical_config_sha256(observation_correction_response_format(self.config, error))}
            self.assertEqual(replay_records(ROOT, [case], [record], self.config)["failures"], 0)
            wrong = None if error == FOOD_SUBJECT_CONFLICT else observation_correction_response_format(self.config, FOOD_SUBJECT_CONFLICT)
            with self.assertRaisesRegex(ValueError, "decoder constraints"):
                replay_records(ROOT, [case], [{**record, "response_format_sha256": canonical_config_sha256(wrong)}], self.config)


if __name__ == "__main__":
    unittest.main()

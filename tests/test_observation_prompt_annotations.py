import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from src.inference.product_observation import (
    canonical_config_sha256, generate_observation, load_observation_config,
    map_observation, observation_messages, observation_prompt_schema, observation_schema,
)
from src.inference.product_style_refinement import refinement_schema
from src.inference.system_runtime import GenerationResult


ROOT = Path(__file__).resolve().parents[1]


class ObservationPromptAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.legacy = json.loads((ROOT / "configs/week8/product_observation_scope_repair_v2.json").read_text(encoding="utf-8"))
        self.path = ROOT / "configs/week8/product_observation_guarded_v1.json"
        self.config = json.loads(self.path.read_text(encoding="utf-8"))
        self.food = {"subject_kind": "food_closeup", "subject_fact": "Bowl of noodles",
                     "style_evidence": [], "facility_evidence": [], "price_text": []}

    def test_annotations_only_change_first_stage_prompt_schema_metadata(self):
        loaded = load_observation_config(self.path, canonical_config_sha256(self.config))
        without_annotations = {key: value for key, value in loaded.items() if key != "prompt_schema_annotations"}
        self.assertEqual(without_annotations, self.legacy)
        self.assertEqual(observation_schema(self.legacy), observation_schema(loaded))
        self.assertEqual(refinement_schema(self.legacy), refinement_schema(loaded))
        annotated = observation_prompt_schema(loaded)
        for field, description in loaded["prompt_schema_annotations"].items():
            self.assertEqual(annotated["properties"][field].pop("description"), description)
        self.assertEqual(annotated, observation_schema(self.legacy))
        self.assertEqual(map_observation(self.food, loaded), map_observation(self.food, self.legacy))

    def test_old_expanded_and_compact_prompt_schemas_remain_unchanged(self):
        self.assertEqual(observation_prompt_schema(self.legacy), observation_schema(self.legacy))
        messages = observation_messages("image.jpg", self.legacy)
        text = messages[1]["content"][1]["text"]
        self.assertEqual(text, self.legacy["task_prompt"] + "\nJSON Schema: " + json.dumps(
            observation_schema(self.legacy), ensure_ascii=False, separators=(",", ":")))
        compact = {**self.legacy, "protocol": "product_visual_observation_v4", "prompt_schema_style": "property_names"}
        shown = observation_prompt_schema(compact)
        self.assertEqual(shown["properties"]["style_evidence"]["propertyNames"]["enum"], self.legacy["style_vocabulary"])
        self.assertNotIn("description", shown["properties"]["price_text"])

    def test_existing_contradictions_and_required_fields_still_fail(self):
        contradictions = [
            {**self.food, "style_evidence": [{"label": "casual", "fact": "Paper food bowl"}]},
            {**self.food, "facility_evidence": [{"label": "seating", "fact": "Chair"}]},
            {key: value for key, value in self.food.items() if key != "price_text"},
            {**self.food, "subject_kind": "dining_space", "facility_evidence": [{"label": "parking", "fact": "No parking"}]},
        ]
        for value in contradictions:
            for config in (self.legacy, self.config):
                with self.subTest(value=value, annotated=config is self.config), self.assertRaises(ValueError):
                    map_observation(value, config)

    def test_no_auto_repair_or_additional_attempt_is_introduced(self):
        invalid = {**self.food, "facility_evidence": [{"label": "seating", "fact": "Chair"}]}
        backend = Mock()
        backend.generate_with_usage.return_value = GenerationResult(content=json.dumps(invalid), input_tokens=10, output_tokens=20)
        result = generate_observation(backend, "image.jpg", self.config)
        self.assertFalse(result["passed"])
        self.assertEqual(backend.generate_with_usage.call_count, 2)
        self.assertEqual([json.loads(item.raw_output) for item in result["attempts"]], [invalid, invalid])
        self.assertEqual(len(backend.generate_with_usage.call_args_list[1].args[0]), 3)

    def test_invalid_annotation_config_is_rejected_before_model_call(self):
        values = [None, [], {}, {"new_field": "rule"}, {"price_text": ""}, {"price_text": 1}, {"price_text": "x" * 401}]
        for annotations in values:
            backend = Mock()
            config = {**copy.deepcopy(self.config), "prompt_schema_annotations": annotations}
            with self.subTest(annotations=annotations), self.assertRaisesRegex(ValueError, "annotations"):
                generate_observation(backend, "image.jpg", config)
            backend.generate_with_usage.assert_not_called()
        with self.assertRaisesRegex(ValueError, "annotations"):
            observation_prompt_schema({**self.config, "protocol": "product_visual_observation_v4"})


if __name__ == "__main__":
    unittest.main()

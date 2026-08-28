import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from contextlib import nullcontext
from types import SimpleNamespace

from src.inference.observation_constraints import PROTOCOL, observation_constraint_schemas, build_observation_constraint_parser
from src.inference.product_observation import (canonical_config_sha256, generate_observation, load_observation_config,
    map_observation, observation_schema, observation_messages, observation_correction_messages, observation_correction_response_format)
from src.inference.system_runtime import GenerationResult, TransformersPeftBackend, ModelGenerationError
from scripts.verify_week8_observation_retry import replay_records
from tests.test_system_runtime import settings


ROOT = Path(__file__).resolve().parents[1]


class ObservationSubjectConstraintTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "configs/week8/product_observation_schema_retry_v1.json"
        self.config = json.loads(self.path.read_text(encoding="utf-8"))
        self.legacy = {key: value for key, value in self.config.items() if key != "correction_protocol"}
        self.food = {"subject_kind": "food_closeup", "subject_fact": "Bowl of noodles", "style_evidence": [], "facility_evidence": [], "price_text": []}
        self.invalid = {**self.food, "facility_evidence": [{"label": "seating", "fact": "Chair"}]}

    def test_first_stage_and_correction_text_are_identical_to_the_original(self):
        self.assertEqual(load_observation_config(self.path, canonical_config_sha256(self.config)), self.config)
        self.assertEqual(observation_schema(self.config), observation_schema(self.legacy))
        messages = observation_messages("image.jpg", self.config)
        self.assertEqual(messages, observation_messages("image.jpg", self.legacy))
        self.assertEqual(observation_correction_messages(messages, "invalid", "error", self.config),
                         observation_correction_messages(messages, "invalid", "error", self.legacy))
        self.assertIsNone(observation_correction_response_format(self.legacy))

    def test_decoder_is_used_only_for_the_single_correction(self):
        backend = Mock()
        backend.generate_with_usage.side_effect = [GenerationResult(content=json.dumps(value), input_tokens=10, output_tokens=20)
                                                   for value in (self.invalid, self.food)]
        result = generate_observation(backend, "image.jpg", self.config)
        self.assertTrue(result["passed"])
        self.assertIsNone(backend.generate_with_usage.call_args_list[0].kwargs["response_format"])
        response_format = backend.generate_with_usage.call_args_list[1].kwargs["response_format"]
        self.assertEqual(response_format["constraint_protocol"], PROTOCOL)
        self.assertEqual(response_format["json_schema"]["schema"], observation_schema(self.legacy))
        self.assertEqual(json.loads(result["attempts"][0].raw_output), self.invalid)
        self.assertEqual(backend.generate_with_usage.call_count, 2)

    def test_unsupported_backend_output_cannot_bypass_post_validation(self):
        backend = Mock()
        backend.generate_with_usage.return_value = GenerationResult(content=json.dumps(self.invalid), input_tokens=10, output_tokens=20)
        result = generate_observation(backend, "image.jpg", self.config)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertTrue(all(item.error for item in result["attempts"]))

    def test_nonfood_branch_keeps_all_other_categories_fields_and_support(self):
        schema = observation_schema(self.config)
        original = copy.deepcopy(schema)
        other, fact, price = observation_constraint_schemas(schema)
        self.assertEqual(schema, original)
        self.assertEqual(set(other["properties"]["subject_kind"]["enum"]), set(self.config["subject_categories"]) - {"food_closeup"})
        self.assertEqual(other["required"], schema["required"])
        for field in ("style_evidence", "facility_evidence", "price_text"):
            self.assertEqual(other["properties"][field], schema["properties"][field])
        self.assertEqual(fact, schema["properties"]["subject_fact"])
        self.assertEqual(price, schema["properties"]["price_text"])

    def test_food_branch_uses_empty_array_literals_not_broken_maxitems_zero(self):
        fake = Mock()
        fake.JsonSchemaParser.side_effect = lambda schema: ("schema", schema)
        fake.StringParser.side_effect = lambda value: ("literal", value)
        fake.SequenceParser.side_effect = lambda values: ("sequence", values)
        fake.UnionParser.side_effect = lambda values: ("union", values)
        with patch.dict("sys.modules", {"lmformatenforcer": fake}):
            result = build_observation_constraint_parser(observation_schema(self.config))
        food = result[1][0]
        self.assertEqual(food[1][2], ("literal", ',"style_evidence":[],"facility_evidence":[],"price_text":'))
        self.assertEqual(food[1][0], ("literal", '{"subject_kind":"food_closeup","subject_fact":'))
        self.assertNotIn("food_closeup", result[1][1][1]["properties"]["subject_kind"]["enum"])

    def test_schema_cannot_omit_required_fields_or_remove_venue_alternatives(self):
        for mutation in ("required", "food", "array", "extras"):
            schema = observation_schema(self.config)
            if mutation == "required":
                schema["required"].remove("price_text")
            elif mutation == "food":
                schema["properties"]["subject_kind"]["enum"] = ["food_closeup"]
            elif mutation == "array":
                schema["properties"]["style_evidence"]["type"] = "object"
            else:
                schema["additionalProperties"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                observation_constraint_schemas(schema)

    def test_replay_binds_the_actual_decoder_format_not_only_the_prompt(self):
        case = {"case_id": "case-1", "sample_id": "dev-1", "image_path": "image.jpg", "previous_raw": json.dumps(self.invalid), "validation_error": "food conflict"}
        messages = observation_correction_messages(observation_messages(str(ROOT / "image.jpg"), self.config), case["previous_raw"], case["validation_error"], self.config)
        row = {"case_id": "case-1", "sample_id": "dev-1", "input_messages_sha256": canonical_config_sha256(messages),
               "response_format_sha256": canonical_config_sha256(observation_correction_response_format(self.config)),
               "passed": True, "error": None, "result": map_observation(self.food, self.config), "raw_output": json.dumps(self.food), "elapsed_ms": 1.0}
        self.assertEqual(replay_records(ROOT, [case], [row], self.config)["failures"], 0)
        for digest in (None, "incorrect"):
            with self.assertRaisesRegex(ValueError, "decoder constraints"):
                replay_records(ROOT, [case], [{**row, "response_format_sha256": digest}], self.config)

    def test_real_backend_dispatches_only_explicit_subject_constraint(self):
        for protocol in (PROTOCOL, None, "unknown"):
            backend = TransformersPeftBackend(settings())
            generate = Mock(side_effect=RuntimeError("decoder dispatch reached model"))
            backend._model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]), generate=generate)
            backend._processor = SimpleNamespace(tokenizer=object(), image_processor=None,
                apply_chat_template=lambda *args, **kwargs: {})
            backend._torch = SimpleNamespace(inference_mode=nullcontext)
            standard_parser, product_parser, prefix = object(), object(), Mock()
            build_prefix = Mock(return_value=prefix)
            json_parser = Mock(return_value=standard_parser)
            modules = {"lmformatenforcer": SimpleNamespace(JsonSchemaParser=json_parser),
                       "lmformatenforcer.integrations.transformers": SimpleNamespace(build_transformers_prefix_allowed_tokens_fn=build_prefix)}
            response_format = observation_correction_response_format(self.config)
            if protocol is None:
                del response_format["constraint_protocol"]
            else:
                response_format["constraint_protocol"] = protocol
            with patch.dict("sys.modules", modules), patch(
                    "src.inference.observation_constraints.build_observation_constraint_parser", return_value=product_parser) as build_product:
                with self.subTest(protocol=protocol), self.assertRaises(ModelGenerationError):
                    backend.generate_with_usage([{"role": "user", "content": "test"}], response_format=response_format, max_new_tokens=8)
                if protocol == "unknown":
                    generate.assert_not_called()
                    build_prefix.assert_not_called()
                else:
                    self.assertEqual(generate.call_args.kwargs["prefix_allowed_tokens_fn"], prefix)
                    self.assertIs(build_prefix.call_args.args[1], product_parser if protocol == PROTOCOL else standard_parser)
                    self.assertEqual(build_product.call_count, int(protocol == PROTOCOL))


if __name__ == "__main__":
    unittest.main()

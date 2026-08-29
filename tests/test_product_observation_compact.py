import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.compare_week8_incumbent import compare
from src.evaluation.week8_visual_silver import replay_record
from src.inference.product_observation import (
    canonical_config_sha256, generate_observation, load_observation_config,
    map_observation, observation_messages, parse_observation, validate_observation,
)
from src.inference.system_runtime import GenerationResult

ROOT = Path(__file__).resolve().parents[1]


class CompactObservationTests(unittest.TestCase):
    def setUp(self):
        self.old = json.loads((ROOT / "configs/week8/product_observation_v3.json").read_text(encoding="utf-8"))
        self.config = json.loads((ROOT / "configs/week8/product_observation_v4.json").read_text(encoding="utf-8"))
        self.value = {"subject_kind": "dining_space", "subject_fact": "Wooden dining tables",
                      "style_evidence": {"casual": "Simple chairs and tables"},
                      "facility_evidence": {"seating": "Simple chairs and tables", "dining_tables": "Wooden dining tables"},
                      "price_text": []}

    def test_compact_mapping_exactly_preserves_legacy_contract(self):
        old = {**self.value, **{field: [{"label": k, "fact": v} for k, v in self.value[field].items()]
                               for field in ("style_evidence", "facility_evidence")}}
        snapshot = copy.deepcopy(self.value)
        self.assertEqual(map_observation(self.value, self.config), map_observation(old, self.old))
        self.assertEqual(self.value, snapshot)

    def test_empty_maps_and_unknown_category_keep_known_facilities(self):
        self.value.update(subject_kind="unidentified_space", style_evidence={})
        result = map_observation(self.value, self.config)
        self.assertEqual(result["business_category"], "unknown")
        self.assertEqual(result["visible_facilities"], ["dining_tables", "seating"])
        self.assertEqual(result["unknown_fields"], ["business_category", "price_range", "style_tags"])

    def test_validation_is_idempotent_and_does_not_change_wire_representation(self):
        validated = validate_observation(self.value, self.config)
        self.assertEqual(validated, self.value)
        self.assertEqual(validate_observation(validated, self.config), self.value)
        self.assertEqual(map_observation(validated, self.config), map_observation(self.value, self.config))

    def test_nonvisual_inference_is_not_an_observed_fact(self):
        for fact in ("Menu implies seating", "Chairs are likely", "菜单暗示座位", "Must have chairs"):
            with self.subTest(fact=fact):
                self.value["facility_evidence"] = {"seating": fact}
                with self.assertRaisesRegex(ValueError, "inferred rather than observable"):
                    map_observation(self.value, self.config)

    def test_unsupported_labels_or_wrong_fact_types_fail_closed(self):
        for evidence in ({"secret_amenity": "Object"}, {"seating": []}, {"seating": ""}, {"seating": "x" * 81}):
            with self.subTest(evidence=evidence):
                self.value["facility_evidence"] = evidence
                with self.assertRaises(ValueError):
                    map_observation(self.value, self.config)

    def test_duplicate_json_keys_are_rejected_not_silently_overwritten(self):
        raw = json.dumps(self.value).replace('"seating": "Simple chairs and tables"', '"seating": "no chairs", "seating": "chairs"')
        with self.assertRaisesRegex(ValueError, "duplicate observation key"):
            parse_observation(raw, self.config)

    def test_food_closeup_still_cannot_establish_venue_labels(self):
        self.value["subject_kind"] = "food_closeup"
        with self.assertRaisesRegex(ValueError, "food closeup"):
            map_observation(self.value, self.config)

    def test_fact_count_limit_is_not_bypassed_by_compact_objects(self):
        self.value["facility_evidence"] = {name: f"Object {i}" for i, name in enumerate(self.config["facility_vocabulary"][:11])}
        with self.assertRaisesRegex(ValueError, "ten distinct facts"):
            map_observation(self.value, self.config)

    def test_new_configs_are_hash_bound_and_messages_have_no_targets(self):
        for name in ("v4", "v5", "v6"):
            path = ROOT / f"configs/week8/product_observation_{name}.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(load_observation_config(path, canonical_config_sha256(config)), config)
            with self.assertRaises(ValueError):
                load_observation_config(path, "0" * 64)
            messages = observation_messages("fixture.jpg", config)
            self.assertIn("file://fixture.jpg", json.dumps(messages))
            self.assertNotIn("week8-product-v2-development", json.dumps(messages))

    def test_compact_prompt_schema_retains_full_strict_validation(self):
        self.config["prompt_schema_style"] = "property_names"
        text = observation_messages("fixture.jpg", self.config)[1]["content"][1]["text"]
        schema = json.loads(text.split("\nJSON Schema: ")[1])
        facilities = schema["properties"]["facility_evidence"]
        self.assertEqual(facilities["propertyNames"]["enum"], self.config["facility_vocabulary"])
        self.assertEqual(facilities["additionalProperties"]["maxLength"], 80)
        self.assertEqual(map_observation(self.value, self.config)["visible_facilities"], ["dining_tables", "seating"])
        self.value["facility_evidence"]["invented"] = "anything"
        with self.assertRaises(ValueError):
            map_observation(self.value, self.config)

    def test_retry_has_previous_output_and_retains_measured_usage(self):
        value = self.value
        class Backend:
            def __init__(self):
                self.calls = []
            def generate_with_usage(self, messages, **kwargs):
                self.calls.append(copy.deepcopy(messages))
                return GenerationResult(content="bad-json" if len(self.calls) == 1 else json.dumps(value), input_tokens=10, output_tokens=20)
        backend = Backend()
        result = generate_observation(backend, "image.jpg", self.config)
        self.assertTrue(result["passed"])
        self.assertEqual(backend.calls[1][-2], {"role": "assistant", "content": "bad-json"})
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(sum(item.output_tokens for item in result["attempts"]), 40)

    def test_raw_score_replay_verifies_compact_mapping_and_rejects_tampering(self):
        record = {"passed": True, "attempts": [{"raw_output": json.dumps(self.value), "error": None}],
                  "result": map_observation(self.value, self.config)}
        self.assertEqual(replay_record(ROOT, record, self.config), record["result"])
        record["result"]["visible_facilities"].append("parking")
        with self.assertRaisesRegex(ValueError, "differs from raw"):
            replay_record(ROOT, record, self.config)


class IncumbentSelectionTests(unittest.TestCase):
    def payload(self):
        from scripts.compare_week8_development_revision import SEMANTIC_FIELDS
        value = {"metrics": {**dict.fromkeys(SEMANTIC_FIELDS, 0.8), "composite": 0.8},
                 "supports": {"samples": 60}, "reference_audit": {"same": True},
                 "latency_ms": {"mean": 100, "p50": 100, "p95": 110}, "tokens": {"output_mean": 100}}
        return {"test_rows_read": False, "summaries": {"formal_adapter": copy.deepcopy(value),
                "observation_base": copy.deepcopy(value), "revision": copy.deepcopy(value)}}

    def checked(self, payload):
        with patch("scripts.compare_week8_incumbent.select_development_candidate", return_value={"failures": dict.fromkeys(payload["summaries"], [])}):
            return compare(payload, "observation_base")

    def test_higher_composite_cannot_hide_incumbent_precision_regression(self):
        payload = self.payload()
        payload["summaries"]["revision"]["metrics"].update(composite=0.9, style_precision=0.7)
        result = self.checked(payload)
        self.assertEqual(result["status"], "KEEP_INCUMBENT_CANDIDATE")
        self.assertIn("style_precision_below_incumbent", result["candidates"]["revision"]["failures"])

    def test_material_speed_gain_requires_token_reduction_and_no_quality_loss(self):
        payload = self.payload()
        revision = payload["summaries"]["revision"]
        revision["latency_ms"] = {"mean": 90, "p50": 90, "p95": 100}
        self.assertIsNone(self.checked(payload)["selected_role"])
        revision["tokens"]["output_mean"] = 80
        result = self.checked(payload)
        self.assertEqual(result["selected_role"], "revision")
        self.assertFalse(result["promotion_allowed"])

    def test_test_comparison_and_changed_support_are_rejected(self):
        payload = self.payload()
        payload["test_rows_read"] = True
        with self.assertRaisesRegex(ValueError, "only development"):
            self.checked(payload)
        payload["test_rows_read"] = False
        payload["summaries"]["revision"]["supports"]["samples"] = 59
        with self.assertRaisesRegex(ValueError, "identical references"):
            self.checked(payload)


if __name__ == "__main__":
    unittest.main()

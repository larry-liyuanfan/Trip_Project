import copy
import json
import unittest
from pathlib import Path


class EvaluationSchemaTest(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def _valid_payloads(self):
        return {
            "image_product_search": {
                "business_category": "hotel",
                "style_tags": ["现代"],
                "visible_facilities": ["泳池"],
                "price_range": "premium",
                "observed_evidence": ["画面中可见泳池"],
                "inferred_attributes": [],
                "unknown_fields": [],
                "confidence": None,
            },
            "after_sales": {
                "issue_type": "facility_damage",
                "severity": "high",
                "issue_location": None,
                "key_information": ["窗户玻璃破损"],
                "ocr_text": None,
                "observed_evidence": ["窗户玻璃存在裂纹"],
                "unknown_fields": [],
                "confidence": 0.8,
            },
            "itinerary_planning": {
                "style_preferences": ["慢节奏"],
                "hard_constraints": ["第二天18:00前结束"],
                "soft_constraints": ["优先咖啡馆"],
                "required_itinerary_elements": ["交通方式"],
                "itinerary": [
                    {
                        "day_index": 1,
                        "date": None,
                        "summary": "市中心慢节奏游览",
                        "activities": [
                            {
                                "start_time": "09:00",
                                "end_time": "11:00",
                                "place_name": "城市博物馆",
                                "activity": "参观常设展览",
                                "transport": "步行",
                                "source_evidence": ["参考图中可见博物馆外立面"],
                            }
                        ],
                    }
                ],
                "constraint_check": [
                    {
                        "constraint": "第二天18:00前结束",
                        "constraint_type": "hard",
                        "status": "satisfied",
                        "evidence": "行程在17:30结束",
                    }
                ],
                "observed_evidence": ["参考图中可见安静咖啡馆"],
                "unknown_fields": [],
                "confidence": 0.7,
            },
        }

    def test_three_schema_files_define_strict_required_contracts(self):
        from src.evaluation.schema_validation import load_output_schema

        for scenario in self._valid_payloads():
            with self.subTest(scenario=scenario):
                schema = load_output_schema(self.PROJECT_ROOT, scenario)
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertEqual(schema["type"], "object")
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(set(schema["required"]), set(schema["properties"]))
                self.assertEqual(schema["properties"]["confidence"]["type"], ["number", "null"])
                self.assertEqual(schema["properties"]["observed_evidence"]["type"], "array")
                self.assertIn('"enum"', json.dumps(schema))

    def test_representative_outputs_validate_for_all_scenarios(self):
        from src.evaluation.schema_validation import validate_output

        for scenario, payload in self._valid_payloads().items():
            with self.subTest(scenario=scenario):
                self.assertEqual(validate_output(self.PROJECT_ROOT, scenario, payload), payload)

    def test_itinerary_schema_requires_strict_day_and_activity_framework(self):
        from src.evaluation.schema_validation import SchemaValidationError, validate_output

        payload = self._valid_payloads()["itinerary_planning"]
        for mutation in ("missing_itinerary", "missing_activity_field", "extra_activity_field"):
            with self.subTest(mutation=mutation):
                invalid = copy.deepcopy(payload)
                if mutation == "missing_itinerary":
                    invalid.pop("itinerary")
                elif mutation == "missing_activity_field":
                    invalid["itinerary"][0]["activities"][0].pop("transport")
                else:
                    invalid["itinerary"][0]["activities"][0]["notes"] = "额外内容"
                with self.assertRaisesRegex(SchemaValidationError, "required|additional properties"):
                    validate_output(self.PROJECT_ROOT, "itinerary_planning", invalid)

    def test_itinerary_place_name_accepts_null_when_source_does_not_provide_it(self):
        from src.evaluation.schema_validation import validate_output

        payload = copy.deepcopy(self._valid_payloads()["itinerary_planning"])
        payload["itinerary"][0]["activities"][0]["place_name"] = None

        self.assertEqual(
            validate_output(self.PROJECT_ROOT, "itinerary_planning", payload),
            payload,
        )

    def test_missing_required_fields_are_rejected(self):
        from src.evaluation.schema_validation import SchemaValidationError, validate_output

        for scenario, payload in self._valid_payloads().items():
            with self.subTest(scenario=scenario):
                invalid = copy.deepcopy(payload)
                invalid.pop(next(iter(invalid)))
                with self.assertRaisesRegex(SchemaValidationError, "required"):
                    validate_output(self.PROJECT_ROOT, scenario, invalid)

    def test_invalid_enums_and_array_types_are_rejected(self):
        from src.evaluation.schema_validation import SchemaValidationError, validate_output

        invalid_cases = {
            "image_product_search": ("business_category", "spaceship", "style_tags"),
            "after_sales": ("severity", "catastrophic", "key_information"),
            "itinerary_planning": ("constraint_check", [], "hard_constraints"),
        }
        for scenario, (enum_field, invalid_enum, array_field) in invalid_cases.items():
            with self.subTest(scenario=scenario, violation="enum"):
                invalid = copy.deepcopy(self._valid_payloads()[scenario])
                if scenario == "itinerary_planning":
                    invalid[enum_field] = [{
                        "constraint": "两天",
                        "constraint_type": "mandatory",
                        "status": "satisfied",
                        "evidence": None,
                    }]
                else:
                    invalid[enum_field] = invalid_enum
                with self.assertRaisesRegex(SchemaValidationError, "enum"):
                    validate_output(self.PROJECT_ROOT, scenario, invalid)

            with self.subTest(scenario=scenario, violation="array"):
                invalid = copy.deepcopy(self._valid_payloads()[scenario])
                invalid[array_field] = "not-an-array"
                with self.assertRaisesRegex(SchemaValidationError, "array"):
                    validate_output(self.PROJECT_ROOT, scenario, invalid)

    def test_nullable_fields_accept_null_but_reject_wrong_types(self):
        from src.evaluation.schema_validation import SchemaValidationError, validate_output

        for scenario, payload in self._valid_payloads().items():
            with self.subTest(scenario=scenario):
                valid = copy.deepcopy(payload)
                valid["confidence"] = None
                validate_output(self.PROJECT_ROOT, scenario, valid)

                invalid = copy.deepcopy(payload)
                invalid["confidence"] = "high"
                with self.assertRaisesRegex(SchemaValidationError, "number|null"):
                    validate_output(self.PROJECT_ROOT, scenario, invalid)

    def test_reasoning_is_rejected_at_top_level_and_nested_levels(self):
        from src.evaluation.schema_validation import SchemaValidationError, validate_output

        for scenario, payload in self._valid_payloads().items():
            with self.subTest(scenario=scenario, level="top"):
                invalid = copy.deepcopy(payload)
                invalid["reasoning"] = "首先分析图片，然后得出结论"
                with self.assertRaisesRegex(SchemaValidationError, "additional properties"):
                    validate_output(self.PROJECT_ROOT, scenario, invalid)

        itinerary = self._valid_payloads()["itinerary_planning"]
        for path in ("constraint_check", "activity"):
            with self.subTest(scenario="itinerary_planning", level=path):
                invalid = copy.deepcopy(itinerary)
                target = (
                    invalid["constraint_check"][0]
                    if path == "constraint_check"
                    else invalid["itinerary"][0]["activities"][0]
                )
                target["reasoning"] = "长篇内部推理"
                with self.assertRaisesRegex(SchemaValidationError, "additional properties"):
                    validate_output(self.PROJECT_ROOT, "itinerary_planning", invalid)

    def test_every_evidence_string_is_length_bounded(self):
        from src.evaluation.schema_validation import SchemaValidationError, validate_output

        cases = []
        for scenario, payload in self._valid_payloads().items():
            invalid = copy.deepcopy(payload)
            invalid["observed_evidence"] = ["证" * 121]
            cases.append((scenario, "observed_evidence", invalid))

        nested = copy.deepcopy(self._valid_payloads()["itinerary_planning"])
        nested["constraint_check"][0]["evidence"] = "证" * 121
        cases.append(("itinerary_planning", "constraint_check.evidence", nested))

        source = copy.deepcopy(self._valid_payloads()["itinerary_planning"])
        source["itinerary"][0]["activities"][0]["source_evidence"] = ["证" * 121]
        cases.append(("itinerary_planning", "source_evidence", source))

        for scenario, field, invalid in cases:
            with self.subTest(scenario=scenario, field=field):
                with self.assertRaisesRegex(SchemaValidationError, "maxLength"):
                    validate_output(self.PROJECT_ROOT, scenario, invalid)

    def test_every_evidence_array_rejects_more_than_ten_items(self):
        from src.evaluation.schema_validation import SchemaValidationError, validate_output

        cases = []
        for scenario, payload in self._valid_payloads().items():
            invalid = copy.deepcopy(payload)
            invalid["observed_evidence"] = [f"证据 {index}" for index in range(11)]
            cases.append((scenario, "observed_evidence", invalid))

        source = copy.deepcopy(self._valid_payloads()["itinerary_planning"])
        source["itinerary"][0]["activities"][0]["source_evidence"] = [
            f"来源证据 {index}" for index in range(11)
        ]
        cases.append(("itinerary_planning", "source_evidence", source))

        for scenario, field, invalid in cases:
            with self.subTest(scenario=scenario, field=field):
                with self.assertRaisesRegex(SchemaValidationError, "maxItems"):
                    validate_output(self.PROJECT_ROOT, scenario, invalid)


if __name__ == "__main__":
    unittest.main()

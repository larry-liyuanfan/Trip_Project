import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.review_week8_contracts import build_requests
from src.evaluation.prompting import render_standard_prompt
from src.inference.business_validation import itinerary_business_errors
from src.inference.system_runtime import _json_schema_response_format
from src.inference.system_runtime import ScenarioService
from dataclasses import replace
from src.inference.schemas import ItineraryTaskRequest
from scripts.collect_week8_visual_silver import teacher_payload, load_inputs, collect_row
from tests.test_system_runtime import ITINERARY_OUTPUT, PRODUCT_OUTPUT, settings, FakeBackend

ROOT = Path(__file__).resolve().parents[1]


class ContractAblationTests(unittest.TestCase):
    def test_prompt_has_real_schema_and_no_null_evidence_instruction(self):
        rendered = render_standard_prompt(ROOT, "itinerary_planning", {
            "images": [{"path": "fixture.jpg"}], "text_constraints": "上海两日行程"},
            "week8_itinerary_actionable_v1")
        self.assertIn('"constraint_check"', rendered["layers"]["output_constraint"])
        self.assertNotIn("evidence 使用 null", rendered["layers"]["task_instruction"])
        self.assertIn("城市来自用户文字", rendered["layers"]["task_instruction"])
        self.assertIn("不能伪造 satisfied", rendered["layers"]["output_constraint"])

    def test_product_prompt_requires_visual_not_merchant_evidence(self):
        rendered = render_standard_prompt(ROOT, "image_product_search", {
            "images": [{"path": "fixture.jpg"}], "text_constraints": None},
            "week8_product_visual_facts_v3")
        self.assertIn("不使用商家属性", rendered["layers"]["system_role"])
        self.assertIn("装修档次不是价格证据", rendered["layers"]["task_instruction"])
        self.assertIn('"visible_facilities"', rendered["layers"]["output_constraint"])

    def test_requests_use_locked_image_path_without_target_metadata(self):
        rows = [{"sample_id": "dev01", "image_path": "images/a.jpg",
                 "target": {"parking": True}, "business_description": "secret-reference"}]
        requests = build_requests(ROOT, rows, ["上海两日行程"])
        self.assertEqual(requests[0][2].image_urls, [str(ROOT / "images/a.jpg")])
        self.assertEqual(requests[1][2].text_context, "上海两日行程")
        self.assertNotIn("secret-reference", str(requests))
        self.assertEqual(len(requests), 2)

    def test_pilot_is_diagnostic_development_not_a_final_run(self):
        config = json.loads((ROOT / "configs/week8/contract_ablation_v1.json").read_text(encoding="utf-8"))
        self.assertFalse(config["final_test_access"])
        self.assertEqual(config["human_annotation_count"], 0)
        self.assertEqual(config["selection_policy"], "development_diagnostic_only")
        self.assertEqual(len(config["development_indices"]), len(set(config["development_indices"])))

    def test_retry_schema_preserves_five_day_request_and_frozen_schema(self):
        schema = _json_schema_response_format(ROOT, "itinerary_planning", "v2", required_day_count=5)["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["itinerary"]["minItems"], 5)
        self.assertEqual(schema["properties"]["itinerary"]["maxItems"], 5)
        original = _json_schema_response_format(ROOT, "itinerary_planning", "v2")["json_schema"]["schema"]
        self.assertEqual(original["properties"]["itinerary"]["maxItems"], 14)

    def test_false_satisfied_claim_cannot_hide_late_or_private_transport(self):
        output = copy.deepcopy(ITINERARY_OUTPUT)
        output["constraint_check"] = [{"constraint": "公共交通18:00前结束", "constraint_type": "hard",
                                       "status": "satisfied", "evidence": "已满足要求"}]
        activity = output["itinerary"][0]["activities"][0]
        activity.update(start_time="17:00", end_time="20:00", transport="出租车")
        errors = itinerary_business_errors(output, "一天行程，公共交通，每天18:00前结束")
        self.assertIn("activity_ends_after_requested_deadline", errors)
        self.assertIn("private_transport_violates_public_transport", errors)

    def test_required_and_excluded_places_are_checked_in_activities(self):
        output = copy.deepcopy(ITINERARY_OUTPUT)
        output["constraint_check"] = [{"constraint": "必须包含故宫，不去长城", "constraint_type": "hard",
                                       "status": "satisfied", "evidence": "符合要求"}]
        output["itinerary"][0]["activities"][0]["place_name"] = "长城"
        errors = itinerary_business_errors(output, "必须包含故宫，不去长城")
        self.assertIn("required_place_missing_from_activities:故宫", errors)
        self.assertIn("excluded_place_in_activities:长城", errors)

    def test_impossible_day_counts_rejected_before_generation(self):
        for text in ("0天行程", "15天行程"):
            with self.assertRaises(ValueError):
                ItineraryTaskRequest(image_urls=["image.jpg"], text_context=text)

    def test_visual_teacher_payload_has_no_candidate_or_metadata(self):
        config = json.loads((ROOT / "configs/week8/visual_teacher_v1.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.jpg"
            path.write_bytes(b"fixture")
            payload = teacher_payload(config, path)
        self.assertEqual(payload["model"], "qwen3.7-plus")
        self.assertNotIn(str(path), json.dumps(payload))
        self.assertTrue(payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_visual_teacher_rejects_final_path_before_read(self):
        config = json.loads((ROOT / "configs/week8/visual_teacher_v1.json").read_text(encoding="utf-8"))
        config["manifest"] = "data/test/image_product_search.jsonl"
        with self.assertRaisesRegex(ValueError, "development manifest"):
            load_inputs(config, ROOT)

    def test_unconstrained_correction_still_validates_complete_output(self):
        from src.inference.schemas import TaskRequest
        backend = FakeBackend(["not-json", json.dumps(PRODUCT_OUTPUT)])
        service = ScenarioService(replace(settings(), schema_constrained_retry=False), backend)
        result = service.run_task("image_product_search", TaskRequest(image_urls=["image.jpg"]))
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(backend.response_formats[-1], {"type": "json_object"})
        self.assertTrue(result.schema_valid)

    def test_teacher_retry_keeps_failed_attempt_and_never_upgrades_silver(self):
        config = json.loads((ROOT / "configs/week8/visual_teacher_v1.json").read_text(encoding="utf-8"))
        config["max_attempts"] = 2
        responses = [Mock(status_code=200), Mock(status_code=200)]
        for response, raw in zip(responses, ("not-json", json.dumps(PRODUCT_OUTPUT))):
            response.json.return_value = {"model": config["model"], "choices": [{"message": {"content": raw}}]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "image.jpg").write_bytes(b"fixture")
            schema = Path("configs/evaluation/schemas/image_product_search_v1.schema.json")
            (root / schema).parent.mkdir(parents=True)
            (root / schema).write_bytes((ROOT / schema).read_bytes())
            with patch("scripts.collect_week8_visual_silver.requests.post", side_effect=responses) as post:
                result = collect_row({"sample_id": "dev", "image_path": "image.jpg"}, config, None, root, "https://example.invalid/v1", "test-key")
        self.assertEqual(post.call_count, 2)
        self.assertIsNotNone(result["attempts"][0]["error"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["label_source"], "model_generated_silver")
        self.assertFalse(result["visual_accuracy_claim_supported"])
        self.assertNotIn("test-key", json.dumps(result))


if __name__ == "__main__":
    unittest.main()

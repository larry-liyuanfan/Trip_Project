import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.review_week8_contracts import build_requests, resolve_profile_adapter_modes
from scripts.verify_week8_candidate_runtime import attempt_metrics
from src.evaluation.prompting import render_standard_prompt
from src.inference.business_validation import itinerary_business_errors, itinerary_request_contract
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

    def test_evidence_guard_prompt_rejects_cross_field_object_substitution(self):
        config = json.loads((
            ROOT / "configs/week8/product_observation_evidence_guard_v1.json"
        ).read_text(encoding="utf-8"))
        prompt = config["task_prompt"]
        self.assertIn("a person, menu, plate, glass or napkin alone is not seating", prompt)
        self.assertIn("bottles, a drink, glassware, shaker or beer sign alone are not a bar", prompt)
        self.assertIn("not traffic or cars merely seen through a window", prompt)
        self.assertIn("remove the label when its fact names only a different object", prompt)

    def test_itinerary_v5_starts_with_complete_schema_order(self):
        rendered = render_standard_prompt(ROOT, "itinerary_planning", {
            "images": [{"path": "fixture.jpg"}], "text_constraints": "上海两日行程"},
            "week8_itinerary_actionable_v5")
        constraint = rendered["layers"]["output_constraint"]
        self.assertIn("all nine top-level keys exactly once", constraint)
        self.assertIn("beginning with style_preferences and hard_constraints", constraint)
        self.assertIn("do not begin with itinerary or constraint_check", constraint)

    def test_v13_changes_only_the_itinerary_prompt_and_has_fixed_probe(self):
        incumbent = json.loads((
            ROOT / "configs/releases/qwen3_vl_system_week8_v12.json"
        ).read_text(encoding="utf-8"))
        candidate = json.loads((
            ROOT / "configs/releases/qwen3_vl_system_week8_v13.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(candidate["prompts"]["itinerary_planning"], "week8_itinerary_actionable_v5")
        incumbent["release_id"] = candidate["release_id"]
        incumbent["prompts"]["itinerary_planning"] = candidate["prompts"]["itinerary_planning"]
        self.assertEqual(candidate, incumbent)
        probe = json.loads((
            ROOT / "configs/week8/candidate_runtime_probe_v8.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(probe["release_config"], "configs/releases/qwen3_vl_system_week8_v13.json")
        self.assertEqual(len(probe["itinerary_requests"]), 3)
        self.assertFalse(probe["test_rows_read"])
        self.assertEqual(probe["human_annotation_count"], 0)

    def test_runtime_attempt_metrics_distinguish_corrected_success(self):
        summary = attempt_metrics([
            {"passed": True, "attempts": [
                {"error": "missing fields", "input_tokens": 10, "output_tokens": 3, "latency_ms": 20},
                {"error": None, "input_tokens": 12, "output_tokens": 4, "latency_ms": 25},
            ]},
            {"passed": True, "attempts": [
                {"error": None, "input_tokens": 8, "output_tokens": 2, "latency_ms": 15},
            ]},
        ])
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["first_attempt_pass"], 1)
        self.assertEqual(summary["attempts_total"], 3)
        self.assertEqual(summary["input_tokens_total"], 30)
        self.assertEqual(summary["latency_ms_total"], 60)

    def test_v17_is_development_only_and_keeps_all_rows(self):
        config = json.loads((
            ROOT / "configs/week8/contract_ablation_v17.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(config["development_indices"], "all")
        self.assertFalse(config["final_test_access"])
        self.assertEqual(config["human_annotation_count"], 0)
        self.assertEqual(config["incumbent_role"], "observation_incumbent")
        runner = (ROOT / "scripts/review_week8_contracts.py").read_text(encoding="utf-8")
        for profile in config["profiles"]:
            self.assertIn(f'"{profile}"', runner)

        recovery = json.loads((
            ROOT / "configs/week8/contract_ablation_v17_recovery_v1.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(recovery["recovery_of"], config["run_id"])
        self.assertNotEqual(recovery["output_root"], config["output_root"])
        self.assertEqual(recovery["development_indices"], "all")
        self.assertFalse(recovery["final_test_access"])

        recovery_v2 = json.loads((
            ROOT / "configs/week8/contract_ablation_v17_recovery_v2.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(recovery_v2["recovery_of"], recovery["run_id"])
        self.assertTrue((
            ROOT / "configs/evaluation/prompts" / recovery_v2["product_prompt"] / "common.yaml"
        ).is_file())
        self.assertTrue((
            ROOT / "configs/evaluation/prompts" / recovery_v2["itinerary_prompt"] / "common.yaml"
        ).is_file())

    def test_observation_profile_adapter_route_must_be_explicit(self):
        with self.assertRaisesRegex(ValueError, "explicit adapter mode"):
            resolve_profile_adapter_modes({"profiles": ["observation_candidate"]})
        self.assertEqual(resolve_profile_adapter_modes({
            "profiles": ["observation_incumbent", "observation_candidate"],
            "profile_adapter_modes": {
                "observation_incumbent": "base", "observation_candidate": "adapter",
            },
        }), {"observation_incumbent": "base", "observation_candidate": "adapter"})
        self.assertEqual(resolve_profile_adapter_modes({
            "profiles": ["observation_enhanced_base"],
        }), {"observation_enhanced_base": "base"})

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

    def test_placeholder_venue_is_not_a_completed_itinerary(self):
        output = copy.deepcopy(ITINERARY_OUTPUT)
        output["itinerary"][0]["activities"][0]["place_name"] = "上海某文化空间（参考图中环境）"
        self.assertIn("activity_place_is_placeholder", itinerary_business_errors(output, "一天行程"))

    def test_model_cannot_fabricate_retrieval_or_transport_details(self):
        output = copy.deepcopy(ITINERARY_OUTPUT)
        activity = output["itinerary"][0]["activities"][0]
        activity.update(source_evidence=["imagined citation"], transport="地铁1号线至静安寺站步行5分钟")
        errors = itinerary_business_errors(output, "一天行程")
        self.assertIn("source_evidence_must_be_empty_without_retrieval", errors)
        self.assertIn("transport_must_use_generic_mode_without_unverified_route", errors)
        activity.update(source_evidence=[], transport="地铁或步行")
        self.assertEqual(itinerary_business_errors(output, "一天行程"), [])

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

    def test_equivalent_visit_constraints_are_not_rejected_as_missing(self):
        output = copy.deepcopy(ITINERARY_OUTPUT)
        output["itinerary"][0]["activities"][0]["place_name"] = "故宫博物院"
        output["constraint_check"] = [{"constraint": "必去：故宫", "constraint_type": "hard", "status": "satisfied", "evidence": "Day 1故宫博物院"},
                                      {"constraint": "禁去：长城", "constraint_type": "hard", "status": "satisfied", "evidence": "所有活动不含长城"}]
        self.assertEqual(itinerary_business_errors(output, "必须包含故宫，不去长城"), [])
        output["constraint_check"][1]["constraint"] = "必去：长城"
        self.assertTrue(any("explicit_constraint_not_verified" in value for value in itinerary_business_errors(output, "必须包含故宫，不去长城")))

    def test_dates_are_not_invented_or_shifted(self):
        output = copy.deepcopy(ITINERARY_OUTPUT)
        output["itinerary"][0]["date"] = "2025-04-01"
        self.assertIn("calendar_date_invented_without_user_request", itinerary_business_errors(output, "一天行程"))
        self.assertIn("calendar_dates_do_not_match_requested_start", itinerary_business_errors(output, "2026-08-28开始，一天行程"))
        with self.assertRaises(ValueError):
            ItineraryTaskRequest(image_urls=["x"], text_context="2026-02-30开始，一天行程")

    def test_request_contract_is_input_derived_not_an_acceptance(self):
        context = itinerary_request_contract("北京3天行程，必须包含故宫，不去长城")
        self.assertEqual(context["days"], 3)
        self.assertIsNone(context["start_date"])
        self.assertTrue(context["not_a_completed_plan"])
        self.assertEqual(set(context["required_checks"]), {"北京", "3天", "必须包含故宫", "不去长城"})

    def test_city_alias_is_accepted_without_removing_city_requirement(self):
        output = copy.deepcopy(ITINERARY_OUTPUT)
        output["itinerary"][0]["activities"][0]["place_name"] = "上海博物馆"
        output["constraint_check"] = [{"constraint": "上海", "constraint_type": "hard", "status": "satisfied", "evidence": "上海博物馆"}]
        self.assertEqual(itinerary_business_errors(output, "城市：Shanghai；一天行程"), [])
        self.assertTrue(itinerary_business_errors(output, "城市：Beijing；一天行程"))

    def test_probe_cannot_pass_without_repeated_business_inputs(self):
        from scripts.verify_week8_candidate_runtime import validate_probe_config
        config = json.loads((ROOT / "configs/week8/candidate_runtime_probe_v1.json").read_text(encoding="utf-8"))
        validate_probe_config(config)
        for field, value in (("cache_modes", []), ("itinerary_requests", []), ("latency_repetitions_per_mode", 1), ("test_rows_read", True)):
            with self.assertRaises(ValueError):
                validate_probe_config(dict(config, **{field: value}))

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

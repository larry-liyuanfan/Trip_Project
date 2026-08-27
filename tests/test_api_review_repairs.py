import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from src.api import routes
from src.evaluation.week8_runtime_optimization import _expected_state_score
from src.inference.schemas import DialogueRequest, ImageUnderstandingRequest, TravelPlanningRequest, VisualSearchRequest
from src.inference.system_runtime import ScenarioService, _deterministic_state_updates
from tests.test_system_runtime import FakeBackend, settings


class APIReviewRepairTests(unittest.TestCase):
    def test_concurrent_first_requests_create_only_one_service(self):
        routes._cached_scenario_service.cache_clear()
        created = []

        def build():
            time.sleep(0.01)
            value = object()
            created.append(value)
            return value

        try:
            with patch.object(routes, "build_service", side_effect=build):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    values = list(pool.map(lambda _: routes.get_scenario_service(), range(8)))
            self.assertEqual(len(created), 1)
            self.assertTrue(all(value is values[0] for value in values))
        finally:
            routes._cached_scenario_service.cache_clear()

    def test_sample_planner_cannot_masquerade_as_production(self):
        with patch.dict("os.environ", {"APP_ENV": "production"}, clear=True):
            with patch.object(routes, "_load_sample_catalog") as catalog:
                with self.assertRaises(HTTPException) as raised:
                    routes.travel_planning(TravelPlanningRequest())
        self.assertEqual(raised.exception.status_code, 404)
        catalog.assert_not_called()

    def test_missing_legacy_model_is_503_not_an_uncaught_500(self):
        with patch.object(routes, "VLLMClient") as client:
            client.return_value.understand_images.side_effect = RuntimeError("offline")
            with self.assertRaises(HTTPException) as raised:
                routes.image_understanding(ImageUnderstandingRequest())
        self.assertEqual(raised.exception.status_code, 503)

    def test_search_limits_are_validated_at_api_boundary(self):
        for value in (0, -1, True, 101, 1.5):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                VisualSearchRequest(top_k=value)
        self.assertEqual(VisualSearchRequest(top_k=100).top_k, 100)

    def test_absent_null_state_does_not_count_as_recalled_value(self):
        self.assertEqual(_expected_state_score({}, {"budget": None}), (0, 0, 1))

    def test_last_explicit_update_wins_without_unnecessary_fallback(self):
        request = DialogueRequest(messages=[{"role": "user", "content": "预算改成2000元，不是，预算改成3000元。"}])
        updates, fallback = _deterministic_state_updates(request)
        self.assertEqual(updates, {"budget": 3000})
        self.assertFalse(fallback)

    def test_negation_is_local_to_its_clause(self):
        request = DialogueRequest(messages=[{"role": "user", "content": "不要改城市，预算改成2000元。"}])
        updates, fallback = _deterministic_state_updates(request)
        self.assertEqual(updates, {"budget": 2000})
        self.assertFalse(fallback)

    def test_partial_parse_does_not_suppress_remaining_state_changes(self):
        request = DialogueRequest(messages=[{"role": "user", "content": "预算改成2000元，并增加一位儿童。"}])
        updates, fallback = _deterministic_state_updates(request)
        self.assertEqual(updates, {"budget": 2000})
        self.assertTrue(fallback)

    def test_fallback_cannot_overwrite_unambiguous_explicit_update(self):
        service = ScenarioService(settings(), FakeBackend([json.dumps({"state_updates": {"budget": 5000, "children": 1}})]), dialogue_execution_mode="deterministic_contract_v1", dialogue_semantic_fallback_enabled=True)
        response = service.run_dialogue(DialogueRequest(messages=[{"role": "user", "content": "预算改成2000元，并增加一位儿童。"}]))
        self.assertEqual(response.state, {"budget": 2000, "children": 1})

    def test_invalid_fallback_days_does_not_corrupt_existing_state(self):
        raw = json.dumps({"state_updates": {"days": -1}})
        service = ScenarioService(settings(), FakeBackend([raw, raw]), dialogue_execution_mode="deterministic_contract_v1", dialogue_semantic_fallback_enabled=True)
        response = service.run_dialogue(DialogueRequest(messages=[{"role": "user", "content": "把行程改成负一天。"}], state={"days": 2}))
        self.assertEqual(response.state, {"days": 2})
        self.assertEqual(response.semantic_fallback_status, "FAILED_SAFE")

    def test_explicit_budget_cancellation_accepts_null(self):
        service = ScenarioService(settings(), FakeBackend(['{"state_updates":{"budget":null}}']), dialogue_execution_mode="deterministic_contract_v1", dialogue_semantic_fallback_enabled=True)
        response = service.run_dialogue(DialogueRequest(messages=[{"role": "user", "content": "取消预算限制。"}], state={"budget": 2000}))
        self.assertIsNone(response.state["budget"])
        self.assertEqual(response.semantic_fallback_status, "SUCCEEDED")

    def test_partial_failure_reply_preserves_successful_updates(self):
        raw = '{"state_updates":{"days":-1}}'
        service = ScenarioService(settings(), FakeBackend([raw, raw]), dialogue_execution_mode="deterministic_contract_v1", dialogue_semantic_fallback_enabled=True)
        response = service.run_dialogue(DialogueRequest(messages=[{"role": "user", "content": "预算改成2000元，并把行程改成负一天。"}], state={"days": 2}))
        self.assertEqual(response.state, {"budget": 2000, "days": 2})
        self.assertIn("已更新预算", response.reply)
        self.assertIn("其余变化未能可靠解析", response.reply)


if __name__ == "__main__":
    unittest.main()

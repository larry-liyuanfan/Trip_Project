import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from src.api.app import app
from src.data.product_labels import caption_labels, merchant_tags, silver_row
from src.inference.business_validation import itinerary_business_errors
from src.inference.schemas import DialogueRequest, TaskRequest
from src.inference.system_runtime import ScenarioService, ModelGenerationError, ReleaseSettings, RuntimeConfigurationError, _dialogue_messages
from src.training.week7_data import _repair_business_tags
from src.training.week8_product import select_prompt, product_silver_target, load_week8_product_config
from src.training.week8_product_two_stage import caption_to_silver_evidence
from src.retrieval.visual_search import VisualSearchService
from src.retrieval.query_inputs import ranking_query_attributes, user_query_attributes
from src.retrieval.week8_relevance import _rank
from src.retrieval.week8_hybrid import OfflineImageChannel
from scripts import tripctl, rebuild_week8_product_labels, run_system_model_smoke
from tests.test_system_runtime import FakeBackend, settings, PRODUCT_OUTPUT, ITINERARY_OUTPUT
from tests.test_week8_product import summary


class Week8AuditFixTests(unittest.TestCase):
    def test_false_parking_and_nested_strings_never_create_parking(self):
        for value in ({"BusinessParking": {"garage": False, "lot": False}},
                      {"BusinessParking": "{'garage': False, 'lot': False}"},
                      "x | Hotels | BusinessParking: {'garage': False, 'lot': False}, BikeParking: True"):
            self.assertNotIn("parking", merchant_tags(value)[1])
        self.assertIn("parking", _repair_business_tags("BusinessParking: {'garage': True}")[1])

    def test_full_words_and_negative_caption_facts(self):
        for text, absent in (("mushroom soup", "hotel"), ("spacious restaurant", "spa"), ("no parking", "parking"),
                             ("parking is not available", "parking"), ("restaurant without pool or spa", "pool")):
            target = caption_labels(text)
            self.assertNotEqual(target["business_category"], absent)
            self.assertNotIn(absent, target["visible_facilities"])
        self.assertEqual(caption_to_silver_evidence("mushroom soup")["subject_category"], "unknown")
        self.assertEqual(product_silver_target({"caption": "no parking"})["visible_facilities"], [])

    def test_merchant_information_is_separate_and_labels_remain_silver(self):
        row = silver_row({"caption": "bowl of soup", "attributes": {"BusinessParking": {"lot": True}}})
        self.assertIn("parking", row["merchant_metadata"]["facilities"])
        self.assertEqual(row["target"]["visible_facilities"], [])
        self.assertEqual(row["label_source"], "programmatic_silver")
        self.assertFalse(row["visual_accuracy_claim_supported"])

    def test_invalid_or_missing_reference_audit_cannot_lock_prompt(self):
        config = load_week8_product_config(Path("configs/week8/product_understanding_v1.json"))
        for audit in ({}, {"visual_accuracy_claim_supported": False}, {"visual_accuracy_claim_supported": True, "issue_counts": {"contradiction": 1}}):
            runs = {role: {**summary(score), "reference_semantics": audit} for role, score in
                    zip(config["prompts"], (0.7, 0.8, 0.9))}
            self.assertEqual(select_prompt(config, runs)["status"], "DIAGNOSTIC_ONLY_INVALID_REFERENCES")

    def test_zero_price_support_does_not_block_other_valid_fields(self):
        config = load_week8_product_config(Path("configs/week8/product_understanding_v1.json"))
        runs = {role: {**summary(score), "reference_semantics": {"visual_accuracy_claim_supported": True, "issue_counts": {}, "metadata_proxy_samples": 0}} for role, score in zip(config["prompts"], (0.7, 0.8, 0.9))}
        self.assertEqual(select_prompt(config, runs)["status"], "PROMPT_LOCKED")

    def test_complete_budget_and_unsupported_formats(self):
        for text, expected in (("2,000", 2000), ("1e3", 500), ("2,00", 500), ("2000-3000", 500), ("1.5千", 1500)):
            service = ScenarioService(settings(), FakeBackend([]), dialogue_execution_mode="deterministic_contract_v1")
            result = service.run_dialogue(DialogueRequest(messages=[{"role": "user", "content": f"预算改成{text}元"}], state={"budget": 500}))
            self.assertEqual(result.state["budget"], expected, text)
            self.assertEqual(result.attempts, [])
            if expected == 500:
                self.assertEqual(result.task_status, "NOT_COMPLETED")

    def test_turn_images_and_legacy_images_bind_to_latest_user(self):
        request = DialogueRequest(messages=[{"role": "user", "content": "第一张", "image_urls": ["old.jpg"]},
                                             {"role": "assistant", "content": "看到了"}, {"role": "user", "content": "第二张相比如何"}], image_urls=["new.jpg"])
        messages = _dialogue_messages(request, prompt_version="system_repair_dialogue_v1")
        self.assertIn("old.jpg", json.dumps(messages[1]))
        self.assertNotIn("new.jpg", json.dumps(messages[1]))
        self.assertIn("new.jpg", json.dumps(messages[3]))
        self.assertEqual(request.messages[2].image_urls, [])

    def test_product_dialogue_actually_dispatches(self):
        backend = FakeBackend([json.dumps(PRODUCT_OUTPUT)])
        service = ScenarioService(settings(), backend, dialogue_execution_mode="deterministic_contract_v1")
        response = service.run_dialogue(DialogueRequest(messages=[{"role": "user", "content": "识别这是什么"}], image_urls=["image.jpg"]))
        self.assertEqual(response.task_status, "COMPLETED")
        self.assertEqual(len(response.attempts), 1)
        self.assertEqual(response.tool_calls[0]["function"], "image_product_search")
        self.assertEqual(response.task_result["result"], PRODUCT_OUTPUT)

    def test_recommendation_executes_retrieval_or_reports_unavailable(self):
        runner = Mock(return_value={"results": [{"business_id": "b1"}]})
        service = ScenarioService(settings(), FakeBackend([]), dialogue_execution_mode="deterministic_contract_v1", retrieval_runner=runner)
        request = DialogueRequest(messages=[{"role": "user", "content": "推荐咖啡馆"}])
        self.assertEqual(service.run_dialogue(request).task_status, "COMPLETED")
        runner.assert_called_once()
        service.retrieval_runner = None
        response = service.run_dialogue(request)
        self.assertEqual(response.task_status, "NOT_COMPLETED")
        self.assertIn("未完成", response.reply)

    def test_itinerary_business_failure_retries_and_fails_closed(self):
        raw = json.dumps(ITINERARY_OUTPUT)
        service = ScenarioService(settings(), FakeBackend([raw, raw]))
        with self.assertRaises(ModelGenerationError) as raised:
            service.run_task("itinerary_planning", TaskRequest(image_urls=["image.jpg"], text_context="上海两日行程"))
        self.assertEqual(len(raised.exception.attempts), 2)
        self.assertIn("day_count_expected_2", str(raised.exception))

    def test_itinerary_business_checks_days_order_constraints_placeholders(self):
        output = copy.deepcopy(ITINERARY_OUTPUT)
        output["itinerary"][0]["summary"] = "简短摘要"
        errors = itinerary_business_errors(output, "上海两日行程，使用公共交通")
        self.assertIn("template_placeholder_content", errors)
        self.assertTrue(any("explicit_constraint" in error for error in errors))
        output["itinerary"][0]["day_index"] = 2
        self.assertIn("day_indices_must_be_contiguous_from_1", itinerary_business_errors(output, "one day"))

    def test_invalid_scenario_inputs_are_422_before_loading_model(self):
        with patch("src.api.routes.get_scenario_service") as service:
            client = TestClient(app)
            for path, payload in (("image-product-search", {"image_urls": ["a", "b"]}),
                                  ("itinerary-planning", {"image_urls": ["a"]}),
                                  ("itinerary-planning", {"image_urls": ["a"], "text_context": " "})):
                self.assertEqual(client.post("/v1/tasks/" + path, json=payload).status_code, 422)
            service.assert_not_called()

    def test_reference_metadata_never_becomes_ranking_input(self):
        query = {"metadata": {"city": "gold-city", "business_category": "hotel"},
                 "query_inputs": {"source": "user", "query_text": "restaurant", "attributes": {"city": "requested"}}}
        first = ranking_query_attributes(query)
        query["metadata"] = {"city": "different", "business_category": "attraction"}
        self.assertEqual(first, ranking_query_attributes(query))
        self.assertEqual(ranking_query_attributes({"metadata": {"city": "gold"}}), {})

    def test_keyword_uses_text_and_filters_without_image_encoder(self):
        encoder = Mock()
        store = Mock()
        store.query_metadata.return_value = [{"image_id": "i", "business_id": "b", "business_category": "hotel", "city": "Shanghai"}]
        service = VisualSearchService(encoder, store)
        result = service.search(None, retrieval_mode="keyword", query_text="hotel", filters={"city": "Shanghai"})
        encoder.encode.assert_not_called()
        self.assertEqual(store.query_metadata.call_args.kwargs["filters"]["business_category"], "hotel")
        self.assertEqual(result[0]["business_id"], "b")

    def test_keyword_disjunction_is_forwarded_as_in_filter(self):
        encoder = Mock()
        store = Mock()
        store.query_metadata.return_value = [
            {"image_id": "i", "business_id": "b", "business_category": "hotel"}
        ]
        service = VisualSearchService(encoder, store)
        result = service.search(
            None, retrieval_mode="keyword", query_text="推荐酒店或餐厅"
        )
        encoder.encode.assert_not_called()
        self.assertEqual(
            store.query_metadata.call_args.kwargs["filters"]["business_category"],
            ["hotel", "restaurant"],
        )
        self.assertEqual(result[0]["business_id"], "b")

    def test_cli_and_runtime_reject_same_missing_release(self):
        with patch.dict("os.environ", {"TRIP_RELEASE_CONFIG": "nonexistent/release.json"}):
            self.assertEqual(tripctl.validate()["status"], "failed")
            self.assertFalse(tripctl.doctor()["checks"]["release_config"]["ok"])
            with self.assertRaises(RuntimeConfigurationError):
                ReleaseSettings.load()

    def test_compose_and_cli_use_same_absolute_explicit_release(self):
        path = Path("configs/releases/qwen3_vl_system_week8_v7.json").resolve()
        with patch.dict("os.environ", {"TRIP_RELEASE_CONFIG": str(path)}):
            self.assertEqual(tripctl.validate()["status"], "ok")
            self.assertEqual(tripctl.validate()["release_config"], str(path))
            with patch.object(tripctl.subprocess, "call", return_value=0) as call:
                self.assertEqual(tripctl.compose(["config", "--quiet"]), 0)
                self.assertEqual(call.call_args.kwargs["env"]["TRIP_RELEASE_CONFIG"], str(path))

    def test_label_rebuild_is_new_version_and_enforces_all_identity_dimensions(self):
        row = {"sample_id": "s1", "source_id": "p1", "image_sha256": "a" * 64,
               "group_id": "g1", "constraint_template_id": "t1", "split": "train", "caption": "no parking"}
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.jsonl", Path(directory) / "new"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            original = source.read_bytes()
            manifest = rebuild_week8_product_labels.rebuild(source, output)
            self.assertEqual(manifest["human_count"], 0)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                rebuild_week8_product_labels.rebuild(source, output)
            for key in rebuild_week8_product_labels.IDENTITY_KEYS:
                other = {**row, "sample_id": "s2", "source_id": "p2", "image_sha256": "b" * 64,
                         "group_id": "g2", "constraint_template_id": "t2", "split": "development"}
                other[key] = row[key]
                source.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, key):
                    rebuild_week8_product_labels.rebuild(source, Path(directory) / key)

    def test_invalid_silver_weights_are_rejected(self):
        for weight in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                silver_row({"caption": "cafe", "sample_weight": weight})
        self.assertEqual(silver_row({"caption": "cafe", "sample_weight": 1})["sample_weight"], 0.5)

    def test_chinese_negation_is_not_a_positive_query_filter(self):
        self.assertNotIn("business_category", user_query_attributes("不要酒店"))
        self.assertEqual(user_query_attributes("不要酒店，而是咖啡馆")["business_category"], "restaurant")

    def test_explicit_disjunction_reaches_safe_in_filter_and_metadata_score(self):
        from src.retrieval.milvus_vectors import build_filter_expression
        from src.retrieval.week8_hybrid import metadata_ranking

        attrs = user_query_attributes("推荐酒店或餐厅")
        self.assertEqual(
            build_filter_expression(attrs),
            'business_category in ["hotel", "restaurant"]',
        )
        config = {"evaluation": {"relevance_weights": {"business_category": 1.0}}}
        rows = [
            {"sample_id": "a", "metadata": {"business_category": "hotel"}},
            {"sample_id": "b", "metadata": {"business_category": "attraction"}},
        ]
        hits = metadata_ranking(config, attrs, rows, top_k=2, filters=attrs)
        self.assertEqual([hit["row"]["sample_id"] for hit in hits], ["a"])

    def test_disjunction_cache_key_is_hashable_and_order_stable(self):
        from src.retrieval.week8_hybrid import MetadataRankingCache

        config = {"evaluation": {"relevance_weights": {"business_category": 1.0}}}
        rows = [{"sample_id": "a", "metadata": {"business_category": "hotel"}}]
        cache = MetadataRankingCache(config, rows, capacity=2)
        filters = {"business_category": ["hotel", "restaurant"]}
        self.assertEqual(len(cache.search(filters, top_k=1, filters=filters)), 1)
        self.assertEqual(len(cache.search(filters, top_k=1, filters=filters)), 1)
        self.assertEqual(cache.stats()["hits"], 1)

    def test_actual_ranking_is_invariant_to_query_reference_metadata(self):
        config = {"evaluation": {"candidate_pool_size": 2, "rerank_weights": {"business_category": 2}}}
        vectors = [[1.0, 0.0], [0.9, 0.1]]
        rows = [{"sample_id": str(i), "vector_index": i, "metadata": {"business_category": cat}}
                for i, cat in enumerate(("hotel", "restaurant"))]
        query = {"metadata": {"business_category": "hotel"}, "query_inputs": {"source": "user", "query_text": "restaurant"}}
        def ranked():
            return _rank(config, vectors, rows, query, vectors[0], "metadata_rerank", 2,
                         image_channel=OfflineImageChannel(vectors))
        before = ranked()
        query["metadata"]["business_category"] = "attraction"
        self.assertEqual(before, ranked())
        self.assertEqual(before[0]["row"]["sample_id"], "1")

    def test_production_keyword_api_forwards_mode_and_user_query(self):
        service = Mock()
        service.search.return_value = [{"business_id": "b"}]
        with patch.dict("os.environ", {"APP_ENV": "production"}), patch("src.api.routes.get_visual_search_service", return_value=service):
            response = TestClient(app).post("/v1/visual-search", json={"query_text": "hotel", "retrieval_mode": "keyword"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(service.search.call_args.kwargs["query_text"], "hotel")
            self.assertEqual(service.search.call_args.kwargs["retrieval_mode"], "keyword")
            self.assertEqual(response.json()["query_attributes"], {"business_category": "hotel"})
            self.assertEqual(TestClient(app).post("/v1/visual-search", json={}).status_code, 422)

    def test_pending_confirmation_is_not_a_completed_answer(self):
        backend = FakeBackend([json.dumps({"reply": "继续处理。", "state_updates": {}, "tool_calls": []})])
        service = ScenarioService(settings(), backend, dialogue_execution_mode="deterministic_contract_v1")
        response = service.run_dialogue(DialogueRequest(messages=[{"role": "user", "content": "介绍一下上海"}], task="conversation"))
        self.assertEqual(response.task_status, "NOT_COMPLETED")
        self.assertEqual(len(response.attempts), 1)

    def test_technical_smoke_does_not_mask_business_failure(self):
        service = Mock()
        service.run_task.return_value.model_dump.return_value = {"schema_valid": True, "business_valid": False}
        service.run_dialogue.return_value.model_dump.return_value = {"quality_tier": "DIALOGUE_BETA", "task_status": "NOT_COMPLETED"}
        result = run_system_model_smoke.run_model_smoke(service, Path("fixture.jpg"))
        self.assertEqual(result["technical_status"], "PASS")
        self.assertEqual(result["business_status"], "FAIL")
        self.assertEqual(result["status"], "FAIL")

    def test_retrieval_does_not_claim_unapplied_constraints_are_completed(self):
        runner = Mock(return_value={"query_status": "PARTIAL_UNSUPPORTED_CONSTRAINTS", "unapplied_query_text": "安静", "results": [{"business_id": "b"}]})
        service = ScenarioService(settings(), FakeBackend([]), dialogue_execution_mode="deterministic_contract_v1", retrieval_runner=runner)
        response = service.run_dialogue(DialogueRequest(messages=[{"role": "user", "content": "推荐安静的咖啡馆"}]))
        self.assertEqual(response.task_status, "NOT_COMPLETED")
        self.assertIn("安静", response.reply)
        self.assertTrue(response.task_result["results"])

    def test_itinerary_calendar_date_is_not_the_requested_duration(self):
        from src.inference.business_validation import requested_days
        self.assertEqual(requested_days("8月27日出发，安排上海两日行程"), 2)


if __name__ == "__main__":
    unittest.main()

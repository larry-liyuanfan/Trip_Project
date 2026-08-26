import json
import unittest
from pathlib import Path

from src.evaluation.week8_runtime_optimization import (
    first_attempt_is_three_key_json,
    load_runtime_benchmark_config,
    run_dialogue_first_turn_comparison,
    run_product_latency_benchmark,
)
from src.inference.schemas import DialogueRequest, DialogueTurn, TaskRequest
from src.inference.system_runtime import (
    GenerationResult,
    ReleaseSettings,
    ScenarioService,
    _dialogue_messages,
)


PRODUCT_OUTPUT = {
    "business_category": "restaurant",
    "style_tags": ["现代"],
    "visible_facilities": ["室内座位"],
    "price_range": "unknown",
    "observed_evidence": ["可见餐桌和座椅"],
    "inferred_attributes": [],
    "unknown_fields": ["price_range"],
    "confidence": 0.8,
}


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages = []
        self.max_new_tokens = []
        self.response_formats = []

    def ready(self):
        return True, "ok"

    def generate(self, messages, *, response_format, max_new_tokens):
        self.messages.append(messages)
        self.max_new_tokens.append(max_new_tokens)
        self.response_formats.append(response_format)
        return self.outputs.pop(0)


class FakeUsageBackend(FakeBackend):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.cache_configurations = []
        self.active_cache_entries = 0

    def configure_processor_cache(self, max_entries):
        self.active_cache_entries = max_entries
        self.cache_configurations.append(max_entries)

    def processor_cache_snapshot(self):
        return {"max_entries": self.active_cache_entries}

    def generate_with_usage(self, messages, *, response_format, max_new_tokens):
        self.messages.append(messages)
        self.max_new_tokens.append(max_new_tokens)
        self.response_formats.append(response_format)
        return GenerationResult(
            content=self.outputs.pop(0),
            input_tokens=10,
            output_tokens=5,
        )


def settings() -> ReleaseSettings:
    return ReleaseSettings(
        root=Path.cwd(),
        release_id="test-release",
        base_model="Qwen/Qwen3-VL-8B-Instruct",
        base_revision="revision",
        backend_name="transformers-peft",
        adapter_name="test-adapter",
        adapter_path=None,
        adapter_model_sha256="0" * 64,
        prompt_versions={
            "image_product_search": "system_repair_product_compact_v3",
            "after_sales": "system_repair_after_sales_evidence_v3",
            "itinerary_planning": "system_repair_itinerary_structured_v4",
        },
        schema_versions={
            "image_product_search": "v1",
            "after_sales": "v1",
            "itinerary_planning": "v2",
        },
        max_new_tokens=512,
        max_new_tokens_by_scenario={
            "image_product_search": 512,
            "after_sales": 512,
            "itinerary_planning": 1024,
        },
        max_schema_retries=1,
        dialogue_prompt_version="week8_dialogue_first_turn_v1",
        dialogue_max_new_tokens=384,
    )


class Week8RuntimeOptimizationTest(unittest.TestCase):
    def test_v4_release_binds_selected_product_and_dialogue_prompts(self):
        release = ReleaseSettings.load(
            root=Path.cwd(),
            config_path=Path("configs/releases/qwen3_vl_system_week8_v4.json"),
        )

        self.assertEqual(
            release.prompt_versions["image_product_search"],
            "week8_product_field_check_v1",
        )
        self.assertEqual(
            release.dialogue_prompt_version,
            "week8_dialogue_first_turn_v2",
        )
        self.assertEqual(
            release.max_new_tokens_by_scenario["image_product_search"], 384
        )

    def test_v5_release_binds_deterministic_contract_and_optional_fallback(self):
        release = ReleaseSettings.load(
            root=Path.cwd(),
            config_path=Path("configs/releases/qwen3_vl_system_week8_v5.json"),
        )

        self.assertEqual(
            release.prompt_versions["image_product_search"],
            "week8_product_field_check_v1",
        )
        self.assertEqual(
            release.dialogue_prompt_version,
            "week8_dialogue_deterministic_v4",
        )
        self.assertEqual(
            release.dialogue_execution_mode,
            "deterministic_contract_v1",
        )
        self.assertTrue(release.dialogue_semantic_fallback_enabled)

    def test_release_binds_candidate_dialogue_prompt_and_bounded_output(self):
        release = ReleaseSettings.load(
            root=Path.cwd(),
            config_path=Path("configs/releases/qwen3_vl_system_week8_v1.json"),
        )

        self.assertEqual(
            release.dialogue_prompt_version,
            "week8_dialogue_first_turn_v1",
        )
        self.assertEqual(release.dialogue_max_new_tokens, 384)

    def test_v2_release_binds_new_prompt_without_mutating_v1(self):
        v1 = ReleaseSettings.load(
            root=Path.cwd(),
            config_path=Path("configs/releases/qwen3_vl_system_week8_v1.json"),
        )
        v2 = ReleaseSettings.load(
            root=Path.cwd(),
            config_path=Path("configs/releases/qwen3_vl_system_week8_v2.json"),
        )

        self.assertEqual(v1.dialogue_prompt_version, "week8_dialogue_first_turn_v1")
        self.assertEqual(v2.dialogue_prompt_version, "week8_dialogue_first_turn_v2")
        self.assertNotEqual(v1.release_id, v2.release_id)

    def test_v3_release_binds_strict_schema_prompt_without_mutating_v1_v2(self):
        versions = [
            ReleaseSettings.load(
                root=Path.cwd(),
                config_path=Path(
                    f"configs/releases/qwen3_vl_system_week8_v{version}.json"
                ),
            )
            for version in (1, 2, 3)
        ]

        self.assertEqual(
            [item.dialogue_prompt_version for item in versions],
            [
                "week8_dialogue_first_turn_v1",
                "week8_dialogue_first_turn_v2",
                "week8_dialogue_first_turn_v3",
            ],
        )
        self.assertEqual(len({item.release_id for item in versions}), 3)

    def test_dialogue_contract_rejects_extra_single_task_keys(self):
        self.assertTrue(
            first_attempt_is_three_key_json(
                '{"reply":"继续","state_updates":{},"tool_calls":[]}'
            )
        )
        self.assertFalse(
            first_attempt_is_three_key_json(
                '{"reply":"继续","state_updates":{},"tool_calls":[],'
                '"confidence":0.9}'
            )
        )

    def test_candidate_prompt_routes_image_dialogue_and_binds_first_user(self):
        request = DialogueRequest(
            messages=[
                DialogueTurn(role="assistant", content="承接历史"),
                DialogueTurn(role="user", content="结合新图片继续"),
            ],
            image_urls=["image.jpg"],
            state={"city": "Shanghai"},
        )

        messages = _dialogue_messages(
            request,
            prompt_version="week8_dialogue_first_turn_v1",
        )

        self.assertIn("多轮对话端点", messages[0]["content"])
        self.assertIn("顶层必须恰好", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "承接历史")
        self.assertEqual(messages[2]["content"][1]["type"], "image_url")
        self.assertEqual(len(messages), 3)

    def test_v2_appends_short_route_control_after_image_bound_user(self):
        request = DialogueRequest(
            messages=[
                DialogueTurn(role="assistant", content="承接历史"),
                DialogueTurn(role="user", content="结合新图片继续"),
            ],
            image_urls=["image.jpg"],
            state={"city": "Shanghai"},
        )

        messages = _dialogue_messages(
            request,
            prompt_version="week8_dialogue_first_turn_v2",
        )

        self.assertEqual(messages[2]["content"][1]["type"], "image_url")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("首字符必须是 {", messages[-1]["content"])
        self.assertIn('"reply":"简短直接回复"', messages[-1]["content"])
        self.assertEqual(len(messages), 4)

    def test_v2_rejects_non_object_first_character_and_corrects_once(self):
        valid = json.dumps(
            {"reply": "已按对话处理。", "state_updates": {}, "tool_calls": []},
            ensure_ascii=False,
        )
        backend = FakeBackend([f"```json\n{valid}\n```", valid])
        service = ScenarioService(
            settings(),
            backend,
            dialogue_prompt_version="week8_dialogue_first_turn_v2",
        )

        response = service.run_dialogue(
            DialogueRequest(messages=[DialogueTurn(role="user", content="继续")])
        )

        self.assertEqual(len(response.attempts), 2)
        self.assertIn("must start", response.attempts[0].error)
        self.assertIsNone(response.attempts[1].error)

    def test_v3_passes_lmfe_compatible_strict_schema_to_backend(self):
        valid = json.dumps(
            {
                "reply": "正在查询。",
                "state_updates": {"budget": 2000},
                "tool_calls": [
                    {
                        "function": "search_itinerary",
                        "arguments": {"city": "Shanghai", "days": 2},
                    }
                ],
            },
            ensure_ascii=False,
        )
        backend = FakeBackend([valid])
        service = ScenarioService(
            settings(),
            backend,
            dialogue_prompt_version="week8_dialogue_first_turn_v3",
        )

        response = service.run_dialogue(
            DialogueRequest(messages=[DialogueTurn(role="user", content="继续")])
        )

        self.assertEqual(len(response.attempts), 1)
        response_format = backend.response_formats[0]
        self.assertEqual(response_format["type"], "json_schema")
        contract = response_format["json_schema"]
        self.assertTrue(contract["strict"])
        schema = contract["schema"]
        self.assertEqual(
            schema["required"],
            ["reply", "state_updates", "tool_calls"],
        )
        self.assertFalse(schema["additionalProperties"])
        state_values = schema["properties"]["state_updates"][
            "additionalProperties"
        ]
        self.assertIsInstance(state_values, dict)
        self.assertEqual(
            state_values["type"],
            ["string", "number", "boolean", "null"],
        )
        tool_item = schema["properties"]["tool_calls"]["items"]
        self.assertEqual(tool_item["required"], ["function", "arguments"])
        self.assertIsInstance(
            tool_item["properties"]["arguments"]["additionalProperties"],
            dict,
        )

    def test_v3_keeps_strict_schema_during_model_level_correction(self):
        invalid = json.dumps(
            {
                "reply": "继续",
                "state_updates": {"tool_calls": []},
                "tool_calls": [],
            },
            ensure_ascii=False,
        )
        valid = json.dumps(
            {"reply": "继续", "state_updates": {}, "tool_calls": []},
            ensure_ascii=False,
        )
        backend = FakeBackend([invalid, valid])
        service = ScenarioService(
            settings(),
            backend,
            dialogue_prompt_version="week8_dialogue_first_turn_v3",
        )

        response = service.run_dialogue(
            DialogueRequest(messages=[DialogueTurn(role="user", content="继续")])
        )

        self.assertEqual(len(response.attempts), 2)
        self.assertIn("finite JSON scalars", response.attempts[0].error)
        self.assertEqual(
            [item["type"] for item in backend.response_formats],
            ["json_schema", "json_schema"],
        )

    def test_v1_v2_keep_unconstrained_json_object_response_format(self):
        valid = json.dumps(
            {"reply": "继续", "state_updates": {}, "tool_calls": []},
            ensure_ascii=False,
        )
        for prompt_version in (
            "week8_dialogue_first_turn_v1",
            "week8_dialogue_first_turn_v2",
        ):
            backend = FakeBackend([valid])
            service = ScenarioService(
                settings(),
                backend,
                dialogue_prompt_version=prompt_version,
            )

            service.run_dialogue(
                DialogueRequest(messages=[DialogueTurn(role="user", content="继续")])
            )

            self.assertEqual(backend.response_formats, [{"type": "json_object"}])

    def test_deterministic_contract_updates_budget_without_model(self):
        backend = FakeBackend([])
        service = ScenarioService(
            settings(),
            backend,
            dialogue_prompt_version="week8_dialogue_deterministic_v4",
            dialogue_execution_mode="deterministic_contract_v1",
            dialogue_semantic_fallback_enabled=True,
        )

        response = service.run_dialogue(
            DialogueRequest(
                messages=[
                    DialogueTurn(role="user", content="预算改成 2000 元，其余不变")
                ],
                state={"city": "Shanghai", "days": 2},
            )
        )

        self.assertEqual(
            response.state,
            {"city": "Shanghai", "days": 2, "budget": 2000},
        )
        self.assertEqual(response.tool_calls, [])
        self.assertEqual(response.attempts, [])
        self.assertEqual(response.execution_mode, "DETERMINISTIC_CONTRACT")
        self.assertEqual(response.semantic_fallback_status, "NOT_USED")
        self.assertEqual(backend.messages, [])

    def test_deterministic_contract_parses_chinese_days_and_negative_city(self):
        backend = FakeBackend([])
        service = ScenarioService(
            settings(),
            backend,
            dialogue_prompt_version="week8_dialogue_deterministic_v4",
            dialogue_execution_mode="deterministic_contract_v1",
            dialogue_semantic_fallback_enabled=True,
        )

        days = service.run_dialogue(
            DialogueRequest(
                messages=[DialogueTurn(role="user", content="把行程调整为三天")],
                state={"days": 2},
            )
        )
        negative = service.run_dialogue(
            DialogueRequest(
                messages=[DialogueTurn(role="user", content="不要改变城市，继续推荐")],
                state={"city": "Shanghai"},
            )
        )

        self.assertEqual(days.state["days"], 3)
        self.assertEqual(negative.state, {"city": "Shanghai"})
        self.assertEqual(negative.semantic_fallback_status, "NOT_USED")
        self.assertEqual(backend.messages, [])

    def test_ambiguous_update_skips_model_when_fallback_disabled(self):
        backend = FakeBackend([])
        service = ScenarioService(
            settings(),
            backend,
            dialogue_prompt_version="week8_dialogue_deterministic_v4",
            dialogue_execution_mode="deterministic_contract_v1",
            dialogue_semantic_fallback_enabled=False,
        )

        response = service.run_dialogue(
            DialogueRequest(
                messages=[DialogueTurn(role="user", content="把安排改得更有氛围一些")],
                state={"city": "Shanghai"},
            )
        )

        self.assertEqual(response.state, {"city": "Shanghai"})
        self.assertEqual(response.attempts, [])
        self.assertEqual(response.semantic_fallback_status, "NOT_USED")

    def test_ambiguous_update_uses_model_only_for_state_extraction(self):
        backend = FakeBackend(['{"state_updates":{"pace":"relaxed"}}'])
        service = ScenarioService(
            settings(),
            backend,
            dialogue_prompt_version="week8_dialogue_deterministic_v4",
            dialogue_execution_mode="deterministic_contract_v1",
            dialogue_semantic_fallback_enabled=True,
        )

        response = service.run_dialogue(
            DialogueRequest(
                messages=[DialogueTurn(role="user", content="把安排改得更有氛围一些")],
                state={"city": "Shanghai"},
            )
        )

        self.assertEqual(response.state, {"city": "Shanghai", "pace": "relaxed"})
        self.assertEqual(response.tool_calls, [])
        self.assertEqual(response.semantic_fallback_status, "SUCCEEDED")
        self.assertEqual(len(response.attempts), 1)
        self.assertEqual(backend.response_formats[0]["type"], "json_schema")
        schema = backend.response_formats[0]["json_schema"]["schema"]
        self.assertEqual(schema["required"], ["state_updates"])

    def test_semantic_fallback_failure_returns_safe_deterministic_contract(self):
        backend = FakeBackend(["not-json", '{"wrong":{}}'])
        service = ScenarioService(
            settings(),
            backend,
            dialogue_prompt_version="week8_dialogue_deterministic_v4",
            dialogue_execution_mode="deterministic_contract_v1",
            dialogue_semantic_fallback_enabled=True,
        )

        response = service.run_dialogue(
            DialogueRequest(
                messages=[DialogueTurn(role="user", content="把安排改得更有氛围一些")],
                state={"city": "Shanghai"},
            )
        )

        self.assertEqual(response.state, {"city": "Shanghai"})
        self.assertEqual(response.tool_calls, [])
        self.assertEqual(response.semantic_fallback_status, "FAILED_SAFE")
        self.assertEqual(len(response.attempts), 2)
        self.assertTrue(all(item.error for item in response.attempts))
        self.assertIn("未能可靠解析", response.reply)

    def test_relaxed_pace_update_is_deterministic_without_model(self):
        backend = FakeBackend([])
        service = ScenarioService(
            settings(),
            backend,
            dialogue_prompt_version="week8_dialogue_deterministic_v4",
            dialogue_execution_mode="deterministic_contract_v1",
            dialogue_semantic_fallback_enabled=True,
        )

        response = service.run_dialogue(
            DialogueRequest(
                messages=[DialogueTurn(role="user", content="把安排改得更松弛一些")],
                state={"city": "Shanghai", "days": 2},
            )
        )

        self.assertEqual(
            response.state,
            {"city": "Shanghai", "days": 2, "pace": "relaxed"},
        )
        self.assertEqual(response.semantic_fallback_status, "NOT_USED")
        self.assertEqual(response.attempts, [])
        self.assertEqual(backend.messages, [])

    def test_fixed_dialogue_comparison_reports_first_turn_correction(self):
        valid = json.dumps(
            {
                "reply": "已更新预算。",
                "state_updates": {"budget": 2000},
                "tool_calls": [],
            },
            ensure_ascii=False,
        )
        backend = FakeBackend(
            [
                '{"reply":"已收到"}',
                valid,
                valid,
            ]
        )
        config = {
            "dialogue": {
                "run_id": "fixed-dialogue-test",
                "profiles": [
                    {
                        "role": "current",
                        "prompt_version": "system_repair_dialogue_v1",
                        "max_new_tokens": 512,
                    },
                    {
                        "role": "candidate",
                        "prompt_version": "week8_dialogue_first_turn_v1",
                        "max_new_tokens": 384,
                    },
                ],
                "cases": [
                    {
                        "case_id": "budget",
                        "request": {
                            "messages": [{"role": "user", "content": "预算 2000"}],
                            "state": {"city": "Shanghai"},
                        },
                        "expected_state": {"city": "Shanghai", "budget": 2000},
                    }
                ],
            }
        }

        result = run_dialogue_first_turn_comparison(settings(), backend, config)

        current = result["profiles"]["current"]
        candidate = result["profiles"]["candidate"]
        self.assertEqual(current["first_turn_format_compliance"], 0.0)
        self.assertEqual(current["correction_trigger_rate"], 1.0)
        self.assertIn(
            "must contain exactly",
            result["records"][0]["response"]["attempts"][0]["error"],
        )
        self.assertEqual(candidate["first_turn_format_compliance"], 1.0)
        self.assertEqual(candidate["correction_trigger_rate"], 0.0)
        self.assertEqual(candidate["context_recall"], 1.0)
        self.assertEqual(candidate["context_state_value_accuracy"], 1.0)
        self.assertEqual(backend.max_new_tokens, [512, 512, 384])

    def test_model_attempt_records_measured_tokens(self):
        backend = FakeUsageBackend([json.dumps(PRODUCT_OUTPUT, ensure_ascii=False)])
        service = ScenarioService(settings(), backend)

        response = service.run_task(
            "image_product_search",
            TaskRequest(image_urls=["image.jpg"]),
        )

        self.assertEqual(response.attempts[0].input_tokens, 10)
        self.assertEqual(response.attempts[0].output_tokens, 5)

    def test_product_latency_reports_p50_p95_tokens_and_failure_rate(self):
        outputs = [json.dumps(PRODUCT_OUTPUT, ensure_ascii=False)] * 8
        backend = FakeUsageBackend(outputs)
        config = {
            "product_latency": {
                "run_id": "fixed-product-latency-test",
                "image": "data/samples/images/cafe_001.jpg",
                "text_context": None,
                "warmup_runs": 1,
                "measured_runs": 3,
                "profiles": [
                    {
                        "role": "current",
                        "max_new_tokens": 512,
                        "processor_cache_entries": 0,
                    },
                    {
                        "role": "bounded_output",
                        "max_new_tokens": 384,
                        "processor_cache_entries": 4,
                    },
                ],
            }
        }

        result = run_product_latency_benchmark(settings(), backend, config)

        for role in ("current", "bounded_output"):
            metrics = result["profiles"][role]["metrics"]
            self.assertEqual(metrics["failure_rate"], 0.0)
            self.assertEqual(metrics["schema_pass_rate"], 1.0)
            self.assertIsNotNone(metrics["latency_ms_p50"])
            self.assertIsNotNone(metrics["latency_ms_p95"])
            self.assertEqual(metrics["input_tokens_total"], 30)
            self.assertEqual(metrics["output_tokens_total"], 15)
            self.assertEqual(
                result["profiles"][role]["quality_consistency"][
                    "exact_result_match_rate"
                ],
                1.0,
            )
        self.assertEqual(backend.max_new_tokens[:4], [512] * 4)
        self.assertEqual(backend.max_new_tokens[4:], [384] * 4)
        self.assertEqual(backend.cache_configurations, [0, 4])
        self.assertEqual(
            result["profiles"]["bounded_output"]["processor_cache"],
            {"max_entries": 4},
        )

    def test_tracked_runtime_config_is_valid(self):
        v1 = load_runtime_benchmark_config(
            Path.cwd(),
            Path("configs/week8/runtime_optimization_v1.json"),
        )
        v2 = load_runtime_benchmark_config(
            Path.cwd(),
            Path("configs/week8/runtime_optimization_v2.json"),
        )
        v3 = load_runtime_benchmark_config(
            Path.cwd(),
            Path("configs/week8/runtime_optimization_v3.json"),
        )
        v5 = load_runtime_benchmark_config(
            Path.cwd(),
            Path("configs/week8/runtime_optimization_v5.json"),
        )
        v6 = load_runtime_benchmark_config(
            Path.cwd(),
            Path("configs/week8/runtime_optimization_v6.json"),
        )
        v7 = load_runtime_benchmark_config(
            Path.cwd(),
            Path("configs/week8/runtime_optimization_v7.json"),
        )

        self.assertEqual(len(v1["dialogue"]["cases"]), 4)
        self.assertEqual(v1["product_latency"]["measured_runs"], 5)
        self.assertEqual(v1["dialogue"]["cases"], v2["dialogue"]["cases"])
        self.assertEqual(v2["dialogue"]["cases"], v3["dialogue"]["cases"])
        self.assertEqual(
            v3["dialogue"]["cases"],
            v5["dialogue"]["cases"][:4],
        )
        self.assertEqual(
            v5["dialogue"]["cases"][4]["case_id"],
            "dialogue-semantic-fallback-probe",
        )
        self.assertEqual(
            v2["dialogue"]["profiles"][1]["prompt_version"],
            "week8_dialogue_first_turn_v2",
        )
        self.assertEqual(
            v3["dialogue"]["profiles"][1]["prompt_version"],
            "week8_dialogue_first_turn_v3",
        )
        self.assertEqual(
            v5["dialogue"]["profiles"][1]["execution_mode"],
            "deterministic_contract_v1",
        )
        self.assertEqual(
            v6["product_latency"]["profiles"][1]["processor_cache_entries"],
            4,
        )
        self.assertEqual(
            v6["product_latency"]["profiles"][1]["visual_max_pixels"],
            200704,
        )
        self.assertEqual(
            v7["dialogue"]["cases"][-1]["expected_state"]["pace"],
            "relaxed",
        )

    def test_v6_release_binds_bounded_visual_processor_cache(self):
        release = ReleaseSettings.load(
            root=Path.cwd(),
            config_path=Path("configs/releases/qwen3_vl_system_week8_v6.json"),
        )

        self.assertEqual(release.max_new_tokens_by_scenario["image_product_search"], 384)
        self.assertEqual(release.visual_max_pixels, 200704)
        self.assertEqual(release.processor_cache_max_entries, 8)

    def test_v7_release_rejects_unproven_cache_and_visual_cap(self):
        release = ReleaseSettings.load(
            root=Path.cwd(),
            config_path=Path("configs/releases/qwen3_vl_system_week8_v7.json"),
        )

        self.assertEqual(release.max_new_tokens_by_scenario["image_product_search"], 384)
        self.assertIsNone(release.visual_max_pixels)
        self.assertEqual(release.processor_cache_max_entries, 0)

    def test_v5_fixed_comparison_scores_code_assembled_contract(self):
        release = ReleaseSettings.load(
            root=Path.cwd(),
            config_path=Path("configs/releases/qwen3_vl_system_week8_v5.json"),
        )
        config = load_runtime_benchmark_config(
            Path.cwd(),
            Path("configs/week8/runtime_optimization_v5.json"),
        )
        current_outputs = [
            json.dumps(
                {
                    "reply": "继续处理。",
                    "state_updates": {},
                    "tool_calls": [],
                },
                ensure_ascii=False,
            )
        ] * 5
        backend = FakeBackend(
            [
                *current_outputs,
                '{"state_updates":{"pace":"relaxed"}}',
            ]
        )

        result = run_dialogue_first_turn_comparison(release, backend, config)

        candidate = result["profiles"]["candidate"]
        self.assertEqual(candidate["first_turn_format_compliance"], 1.0)
        self.assertEqual(candidate["correction_trigger_rate"], 0.0)
        self.assertEqual(candidate["context_recall"], 1.0)
        self.assertEqual(candidate["context_state_value_accuracy"], 1.0)
        self.assertEqual(candidate["context_state_precision"], 1.0)
        self.assertEqual(candidate["context_state_exact_rate"], 1.0)
        self.assertEqual(candidate["unexpected_state_key_count"], 0)
        self.assertEqual(candidate["failure_rate"], 0.0)
        self.assertEqual(candidate["deterministic_route_rate"], 1.0)
        self.assertEqual(candidate["semantic_fallback_rate"], 0.0)
        self.assertEqual(candidate["semantic_fallback_safe_failure_rate"], 0.0)
        self.assertEqual(len(backend.messages), 5)

    def test_spartan_job_allows_versioned_config_overrides(self):
        script = Path("scripts/spartan/week8_runtime_optimization.sbatch").read_text(
            encoding="utf-8"
        )

        self.assertIn('TRIP_RELEASE_CONFIG="${TRIP_RELEASE_CONFIG:-', script)
        self.assertIn('TRIP_RUNTIME_CONFIG="${TRIP_RUNTIME_CONFIG:-', script)
        self.assertIn('--benchmark-config "${TRIP_RUNTIME_CONFIG}"', script)

        smoke = Path("scripts/spartan/system_release_model_smoke.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn('release_config="${TRIP_RELEASE_CONFIG:-', smoke)
        self.assertIn('--release-config "${release_config}"', smoke)
        self.assertIn('--image "${smoke_image}"', smoke)


if __name__ == "__main__":
    unittest.main()

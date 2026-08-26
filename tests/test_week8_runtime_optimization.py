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

    def ready(self):
        return True, "ok"

    def generate(self, messages, *, response_format, max_new_tokens):
        self.messages.append(messages)
        self.max_new_tokens.append(max_new_tokens)
        return self.outputs.pop(0)


class FakeUsageBackend(FakeBackend):
    def generate_with_usage(self, messages, *, response_format, max_new_tokens):
        self.messages.append(messages)
        self.max_new_tokens.append(max_new_tokens)
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
                    {"role": "current", "max_new_tokens": 512},
                    {"role": "bounded_output", "max_new_tokens": 384},
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

    def test_tracked_runtime_config_is_valid(self):
        config = load_runtime_benchmark_config(
            Path.cwd(),
            Path("configs/week8/runtime_optimization_v1.json"),
        )

        self.assertEqual(len(config["dialogue"]["cases"]), 4)
        self.assertEqual(config["product_latency"]["measured_runs"], 5)


if __name__ == "__main__":
    unittest.main()

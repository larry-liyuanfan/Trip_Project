import hashlib
import json
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from src.api.routes import dialogue, readiness, visual_search
from src.evaluation.prompting import render_standard_prompt
from src.inference.client import OpenAICompatibleClient
from src.inference.schemas import (
    DialogueRequest,
    DialogueTurn,
    ImageUnderstandingRequest,
    TaskRequest,
    VisualSearchRequest,
)
from src.inference.system_runtime import (
    ModelGenerationError,
    ReleaseSettings,
    ScenarioService,
    TransformersPeftBackend,
    _transformers_messages,
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
    def __init__(self, outputs, ready=True):
        self.outputs = list(outputs)
        self.ready_value = ready
        self.messages = []

    def ready(self):
        return self.ready_value, "ok" if self.ready_value else "backend unavailable"

    def generate(self, messages, *, response_format, max_new_tokens):
        self.messages.append(messages)
        return self.outputs.pop(0)


class FakeReadyService:
    def ready(self):
        return {
            "status": "ready",
            "release_id": "test-release",
            "checks": {},
        }


class FakeVisualSearchService:
    def __init__(self, results=None):
        self.results = results or []
        self.call = None

    def search(self, image_path, *, top_k, filters):
        self.call = {"image_path": image_path, "top_k": top_k, "filters": filters}
        return self.results


def settings(adapter_path=None, adapter_sha="0" * 64):
    return ReleaseSettings(
        root=Path.cwd(),
        release_id="test-release",
        base_model="Qwen/Qwen3-VL-8B-Instruct",
        base_revision="revision",
        backend_name="transformers-peft",
        adapter_name="test-adapter",
        adapter_path=adapter_path,
        adapter_model_sha256=adapter_sha,
        prompt_versions={
            "image_product_search": "standardized_v2",
            "after_sales": "standardized_v2",
            "itinerary_planning": "standardized_v4",
        },
        schema_versions={
            "image_product_search": "v1",
            "after_sales": "v1",
            "itinerary_planning": "v2",
        },
        max_new_tokens=512,
        max_schema_retries=1,
    )


class SystemRuntimeTest(unittest.TestCase):
    def test_release_uses_prompt_pilot_winners(self):
        release = ReleaseSettings.load(root=Path.cwd())
        expected = {
            "image_product_search": (
                "system_repair_product_compact_v1",
                "逐个检查必填键",
            ),
            "after_sales": (
                "system_repair_after_sales_evidence_v1",
                "核对证据来源",
            ),
            "itinerary_planning": (
                "system_repair_itinerary_current_v1",
                "保留未知字段",
            ),
        }

        for scenario, (version, expected_text) in expected.items():
            context = {
                "images": [{"path": "sample.jpg"}],
                "text_constraints": "行程共2天" if scenario == "itinerary_planning" else None,
            }
            rendered = render_standard_prompt(Path.cwd(), scenario, context, version)
            self.assertEqual(release.prompt_versions[scenario], version)
            self.assertIn(expected_text, rendered["layers"]["task_instruction"])

    def test_transformers_backend_preserves_underlying_generation_error(self):
        backend = TransformersPeftBackend(settings())
        backend._model = types.SimpleNamespace(device="cpu")
        backend._processor = types.SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("diagnostic failure")
            )
        )
        with self.assertRaisesRegex(
            ModelGenerationError,
            "ValueError: diagnostic failure",
        ):
            backend.generate_with_usage(
                [{"role": "user", "content": "test"}],
                response_format={"type": "json_object"},
                max_new_tokens=8,
            )

    def test_transformers_messages_convert_openai_image_blocks(self):
        messages = [
            {
                "role": "system",
                "content": "system instruction",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                    {"type": "text", "text": "inspect"},
                ],
            }
        ]

        converted = _transformers_messages(messages)

        self.assertEqual(
            converted[0]["content"],
            [{"type": "text", "text": "system instruction"}],
        )
        self.assertEqual(
            converted[1]["content"][0],
            {"type": "image", "image": "https://example.com/image.jpg"},
        )
        self.assertEqual(messages[0]["content"], "system instruction")
        self.assertEqual(messages[1]["content"][0]["type"], "image_url")

    def test_valid_first_output_is_returned_without_retry(self):
        backend = FakeBackend([json.dumps(PRODUCT_OUTPUT, ensure_ascii=False)])
        service = ScenarioService(settings(), backend)

        result = service.run_task(
            "image_product_search",
            TaskRequest(image_urls=["https://example.com/product.jpg"]),
        )

        self.assertTrue(result.schema_valid)
        self.assertEqual(result.result["business_category"], "restaurant")
        self.assertEqual(len(result.attempts), 1)
        self.assertIsNone(result.attempts[0].error)

    def test_invalid_output_gets_one_model_level_correction(self):
        backend = FakeBackend(
            [
                '{"business_category":"restaurant"}',
                json.dumps(PRODUCT_OUTPUT, ensure_ascii=False),
            ]
        )
        service = ScenarioService(settings(), backend)

        result = service.run_task(
            "image_product_search",
            TaskRequest(image_urls=["https://example.com/product.jpg"]),
        )

        self.assertEqual(len(result.attempts), 2)
        self.assertIn("missing required properties", result.attempts[0].error)
        self.assertIsNone(result.attempts[1].error)
        retry_text = backend.messages[1][-1]["content"]
        self.assertIn("不得解释", retry_text)
        self.assertIn("全部字段", retry_text)
        self.assertIn("minItems/maxItems", retry_text)
        self.assertIn("完整闭合", retry_text)

    def test_two_invalid_outputs_fail_closed(self):
        backend = FakeBackend(["not-json", "still-not-json"])
        service = ScenarioService(settings(), backend)

        with self.assertRaisesRegex(ModelGenerationError, "after 2 attempts"):
            service.run_task(
                "image_product_search",
                TaskRequest(image_urls=["https://example.com/product.jpg"]),
            )

    def test_readiness_verifies_adapter_file_hash(self):
        with TemporaryDirectory() as tmpdir:
            adapter = Path(tmpdir)
            model_bytes = b"adapter"
            (adapter / "adapter_model.safetensors").write_bytes(model_bytes)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            expected = hashlib.sha256(model_bytes).hexdigest()
            service = ScenarioService(
                settings(adapter.resolve(), expected),
                FakeBackend([], ready=True),
            )

            result = service.ready()

        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["checks"]["adapter"]["ok"])

    def test_missing_adapter_is_not_ready(self):
        service = ScenarioService(settings(), FakeBackend([], ready=True))

        result = service.ready()

        self.assertEqual(result["status"], "not_ready")
        self.assertFalse(result["checks"]["adapter"]["ok"])

    def test_dialogue_state_updates_are_explicit(self):
        output = json.dumps(
            {
                "reply": "已将预算更新为 2000 元。",
                "state_updates": {"budget": 2000},
                "tool_calls": [],
            },
            ensure_ascii=False,
        )
        service = ScenarioService(settings(), FakeBackend([output]))

        result = service.run_dialogue(
            DialogueRequest(
                messages=[DialogueTurn(role="user", content="预算改成两千")],
                state={"city": "上海"},
            )
        )

        self.assertEqual(result.quality_tier, "DIALOGUE_BETA")
        self.assertEqual(result.state, {"city": "上海", "budget": 2000})

    def test_dialogue_endpoint_is_disabled_by_default(self):
        request = DialogueRequest(
            messages=[DialogueTurn(role="user", content="继续规划")]
        )
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(HTTPException) as raised:
                dialogue(request)

        self.assertEqual(raised.exception.status_code, 404)

    def test_legacy_client_disables_fallback_in_production(self):
        with patch.dict("os.environ", {"APP_ENV": "production"}, clear=True):
            client = OpenAICompatibleClient()

        self.assertFalse(client.fallback_enabled)
        self.assertEqual(client.model_name, "Qwen/Qwen3-VL-8B-Instruct")

    def test_legacy_client_fails_closed_without_production_endpoint(self):
        request = ImageUnderstandingRequest(image_urls=["https://example.com/image.jpg"])
        with patch.dict("os.environ", {"APP_ENV": "production"}, clear=True):
            client = OpenAICompatibleClient()
            with self.assertRaisesRegex(RuntimeError, "endpoint is not configured"):
                client.understand_images(request)

    def test_readiness_reports_retrieval_initialization_failure(self):
        with patch("src.api.routes.get_scenario_service", return_value=FakeReadyService()):
            with patch(
                "src.api.routes.get_visual_search_service",
                side_effect=RuntimeError("Milvus config missing"),
            ):
                with self.assertRaises(HTTPException) as raised:
                    readiness()

        self.assertEqual(raised.exception.status_code, 503)
        self.assertFalse(raised.exception.detail["checks"]["clip"]["ok"])
        self.assertFalse(raised.exception.detail["checks"]["milvus"]["ok"])

    def test_production_visual_search_uses_clip_milvus_without_fallback(self):
        service = FakeVisualSearchService(
            [{"image_id": "photo-1", "score": 0.9}]
        )
        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "query.jpg"
            image_path.write_bytes(b"image")
            request = VisualSearchRequest(
                image_urls=[str(image_path)],
                city="Shanghai",
                top_k=3,
            )
            with patch.dict("os.environ", {"APP_ENV": "production"}, clear=True):
                with patch(
                    "src.api.routes.get_visual_search_service", return_value=service
                ):
                    result = visual_search(request)

        self.assertEqual(result["retrieval_mode"], "clip_milvus_hnsw_cosine")
        self.assertEqual(result["results"][0]["image_id"], "photo-1")
        self.assertEqual(service.call["filters"], {"city": "Shanghai"})

    def test_production_visual_search_rejects_remote_url(self):
        request = VisualSearchRequest(
            image_urls=["https://example.com/query.jpg"],
            top_k=3,
        )
        with patch.dict("os.environ", {"APP_ENV": "production"}, clear=True):
            with self.assertRaises(HTTPException) as raised:
                visual_search(request)

        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()

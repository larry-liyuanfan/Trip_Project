import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.repair_app import create_repair_app
from src.inference.system_runtime import GenerationResult, ReleaseSettings


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


class FakeRepairBackend:
    def __init__(self) -> None:
        self.messages = []

    def ready(self):
        return True, "ok"

    def generate_with_usage(self, messages, *, response_format, max_new_tokens):
        self.messages.append((messages, response_format, max_new_tokens))
        return GenerationResult(
            content=json.dumps(PRODUCT_OUTPUT, ensure_ascii=False),
            input_tokens=17,
            output_tokens=11,
        )

    def generate(self, messages, *, response_format, max_new_tokens):
        return self.generate_with_usage(
            messages,
            response_format=response_format,
            max_new_tokens=max_new_tokens,
        ).content


def repair_settings() -> ReleaseSettings:
    return ReleaseSettings(
        root=Path.cwd(),
        release_id="repair-test",
        base_model="Qwen/Qwen3-VL-8B-Instruct",
        base_revision="revision",
        backend_name="transformers-peft",
        adapter_name="repair-adapter",
        adapter_path=None,
        adapter_model_sha256="0" * 64,
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


class RepairAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeRepairBackend()
        self.client = TestClient(
            create_repair_app(settings=repair_settings(), backend=self.backend)
        )

    def test_model_registry_waits_for_backend_readiness(self):
        response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "repair-adapter")

    def test_completion_reports_measured_token_usage(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "repair-adapter",
                "messages": [{"role": "user", "content": "分析图片"}],
                "max_tokens": 100,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["usage"]["total_tokens"], 28)
        self.assertEqual(
            response.json()["choices"][0]["message"]["role"], "assistant"
        )

    def test_completion_rejects_wrong_model_identity(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "wrong-model",
                "messages": [{"role": "user", "content": "分析图片"}],
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_task_endpoint_uses_same_backend_and_validates_schema(self):
        response = self.client.post(
            "/v1/tasks/image-product-search",
            json={"image_urls": ["https://example.com/product.jpg"]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["schema_valid"])
        self.assertEqual(
            response.json()["result"]["business_category"], "restaurant"
        )


if __name__ == "__main__":
    unittest.main()

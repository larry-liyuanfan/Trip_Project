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
    _prepared_input_cache_eligible,
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

ITINERARY_OUTPUT = {
    "style_preferences": [],
    "hard_constraints": ["2 days"],
    "soft_constraints": ["public transport"],
    "required_itinerary_elements": ["daily_schedule"],
    "itinerary": [
        {
            "day_index": 1,
            "date": None,
            "summary": "Day 1",
            "activities": [
                {
                    "start_time": None,
                    "end_time": None,
                    "place_name": None,
                    "activity": "Explore",
                    "transport": "public transport",
                    "source_evidence": [],
                }
            ],
        }
    ],
    "constraint_check": [],
    "observed_evidence": [],
    "unknown_fields": [],
    "confidence": None,
}


class FakeBackend:
    def __init__(self, outputs, ready=True):
        self.outputs = list(outputs)
        self.ready_value = ready
        self.messages = []
        self.response_formats = []

    def ready(self):
        return self.ready_value, "ok" if self.ready_value else "backend unavailable"

    def generate(self, messages, *, response_format, max_new_tokens):
        self.messages.append(messages)
        self.response_formats.append(response_format)
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
        max_new_tokens_by_scenario={
            "image_product_search": 512,
            "after_sales": 512,
            "itinerary_planning": 1024,
        },
        max_schema_retries=1,
    )


class SystemRuntimeTest(unittest.TestCase):
    def test_prepared_input_cache_rejects_mutable_remote_media(self):
        remote = _transformers_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.jpg"},
                        }
                    ],
                }
            ]
        )
        inline = _transformers_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,AA=="},
                        }
                    ],
                }
            ]
        )

        self.assertFalse(_prepared_input_cache_eligible(remote))
        self.assertTrue(_prepared_input_cache_eligible(inline))

    def test_transformers_backend_reuses_model_device_inputs(self):
        class Vector:
            def __init__(self, values):
                self.values = list(values)

            def __len__(self):
                return len(self.values)

            def __getitem__(self, item):
                selected = self.values[item]
                return Vector(selected) if isinstance(item, slice) else selected

            @property
            def shape(self):
                return (len(self.values),)

        class Batch:
            def __init__(self, rows):
                self.rows = rows
                self.transfer_calls = 0

            def __iter__(self):
                return iter(self.rows)

            def to(self, _device):
                self.transfer_calls += 1
                return self

            @property
            def shape(self):
                return (len(self.rows), len(self.rows[0]))

        class InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        class Processor:
            tokenizer = None
            image_processor = types.SimpleNamespace(max_pixels=1024)

            def __init__(self):
                self.calls = 0
                self.batch = Batch([Vector([1, 2, 3])])

            def apply_chat_template(self, *_args, **_kwargs):
                self.calls += 1
                return {"input_ids": self.batch}

            def batch_decode(self, *_args, **_kwargs):
                return ["prepared"]

        processor = Processor()
        backend = TransformersPeftBackend(settings())
        backend.configure_prepared_input_cache(1)
        backend._processor = processor
        backend._model = types.SimpleNamespace(
            parameters=lambda: iter([types.SimpleNamespace(device="cuda:0")]),
            generate=lambda **_kwargs: [Vector([1, 2, 3, 4])],
        )
        backend._torch = types.SimpleNamespace(
            inference_mode=lambda: InferenceMode()
        )

        for _ in range(2):
            result = backend.generate_with_usage(
                [{"role": "user", "content": "same request"}],
                response_format=None,
                max_new_tokens=8,
            )

        self.assertEqual(result.content, "prepared")
        self.assertEqual(processor.calls, 1)
        self.assertEqual(processor.batch.transfer_calls, 1)
        self.assertEqual(backend.prepared_input_cache_snapshot()["hits"], 1)

    def test_transformers_backend_reuses_bounded_processor_outputs(self):
        class Vector:
            def __init__(self, values):
                self.values = list(values)

            def __len__(self):
                return len(self.values)

            def __getitem__(self, item):
                selected = self.values[item]
                return Vector(selected) if isinstance(item, slice) else selected

            @property
            def shape(self):
                return (len(self.values),)

        class Batch:
            def __init__(self, rows):
                self.rows = rows

            def __iter__(self):
                return iter(self.rows)

            def to(self, _device):
                return self

            @property
            def shape(self):
                return (len(self.rows), len(self.rows[0]))

        class InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        class Processor:
            tokenizer = None
            image_processor = types.SimpleNamespace(max_pixels=1024)

            def __init__(self):
                self.calls = 0

            def apply_chat_template(self, *_args, **_kwargs):
                self.calls += 1
                return {"input_ids": Batch([Vector([1, 2, 3])])}

            def batch_decode(self, *_args, **_kwargs):
                return ["cached"]

        processor = Processor()
        backend = TransformersPeftBackend(settings())
        backend.configure_processor_cache(2)
        backend._processor = processor
        backend._model = types.SimpleNamespace(
            parameters=lambda: iter([types.SimpleNamespace(device="cpu")]),
            generate=lambda **_kwargs: [Vector([1, 2, 3, 4])],
        )
        backend._torch = types.SimpleNamespace(
            inference_mode=lambda: InferenceMode()
        )

        for _ in range(2):
            result = backend.generate_with_usage(
                [{"role": "user", "content": "same request"}],
                response_format=None,
                max_new_tokens=8,
            )

        self.assertEqual(result.content, "cached")
        self.assertEqual(processor.calls, 1)
        self.assertEqual(
            backend.processor_cache_snapshot(),
            {
                "max_entries": 2,
                "entries": 1,
                "hits": 1,
                "misses": 1,
                "hit_rate": 0.5,
            },
        )

    def test_release_uses_prompt_pilot_winners(self):
        release = ReleaseSettings.load(root=Path.cwd())
        expected = {
            "image_product_search": (
                "system_repair_product_compact_v3",
                "八个顶层键",
            ),
            "after_sales": (
                "system_repair_after_sales_evidence_v3",
                "ocr_text",
            ),
            "itinerary_planning": (
                "system_repair_itinerary_structured_v4",
                "constraint_check 必须是对象数组",
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
        self.assertEqual(
            release.max_new_tokens_by_scenario,
            {
                "image_product_search": 512,
                "after_sales": 512,
                "itinerary_planning": 1024,
            },
        )

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
        self.assertEqual(len(backend.messages[1]), len(backend.messages[0]) + 1)
        self.assertNotIn("assistant", [item["role"] for item in backend.messages[1]])
        self.assertEqual(backend.response_formats[0], {"type": "json_object"})
        self.assertEqual(backend.response_formats[1]["type"], "json_schema")
        self.assertEqual(
            backend.response_formats[1]["json_schema"]["name"],
            "image_product_search_v1",
        )
        self.assertIn(
            "business_category",
            backend.response_formats[1]["json_schema"]["schema"]["required"],
        )

    def test_two_invalid_outputs_fail_closed(self):
        backend = FakeBackend(["not-json", "still-not-json"])
        service = ScenarioService(settings(), backend)

        with self.assertRaisesRegex(ModelGenerationError, "after 2 attempts") as raised:
            service.run_task(
                "image_product_search",
                TaskRequest(image_urls=["https://example.com/product.jpg"]),
            )

        self.assertEqual(len(raised.exception.attempts), 2)
        self.assertEqual(raised.exception.attempts[0].raw_output, "not-json")
        self.assertEqual(raised.exception.attempts[1].raw_output, "still-not-json")

    def test_itinerary_retry_uses_minimal_nine_key_contract(self):
        backend = FakeBackend(
            [
                "not-json",
                json.dumps(ITINERARY_OUTPUT, ensure_ascii=False),
            ]
        )
        service = ScenarioService(settings(), backend)

        result = service.run_task(
            "itinerary_planning",
            TaskRequest(
                image_urls=["https://example.com/trip.jpg"],
                text_context="2 days, public transport",
            ),
        )

        self.assertTrue(result.schema_valid)
        retry_text = backend.messages[1][-1]["content"]
        self.assertIn("行程纠错时必须使用以下九键骨架", retry_text)
        self.assertIn('"itinerary"', retry_text)
        self.assertIn('"constraint_check":[]', retry_text)
        self.assertIn("不得在任何 ] 或 } 后插入自然语言", retry_text)
        correction_schema = backend.response_formats[1]["json_schema"]["schema"]
        itinerary_schema = correction_schema["properties"]["itinerary"]
        self.assertEqual(itinerary_schema["maxItems"], 4)
        self.assertEqual(
            itinerary_schema["items"]["properties"]["activities"]["maxItems"],
            2,
        )
        self.assertEqual(
            correction_schema["properties"]["constraint_check"]["maxItems"],
            12,
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
        self.assertTrue(result["checks"]["prompt_schema_contracts"]["ok"])

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

    def test_dialogue_retry_uses_three_key_contract_without_lmfe(self):
        invalid = json.dumps({"confidence": 0.7, "scene_tags": ["attraction"]})
        valid = json.dumps(
            {
                "reply": "已按图片风格保留安静的文化体验。",
                "state_updates": {},
                "tool_calls": [],
            },
            ensure_ascii=False,
        )
        backend = FakeBackend([invalid, valid])
        service = ScenarioService(settings(), backend)

        result = service.run_dialogue(
            DialogueRequest(
                messages=[DialogueTurn(role="user", content="继续规划")]
            )
        )

        self.assertEqual(len(result.attempts), 2)
        retry_text = backend.messages[1][-1]["content"]
        self.assertIn("对话纠错必须使用以下三键骨架", retry_text)
        self.assertIn('"reply":"简短直接回复"', retry_text)
        self.assertEqual(backend.response_formats[1], {"type": "json_object"})

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

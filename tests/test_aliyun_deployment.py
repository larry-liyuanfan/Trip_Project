import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import requests

from src.api.routes import health
from src.inference.client import OpenAICompatibleClient
from src.inference.schemas import ImageUnderstandingRequest


class AliyunDeploymentTest(unittest.TestCase):
    def test_versioned_base_url_and_bearer_key_are_used(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"image_summary":"ok","structured_info":{},'
                            '"confidence":0.9}'
                        )
                    }
                }
            ]
        }
        client = OpenAICompatibleClient(
            base_url="https://workspace.example.com/compatible-mode/v1",
            model_name="qwen3.7-plus",
            api_key="test-key",
        )

        with patch("src.inference.client.requests.post", return_value=response) as post:
            result = client.understand_images(
                ImageUnderstandingRequest(image_urls=["https://example.com/image.jpg"])
            )

        self.assertEqual(result.image_summary, "ok")
        self.assertEqual(
            post.call_args.args[0],
            "https://workspace.example.com/compatible-mode/v1/chat/completions",
        )
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "qwen3.7-plus")
        self.assertFalse(post.call_args.kwargs["json"]["enable_thinking"])

    def test_key_can_be_read_from_mounted_secret_file(self):
        with TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "api_key"
            key_path.write_text("mounted-key\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"MODEL_API_KEY_FILE": str(key_path)},
                clear=True,
            ):
                client = OpenAICompatibleClient()

        self.assertEqual(client.api_key, "mounted-key")

    def test_cloud_mode_does_not_hide_request_failures_with_fallback(self):
        with patch.dict(
            os.environ,
            {"MODEL_FALLBACK_ENABLED": "false"},
            clear=True,
        ):
            client = OpenAICompatibleClient(
                base_url="https://workspace.example.com/compatible-mode/v1",
                model_name="qwen3.7-plus",
                api_key="test-key",
            )
            with patch(
                "src.inference.client.requests.post",
                side_effect=requests.Timeout("timeout"),
            ):
                with self.assertRaisesRegex(RuntimeError, "model request failed"):
                    client.understand_images(
                        ImageUnderstandingRequest(
                            image_urls=["https://example.com/image.jpg"]
                        )
                    )

    def test_health_uses_generic_cloud_model_settings(self):
        with patch.dict(
            os.environ,
            {
                "MODEL_NAME": "qwen3.7-plus",
                "MODEL_PROVIDER": "aliyun-model-studio",
            },
            clear=True,
        ):
            body = health()

        self.assertEqual(body["model"], "qwen3.7-plus")
        self.assertEqual(body["backend"], "aliyun-model-studio")

    def test_cloud_compose_mounts_secret_and_excludes_local_vllm(self):
        compose = Path("docker/aliyun/docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("MODEL_API_KEY_FILE: /run/secrets/dashscope_api_key", compose)
        self.assertIn("MODEL_FALLBACK_ENABLED: \"false\"", compose)
        self.assertIn("qwen3.7-plus", compose)
        self.assertIn("${API_BIND_ADDRESS:-127.0.0.1}", compose)
        self.assertNotIn("DASHSCOPE_API_KEY:", compose)
        self.assertNotIn("vllm/vllm-openai", compose)

    def test_docker_context_excludes_secrets_and_local_cloud_env(self):
        dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

        self.assertIn("secrets/", dockerignore)
        self.assertIn("docker/**/.env", dockerignore)


if __name__ == "__main__":
    unittest.main()

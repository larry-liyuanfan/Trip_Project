import base64
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.training.week6_final_evaluation import (
    adapter_identity,
    processor_messages,
)
from src.training.week6_qlora import Week6TrainingError


ROOT = Path(__file__).resolve().parents[1]


class Week6FinalEvaluationTests(unittest.TestCase):
    def test_config_locks_frozen_dataset_and_serial_generation(self):
        config = (ROOT / "configs/evaluation_week6_final.yaml").read_text(
            encoding="utf-8"
        )
        inference = (ROOT / "configs/inference_week6_final.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("dataset_version: week3_evaluation_v2", config)
        self.assertIn("full_concurrency: 1", config)
        self.assertIn("temperature: 0.0", inference)
        self.assertIn("max_tokens: 2048", inference)

    def test_adapter_identity_requires_exact_model_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            model = b"adapter"
            (adapter / "adapter_model.safetensors").write_bytes(model)
            expected = hashlib.sha256(model).hexdigest()
            identity = adapter_identity(adapter, expected)
            self.assertEqual(
                identity["adapter_file_sha256"]["adapter_model.safetensors"],
                expected,
            )
            with self.assertRaisesRegex(Week6TrainingError, "does not match"):
                adapter_identity(adapter, "0" * 64)

    def test_processor_messages_decodes_normalized_image(self):
        image = Image.new("RGB", (2, 2), "red")
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        data_uri = "data:image/png;base64," + base64.b64encode(
            payload.getvalue()
        ).decode("ascii")
        messages = processor_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ]
        )
        self.assertEqual(messages[0]["content"][0]["text"], "inspect")
        self.assertEqual(messages[0]["content"][1]["type"], "image")
        self.assertEqual(messages[0]["content"][1]["image"].size, (2, 2))


if __name__ == "__main__":
    unittest.main()

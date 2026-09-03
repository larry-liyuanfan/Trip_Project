"""Tests for targeted v4 training inputs that do not require model loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_targeted_exploration_adapter_v4 import build_training_messages


class TargetedExplorationTrainingV4Tests(unittest.TestCase):
    def test_product_messages_use_typed_content_on_both_sides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = {
                "scenario": "product",
                "prompt": "inspect",
                "image_relative_path": "image.ppm",
                "gold": {"business_category": "hotel", "unknown_fields": []},
            }
            messages = build_training_messages(row, root)
        self.assertEqual([item["role"] for item in messages], ["user", "assistant"])
        self.assertEqual(messages[0]["content"][0]["type"], "image")
        self.assertEqual(messages[1]["content"][0]["type"], "text")
        self.assertEqual(
            json.loads(messages[1]["content"][0]["text"]),
            {"business_category": "hotel"},
        )

    def test_dialogue_message_embeds_case_without_image(self) -> None:
        row = {
            "scenario": "dialogue",
            "prompt": "route",
            "dialogue": "user asks for parking",
            "gold": {"route": "search"},
        }
        messages = build_training_messages(row, Path("."))
        self.assertEqual([item["type"] for item in messages[0]["content"]], ["text"])
        self.assertIn("CASE:\nuser asks for parking", messages[0]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()

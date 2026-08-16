from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.training.week6_data import (
    _normalize_training_messages,
    _split_name,
    _validate_silver_label,
)
from src.training.week6_qlora import Week6TrainingError, load_training_config


ROOT = Path(__file__).resolve().parents[1]


class Week6DataLockTests(unittest.TestCase):
    def test_split_is_deterministic(self) -> None:
        first = _split_name("sample-1", seed=20260814, validation_fraction=0.05)
        second = _split_name("sample-1", seed=20260814, validation_fraction=0.05)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "validation"})

    def test_split_changes_only_through_explicit_inputs(self) -> None:
        values = {
            _split_name(f"sample-{index}", seed=20260814, validation_fraction=0.05)
            for index in range(1000)
        }
        self.assertEqual(values, {"train", "validation"})

    def test_final_config_requires_all_300_human_revisions(self) -> None:
        config = load_training_config(
            ROOT / "configs/week6/qwen3_vl_8b_qlora_final300_v4.json"
        )
        self.assertEqual(
            config["dataset"]["expected_human_revised_per_scenario"], 100
        )
        self.assertEqual(
            config["dataset"]["dataset_version"],
            "week6_week5_final_human300_20260817_v4",
        )

    def test_silver_label_requires_schema_but_not_human_vocab(self) -> None:
        # Schema-valid model labels remain silver even when free-form tags are not
        # part of the stricter human annotation vocabulary.
        label = {
            "business_category": "restaurant",
            "style_tags": ["model_free_form_style"],
            "visible_facilities": [],
            "price_range": "unknown",
            "observed_evidence": [],
            "inferred_attributes": [],
            "unknown_fields": [],
            "confidence": None,
        }
        _validate_silver_label(ROOT, "image_product_search", label)
        label.pop("business_category")
        with self.assertRaises(Week6TrainingError):
            _validate_silver_label(ROOT, "image_product_search", label)

    def test_image_references_are_portable_and_project_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "assets" / "image.png"
            image_path.parent.mkdir()
            image_path.write_bytes(b"image")
            messages = [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"file://{image_path.as_posix()}"},
                }],
            }]
            normalized = _normalize_training_messages(root, messages)
            self.assertEqual(
                normalized[0]["content"][0],
                {"type": "image", "path": "assets/image.png"},
            )
            self.assertNotEqual(normalized, messages)


if __name__ == "__main__":
    unittest.main()

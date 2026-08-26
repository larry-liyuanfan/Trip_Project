import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.week7_data import sha256_file
from src.training.week8_product_sft import (
    Week8ProductSFTError,
    _inspect_split_rows,
    _load_no_prompt_winner_evidence,
    _validate_product_lock_header,
    load_week8_product_sft_config,
    product_training_messages,
    validate_week8_product_sft_eligibility,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/week8/product_continuation_sft_v1.json"


def product_row(split: str, sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "scenario": "image_product_search",
        "image_path": "data/samples/images/cafe_001.jpg",
        "label_source": "programmatic_silver",
        "sample_weight": 0.5,
        "target_provenance": {"human_completed": False},
        "error_slices": [
            "style_multilabel",
            "facility_visible",
            "price_without_visual_evidence",
            "should_use_unknown",
            "multiple_or_ambiguous_subject",
        ],
        "target": {
            "business_category": "restaurant",
            "style_tags": ["modern", "casual"],
            "visible_facilities": ["bar"],
            "price_range": "unknown",
            "observed_evidence": ["可见吧台"],
            "inferred_attributes": [],
            "unknown_fields": ["price_range"],
            "confidence": 0.5,
        },
    }


class Week8ProductSFTTests(unittest.TestCase):
    def test_config_locks_low_lr_one_epoch_silver_cap_and_release_lora(self):
        config = load_week8_product_sft_config(CONFIG)
        self.assertEqual(config["training"]["epochs"], 1)
        self.assertLessEqual(config["training"]["learning_rate"], 1e-5)
        self.assertEqual(config["training"]["maximum_silver_sample_weight"], 0.5)
        self.assertEqual(config["lora"]["r"], 16)
        self.assertEqual(config["lora"]["lora_alpha"], 32)
        self.assertEqual(
            set(config["lora"]["expected_target_modules"]),
            {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "visual.merger.linear_fc1",
                "visual.merger.linear_fc2",
            },
        )
        self.assertEqual(
            config["continuation"]["adapter_model_sha256"],
            "c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a",
        )

    def test_config_rejects_scope_expanding_training_settings(self):
        original = json.loads(CONFIG.read_text(encoding="utf-8"))
        mutations = [
            ("epochs", 2),
            ("learning_rate", 5e-5),
            ("maximum_silver_sample_weight", 0.75),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for field, value in mutations:
                payload = copy.deepcopy(original)
                payload["training"][field] = value
                path = Path(directory) / f"{field}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(Week8ProductSFTError):
                    load_week8_product_sft_config(path)

    def test_split_inspection_rejects_gold_impersonation_and_heavy_silver(self):
        row = product_row("train", "train-1")
        row["target_provenance"]["human_completed"] = True
        with self.assertRaisesRegex(Week8ProductSFTError, "falsely marks"):
            _inspect_split_rows([row], split="train", maximum_silver_weight=0.5)
        row["target_provenance"]["human_completed"] = False
        row["sample_weight"] = 0.51
        with self.assertRaisesRegex(Week8ProductSFTError, "exceeds"):
            _inspect_split_rows([row], split="train", maximum_silver_weight=0.5)

    def test_training_messages_use_fixed_evidence_prompt_and_compact_target(self):
        messages = product_training_messages(
            ROOT,
            product_row("train", "train-1"),
            "week8_product_evidence_guard_v1",
        )
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(json.loads(messages[-1]["content"])["price_range"], "unknown")
        rendered = json.dumps(messages[:-1], ensure_ascii=False)
        self.assertIn("observed_evidence", rendered)
        self.assertIn("多主体", rendered)
        self.assertNotIn("思维链", rendered)

    def test_prompt_failure_evidence_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            product_config = root / "product.json"
            product_config.write_text("{}", encoding="utf-8")
            metrics_path = root / "current_release" / "metrics.json"
            metrics_path.parent.mkdir()
            metrics_path.write_text("{}", encoding="utf-8")
            selection = {
                "status": "SFT_ALLOWED_NO_PROMPT_WINNER",
                "selected_role": None,
                "test_consumed": False,
                "config_sha256": sha256_file(product_config),
                "dataset_lock_sha256": "lock-1",
                "selection_id": "selection-1",
                "metrics_sha256": {"current_release": sha256_file(metrics_path)},
            }
            (root / "selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            evidence = _load_no_prompt_winner_evidence(
                root,
                product_config_path=product_config,
                dataset_lock_sha256="lock-1",
            )
            self.assertEqual(evidence["selection_id"], "selection-1")
            metrics_path.write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(Week8ProductSFTError, "changed"):
                _load_no_prompt_winner_evidence(
                    root,
                    product_config_path=product_config,
                    dataset_lock_sha256="lock-1",
                )

    def test_sft_lock_validation_does_not_require_or_open_test_rows(self):
        from src.training.week7_data import canonical_sha256

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            product_config_path = root / "product.json"
            product_config_path.write_text("{}", encoding="utf-8")
            product_config = {
                "dataset": {"output_root": "outputs/locked"},
                "week8": {"dataset_version": "v1"},
            }
            lock_root = root / "outputs/locked/v1"
            files = {}
            for split in ("train", "development"):
                path = lock_root / split / "image_product_search.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
                files[f"{split}/image_product_search.jsonl"] = {
                    "count": 1,
                    "sha256": sha256_file(path),
                }
            core = {
                "dataset_version": "v1",
                "config_sha256": sha256_file(product_config_path),
                "files": files,
                "test_status": "LOCKED_UNCONSUMED",
            }
            lock = {**core, "lock_sha256": canonical_sha256(core)}
            (lock_root / "dataset_lock.json").write_text(
                json.dumps(lock), encoding="utf-8"
            )
            result = _validate_product_lock_header(
                root, product_config_path, product_config
            )
            self.assertEqual(result["status"], "PASS")
            self.assertFalse((lock_root / "test").exists())

    def test_eligibility_reads_only_train_and_development(self):
        config = load_week8_product_sft_config(CONFIG)
        product_config = {
            "model": {
                "adapter_model_sha256": config["continuation"]["adapter_model_sha256"],
                "base_model": config["model"]["base_model"],
                "base_revision": config["model"]["base_revision"],
            },
            "prompts": {
                config["development"]["prompt_role"]: config["development"]["prompt_version"]
            },
            "dataset": {
                "output_root": "outputs/week8/locked_data",
                "continuation_train_count": 1,
                "development_count": 1,
            },
            "week8": {"dataset_version": "week8-test-lock"},
        }
        requested_paths = []

        def fake_iter(path):
            requested_paths.append(Path(path).as_posix())
            if "/test/" in Path(path).as_posix():
                raise AssertionError("SFT must never read final test")
            split = "train" if "/train/" in Path(path).as_posix() else "development"
            return iter([product_row(split, f"{split}-1")])

        with (
            patch(
                "src.training.week8_product_sft.load_week8_product_config",
                return_value=product_config,
            ),
            patch(
                "src.training.week8_product_sft._validate_product_lock_header",
                return_value={
                    "status": "PASS",
                    "test_status": "LOCKED_UNCONSUMED",
                    "lock_sha256": "lock-1",
                    "dataset_version": "week8-test-lock",
                },
            ),
            patch(
                "src.training.week8_product_sft._load_no_prompt_winner_evidence",
                return_value={
                    "selection_sha256": "selection-hash",
                    "selection_id": "selection-1",
                    "metrics_sha256": {},
                },
            ),
            patch("src.training.week8_product_sft.iter_jsonl", side_effect=fake_iter),
        ):
            result = validate_week8_product_sft_eligibility(
                ROOT, CONFIG, ROOT / "fake-prompt-development"
            )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(requested_paths), 2)
        self.assertTrue(all("/test/" not in path for path in requested_paths))


if __name__ == "__main__":
    unittest.main()

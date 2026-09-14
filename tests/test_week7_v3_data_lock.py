from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from src.training import week7_data
from src.training.week7_data import (
    CORE_SCENARIOS,
    IDENTITY_FIELDS,
    Week7DataError,
    build_week7_lock,
    iter_jsonl,
    load_week7_config,
    sha256_file,
    validate_week7_lock,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_V2 = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v2.json"
CONFIG_V3 = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v3.json"


class Week7V3DataLockTests(unittest.TestCase):
    def test_v3_identity_and_exact_dialogue_parent_contract(self) -> None:
        v2 = load_week7_config(CONFIG_V2)
        v3 = load_week7_config(CONFIG_V3)
        self.assertNotEqual(sha256_file(CONFIG_V2), sha256_file(CONFIG_V3))
        self.assertNotEqual(v2["dataset"]["dataset_version"], v3["dataset"]["dataset_version"])
        self.assertEqual(v3["sampling"]["dialogue_parent_scenario_counts"], {
            "train": {scenario: 150 for scenario in CORE_SCENARIOS},
            "development": {scenario: 8 for scenario in CORE_SCENARIOS},
            "test": {scenario: 8 for scenario in CORE_SCENARIOS},
        })
        run_ids = list(v3["experiment_identity"]["development_baseline_run_ids"].values())
        run_ids.extend(
            value for key, value in v3["experiment_identity"].items()
            if key != "development_baseline_run_ids"
        )
        self.assertEqual(len(run_ids), len(set(run_ids)))
        self.assertTrue(all(run_id.endswith("_v3") for run_id in run_ids))

    def test_real_build_and_lock_validation_are_balanced_without_reading_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "configs/evaluation", root / "configs/evaluation")
            config = copy.deepcopy(load_week7_config(CONFIG_V3))
            config["dataset"].update({
                "dataset_version": "week7_v3_balanced_unit_lock",
                "train_total": 600,
                "train_per_core_scenario": 152,
                "general_regularization_count": 54,
                "dialogue_count": 90,
                "development_per_core_scenario": 3,
                "test_per_core_scenario": 3,
                "development_dialogue_count": 6,
                "test_dialogue_count": 6,
            })
            config["evaluation"]["human_review_queue_size"] = 6
            config["sampling"]["dialogue_parent_scenario_counts"] = {
                "train": {scenario: 30 for scenario in CORE_SCENARIOS},
                "development": {scenario: 2 for scenario in CORE_SCENARIOS},
                "test": {scenario: 2 for scenario in CORE_SCENARIOS},
            }
            config_path = root / "configs/week7/config-v3.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            load_week7_config(config_path)

            source_root = root / "source"
            source_images = source_root / "images"
            source_images.mkdir(parents=True)
            public_needed = 2 * (152 + 3 + 3) + 54
            sources = []
            for index in range(public_needed):
                image = source_images / f"{index:04d}.jpg"
                image.write_bytes(f"fresh-image-{index:04d}".encode())
                sources.append({
                    "photo_id": str(index), "business_id": str(index),
                    "source_id": f"source:{index}", "group_id": f"group:{index}",
                    "source_image": image, "image_sha256": sha256_file(image),
                    "caption": "modern hotel lobby with accessible front desk",
                })

            def after_sales_row(
                build_root: Path, output: Path, split: str, ordinal: int, weight: float,
            ) -> dict:
                image = output / "images/synthetic_after_sales" / f"{split}-{ordinal}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"after-sales-{split}-{ordinal}".encode())
                identity = {
                    "source_id": f"after-sales:{split}:{ordinal}",
                    "group_id": f"after-sales-group:{split}:{ordinal}",
                    "constraint_template_id": None,
                    "image_sha256": sha256_file(image),
                    "image_path": image.relative_to(build_root).as_posix(),
                }
                target = {
                    "issue_type": "facility_damage", "severity": "high",
                    "issue_location": "evidence card", "key_information": ["visible damage"],
                    "ocr_text": ["SEVERITY: HIGH"], "observed_evidence": ["visible damage"],
                    "unknown_fields": [], "confidence": 1.0,
                }
                return week7_data._row(
                    f"week7-{split}-after_sales-{ordinal:04d}", "after_sales", split,
                    identity,
                    [week7_data._system(), week7_data._user(identity["image_path"], "inspect")],
                    target, "programmatic_silver", weight,
                )

            empty_consumed = {field: set() for field in IDENTITY_FIELDS}
            with (
                patch(
                    "src.training.week7_data.load_consumed_identities",
                    return_value=(empty_consumed, {"unit_test": True}),
                ),
                patch(
                    "src.training.week7_data.audit_week5_dialogues",
                    return_value={"disposition": "unit_test_excluded"},
                ),
                patch(
                    "src.training.week7_data._collect_public_sources",
                    return_value=sources,
                ),
                patch("src.training.week7_data._after_sales_row", side_effect=after_sales_row),
            ):
                lock_root = build_week7_lock(root, source_root, config_path)

            train_dialogues = list(iter_jsonl(lock_root / "train/dialogue.jsonl"))
            development_dialogues = list(iter_jsonl(lock_root / "development/dialogue.jsonl"))
            self.assertEqual(Counter(row["parent_scenario"] for row in train_dialogues), {
                scenario: 30 for scenario in CORE_SCENARIOS
            })
            self.assertEqual(Counter(row["parent_scenario"] for row in development_dialogues), {
                scenario: 2 for scenario in CORE_SCENARIOS
            })
            self.assertEqual(sum(row["contains_tool_call"] for row in train_dialogues), 9)

            lock = json.loads((lock_root / "dataset_lock.json").read_text(encoding="utf-8"))
            self.assertEqual(lock["schema_version"], "week7_dataset_lock_v3")
            self.assertEqual(lock["dialogue_parent_scenario_counts"]["test"], {
                scenario: 2 for scenario in CORE_SCENARIOS
            })
            self.assertEqual(lock["isolation"]["cross_split_collisions"], [])

            original_iter_jsonl = week7_data.iter_jsonl

            def reject_test_read(path: Path):
                if Path(path).name == "test.jsonl":
                    raise AssertionError("test content was read before the parameter lock")
                return original_iter_jsonl(path)

            with patch("src.training.week7_data.iter_jsonl", side_effect=reject_test_read):
                result = validate_week7_lock(root, config_path)
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(result["test_consumed"])
            with self.assertRaisesRegex(Week7DataError, "parameter-locked"):
                validate_week7_lock(root, config_path, include_test=True)


if __name__ == "__main__":
    unittest.main()

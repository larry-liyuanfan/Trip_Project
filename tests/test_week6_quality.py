import json
import tempfile
import unittest
from pathlib import Path

from src.training.week6_quality import (
    audit_itinerary_target,
    expected_itinerary_elements,
    repair_itinerary_target,
    summarize_itinerary_targets,
)
from src.training.week6_refinement import build_itinerary_refinement_lock


ROOT = Path(__file__).resolve().parents[1]


def itinerary_row(*, complete: bool, source: str = "model_preannotation") -> dict:
    constraints = (
        "计划2天前往Melbourne，solo出行，预算档位为budget；"
        "偏好慢节奏，优先公共交通，每日包含用餐安排。"
    )
    clauses = [
        "计划2天前往Melbourne",
        "solo出行",
        "预算档位为budget",
        "偏好慢节奏",
        "优先公共交通",
        "每日包含用餐安排",
    ]
    output = {
        "style_preferences": [],
        "hard_constraints": clauses[:3] if complete else [],
        "soft_constraints": clauses[3:] if complete else [],
        "required_itinerary_elements": (
            ["daily_schedule", "meals", "transport", "budget_check"]
            if complete
            else ["daily_schedule"]
        ),
        "itinerary": [
            {
                "day_index": index,
                "date": None,
                "summary": "简短摘要",
                "activities": [
                    {
                        "start_time": None,
                        "end_time": None,
                        "place_name": None,
                        "activity": "简短活动",
                        "transport": None,
                        "source_evidence": [],
                    }
                ],
            }
            for index in range(1, 3 if complete else 2)
        ],
        "constraint_check": [
            {
                "constraint": clause,
                "constraint_type": "hard" if index < 3 else "soft",
                "status": "unknown",
                "evidence": None,
            }
            for index, clause in enumerate(clauses if complete else [])
        ],
        "observed_evidence": [],
        "unknown_fields": [],
        "confidence": None,
    }
    return {
        "sample_id": f"sample-{complete}-{source}",
        "scenario": "itinerary_planning",
        "label_source": source,
        "sample_weight": 1.0 if source == "human_revised" else 0.5,
        "dataset_lock": {
            "dataset_version": "test-v1",
            "manifest_sha256": "a" * 64,
            "split_sha256": "b" * 64,
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"原始文字约束：{constraints}"}
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps(output, ensure_ascii=False),
            },
        ],
    }


class Week6QualityTests(unittest.TestCase):
    def test_expected_elements_follow_explicit_prompt_mapping(self):
        self.assertEqual(
            expected_itinerary_elements("预算有限，优先公共交通，每日包含用餐安排"),
            {"daily_schedule", "budget_check", "transport", "meals"},
        )

    def test_audit_accepts_structurally_complete_target(self):
        result = audit_itinerary_target(ROOT, itinerary_row(complete=True))
        self.assertTrue(result["passed"])
        self.assertEqual(result["expected_days"], 2)
        self.assertEqual(result["actual_days"], 2)

    def test_audit_slices_incomplete_target(self):
        result = audit_itinerary_target(ROOT, itinerary_row(complete=False))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["day_count_match"])
        self.assertFalse(result["checks"]["constraint_text_exact_match"])
        self.assertFalse(result["checks"]["required_elements_complete"])

    def test_repair_produces_complete_derived_target(self):
        row = itinerary_row(complete=False, source="human_revised")
        repaired = repair_itinerary_target(ROOT, row)
        row["messages"][-1]["content"] = json.dumps(repaired, ensure_ascii=False)
        result = audit_itinerary_target(ROOT, row)
        self.assertTrue(result["passed"])
        self.assertEqual(len(repaired["itinerary"]), 2)
        self.assertEqual(
            repaired["required_itinerary_elements"],
            ["daily_schedule", "meals", "transport", "budget_check"],
        )

    def test_summary_groups_sources_and_bounds_examples(self):
        rows = [
            itinerary_row(complete=True),
            itinerary_row(complete=False, source="human_revised"),
        ]
        summary = summarize_itinerary_targets(
            ROOT, rows, max_examples_per_issue=1
        )
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["by_label_source"]["human_revised"]["rows"], 1)
        self.assertEqual(
            len(summary["issue_examples"]["failed_day_count_match"]), 1
        )

    def test_refinement_lock_downgrades_derived_human_target_to_silver(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            train = temp / "source-train.jsonl"
            validation = temp / "source-validation.jsonl"
            train.write_text(
                json.dumps(
                    itinerary_row(complete=False, source="human_revised"),
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            validation.write_text(
                json.dumps(itinerary_row(complete=False), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            output = temp / "refined"
            summary = build_itinerary_refinement_lock(
                ROOT,
                train_path=train,
                validation_path=validation,
                output_dir=output,
                dataset_version="refinement-test-v1",
            )
            self.assertEqual(summary["counts"], {"train": 1, "validation": 1})
            row = json.loads(
                (output / "itinerary_planning/train.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(row["label_source"], "model_preannotation")
            self.assertEqual(row["sample_weight"], 0.5)
            self.assertFalse(row["target_derivation"]["human_identity_inherited"])


if __name__ == "__main__":
    unittest.main()

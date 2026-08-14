from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.api.week5_annotation_station import HumanSubmission, Week5AnnotationStore
from src.data.week5_dataset import Week5DataError


ROOT = Path(__file__).resolve().parents[1]


class Week5AnnotationStationTests(unittest.TestCase):
    def test_station_prefills_authorized_operator_and_shows_chinese_preview(self) -> None:
        html = (ROOT / "src/api/templates/week5_annotation_station.html").read_text(
            encoding="utf-8"
        )
        self.assertIn('value="Larry Fan"', html)
        self.assertIn('id="reference-cn"', html)
        self.assertIn('business_category:"商家类别"', html)
        self.assertIn("newSessionId()", html)

    def fixture(self, directory: str) -> tuple[Week5AnnotationStore, str]:
        root = Path(directory)
        (root / "configs/evaluation/schemas").mkdir(parents=True)
        shutil.copy2(
            ROOT / "configs/evaluation/schemas/image_product_search_v1.schema.json",
            root / "configs/evaluation/schemas/image_product_search_v1.schema.json",
        )
        (root / "configs/week5").mkdir(parents=True)
        shutil.copy2(
            ROOT / "configs/week5/annotation_tool.json",
            root / "configs/week5/annotation_tool.json",
        )
        image = root / "image.jpg"
        image.write_bytes(b"image")
        sample_id = "week5-image_product_search-station"
        pool_dir = root / "outputs/week5/pools"
        pool_dir.mkdir(parents=True)
        candidate = {
            "sample_id": sample_id,
            "scenario": "image_product_search",
            "input": {"images": [{"path": "image.jpg"}]},
            "sampling_metadata": {},
            "isolation": {},
        }
        (pool_dir / "image_product_search.jsonl").write_text(
            json.dumps(candidate) + "\n", encoding="utf-8"
        )
        for scenario in ("after_sales", "itinerary_planning"):
            (pool_dir / f"{scenario}.jsonl").write_text("", encoding="utf-8")
        pre_dir = root / "outputs/week5/preannotations"
        pre_dir.mkdir(parents=True)
        output = {
            "business_category": "unknown",
            "style_tags": [],
            "visible_facilities": [],
            "price_range": "unknown",
            "observed_evidence": [],
            "inferred_attributes": [],
            "unknown_fields": ["business_category", "price_range"],
            "confidence": None,
        }
        (pre_dir / "image_product_search.jsonl").write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "status": "completed",
                    "schema_valid": True,
                    "parsed_output": output,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        for scenario in ("after_sales", "itinerary_planning"):
            (pre_dir / f"{scenario}.jsonl").write_text("", encoding="utf-8")
        config = {
            "paths": {"output_dir": "outputs/week5"},
            "quality": {
                "mode": "single_operator_minimal_review_v1",
                "core_scenarios": ["after_sales", "itinerary_planning"],
                "core_cross_review_rate": 1.0,
                "general_cross_review_rate": 1.0,
                "core_audit_rate": 1.0,
                "general_audit_rate": 1.0,
            },
        }
        return Week5AnnotationStore(root, config), sample_id

    def test_human_queue_requires_explicit_self_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, sample_id = self.fixture(directory)
            self.assertEqual(store.queue_ids("image_product_search", "human"), [sample_id])
            annotation = store.preannotations["image_product_search"][sample_id][
                "parsed_output"
            ]
            submission = HumanSubmission(
                sample_id=sample_id,
                scenario="image_product_search",
                annotator="operator-a",
                human_annotation=annotation,
                self_review_confirmed=False,
                review_session_id="human-session-1",
            )
            with self.assertRaises(Week5DataError):
                store.save_human(submission)
            submission.self_review_confirmed = True
            with patch(
                "src.data.week5_workflow.load_pools",
                side_effect=AssertionError("station submit must use its candidate cache"),
            ):
                self.assertEqual(store.save_human(submission), {"applied": 1})
            self.assertEqual(store.queue_ids("image_product_search", "human"), [])
            self.assertEqual(
                store.queue_ids("image_product_search", "cross_review"), [sample_id]
            )

    def test_task_removes_uncontrolled_model_labels_from_human_draft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, sample_id = self.fixture(directory)
            preannotation = store.preannotations["image_product_search"][sample_id]
            preannotation["parsed_output"]["style_tags"] = ["traditional", "decorative"]
            task = store.task("image_product_search", "human", sample_id)
            self.assertEqual(
                task["model_preannotation"]["style_tags"],
                ["traditional", "decorative"],
            )
            self.assertEqual(task["human_draft"]["style_tags"], ["traditional"])
            self.assertEqual(len(task["draft_warnings"]), 1)
            self.assertIn("decorative", task["draft_warnings"][0])

    def test_image_path_cannot_escape_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, _ = self.fixture(directory)
            with self.assertRaises(Week5DataError):
                store.image_path("../outside.jpg")

    def test_bounded_human_queue_preserves_completed_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, sample_id = self.fixture(directory)
            store.config["quality"]["human_review_targets"] = {
                "image_product_search": 1
            }
            self.assertEqual(store.queue_ids("image_product_search", "human"), [sample_id])
            annotation = store.preannotations["image_product_search"][sample_id][
                "parsed_output"
            ]
            store.save_human(
                HumanSubmission(
                    sample_id=sample_id,
                    scenario="image_product_search",
                    annotator="operator-a",
                    human_annotation=annotation,
                    self_review_confirmed=True,
                    review_session_id="bounded-session-1",
                )
            )
            self.assertEqual(store.queue_ids("image_product_search", "human"), [])
            plan = store.summary()["human_review_plan"]["image_product_search"]
            self.assertEqual(plan, {"target": 1, "completed": 1})


if __name__ == "__main__":
    unittest.main()

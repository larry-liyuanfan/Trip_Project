import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class RunBoundCountValidationTest(unittest.TestCase):
    def test_completed_full_run_supplies_tested_ids_by_scenario(self) -> None:
        from scripts.validate_week3_evaluation import load_tested_sample_ids

        metadata = {
            "run_id": "run-1",
            "status": "completed",
            "mode": "live",
            "run_scope": "full",
            "selected_count": 2,
            "record_count": 2,
            "artifact_hashes": {"fixture": "a" * 64},
        }
        results = [
            {"run_id": "run-1", "sample_id": "product-1", "scenario": "image_product_search"},
            {"run_id": "run-1", "sample_id": "sales-1", "scenario": "after_sales"},
        ]
        config = {"paths": {"runs_dir": "runs"}, "scenarios": {
            "image_product_search": {}, "after_sales": {}, "itinerary_planning": {}
        }}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("scripts.validate_week3_evaluation.load_run_metadata", return_value=metadata),
            patch("scripts.validate_week3_evaluation.load_result_records", return_value=results),
            patch("scripts.validate_week3_evaluation.verify_artifact_hashes", return_value=None),
        ):
            tested = load_tested_sample_ids(
                config, root=Path(directory), run_id="run-1"
            )

        self.assertEqual(tested["image_product_search"], {"product-1"})
        self.assertEqual(tested["after_sales"], {"sales-1"})
        self.assertEqual(tested["itinerary_planning"], set())

    def test_failed_run_cannot_supply_tested_counts(self) -> None:
        from scripts.validate_week3_evaluation import load_tested_sample_ids

        config = {"paths": {"runs_dir": "runs"}, "scenarios": {}}
        metadata = {
            "run_id": "run-1", "status": "failed", "mode": "live",
            "run_scope": "full", "selected_count": 1, "record_count": 0,
            "artifact_hashes": {"fixture": "a" * 64},
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.validate_week3_evaluation.load_run_metadata",
            return_value=metadata,
        ):
            with self.assertRaisesRegex(ValueError, "completed full live run"):
                load_tested_sample_ids(config, root=Path(directory), run_id="run-1")


if __name__ == "__main__":
    unittest.main()

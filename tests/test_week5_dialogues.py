from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.week5_dataset import SCENARIOS, Week5DataError
from src.data.week5_workflow import generate_dialogue_candidates


class Week5DialogueTests(unittest.TestCase):
    def test_resume_requires_identical_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "paths": {"output_dir": "outputs/week5"},
                "targets": {"dialogues": 0},
                "runtime": {"base_url": "http://127.0.0.1:8001/v1"},
            }
            qualified = {scenario: [f"{scenario}-1"] for scenario in SCENARIOS}
            pools = {scenario: [] for scenario in SCENARIOS}
            with (
                patch(
                    "src.data.week5_workflow._qualified_sample_ids",
                    return_value=qualified,
                ),
                patch("src.data.week5_workflow.load_pools", return_value=pools),
            ):
                self.assertEqual(
                    generate_dialogue_candidates(root, config, run_id="resume-test"),
                    {"generated": 0, "failed": 0, "existing": 0},
                )
                with self.assertRaises(Week5DataError):
                    generate_dialogue_candidates(root, config, run_id="resume-test")
                self.assertEqual(
                    generate_dialogue_candidates(
                        root, config, run_id="resume-test", resume=True
                    ),
                    {"generated": 0, "failed": 0, "existing": 0},
                )


if __name__ == "__main__":
    unittest.main()

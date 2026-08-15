from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.week5_dataset import SCENARIOS, Week5DataError
from src.data.week5_workflow import generate_dialogue_candidates


class Week5DialogueTests(unittest.TestCase):
    def test_generation_stops_after_eight_consecutive_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "paths": {"output_dir": "outputs/week5"},
                "targets": {"dialogues": 20},
                "runtime": {"base_url": "http://127.0.0.1:8001/v1"},
            }
            qualified = {scenario: [f"{scenario}-1"] for scenario in SCENARIOS}
            pools = {
                scenario: [
                    {
                        "sample_id": f"{scenario}-1",
                        "input": {
                            "images": [
                                {"path": "image.jpg", "sha256": "a" * 64}
                            ],
                            "text_constraints": None,
                        },
                    }
                ]
                for scenario in SCENARIOS
            }
            runtime = {
                "model_name": "model",
                "served_model_name": "model",
                "live_base_url": "http://127.0.0.1:8001/v1",
                "timeout_seconds": 1,
                "generation": {},
            }
            with (
                patch(
                    "src.data.week5_workflow._qualified_sample_ids",
                    return_value=qualified,
                ),
                patch("src.data.week5_workflow.load_pools", return_value=pools),
                patch("src.data.week5_workflow._runtime", return_value=runtime),
                patch(
                    "src.data.week5_workflow.post_chat_completion",
                    side_effect=RuntimeError("server failed"),
                ),
            ):
                with self.assertRaisesRegex(
                    Week5DataError,
                    "stopped after 8 consecutive failures",
                ):
                    generate_dialogue_candidates(root, config, run_id="breaker-test")

            failures = (
                root
                / "outputs/week5/runs/dialogue-breaker-test/failures.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(failures), 8)

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

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.week5_dataset import SCENARIOS, Week5DataError
from src.data.week5_workflow import generate_dialogue_candidates, merge_dialogue_runs


class Week5DialogueTests(unittest.TestCase):
    def test_merge_dialogue_runs_validates_complete_unique_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "paths": {"output_dir": "outputs/week5"},
                "targets": {"dialogues": 2},
                "runtime": {"base_url": "http://127.0.0.1:8001/v1"},
            }
            qualified = {scenario: [f"{scenario}-1"] for scenario in SCENARIOS}
            config_sha256 = hashlib.sha256(
                json.dumps(config, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            qualified_sha256 = {
                scenario: hashlib.sha256(
                    json.dumps(ids, ensure_ascii=False).encode()
                ).hexdigest()
                for scenario, ids in qualified.items()
            }
            for index, source in enumerate(("source-a", "source-b")):
                scenario = SCENARIOS[index % 3]
                sample_id = qualified[scenario][0]
                dialogue_id = (
                    f"week5-dialogue-{index:05d}-"
                    f"{hashlib.sha256(sample_id.encode()).hexdigest()[:8]}"
                )
                run_dir = root / f"outputs/week5/runs/dialogue-{source}"
                run_dir.mkdir(parents=True)
                (run_dir / "run_manifest.json").write_text(json.dumps({
                    "target": 2,
                    "config_sha256": config_sha256,
                    "qualified_sample_ids_sha256": qualified_sha256,
                }), encoding="utf-8")
                (run_dir / "candidates.jsonl").write_text(
                    json.dumps({"dialogue_id": dialogue_id}) + "\n",
                    encoding="utf-8",
                )
            with (
                patch("src.data.week5_workflow._qualified_sample_ids", return_value=qualified),
                patch("src.data.week5_workflow.validate_dialogue_v2"),
            ):
                merged = merge_dialogue_runs(
                    root, config, source_run_ids=["source-a", "source-b"],
                    merged_run_id="merged",
                )
            self.assertEqual(merged["status"], "completed")
            self.assertEqual(merged["unique_candidates"], 2)
            self.assertEqual(merged["missing_count"], 0)

    def test_bounded_shard_is_disjoint_and_records_parallel_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "paths": {"output_dir": "outputs/week5"},
                "targets": {"dialogues": 8},
                "runtime": {"base_url": "http://127.0.0.1:8001/v1"},
            }
            qualified = {scenario: [f"{scenario}-1"] for scenario in SCENARIOS}
            pools = {
                scenario: [{
                    "sample_id": f"{scenario}-1",
                    "input": {
                        "images": [{"path": "image.jpg", "sha256": "a" * 64}],
                        "text_constraints": None,
                    },
                }]
                for scenario in SCENARIOS
            }
            runtime = {
                "model_name": "model",
                "served_model_name": "model",
                "live_base_url": "http://127.0.0.1:8001/v1",
                "timeout_seconds": 1,
                "generation": {},
            }

            def response(_url: str, payload: dict, _timeout: int) -> dict:
                prompt = payload["messages"][0]["content"][0]["text"]
                scenario = re.search(r"\nscenario=([^\n]+)", prompt).group(1)
                message_count = int(re.search(r"必须输出恰好 (\d+) 条消息", prompt).group(1))
                turns = [{
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": "这是满足长度要求的测试对话内容",
                    "image_refs": ["img_1"] if index == 0 else [],
                } for index in range(message_count)]
                return {"choices": [{"message": {"content": json.dumps({
                    "scenario": scenario, "turns": turns,
                }, ensure_ascii=False)}}]}

            with (
                patch("src.data.week5_workflow._qualified_sample_ids", return_value=qualified),
                patch("src.data.week5_workflow.load_pools", return_value=pools),
                patch("src.data.week5_workflow._runtime", return_value=runtime),
                patch("src.data.week5_workflow.post_chat_completion", side_effect=response),
                patch("src.data.week5_workflow.validate_dialogue_v2"),
                patch.dict("os.environ", {"TRIP_DIALOGUE_CONCURRENCY": "2"}),
            ):
                result = generate_dialogue_candidates(
                    root, config, run_id="shard-test", start_index=2,
                    end_index=8, shard_index=1, shard_count=3,
                )

            self.assertEqual(result, {"generated": 2, "failed": 0, "existing": 0})
            run_dir = root / "outputs/week5/runs/dialogue-shard-test"
            rows = [json.loads(line) for line in (run_dir / "candidates.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [row["dialogue_id"].split("-")[2] for row in rows],
                ["00003", "00006"],
            )
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["selection"], {
                "start_index": 2, "end_index": 8,
                "shard_index": 1, "shard_count": 3,
                "strategy": "bounded_modulo_v1",
            })
            self.assertEqual(manifest["execution"], {"dialogue_concurrency": 2})

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

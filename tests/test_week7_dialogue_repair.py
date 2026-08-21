from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.week7_data import DIALOGUE_DIMENSIONS, iter_jsonl, sha256_file
from src.training.week7_dialogue_repair import build_dialogue_review_v2
from src.training.week7_qlora import Week7TrainingError


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class Week7DialogueRepairTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        base_config = root / "configs/week7/base.json"
        base_config.parent.mkdir(parents=True, exist_ok=True)
        base_config.write_text("{}\n", encoding="utf-8")
        source_dir = root / "outputs/source-v3"
        development = []
        queue = []
        for index in range(24):
            sample_id = f"source-dialogue-{index:04d}"
            target = {
                "task_result": {
                    "business_category": "unknown",
                    "observed_evidence": [f"evidence-{index}"],
                    "unknown_fields": ["business_category"],
                },
                "context_state": {
                    "historical_image_reference": [f"evidence-{index}"],
                    "updated_requirement": "预算优先",
                    "retained_hard_constraints": ["不得猜测"],
                },
            }
            development.append({
                "sample_id": sample_id,
                "scenario": "dialogue",
                "parent_scenario": "image_product_search",
                "contains_tool_call": index % 10 == 0,
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": [
                        {"type": "image", "path": "outputs/source-v3/images/example.jpg"},
                        {"type": "text", "text": "请识别图片"},
                    ]},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                ],
                "target": target,
            })
            queue.append({
                "queue_id": f"source-queue-{index:03d}",
                "sample_id": sample_id,
                "required_dimensions": list(DIALOGUE_DIMENSIONS),
            })
        development_path = source_dir / "development.jsonl"
        queue_path = source_dir / "dialogue_human_review_queue.jsonl"
        _write_jsonl(development_path, development)
        _write_jsonl(queue_path, queue)
        source_lock = {
            "dataset_version": "source-v3",
            "lock_sha256": "source-lock-sha",
        }
        (source_dir / "dataset_lock.json").write_text(
            json.dumps(source_lock), encoding="utf-8",
        )
        review_config = root / "configs/week7/dialogue_review_v2.json"
        review_config.write_text(json.dumps({
            "schema_version": "week7_dialogue_review_config_v2",
            "dataset_version": "review-v2",
            "construction_version": "aligned_concrete_turns_v2",
            "base_config": {
                "path": "configs/week7/base.json",
                "sha256": sha256_file(base_config),
            },
            "source": {
                "dataset_version": "source-v3",
                "dataset_dir": "outputs/source-v3",
                "dataset_lock_sha256": "source-lock-sha",
                "development_sha256": sha256_file(development_path),
                "queue_sha256": sha256_file(queue_path),
                "sample_count": 24,
            },
            "output_root": "outputs/review-v2",
            "selected_checkpoint": {"name": "checkpoint-151", "adapter_sha256": "adapter"},
            "inference": {},
            "human_review": {
                "dimensions": list(DIALOGUE_DIMENSIONS),
                "score_min": 1,
                "score_max": 5,
                "real_self_review_required": True,
            },
            "scope": {
                "split": "development",
                "test_allowed": False,
                "training_allowed": False,
                "may_change_final_test_claims": False,
            },
        }, ensure_ascii=False), encoding="utf-8")
        return review_config, root / "outputs/review-v2/review-v2"

    def test_builds_aligned_five_to_eight_round_dialogues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, output = self._fixture(root)
            lock = build_dialogue_review_v2(root, config)
            rows = list(iter_jsonl(output / "development.jsonl"))
            self.assertEqual(lock["scope"]["test_allowed"], False)
            self.assertEqual(lock["config_path"], "configs/week7/dialogue_review_v2.json")
            self.assertEqual(len(rows), 24)
            self.assertEqual({row["dialogue_rounds"] for row in rows}, {5, 6, 7, 8})
            for row in rows:
                messages = row["messages"]
                self.assertEqual(messages[0]["role"], "system")
                self.assertEqual(messages[1]["role"], "user")
                self.assertEqual(messages[2]["role"], "assistant")
                self.assertIn("business_category", messages[2]["content"])
                self.assertEqual(messages[3]["role"], "user")
                self.assertIn("图片中的证据", messages[3]["content"][0]["text"])
                self.assertEqual(messages[4]["role"], "assistant")
                image_indexes = [
                    index for index, message in enumerate(messages)
                    if isinstance(message.get("content"), list)
                    and any(part.get("type") == "image" for part in message["content"])
                ]
                self.assertEqual(image_indexes, [1])
                self.assertEqual(json.loads(messages[-1]["content"]), row["target"])

    def test_refuses_to_overwrite_corrected_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self._fixture(root)
            build_dialogue_review_v2(root, config)
            with self.assertRaisesRegex(Week7TrainingError, "refusing to overwrite"):
                build_dialogue_review_v2(root, config)


if __name__ == "__main__":
    unittest.main()

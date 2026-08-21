from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.api.week7_dialogue_review import (
    DialogueReviewSubmission,
    EXPECTED_DATASET_VERSION,
    EXPECTED_MODEL_NAME,
    EXPECTED_RUN_ID,
    Week7DialogueReviewError,
    Week7DialogueReviewStore,
)
from src.training.week7_data import DIALOGUE_DIMENSIONS, sha256_file


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class Week7DialogueReviewTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, str]:
        dataset = root / "outputs/week7/locked_data/fixed"
        raw_path = root / "outputs/week7/human_review/source/raw.jsonl"
        image = dataset / "images/example.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"review-image")
        image_relative = image.relative_to(root).as_posix()
        development = []
        queue = []
        raw = []
        for index in range(24):
            sample_id = f"week7-development-dialogue-{index:04d}"
            development.append(
                {
                    "sample_id": sample_id,
                    "scenario": "dialogue",
                    "parent_scenario": "image_product_search",
                    "dialogue_rounds": 5,
                    "contains_tool_call": False,
                    "context_expectations": {"updated_requirement": "预算降低"},
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": [{"type": "image", "path": image_relative}, {"type": "text", "text": "请看图"}]},
                        {"role": "user", "content": "预算降低"},
                        {"role": "assistant", "content": "locked silver target"},
                    ],
                }
            )
            queue.append(
                {
                    "queue_id": f"week7-dialogue-human-{index:03d}",
                    "sample_id": sample_id,
                    "required_dimensions": list(DIALOGUE_DIMENSIONS),
                }
            )
            raw.append(
                {
                    "sample_id": sample_id,
                    "run_id": EXPECTED_RUN_ID,
                    "model_name": EXPECTED_MODEL_NAME,
                    "failed": False,
                    "raw_output": '{"answer":"ok"}',
                }
            )
        development_path = dataset / "development.jsonl"
        queue_path = dataset / "dialogue_human_review_queue.jsonl"
        _write_jsonl(development_path, development)
        _write_jsonl(queue_path, queue)
        _write_jsonl(raw_path, raw)
        lock = {
            "dataset_version": EXPECTED_DATASET_VERSION,
            "lock_sha256": "dataset-lock",
            "files": {
                "development.jsonl": {"sha256": sha256_file(development_path)},
                "dialogue_human_review_queue.jsonl": {"sha256": sha256_file(queue_path)},
            },
        }
        (dataset / "dataset_lock.json").write_text(json.dumps(lock), encoding="utf-8")
        return dataset.relative_to(root), raw_path.relative_to(root), sha256_file(raw_path)

    def test_fixed_queue_and_selected_output_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, raw_path, raw_sha = self._fixture(root)
            store = Week7DialogueReviewStore(
                root, dataset, raw_path, Path("outputs/week7/human_review/results"),
                expected_raw_sha256=raw_sha,
            )
            self.assertEqual(store.summary()["total"], 24)
            self.assertEqual(store.summary()["remaining"], 24)
            self.assertEqual(store.summary()["evidence"]["raw_outputs_sha256"], raw_sha)
            task = store.task(0)
            self.assertNotIn("locked silver target", json.dumps(task["input_messages"]))
            self.assertEqual(task["model_output"], '{"answer":"ok"}')
            self.assertTrue(store.image_path(task["image_urls"][0].split("path=", 1)[1]).is_file())

    def test_save_requires_self_review_and_appends_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, raw_path, raw_sha = self._fixture(root)
            store = Week7DialogueReviewStore(
                root, dataset, raw_path, Path("outputs/week7/human_review/results"),
                expected_raw_sha256=raw_sha,
            )
            task = store.task(0)
            values = {dimension: 4 for dimension in DIALOGUE_DIMENSIONS}
            pending = DialogueReviewSubmission(
                queue_id=task["queue_id"], sample_id=task["sample_id"], reviewer="human",
                review_session_id="session-1", scores=values, decision="pass", notes="checked",
                self_review_confirmed=False,
            )
            with self.assertRaises(Week7DialogueReviewError):
                store.save(pending)
            accepted = pending.model_copy(update={"self_review_confirmed": True})
            self.assertEqual(store.save(accepted)["revision"], 1)
            self.assertEqual(store.save(accepted)["revision"], 2)
            records = list(store.results_path.read_text(encoding="utf-8").splitlines())
            self.assertEqual(len(records), 2)
            last = json.loads(records[-1])
            self.assertEqual(last["raw_outputs_sha256"], raw_sha)
            self.assertTrue(last["self_review_confirmed"])
            self.assertEqual(store.summary()["completed"], 1)

    def test_changed_raw_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, raw_path, raw_sha = self._fixture(root)
            absolute_raw = root / raw_path
            absolute_raw.write_text(absolute_raw.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(Week7DialogueReviewError, "raw output hash changed"):
                Week7DialogueReviewStore(
                    root, dataset, raw_path, Path("outputs/week7/human_review/results"),
                    expected_raw_sha256=raw_sha,
                )


if __name__ == "__main__":
    unittest.main()

"""Local-only Week 7 dialogue scoring station for the real human operator."""

from __future__ import annotations

import json
import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.training.week7_data import (
    DIALOGUE_DIMENSIONS,
    Week7DataError,
    iter_jsonl,
    sha256_file,
)


EXPECTED_DATASET_VERSION = "week7_fresh_multitask_context_20260820_v3"
EXPECTED_MODEL_NAME = "multitask_step_000151"
EXPECTED_RUN_ID = "week7_dev_bf16_static_compile_20260820_v5"
EXPECTED_RAW_SHA256 = "aee27cf1cab1d97d26f9ba81c1319d3fe5532e8328b6738c59416c78bfa37090"


class Week7DialogueReviewError(ValueError):
    """Raised when the fixed queue, evidence, or a human submission is invalid."""


class DialogueReviewSubmission(BaseModel):
    queue_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    review_session_id: str = Field(min_length=1)
    scores: dict[str, int]
    decision: Literal["pass", "rework", "reject"]
    notes: str | None = None
    self_review_confirmed: bool


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _within(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise Week7DialogueReviewError("review path must stay inside the project root")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week7DialogueReviewError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise Week7DialogueReviewError(f"JSON evidence must be an object: {path}")
    return value


class Week7DialogueReviewStore:
    """Join the immutable queue with selected-checkpoint raw output and append reviews."""

    def __init__(
        self,
        root: Path,
        dataset_dir: Path,
        raw_outputs_path: Path,
        output_dir: Path,
        *,
        expected_raw_sha256: str = EXPECTED_RAW_SHA256,
    ) -> None:
        self.root = Path(root).resolve()
        self.dataset_dir = _within(self.root, self.root / dataset_dir)
        self.raw_outputs_path = _within(self.root, self.root / raw_outputs_path)
        self.output_dir = _within(self.root, self.root / output_dir)
        self.results_path = self.output_dir / "week7_dialogue_human_scores_v1.jsonl"
        self._lock = threading.RLock()
        self._load_inputs(expected_raw_sha256)
        self._load_reviews()

    def _load_inputs(self, expected_raw_sha256: str) -> None:
        lock_path = self.dataset_dir / "dataset_lock.json"
        queue_path = self.dataset_dir / "dialogue_human_review_queue.jsonl"
        development_path = self.dataset_dir / "development.jsonl"
        for path in (lock_path, queue_path, development_path, self.raw_outputs_path):
            if not path.is_file():
                raise Week7DialogueReviewError(f"required review evidence is missing: {path}")

        lock = _read_json(lock_path)
        if lock.get("dataset_version") != EXPECTED_DATASET_VERSION:
            raise Week7DialogueReviewError("unexpected Week 7 dataset identity")
        files = lock.get("files", {})
        expected_files = {
            "dialogue_human_review_queue.jsonl": queue_path,
            "development.jsonl": development_path,
        }
        for name, path in expected_files.items():
            evidence = files.get(name, {})
            if evidence.get("sha256") != sha256_file(path):
                raise Week7DialogueReviewError(f"locked file hash changed: {name}")

        raw_sha256 = sha256_file(self.raw_outputs_path)
        if raw_sha256 != expected_raw_sha256:
            raise Week7DialogueReviewError("selected checkpoint raw output hash changed")

        dialogues = {
            str(row["sample_id"]): row
            for row in iter_jsonl(development_path)
            if row.get("scenario") == "dialogue"
        }
        raw_records = {str(row["sample_id"]): row for row in iter_jsonl(self.raw_outputs_path)}
        queue = list(iter_jsonl(queue_path))
        if len(queue) != 24 or len({row.get("sample_id") for row in queue}) != 24:
            raise Week7DialogueReviewError("human queue must contain 24 unique samples")

        tasks: list[dict[str, Any]] = []
        allowed_images: set[Path] = set()
        for item in queue:
            sample_id = str(item.get("sample_id"))
            row = dialogues.get(sample_id)
            raw = raw_records.get(sample_id)
            if row is None or raw is None:
                raise Week7DialogueReviewError("queue sample is missing dialogue or raw output")
            if tuple(item.get("required_dimensions", [])) != DIALOGUE_DIMENSIONS:
                raise Week7DialogueReviewError("queue dimensions changed")
            if raw.get("run_id") != EXPECTED_RUN_ID or raw.get("model_name") != EXPECTED_MODEL_NAME:
                raise Week7DialogueReviewError("raw output does not belong to selected checkpoint-151")
            if raw.get("failed") is not False or not isinstance(raw.get("raw_output"), str):
                raise Week7DialogueReviewError("queued dialogue has no successful raw output")

            input_messages = deepcopy(row.get("messages", []))
            if input_messages and input_messages[-1].get("role") == "assistant":
                input_messages.pop()
            image_urls: list[str] = []
            for message in input_messages:
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    image_path = part.get("path") if isinstance(part, dict) else None
                    if not image_path:
                        continue
                    resolved = _within(self.root, self.root / str(image_path))
                    if not resolved.is_file():
                        raise Week7DialogueReviewError("queued dialogue image is missing")
                    allowed_images.add(resolved)
                    image_urls.append(f"/api/image?path={image_path}")

            tasks.append(
                {
                    "queue_id": str(item["queue_id"]),
                    "sample_id": sample_id,
                    "parent_scenario": row.get("parent_scenario"),
                    "dialogue_rounds": row.get("dialogue_rounds"),
                    "contains_tool_call": bool(row.get("contains_tool_call")),
                    "input_messages": input_messages,
                    "model_output": raw["raw_output"],
                    "image_urls": image_urls,
                    "silver_context_reference": row.get("context_expectations", {}),
                }
            )

        self.tasks = tasks
        self.task_by_sample = {task["sample_id"]: task for task in tasks}
        self.queue_sha256 = sha256_file(queue_path)
        self.development_sha256 = sha256_file(development_path)
        self.raw_outputs_sha256 = raw_sha256
        self.dataset_lock_sha256 = str(lock.get("lock_sha256"))
        self.allowed_images = allowed_images

    def _load_reviews(self) -> None:
        latest: dict[str, dict[str, Any]] = {}
        if self.results_path.is_file():
            for record in iter_jsonl(self.results_path):
                sample_id = str(record.get("sample_id"))
                if sample_id not in self.task_by_sample:
                    raise Week7DialogueReviewError("review file contains an unknown sample")
                if record.get("raw_outputs_sha256") != self.raw_outputs_sha256:
                    raise Week7DialogueReviewError("review is bound to different raw outputs")
                revision = int(record.get("revision", 0))
                if revision > int(latest.get(sample_id, {}).get("revision", 0)):
                    latest[sample_id] = record
        self.latest_reviews = latest

    def summary(self) -> dict[str, Any]:
        completed = len(self.latest_reviews)
        dimension_means: dict[str, float | None] = {}
        for dimension in DIALOGUE_DIMENSIONS:
            values = [int(row["scores"][dimension]) for row in self.latest_reviews.values()]
            dimension_means[dimension] = sum(values) / len(values) if values else None
        return {
            "dataset_version": EXPECTED_DATASET_VERSION,
            "model_name": EXPECTED_MODEL_NAME,
            "total": len(self.tasks),
            "completed": completed,
            "remaining": len(self.tasks) - completed,
            "dimension_means": dimension_means,
            "evidence": {
                "dataset_lock_sha256": self.dataset_lock_sha256,
                "queue_sha256": self.queue_sha256,
                "development_sha256": self.development_sha256,
                "raw_outputs_sha256": self.raw_outputs_sha256,
            },
        }

    def task(self, offset: int) -> dict[str, Any]:
        if offset < 0 or offset >= len(self.tasks):
            raise Week7DialogueReviewError("review offset is outside the fixed queue")
        task = deepcopy(self.tasks[offset])
        task["offset"] = offset
        task["total"] = len(self.tasks)
        task["saved_review"] = deepcopy(self.latest_reviews.get(task["sample_id"]))
        return task

    def image_path(self, relative_path: str) -> Path:
        candidate = _within(self.root, self.root / relative_path)
        if candidate not in self.allowed_images or not candidate.is_file():
            raise Week7DialogueReviewError("image is not part of the fixed review queue")
        return candidate

    def save(self, submission: DialogueReviewSubmission) -> dict[str, Any]:
        if not submission.self_review_confirmed:
            raise Week7DialogueReviewError("保存前必须由本人确认已逐项自审")
        if not submission.reviewer.strip() or not submission.review_session_id.strip():
            raise Week7DialogueReviewError("真实评分人和评分会话不能为空")
        if set(submission.scores) != set(DIALOGUE_DIMENSIONS):
            raise Week7DialogueReviewError("四个评分维度必须全部填写")
        if any(isinstance(value, bool) or not 1 <= int(value) <= 5 for value in submission.scores.values()):
            raise Week7DialogueReviewError("每个维度必须是 1 到 5 的整数")
        task = self.task_by_sample.get(submission.sample_id)
        if task is None or task["queue_id"] != submission.queue_id:
            raise Week7DialogueReviewError("submission is not part of the fixed queue")

        with self._lock:
            current = self.latest_reviews.get(submission.sample_id)
            revision = int(current.get("revision", 0)) + 1 if current else 1
            record = {
                "schema_version": "week7_dialogue_human_score_v1",
                "review_id": f"week7-human-{uuid.uuid4().hex}",
                "queue_id": submission.queue_id,
                "sample_id": submission.sample_id,
                "reviewer": submission.reviewer.strip(),
                "review_session_id": submission.review_session_id.strip(),
                "scores": {name: int(submission.scores[name]) for name in DIALOGUE_DIMENSIONS},
                "decision": submission.decision,
                "notes": submission.notes.strip() if submission.notes else None,
                "self_review_confirmed": True,
                "revision": revision,
                "reviewed_at": _now(),
                "dataset_version": EXPECTED_DATASET_VERSION,
                "dataset_lock_sha256": self.dataset_lock_sha256,
                "queue_sha256": self.queue_sha256,
                "development_sha256": self.development_sha256,
                "raw_outputs_sha256": self.raw_outputs_sha256,
                "model_name": EXPECTED_MODEL_NAME,
                "run_id": EXPECTED_RUN_ID,
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            payload = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            descriptor = os.open(self.results_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                with os.fdopen(descriptor, "ab", buffering=0) as handle:
                    written = handle.write(payload)
                    if written != len(payload):
                        raise Week7DialogueReviewError("human review append was incomplete")
                    os.fsync(handle.fileno())
            except Exception:
                raise
            self.latest_reviews[submission.sample_id] = record
            return {"status": "SAVED", "revision": revision, "summary": self.summary()}


def create_week7_dialogue_review_app(
    root: Path | None = None,
    *,
    dataset_dir: Path = Path("outputs/week7/locked_data/week7_fresh_multitask_context_20260820_v3"),
    raw_outputs_path: Path = Path("outputs/week7/human_review/source/multitask_step_000151_raw_outputs.jsonl"),
    output_dir: Path = Path("outputs/week7/human_review"),
    expected_raw_sha256: str = EXPECTED_RAW_SHA256,
) -> FastAPI:
    project_root = (root or Path(__file__).resolve().parents[2]).resolve()
    store = Week7DialogueReviewStore(
        project_root,
        dataset_dir,
        raw_outputs_path,
        output_dir,
        expected_raw_sha256=expected_raw_sha256,
    )
    app = FastAPI(title="Trip Week 7 对话人工评分", version="1.0")
    app.state.store = store

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        template = project_root / "src/api/templates/week7_dialogue_review.html"
        return HTMLResponse(template.read_text(encoding="utf-8"))

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return store.summary()

    @app.get("/api/task")
    def task(offset: int = Query(0, ge=0)) -> dict[str, Any]:
        try:
            return store.task(offset)
        except Week7DialogueReviewError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/image")
    def image(path: str) -> FileResponse:
        try:
            return FileResponse(store.image_path(path))
        except Week7DialogueReviewError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/review")
    def review(submission: DialogueReviewSubmission) -> dict[str, Any]:
        try:
            return store.save(submission)
        except (Week7DialogueReviewError, Week7DataError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app

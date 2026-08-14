"""Local-only Week 5 human annotation and QC station."""

from __future__ import annotations

import json
import hashlib
import mimetypes
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.data.week5_dataset import (
    SCENARIOS,
    Week5DataError,
    load_pools,
    load_week5_config,
    qc_audit_selected,
    qc_cross_review_selected,
    read_jsonl,
)
from src.data.week5_workflow import apply_human_corrections, apply_quality_records


Stage = Literal["human", "cross_review", "core_audit"]


class HumanSubmission(BaseModel):
    sample_id: str
    scenario: str
    annotator: str = Field(min_length=1)
    human_annotation: dict[str, Any]
    self_review_confirmed: bool
    review_session_id: str = Field(min_length=1)
    self_review_notes: str | None = None


class QualitySubmission(BaseModel):
    sample_id: str
    scenario: str
    stage: Literal["cross_review", "core_audit"]
    decision: Literal["pass", "rework", "reject"]
    reviewer: str = Field(min_length=1)
    issues: list[str] = Field(default_factory=list)
    notes: str | None = None
    review_session_id: str = Field(min_length=1)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Week5AnnotationStore:
    """Build deterministic queues and route writes through audited workflow APIs."""

    def __init__(self, root: Path, config: dict[str, Any]) -> None:
        self.root = root.resolve()
        self.config = config
        self.output_dir = self.root / config["paths"]["output_dir"]
        self._lock = threading.RLock()
        self.refresh()

    def refresh(self) -> None:
        with self._lock:
            self.pools = load_pools(self.root, self.config)
            self.candidates = {
                scenario: {row["sample_id"]: row for row in rows}
                for scenario, rows in self.pools.items()
            }
            self.preannotations: dict[str, dict[str, dict[str, Any]]] = {}
            self.annotations: dict[str, dict[str, dict[str, Any]]] = {}
            self.quality: dict[str, list[dict[str, Any]]] = {}
            for scenario in SCENARIOS:
                pre_path = self.output_dir / "preannotations" / f"{scenario}.jsonl"
                completed: dict[str, dict[str, Any]] = {}
                for row in read_jsonl(pre_path):
                    if row.get("status") == "completed" and row.get("schema_valid") is True:
                        completed[str(row["sample_id"])] = row
                self.preannotations[scenario] = completed

                latest: dict[str, dict[str, Any]] = {}
                for row in read_jsonl(
                    self.output_dir / "annotations" / f"{scenario}.jsonl"
                ):
                    sample_id = str(row["sample_id"])
                    if int(row.get("revision", 0)) >= int(
                        latest.get(sample_id, {}).get("revision", 0)
                    ):
                        latest[sample_id] = row
                self.annotations[scenario] = latest
                self.quality[scenario] = read_jsonl(
                    self.output_dir / "quality" / f"{scenario}.jsonl"
                )
            self.human_cohorts = {
                scenario: self._compute_human_cohort_ids(scenario)
                for scenario in SCENARIOS
            }

    def _refresh_records(self, scenario: str) -> None:
        """提交后仅刷新小型人工记录，避免重复扫描 80,000 条候选池。"""
        latest: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(
            self.output_dir / "annotations" / f"{scenario}.jsonl"
        ):
            sample_id = str(row["sample_id"])
            if int(row.get("revision", 0)) >= int(
                latest.get(sample_id, {}).get("revision", 0)
            ):
                latest[sample_id] = row
        self.annotations[scenario] = latest
        self.quality[scenario] = read_jsonl(
            self.output_dir / "quality" / f"{scenario}.jsonl"
        )

    def _passed(self, scenario: str) -> dict[str, set[tuple[str, int]]]:
        passed = {stage: set() for stage in ("self_review", "cross_review", "core_audit")}
        for row in self.quality[scenario]:
            stage = row.get("stage")
            if stage in passed and row.get("decision") == "pass":
                passed[stage].add(
                    (str(row.get("sample_id")), int(row.get("annotation_revision", 0)))
                )
        return passed

    def _compute_human_cohort_ids(self, scenario: str) -> set[str]:
        """按固定哈希构造小规模人工验证队列，并保留已经完成的真实记录。"""
        quality = self.config.get("quality", {})
        targets = quality.get("human_review_targets", {})
        target = int(targets.get(scenario, 0))
        available = set(self.preannotations[scenario])
        if target <= 0 or target >= len(available):
            return available

        completed = set(self.annotations[scenario]) & available
        selected = set(completed)
        cross_target = int(
            quality.get("bounded_cross_review_targets", {}).get(scenario, 0)
        )
        audit_target = int(
            quality.get("bounded_core_audit_targets", {}).get(scenario, 0)
        )

        def rank(sample_id: str) -> tuple[str, str]:
            digest = hashlib.sha256(
                f"week5-bounded-human-v1:{scenario}:{sample_id}".encode("utf-8")
            ).hexdigest()
            return digest, sample_id

        ranked = sorted(available, key=rank)
        audit_ids = [
            sample_id
            for sample_id in ranked
            if qc_audit_selected(sample_id, scenario, self.config)
        ]
        selected.update(audit_ids[:audit_target])

        current_cross = sum(
            qc_cross_review_selected(sample_id, scenario, self.config)
            for sample_id in selected
        )
        for sample_id in ranked:
            if current_cross >= cross_target:
                break
            if (
                sample_id not in selected
                and qc_cross_review_selected(sample_id, scenario, self.config)
            ):
                selected.add(sample_id)
                current_cross += 1

        for sample_id in ranked:
            if len(selected) >= target:
                break
            selected.add(sample_id)
        return selected

    def _human_cohort_ids(self, scenario: str) -> set[str]:
        return self.human_cohorts[scenario]

    def queue_ids(self, scenario: str, stage: Stage) -> list[str]:
        if scenario not in SCENARIOS:
            raise Week5DataError(f"unsupported scenario: {scenario}")
        if stage == "human":
            return sorted(self._human_cohort_ids(scenario) - set(self.annotations[scenario]))

        passed = self._passed(scenario)
        ids: list[str] = []
        for sample_id, annotation in self.annotations[scenario].items():
            revision = int(annotation["revision"])
            key = (sample_id, revision)
            if key not in passed["self_review"]:
                continue
            if stage == "cross_review":
                if (
                    qc_cross_review_selected(sample_id, scenario, self.config)
                    and key not in passed["cross_review"]
                ):
                    ids.append(sample_id)
            elif (
                qc_audit_selected(sample_id, scenario, self.config)
                and key in passed["cross_review"]
                and key not in passed["core_audit"]
            ):
                ids.append(sample_id)
        return sorted(ids)

    def task(self, scenario: str, stage: Stage, sample_id: str) -> dict[str, Any]:
        if sample_id not in self.queue_ids(scenario, stage):
            raise Week5DataError("task is not currently ready for this stage")
        candidate = deepcopy(self.candidates[scenario][sample_id])
        task: dict[str, Any] = {
            "sample_id": sample_id,
            "scenario": scenario,
            "stage": stage,
            "input": candidate.get("input", {}),
            "sampling_metadata": candidate.get("sampling_metadata", {}),
            "isolation": candidate.get("isolation", {}),
            "image_url": self._image_url(candidate),
        }
        if stage == "human":
            model_preannotation = deepcopy(
                self.preannotations[scenario][sample_id].get("parsed_output")
            )
            task["model_preannotation"] = model_preannotation
            task["human_draft"], task["draft_warnings"] = self._human_draft(
                scenario, model_preannotation
            )
        else:
            annotation = self.annotations[scenario][sample_id]
            task["annotation_revision"] = int(annotation["revision"])
            task["human_annotation"] = deepcopy(annotation["human_annotation"])
        return task

    def _human_draft(
        self, scenario: str, model_preannotation: Any
    ) -> tuple[Any, list[str]]:
        """移除模型生成的非受控标签，同时保留原预标注作为只读证据。"""
        draft = deepcopy(model_preannotation)
        if not isinstance(draft, dict):
            return draft, []
        tool = json.loads(
            (self.root / "configs/week5/annotation_tool.json").read_text(
                encoding="utf-8"
            )
        )
        vocabularies = tool.get("label_vocabularies", {})
        fields = {
            "image_product_search": ("style_tags", "visible_facilities"),
            "after_sales": (),
            "itinerary_planning": ("style_preferences",),
        }[scenario]
        warnings: list[str] = []
        for field in fields:
            values = draft.get(field, [])
            if not isinstance(values, list):
                continue
            allowed = set(vocabularies.get(field, []))
            removed = [value for value in values if value not in allowed]
            if removed:
                draft[field] = [value for value in values if value in allowed]
                warnings.append(
                    f"{field} 已从人工草稿移除非受控模型标签: "
                    + ", ".join(map(str, removed))
                )
        return draft, warnings

    def _image_url(self, candidate: dict[str, Any]) -> str | None:
        image = candidate.get("image")
        path = image.get("path") if isinstance(image, dict) else None
        if not path:
            images = candidate.get("input", {}).get("images", [])
            if images and isinstance(images[0], dict):
                path = images[0].get("path")
        return f"/api/image?path={path}" if isinstance(path, str) and path else None

    def image_path(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if not candidate.is_relative_to(self.root) or not candidate.is_file():
            raise Week5DataError("image path is unavailable")
        return candidate

    def summary(self) -> dict[str, Any]:
        queues = {
            scenario: {
                stage: len(self.queue_ids(scenario, stage))
                for stage in ("human", "cross_review", "core_audit")
            }
            for scenario in SCENARIOS
        }
        return {
            "queues": queues,
            "preannotations": {
                scenario: len(self.preannotations[scenario]) for scenario in SCENARIOS
            },
            "human_revisions": {
                scenario: len(self.annotations[scenario]) for scenario in SCENARIOS
            },
            "human_review_plan": {
                scenario: {
                    "target": len(self._human_cohort_ids(scenario)),
                    "completed": len(
                        set(self.annotations[scenario]) & self._human_cohort_ids(scenario)
                    ),
                }
                for scenario in SCENARIOS
            },
        }

    def _submission_path(self, kind: str) -> Path:
        directory = self.output_dir / "human_tasks" / "station_submissions"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{kind}-{uuid.uuid4().hex}.jsonl"

    def save_human(self, submission: HumanSubmission) -> dict[str, int]:
        if not submission.self_review_confirmed:
            raise Week5DataError("保存前必须由本人明确确认已完成逐项自审")
        with self._lock:
            if submission.sample_id not in self.queue_ids(submission.scenario, "human"):
                raise Week5DataError("sample is not awaiting human annotation")
            payload = submission.model_dump()
            payload["corrected_at"] = _now()
            path = self._submission_path("human")
            path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = apply_human_corrections(
                self.root,
                self.config,
                submission.scenario,
                path,
                cached_candidates=self.candidates[submission.scenario],
                cached_preannotated_ids=set(
                    self.preannotations[submission.scenario]
                ),
                cached_existing=list(self.annotations[submission.scenario].values()),
            )
            self._refresh_records(submission.scenario)
            return result

    def save_quality(self, submission: QualitySubmission) -> dict[str, int]:
        with self._lock:
            if submission.sample_id not in self.queue_ids(
                submission.scenario, submission.stage
            ):
                raise Week5DataError("sample is not ready for this QC stage")
            payload = submission.model_dump()
            payload["reviewed_at"] = _now()
            path = self._submission_path(submission.stage)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = apply_quality_records(
                self.root,
                self.config,
                submission.scenario,
                path,
                cached_annotations=self.annotations[submission.scenario],
                cached_existing=self.quality[submission.scenario],
            )
            self._refresh_records(submission.scenario)
            return result


def create_annotation_station(
    root: Path | None = None,
    config_path: str = "configs/week5_dataset_qwen3_vl_4b_single_operator.json",
    *,
    config: dict[str, Any] | None = None,
) -> FastAPI:
    project_root = (root or Path(__file__).resolve().parents[2]).resolve()
    active_config = config or load_week5_config(project_root, config_path)
    store = Week5AnnotationStore(project_root, active_config)
    app = FastAPI(title="Trip Week 5 人工标注与质检台", version="1.0")
    app.state.store = store

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = (project_root / "src/api/templates/week5_annotation_station.html").read_text(
            encoding="utf-8"
        )
        return HTMLResponse(html)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        # 服务启动时已加载全量只读候选；提交路径只增量刷新小型人工文件。
        # 避免每次状态查询重新扫描 80,000 条记录导致页面等待。
        return store.summary()

    @app.get("/api/tasks")
    def tasks(
        scenario: str = Query(...),
        stage: Stage = Query(...),
        offset: int = Query(0, ge=0),
        limit: int = Query(1, ge=1, le=50),
    ) -> dict[str, Any]:
        try:
            ids = store.queue_ids(scenario, stage)
            selected = ids[offset : offset + limit]
            return {
                "total": len(ids),
                "offset": offset,
                "tasks": [store.task(scenario, stage, sample_id) for sample_id in selected],
            }
        except Week5DataError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/image")
    def image(path: str) -> FileResponse:
        try:
            resolved = store.image_path(path)
        except Week5DataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        return FileResponse(resolved, media_type=media_type)

    @app.post("/api/human")
    def save_human(submission: HumanSubmission) -> dict[str, int]:
        try:
            return store.save_human(submission)
        except Week5DataError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/quality")
    def save_quality(submission: QualitySubmission) -> dict[str, int]:
        try:
            return store.save_quality(submission)
        except Week5DataError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_annotation_station()

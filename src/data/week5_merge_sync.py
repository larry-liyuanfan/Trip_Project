"""Synchronize a validated Week 5 merge into canonical preannotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.week5_dataset import SCENARIOS, Week5DataError, iter_jsonl


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")


def sync_merged_preannotations(
    *,
    merge_dir: Path,
    preserved_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build canonical files while preserving rows already reviewed by the human."""
    summary_path = merge_dir / "summary.json"
    results_path = merge_dir / "results.jsonl"
    if not summary_path.is_file() or not results_path.is_file():
        raise Week5DataError("final Week 5 merge artifacts are incomplete")
    if output_dir.exists():
        raise Week5DataError(f"refusing to overwrite preannotations: {output_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_success = int(summary.get("unique_success", -1))
    if expected_success < 0:
        raise Week5DataError("merge summary lacks unique_success")

    preserved: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        path = preserved_dir / f"{scenario}.jsonl"
        rows = list(iter_jsonl(path)) if path.is_file() else []
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or row.get("scenario") != scenario:
                raise Week5DataError(f"invalid preserved preannotation: {path}")
            if row.get("status") != "completed" or not row.get("schema_valid"):
                raise Week5DataError(f"preserved row is not schema-valid: {sample_id}")
            if sample_id in by_id:
                raise Week5DataError(f"duplicate preserved sample_id: {sample_id}")
            by_id[sample_id] = row
        preserved[scenario] = by_id

    # 先保留已被人工查看时的原始顺序与内容，再按 merge 顺序补齐其余成功项。
    merged: dict[str, list[dict[str, Any]]] = {
        scenario: list(preserved[scenario].values()) for scenario in SCENARIOS
    }
    seen: set[str] = set()
    preserved_used: set[str] = set()
    for row in iter_jsonl(results_path):
        sample_id = str(row.get("sample_id", ""))
        scenario = str(row.get("scenario", ""))
        if not sample_id or scenario not in merged:
            raise Week5DataError("merge result has invalid sample identity")
        if sample_id in seen:
            raise Week5DataError(f"duplicate merge success: {sample_id}")
        if row.get("status") != "completed" or not row.get("schema_valid"):
            raise Week5DataError(f"merge result is not schema-valid: {sample_id}")
        seen.add(sample_id)
        if sample_id in preserved[scenario]:
            preserved_used.add(sample_id)
            continue
        merged[scenario].append(row)

    if len(seen) != expected_success:
        raise Week5DataError(
            f"merge success count mismatch: expected {expected_success}, got {len(seen)}"
        )
    preserved_ids = {
        sample_id for rows in preserved.values() for sample_id in rows
    }
    if preserved_used != preserved_ids:
        missing = sorted(preserved_ids - preserved_used)
        raise Week5DataError(f"preserved samples are absent from merge: {missing[:3]}")

    try:
        output_dir.mkdir(parents=True)
        for scenario in SCENARIOS:
            _write_jsonl(output_dir / f"{scenario}.jsonl", merged[scenario])
    except Exception:
        # 新目录写入失败时保留现场，禁止静默覆盖既有 canonical 数据。
        raise

    return {
        "status": "synchronized",
        "unique_success": len(seen),
        "preserved_human_review_rows": len(preserved_used),
        "counts": {scenario: len(merged[scenario]) for scenario in SCENARIOS},
    }

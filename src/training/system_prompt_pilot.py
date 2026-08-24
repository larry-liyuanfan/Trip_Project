"""Run three prompt candidates on the fresh development lock only."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests

from src.evaluation.schema_validation import load_output_schema
from src.inference.transport_utils import normalize_image_url
from src.training.week7_data import CORE_SCENARIOS, iter_jsonl, load_week7_config, sha256_file
from src.training.week7_evaluation import summarize_raw_records


class PromptPilotError(ValueError):
    """Raised when a prompt pilot is incomplete or attempts to consume test."""


def run_prompt_pilot(
    root: Path,
    config_path: Path,
    candidates_path: Path,
    output_dir: Path,
    *,
    endpoint: str,
    served_model: str,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    root = Path(root).resolve()
    config = load_week7_config(config_path)
    lock_root = root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    rows = list(iter_jsonl(lock_root / "development.jsonl"))
    core_rows = [row for row in rows if row["scenario"] in CORE_SCENARIOS]
    counts = {scenario: sum(row["scenario"] == scenario for row in core_rows) for scenario in CORE_SCENARIOS}
    if any(count != 48 for count in counts.values()):
        raise PromptPilotError(f"prompt pilot requires 48 development rows per scenario: {counts}")
    candidates = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    versions = candidates.get("versions", {})
    if set(versions) != {"current_week7", "compact_schema_v1", "evidence_state_v1"}:
        raise PromptPilotError("prompt candidate identities changed")
    if output_dir.exists():
        raise PromptPilotError(f"prompt pilot output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    session = requests.Session()
    summaries: dict[str, Any] = {}
    for version, spec in versions.items():
        records = []
        raw_path = output_dir / f"{version}_raw.jsonl"
        with raw_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in core_rows:
                messages = _render_messages(root, row, spec)
                started = time.perf_counter()
                error = None
                raw_output = ""
                usage = {}
                try:
                    response = session.post(
                        endpoint,
                        json={
                            "model": served_model,
                            "messages": messages,
                            "temperature": 0.0,
                            "max_tokens": 3072,
                            "enable_thinking": False,
                            "response_format": {"type": "json_object"},
                        },
                        timeout=timeout_seconds,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    raw_output = payload["choices"][0]["message"]["content"]
                    usage = payload.get("usage") or {}
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                record = {
                    "sample_id": row["sample_id"],
                    "run_id": f"system_repair_prompt_pilot_{version}",
                    "model_name": served_model,
                    "raw_output": raw_output,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "failed": error is not None,
                    "error": error,
                    "usage": usage,
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        summary = summarize_raw_records(
            root,
            config,
            core_rows,
            records,
            metric_support_protocol=config["evaluation"].get("metric_support_protocol"),
        )
        summary["prompt_version"] = version
        summary["raw_sha256"] = sha256_file(raw_path)
        summaries[version] = summary

    winners = {}
    for scenario in CORE_SCENARIOS:
        candidates_for_scenario = []
        for version, summary in summaries.items():
            scenario_metrics = summary["scenarios"][scenario]
            version_records = [
                row for row in iter_jsonl(output_dir / f"{version}_raw.jsonl")
                if next(item for item in core_rows if item["sample_id"] == row["sample_id"])["scenario"] == scenario
            ]
            token_values = [
                int(record.get("usage", {}).get("total_tokens", 0))
                for record in version_records
            ]
            candidates_for_scenario.append(
                (
                    float(scenario_metrics["composite"]),
                    float(scenario_metrics.get("schema_pass", 0.0)),
                    float(scenario_metrics.get("json_compliance", 0.0)),
                    -statistics.fmean(record["latency_ms"] for record in version_records),
                    -statistics.fmean(token_values),
                    version,
                )
            )
        winners[scenario] = max(candidates_for_scenario)[-1]
    result = {
        "status": "COMPLETED",
        "split": "development",
        "counts": counts,
        "config_sha256": sha256_file(config_path),
        "prompt_candidates_sha256": sha256_file(candidates_path),
        "summaries": summaries,
        "winners": winners,
        "test_consumed": False,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def _render_messages(
    root: Path,
    row: dict[str, Any],
    spec: dict[str, str],
) -> list[dict[str, Any]]:
    scenario = row["scenario"]
    schema_version = "v2" if scenario == "itinerary_planning" else "v1"
    schema = json.dumps(
        load_output_schema(root, scenario, schema_version),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    messages = json.loads(json.dumps(row["messages"], ensure_ascii=False))
    messages[0]["content"] = f"{messages[0]['content']} {spec['system_suffix']}"
    user = messages[-1]
    content = user["content"] if isinstance(user["content"], list) else [{"type": "text", "text": user["content"]}]
    for part in content:
        if part.get("type") == "image":
            part.clear()
            part.update(
                {
                    "type": "image_url",
                    "image_url": {"url": normalize_image_url(f"file://{row['image_path']}")},
                }
            )
    content.append(
        {
            "type": "text",
            "text": f"{spec['task_suffix']} 完整 Schema：{schema}",
        }
    )
    user["content"] = content
    return messages

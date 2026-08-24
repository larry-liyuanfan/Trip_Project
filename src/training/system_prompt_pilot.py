"""Run three prompt candidates on the fresh development lock only."""

from __future__ import annotations

import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import requests

from src.evaluation.schema_validation import load_output_schema
from src.inference.system_runtime import GenerationResult
from src.inference.transport_utils import normalize_image_url
from src.training.week7_data import CORE_SCENARIOS, iter_jsonl, load_week7_config, sha256_file
from src.training.week7_evaluation import summarize_raw_records


class PromptPilotError(ValueError):
    """Raised when a prompt pilot is incomplete or attempts to consume test."""


def _prompt_summary_config(config: dict[str, Any]) -> dict[str, Any]:
    """Score the core-only Prompt pilot without requiring dialogue rows."""

    summary_config = copy.deepcopy(config)
    summary_config["evaluation"]["dialogue_automatic_gate"]["enabled"] = False
    return summary_config


def _validate_pilot_identity(
    identity: dict[str, Any],
    config_path: Path,
    candidates_path: Path,
) -> None:
    expected_counts = {scenario: 48 for scenario in CORE_SCENARIOS}
    if (
        identity.get("split") != "development"
        or identity.get("test_consumed") is not False
        or identity.get("config_sha256") != sha256_file(config_path)
        or identity.get("prompt_candidates_sha256") != sha256_file(candidates_path)
        or identity.get("counts") != expected_counts
        or not isinstance(identity.get("endpoint"), str)
        or not identity["endpoint"]
        or not isinstance(identity.get("served_model"), str)
        or not identity["served_model"]
    ):
        raise PromptPilotError("prompt pilot identity is invalid")


def load_completed_prompt_pilot(
    config_path: Path,
    candidates_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate and return an immutable completed pilot for safe resume."""

    selection_path = Path(output_dir) / "selection.json"
    identity_path = Path(output_dir) / "pilot_identity.json"
    if not selection_path.is_file():
        raise PromptPilotError("resumed prompt pilot has no selection.json")
    if not identity_path.is_file():
        raise PromptPilotError("resumed prompt pilot has no pilot_identity.json")
    _validate_pilot_identity(
        json.loads(identity_path.read_text(encoding="utf-8")),
        config_path,
        candidates_path,
    )
    result = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "COMPLETED"
        or result.get("split") != "development"
        or result.get("test_consumed") is not False
        or result.get("config_sha256") != sha256_file(config_path)
        or result.get("prompt_candidates_sha256") != sha256_file(candidates_path)
        or result.get("counts")
        != {scenario: 48 for scenario in CORE_SCENARIOS}
    ):
        raise PromptPilotError("resumed prompt pilot identity is invalid")
    summaries = result.get("summaries", {})
    if set(summaries) != {
        "current_week7",
        "compact_schema_v1",
        "evidence_state_v1",
    }:
        raise PromptPilotError("resumed prompt pilot versions changed")
    for version, summary in summaries.items():
        raw_path = Path(output_dir) / f"{version}_raw.jsonl"
        if (
            not raw_path.is_file()
            or sha256_file(raw_path) != summary.get("raw_sha256")
            or sum(1 for _ in iter_jsonl(raw_path)) != 144
        ):
            raise PromptPilotError(f"resumed prompt pilot raw evidence changed: {version}")
    return result


def run_prompt_pilot(
    root: Path,
    config_path: Path,
    candidates_path: Path,
    output_dir: Path,
    *,
    endpoint: str,
    served_model: str,
    timeout_seconds: int = 300,
    generator: Callable[..., GenerationResult] | None = None,
    resume: bool = False,
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
    identity = {
        "split": "development",
        "test_consumed": False,
        "config_sha256": sha256_file(config_path),
        "prompt_candidates_sha256": sha256_file(candidates_path),
        "counts": counts,
        "endpoint": endpoint,
        "served_model": served_model,
    }
    identity_path = output_dir / "pilot_identity.json"
    if output_dir.exists():
        if not resume or not identity_path.is_file():
            raise PromptPilotError(f"prompt pilot output already exists: {output_dir}")
        stored_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        _validate_pilot_identity(stored_identity, config_path, candidates_path)
        if stored_identity != identity:
            raise PromptPilotError("resumed prompt pilot runtime identity changed")
    else:
        output_dir.mkdir(parents=True)
        identity_path.write_text(
            json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    session = requests.Session()
    summaries: dict[str, Any] = {}
    summary_config = _prompt_summary_config(config)
    expected_sample_ids = {row["sample_id"] for row in core_rows}
    for version, spec in versions.items():
        raw_path = output_dir / f"{version}_raw.jsonl"
        if raw_path.exists():
            records = list(iter_jsonl(raw_path))
            if (
                len(records) != len(core_rows)
                or {record.get("sample_id") for record in records}
                != expected_sample_ids
                or any(
                    record.get("run_id")
                    != f"system_repair_prompt_pilot_{version}"
                    or record.get("model_name") != served_model
                    for record in records
                )
            ):
                raise PromptPilotError(
                    f"resumed prompt version is incomplete or changed: {version}"
                )
        else:
            records = []
            consecutive_failures = 0
            with raw_path.open("x", encoding="utf-8", newline="\n") as handle:
                for row in core_rows:
                    messages = _render_messages(root, row, spec)
                    started = time.perf_counter()
                    error = None
                    raw_output = ""
                    usage = {}
                    try:
                        if generator is not None:
                            generated = generator(
                                messages,
                                response_format={"type": "json_object"},
                                max_new_tokens=3072,
                            )
                            raw_output = generated.content
                            usage = {
                                "prompt_tokens": generated.input_tokens,
                                "completion_tokens": generated.output_tokens,
                                "total_tokens": (
                                    generated.input_tokens + generated.output_tokens
                                ),
                            }
                        else:
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
                    handle.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    if error is None:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            raise PromptPilotError(
                                "prompt pilot stopped after 3 consecutive model failures: "
                                f"{error}"
                            )
        summary = summarize_raw_records(
            root,
            summary_config,
            core_rows,
            records,
            metric_support_protocol=config["evaluation"].get("metric_support_protocol"),
        )
        if float(summary["failure_rate"]) > 0.02:
            raise PromptPilotError(
                f"prompt version request failure rate exceeds 2%: {version}"
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

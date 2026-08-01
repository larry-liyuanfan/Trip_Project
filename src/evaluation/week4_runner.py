"""Immutable Week 4 prompt pilot and winner-run execution."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.yelp_paths import parse_simple_yaml
from src.evaluation.config import load_evaluation_config
from src.evaluation.manifests import load_configured_manifests
from src.evaluation.prompting import render_standard_prompt
from src.evaluation.provenance import canonical_sha256
from src.evaluation.results import ImmutableRunWriter, parse_and_validate_output
from src.evaluation.runner import (
    _build_chat_payload,
    chat_completions_url,
    load_runtime_settings,
    post_chat_completion,
    select_inference_records,
)
from src.evaluation.week4_prompting import (
    WEEK4_PROMPT_VERSIONS,
    load_week4_selection,
    render_week4_request,
    validate_demo_dev_isolation,
    validate_selection_records,
)


PROMPT_CANDIDATES = {"standardized_v2", *WEEK4_PROMPT_VERSIONS}


class Week4RunError(ValueError):
    """Raised when a Week 4 prompt run is not reproducible or safe."""


def load_week4_config(root: Path, path: Path | str) -> dict[str, Any]:
    """Load the small Week 4 orchestration config."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(root) / resolved
    payload = parse_simple_yaml(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Week4RunError("Week 4 config must be a mapping")
    candidates = payload.get("prompt_candidates")
    if not isinstance(candidates, list) or set(candidates) != PROMPT_CANDIDATES:
        raise Week4RunError("Week 4 config must declare exactly three candidates")
    return payload


def run_week4_prompt_evaluation(
    *,
    root: Path,
    config_path: Path,
    run_id: str,
    stage: str,
    variants_by_scenario: dict[str, str],
) -> dict[str, Any]:
    """Run fixed pilot candidates or per-scenario full winners."""
    if stage not in {"pilot", "full"}:
        raise Week4RunError("stage must be pilot or full")
    project_root = Path(root)
    week4 = load_week4_config(project_root, config_path)
    week3_path = project_root / week4["paths"]["week3_config"]
    week3 = load_evaluation_config(week3_path)
    configured = load_configured_manifests(week3, root=project_root)
    records = [
        record
        for scenario in week3["scenarios"]
        for record in configured[scenario]
    ]
    records_by_id = {record["sample_id"]: record for record in records}
    demo_dev_path = project_root / week4["paths"]["demo_dev_config"]
    demo_dev = load_evaluation_config(demo_dev_path)
    demo_dev_configured = load_configured_manifests(
        demo_dev,
        root=project_root,
    )
    example_records = [
        record
        for scenario in demo_dev["scenarios"]
        for record in demo_dev_configured[scenario]
    ]
    example_records_by_id = {
        record["sample_id"]: record for record in example_records
    }
    validate_demo_dev_isolation(example_records, records)
    selection_path = project_root / week4["paths"]["selection_config"]
    selection = load_week4_selection(selection_path)
    validate_selection_records(
        selection,
        example_records_by_id,
        records_by_id,
    )
    if set(variants_by_scenario) != set(week3["scenarios"]):
        raise Week4RunError("one prompt variant is required for every scenario")
    for scenario, variant in variants_by_scenario.items():
        if variant not in PROMPT_CANDIDATES:
            raise Week4RunError(f"unsupported prompt candidate for {scenario}: {variant}")
        if stage == "pilot" and len(set(variants_by_scenario.values())) != 1:
            raise Week4RunError("a pilot run compares one candidate on all scenarios")

    if stage == "pilot":
        selected = [
            records_by_id[sample_id]
            for scenario in week3["scenarios"]
            for sample_id in selection["scenarios"][scenario]["pilot_sample_ids"]
        ]
    else:
        selected = select_inference_records(records)
    runtime = load_runtime_settings(project_root, week3)
    output_root = project_root / week4["paths"]["output_dir"]
    artifact_hashes = _artifact_hashes(
        project_root,
        week3,
        demo_dev,
        selection_path,
        set(variants_by_scenario.values()),
    )
    metadata = {
        "run_id": run_id,
        "mode": "live",
        "prompt_version": (
            next(iter(variants_by_scenario.values()))
            if stage == "pilot"
            else "week4_per_scenario_winners_v1"
        ),
        "prompt_versions_by_scenario": dict(variants_by_scenario),
        "run_scope": stage,
        "model_name": runtime["model_name"],
        "model_config": {
            "model": copy.deepcopy(runtime["model_config"]),
            "served_model_name": runtime["served_model_name"],
            "generation": copy.deepcopy(runtime["generation"]),
            "live_base_url": runtime["live_base_url"],
            "timeout_seconds": runtime["timeout_seconds"],
        },
        "dataset_version": week3["dataset_version"],
        "selection_version": selection["version"],
        "concurrency": (
            int(week4["runtime"].get("full_concurrency", 1))
            if stage == "full"
            else 1
        ),
        "artifact_hashes": artifact_hashes,
        "selected_sample_ids_sha256": canonical_sha256(
            [record["sample_id"] for record in selected]
        ),
        "selected_count": len(selected),
    }

    def execute(record: dict[str, Any]) -> dict[str, Any]:
        return _execute_record(
            project_root=project_root,
            record=record,
            run_id=run_id,
            prompt_version=variants_by_scenario[record["scenario"]],
            selection=selection,
            records_by_id=example_records_by_id,
            runtime=runtime,
            model_config=metadata["model_config"],
        )

    request_errors: list[dict[str, Any]] = []
    with ImmutableRunWriter(output_root / "runs", run_id, metadata) as writer:
        if metadata["concurrency"] == 1:
            result_iterator = map(execute, selected)
            for result in result_iterator:
                writer.write(result)
                _collect_model_request_error(result, request_errors)
        else:
            with ThreadPoolExecutor(max_workers=metadata["concurrency"]) as executor:
                for result in executor.map(execute, selected):
                    writer.write(result)
                    _collect_model_request_error(result, request_errors)
        writer.metadata["model_request_error_count"] = len(request_errors)
        _ensure_no_model_request_errors(request_errors)
    return json.loads(
        (output_root / "runs" / run_id / "metadata.json").read_text(encoding="utf-8")
    )


def _execute_record(
    *,
    project_root: Path,
    record: dict[str, Any],
    run_id: str,
    prompt_version: str,
    selection: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    runtime: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    scenario = record["scenario"]
    rendered = _render(
        project_root,
        record,
        prompt_version,
        selection,
        records_by_id,
    )
    payload = _build_chat_payload(project_root, rendered, runtime)
    endpoint = chat_completions_url(runtime["live_base_url"])
    started = time.perf_counter()
    raw_output: str | None = None
    usage = _empty_usage()
    try:
        response_payload = post_chat_completion(
            endpoint,
            payload,
            runtime["timeout_seconds"],
        )
        raw_output = response_payload["choices"][0]["message"]["content"]
        if not isinstance(raw_output, str):
            raise Week4RunError("chat completion content must be text")
        usage = _normalize_usage(response_payload.get("usage"))
        parsed = parse_and_validate_output(
            project_root,
            scenario,
            raw_output,
            "v2" if scenario == "itinerary_planning" else "v1",
        )
    except Exception as exc:
        parsed = {
            "parsed_output": None,
            "json_valid": False,
            "schema_valid": False,
            "error": f"model_request_error: {type(exc).__name__}: {exc}",
        }
    return {
        "run_id": run_id,
        "sample_id": record["sample_id"],
        "scenario": scenario,
        "mode": "live",
        "model_name": runtime["model_name"],
        "model_config": model_config,
        "prompt_version": prompt_version,
        "request_sha256": canonical_sha256(rendered),
        "prompt_artifact_sha256": canonical_sha256(
            {
                "prompt_version": prompt_version,
                "messages": rendered["messages"],
                "schema": rendered.get("output_schema"),
            }
        ),
        "input_sha256": canonical_sha256(record["input"]),
        "input_metadata": copy.deepcopy(record["input"]),
        "example_ids": list(rendered.get("example_ids", [])),
        "raw_output": raw_output,
        "parsed_output": parsed["parsed_output"],
        "json_valid": parsed["json_valid"],
        "schema_valid": parsed["schema_valid"],
        "latency_ms": (time.perf_counter() - started) * 1000,
        "token_usage": usage,
        "error": parsed["error"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _render(
    root: Path,
    record: dict[str, Any],
    prompt_version: str,
    selection: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if prompt_version == "standardized_v2":
        return render_standard_prompt(
            root,
            record["scenario"],
            record["input"],
            version="standardized_v2",
        )
    return render_week4_request(
        root,
        record["scenario"],
        record["input"],
        prompt_version=prompt_version,
        selection=selection,
        records_by_id=records_by_id,
    )


def _normalize_usage(value: Any) -> dict[str, int | None]:
    if not isinstance(value, dict):
        return _empty_usage()
    normalized: dict[str, int | None] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        count = value.get(name)
        normalized[name] = (
            count
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0
            else None
        )
    return normalized


def _empty_usage() -> dict[str, None]:
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def _collect_model_request_error(
    result: dict[str, Any],
    errors: list[dict[str, Any]],
) -> None:
    error = result.get("error")
    if isinstance(error, str) and error.startswith("model_request_error:"):
        errors.append(
            {
                "sample_id": result.get("sample_id"),
                "scenario": result.get("scenario"),
                "error": error,
            }
        )


def _ensure_no_model_request_errors(errors: list[dict[str, Any]]) -> None:
    """任何模型请求失败都会使 pilot/full run 失败，不能伪装成有效候选。"""
    if not errors:
        return
    scenario_counts: dict[str, int] = {}
    for row in errors:
        scenario = str(row.get("scenario"))
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    raise Week4RunError(
        "model requests failed; run is ineligible: "
        f"count={len(errors)}, scenarios={dict(sorted(scenario_counts.items()))}"
    )


def _artifact_hashes(
    root: Path,
    week3: dict[str, Any],
    demo_dev: dict[str, Any],
    selection_path: Path,
    prompt_versions: set[str],
) -> dict[str, str]:
    paths = [selection_path]
    for settings in week3["scenarios"].values():
        paths.append(root / settings["manifest_path"])
    for settings in demo_dev["scenarios"].values():
        paths.append(root / settings["manifest_path"])
    paths.extend(
        [
            root / "configs/evaluation/schemas/image_product_search_v1.schema.json",
            root / "configs/evaluation/schemas/after_sales_v1.schema.json",
            root / "configs/evaluation/schemas/itinerary_planning_v2.schema.json",
        ]
    )
    for version in prompt_versions:
        directory = (
            root
            / "configs"
            / "evaluation"
            / "prompts"
            / ("standardized_v2" if version == "standardized_v2" else "week4_optimized_v2")
        )
        paths.extend(sorted(directory.glob("*.yaml")))
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(set(paths))
    }

"""Week 8 fixed-input dialogue routing and product latency benchmarks."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.inference.schemas import DialogueModelOutput, DialogueRequest, TaskRequest
from src.inference.system_runtime import (
    ModelGenerationError,
    ReleaseSettings,
    ScenarioService,
)
from src.inference.transport_utils import strip_json_fence


class Week8RuntimeBenchmarkError(ValueError):
    """Raised when fixed benchmark identity or inputs are invalid."""


def load_runtime_benchmark_config(root: Path, path: Path) -> dict[str, Any]:
    """Load the versioned benchmark config and validate its bounded protocol."""

    selected = path if path.is_absolute() else root / path
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week8RuntimeBenchmarkError(
            f"invalid runtime benchmark config: {selected}"
        ) from exc
    if payload.get("schema_version") not in {
        "week8_runtime_optimization_config_v1",
        "week8_runtime_optimization_config_v2",
    }:
        raise Week8RuntimeBenchmarkError("unexpected runtime benchmark schema_version")
    profiles = payload.get("dialogue", {}).get("profiles", [])
    if not isinstance(profiles, list) or len(profiles) != 2:
        raise Week8RuntimeBenchmarkError("dialogue benchmark requires two profiles")
    if {item.get("role") for item in profiles} != {"current", "candidate"}:
        raise Week8RuntimeBenchmarkError(
            "dialogue profiles must define current and candidate"
        )
    cases = payload.get("dialogue", {}).get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise Week8RuntimeBenchmarkError("dialogue cases must be non-empty")
    case_ids = [item.get("case_id") for item in cases]
    if any(not isinstance(item, str) or not item for item in case_ids):
        raise Week8RuntimeBenchmarkError("dialogue case_id must be non-empty text")
    if len(set(case_ids)) != len(case_ids):
        raise Week8RuntimeBenchmarkError("dialogue case_id must be unique")
    latency = payload.get("product_latency", {})
    if int(latency.get("warmup_runs", 0)) < 1 or int(latency.get("measured_runs", 0)) < 3:
        raise Week8RuntimeBenchmarkError(
            "product latency requires at least one warmup and three measured runs"
        )
    image = root / str(latency.get("image", ""))
    if not image.is_file():
        raise Week8RuntimeBenchmarkError(f"fixed product image is missing: {image}")
    return payload


def run_dialogue_first_turn_comparison(
    settings: ReleaseSettings,
    backend: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compare current and candidate routing on identical fixed first turns."""

    records: list[dict[str, Any]] = []
    dialogue = config["dialogue"]
    for profile in dialogue["profiles"]:
        service = ScenarioService(
            settings,
            backend,
            dialogue_prompt_version=str(profile["prompt_version"]),
            dialogue_max_new_tokens=int(profile["max_new_tokens"]),
        )
        for case in dialogue["cases"]:
            request_payload = dict(case["request"])
            request_payload["image_urls"] = [
                str((settings.root / path).resolve())
                for path in request_payload.get("image_urls", [])
            ]
            request = DialogueRequest.model_validate(request_payload)
            started = time.perf_counter()
            try:
                response = service.run_dialogue(request)
                attempts = response.attempts
                expected_state = case.get("expected_state", {})
                recalled, matched, supported = _expected_state_score(
                    response.state,
                    expected_state,
                )
                record = {
                    "case_id": case["case_id"],
                    "profile": profile["role"],
                    "prompt_version": profile["prompt_version"],
                    "success": True,
                    "first_turn_three_key_compliant": bool(
                        attempts and attempts[0].error is None
                    ),
                    "correction_triggered": len(attempts) > 1,
                    "context_state_recalled": recalled,
                    "context_state_matched": matched,
                    "context_state_support": supported,
                    "latency_ms": response.total_latency_ms,
                    "wall_latency_ms": (time.perf_counter() - started) * 1000,
                    "response": response.model_dump(),
                    "error": None,
                }
            except ModelGenerationError as exc:
                attempts = list(exc.attempts)
                record = {
                    "case_id": case["case_id"],
                    "profile": profile["role"],
                    "prompt_version": profile["prompt_version"],
                    "success": False,
                    "first_turn_three_key_compliant": False,
                    "correction_triggered": len(attempts) > 1,
                    "context_state_recalled": 0,
                    "context_state_matched": 0,
                    "context_state_support": len(case.get("expected_state", {})),
                    "latency_ms": sum(item.latency_ms for item in attempts),
                    "wall_latency_ms": (time.perf_counter() - started) * 1000,
                    "response": None,
                    "attempts": [item.model_dump() for item in attempts],
                    "error": str(exc),
                }
            records.append(record)
    return {
        "run_id": dialogue["run_id"],
        "sample_count_per_profile": len(dialogue["cases"]),
        "profiles": {
            profile["role"]: _summarize_dialogue_records(
                [row for row in records if row["profile"] == profile["role"]]
            )
            for profile in dialogue["profiles"]
        },
        "records": records,
    }


def run_product_latency_benchmark(
    settings: ReleaseSettings,
    backend: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Measure fixed-image product latency without changing model or adapter."""

    protocol = config["product_latency"]
    image = str((settings.root / protocol["image"]).resolve())
    request = TaskRequest(
        image_urls=[image],
        text_context=protocol.get("text_context"),
    )
    results: dict[str, Any] = {}
    image_processor = getattr(getattr(backend, "_processor", None), "image_processor", None)
    original_max_pixels = getattr(image_processor, "max_pixels", None)
    for profile in protocol["profiles"]:
        max_pixels = profile.get("visual_max_pixels")
        if image_processor is not None and max_pixels is not None:
            image_processor.max_pixels = int(max_pixels)
        elif image_processor is not None and original_max_pixels is not None:
            image_processor.max_pixels = original_max_pixels
        scenario_limits = dict(settings.max_new_tokens_by_scenario)
        scenario_limits["image_product_search"] = int(profile["max_new_tokens"])
        profile_settings = replace(
            settings,
            max_new_tokens_by_scenario=scenario_limits,
        )
        service = ScenarioService(profile_settings, backend)
        for _ in range(int(protocol["warmup_runs"])):
            service.run_task("image_product_search", request)
        torch_module = getattr(backend, "_torch", None)
        if torch_module is not None and torch_module.cuda.is_available():
            torch_module.cuda.reset_peak_memory_stats()
        records = []
        for index in range(int(protocol["measured_runs"])):
            try:
                response = service.run_task("image_product_search", request)
                records.append(
                    {
                        "run_index": index,
                        "success": True,
                        "schema_valid": response.schema_valid,
                        "latency_ms": response.total_latency_ms,
                        "attempts": [item.model_dump() for item in response.attempts],
                        "result": response.result,
                        "error": None,
                    }
                )
            except ModelGenerationError as exc:
                records.append(
                    {
                        "run_index": index,
                        "success": False,
                        "schema_valid": False,
                        "latency_ms": sum(item.latency_ms for item in exc.attempts),
                        "attempts": [item.model_dump() for item in exc.attempts],
                        "result": None,
                        "error": str(exc),
                    }
                )
        peak_allocated = None
        peak_reserved = None
        if torch_module is not None and torch_module.cuda.is_available():
            peak_allocated = int(torch_module.cuda.max_memory_allocated())
            peak_reserved = int(torch_module.cuda.max_memory_reserved())
        results[profile["role"]] = {
            "settings": profile,
            "metrics": _summarize_latency_records(records),
            "peak_gpu_memory_allocated_bytes": peak_allocated,
            "peak_gpu_memory_reserved_bytes": peak_reserved,
            "records": records,
        }
    current_records = results.get("current", {}).get("records", [])
    for role, payload in results.items():
        paired = list(zip(current_records, payload["records"]))
        comparable = [
            (current, candidate)
            for current, candidate in paired
            if current["success"] and candidate["success"]
        ]
        payload["quality_consistency"] = {
            "paired_success_support": len(comparable),
            "exact_result_match_rate": _ratio(
                sum(
                    current["result"] == candidate["result"]
                    for current, candidate in comparable
                ),
                len(comparable),
            ),
        }
    if image_processor is not None and original_max_pixels is not None:
        image_processor.max_pixels = original_max_pixels
    return {
        "run_id": protocol["run_id"],
        "fixed_image": protocol["image"],
        "warmup_runs": int(protocol["warmup_runs"]),
        "measured_runs_per_profile": int(protocol["measured_runs"]),
        "model_reused": True,
        "processor_reused": True,
        "profiles": results,
    }


def _expected_state_score(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[int, int, int]:
    recalled = sum(key in actual for key in expected)
    matched = sum(actual.get(key) == value for key, value in expected.items())
    return int(recalled), int(matched), len(expected)


def _summarize_dialogue_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    recalled = sum(int(row["context_state_recalled"]) for row in records)
    matched = sum(int(row["context_state_matched"]) for row in records)
    support = sum(int(row["context_state_support"]) for row in records)
    latencies = [float(row["latency_ms"]) for row in records]
    return {
        "sample_count": count,
        "first_turn_format_compliance": _ratio(
            sum(bool(row["first_turn_three_key_compliant"]) for row in records), count
        ),
        "correction_trigger_rate": _ratio(
            sum(bool(row["correction_triggered"]) for row in records), count
        ),
        "context_recall": _ratio(recalled, support),
        "context_state_value_accuracy": _ratio(matched, support),
        "context_state_support": support,
        "failure_rate": _ratio(
            sum(not bool(row["success"]) for row in records), count
        ),
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
    }


def _summarize_latency_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in records]
    successful = [row for row in records if row["success"]]
    attempts = [attempt for row in records for attempt in row["attempts"]]
    input_tokens = [
        int(item["input_tokens"])
        for item in attempts
        if item.get("input_tokens") is not None
    ]
    output_tokens = [
        int(item["output_tokens"])
        for item in attempts
        if item.get("output_tokens") is not None
    ]
    generation_seconds = sum(float(item["latency_ms"]) for item in attempts) / 1000
    return {
        "sample_count": len(records),
        "success_count": len(successful),
        "failure_rate": _ratio(len(records) - len(successful), len(records)),
        "schema_pass_rate": _ratio(
            sum(bool(row["schema_valid"]) for row in records), len(records)
        ),
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "input_tokens_total": sum(input_tokens) if input_tokens else None,
        "output_tokens_total": sum(output_tokens) if output_tokens else None,
        "output_tokens_per_second": (
            sum(output_tokens) / generation_seconds
            if output_tokens and generation_seconds > 0
            else None
        ),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def first_attempt_is_three_key_json(raw: str) -> bool:
    """Expose the exact first-turn contract for deterministic unit tests."""

    try:
        parsed = json.loads(strip_json_fence(raw))
        DialogueModelOutput.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError):
        return False
    return set(parsed) == {"reply", "state_updates", "tool_calls"}

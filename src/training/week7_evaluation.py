"""Week 7 raw-output scoring, development evaluation, and one-shot test gate."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from src.evaluation.metrics import aggregate_scenario_scores, load_metric_aliases, score_sample
from src.evaluation.schema_validation import SchemaValidationError, load_output_schema, validate_output
from src.training.week7_data import CORE_SCENARIOS, Week7DataError, canonical_sha256, iter_jsonl, sha256_file


class Week7EvaluationError(ValueError):
    """Raised when evaluation identities, outputs, or gates are invalid."""


def strict_parse_output(root: Path, scenario: str, raw_output: str) -> tuple[dict[str, Any] | None, bool, bool, str | None]:
    """Parse the full raw string once; never extract or repair a JSON fragment."""
    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError) as exc:
        return None, False, False, str(exc)
    if not isinstance(parsed, dict):
        return None, True, False, "top-level output is not an object"
    try:
        validate_output(root, scenario, parsed, "v1")
    except SchemaValidationError as exc:
        return parsed, True, False, str(exc)
    return parsed, True, True, None


def _annotation(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row["target"])
    if row["scenario"] == "after_sales":
        value["ocr_ground_truth"] = value.get("ocr_text")
    return value


def _weighted_metric(aggregate: dict[str, Any], weights: dict[str, float]) -> tuple[float, dict[str, int]]:
    numerator = 0.0
    denominator = 0.0
    support: dict[str, int] = {}
    for name, weight in weights.items():
        key = f"{name}_macro" if f"{name}_macro" in aggregate else name
        value = aggregate.get(key)
        support_key = f"{key}_support_count"
        support[name] = int(aggregate.get(support_key, aggregate.get("sample_count", 0)))
        if isinstance(value, (int, float)):
            numerator += float(weight) * float(value)
            denominator += float(weight)
    if denominator <= 0:
        raise Week7EvaluationError("no supported metric entered the scenario composite")
    return numerator / denominator, support


def _dialogue_terms(value: Any) -> set[str]:
    terms: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            terms.update(_dialogue_terms(child))
    elif isinstance(value, list):
        for child in value:
            terms.update(_dialogue_terms(child))
    elif isinstance(value, str) and len(value.strip()) >= 4:
        terms.add(value.strip().casefold())
    return terms


def score_dialogue_record(row: dict[str, Any], raw_output: str, latency_ms: float, failed: bool) -> dict[str, Any]:
    expected = _dialogue_terms(row["target"])
    normalized = raw_output.casefold()
    recalled = sum(term in normalized for term in expected)
    parsed = None
    json_valid = False
    try:
        parsed = json.loads(raw_output)
        json_valid = isinstance(parsed, dict)
    except json.JSONDecodeError:
        pass
    return {
        "sample_id": row["sample_id"], "scenario": "dialogue", "latency_ms": latency_ms,
        "failed": failed, "format_compliance": float(json_valid),
        "context_recall": recalled / len(expected) if expected else 1.0,
        "historical_image_reference": float(any(term in normalized for term in expected)),
        "requirement_update": float("预算" in normalized or "budget" in normalized),
        "context_carryover": float(recalled >= min(2, len(expected))),
        "logical_consistency": None,
        "parsed_output": parsed,
        "human_required": True,
    }


def summarize_raw_records(root: Path, config: dict[str, Any], rows: Iterable[dict[str, Any]], records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows_by_id = {row["sample_id"]: row for row in rows}
    records_list = list(records)
    if len(records_list) != len(rows_by_id) or {record.get("sample_id") for record in records_list} != set(rows_by_id):
        raise Week7EvaluationError("raw records do not exactly cover the selected rows")
    aliases = load_metric_aliases(root / "configs/evaluation/metric_aliases_v1.json")
    sample_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dialogue_scores = []
    failures = 0
    latencies = []
    for record in records_list:
        row = rows_by_id[record["sample_id"]]
        raw = record.get("raw_output")
        if not isinstance(raw, str):
            raw = ""
        latency = float(record.get("latency_ms", 0.0))
        failed = bool(record.get("failed"))
        failures += failed
        latencies.append(latency)
        if row["scenario"] == "dialogue":
            dialogue_scores.append(score_dialogue_record(row, raw, latency, failed))
            continue
        parsed, json_valid, schema_valid, error = strict_parse_output(root, row["scenario"], raw)
        result = {
            "run_id": record.get("run_id"), "sample_id": row["sample_id"], "scenario": row["scenario"],
            "model_name": record.get("model_name"), "prompt_version": "week7_locked_v1",
            "raw_output": raw, "parsed_output": parsed, "json_valid": json_valid,
            "schema_valid": schema_valid, "parse_or_schema_error": error, "latency_ms": latency,
        }
        sample_scores[row["scenario"]].append(score_sample(result, _annotation(row), aliases))
    scenario_results = {}
    scenario_composites = {}
    for scenario in CORE_SCENARIOS:
        aggregate = aggregate_scenario_scores(sample_scores[scenario])
        composite, support = _weighted_metric(aggregate, config["evaluation"]["metric_weights"][scenario])
        scenario_results[scenario] = {"aggregate": aggregate, "composite": composite, "metric_support": support}
        scenario_composites[scenario] = composite
    weighted = sum(float(config["evaluation"]["scenario_weights"][scenario]) * scenario_composites[scenario] for scenario in CORE_SCENARIOS)
    dialogue_summary = None
    if dialogue_scores:
        dialogue_summary = {
            "sample_count": len(dialogue_scores),
            "format_compliance": statistics.fmean(item["format_compliance"] for item in dialogue_scores),
            "context_recall": statistics.fmean(item["context_recall"] for item in dialogue_scores),
            "human_dimensions_status": "PENDING_REAL_HUMAN_INPUT",
            "scores": dialogue_scores,
        }
    return {
        "sample_count": len(records_list), "weighted_composite": weighted,
        "scenarios": scenario_results, "dialogue": dialogue_summary,
        "latency_ms_mean": statistics.fmean(latencies), "latency_ms_median": statistics.median(latencies),
        "failure_count": failures, "failure_rate": failures / len(records_list),
    }


def compare_schema_decoding(root: Path, config: dict[str, Any], rows: list[dict[str, Any]], free_records: list[dict[str, Any]], constrained_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare only format/Schema/latency/fallback; semantic fields are intentionally absent."""
    selected = [row for row in rows if row["scenario"] in CORE_SCENARIOS]
    row_ids = {row["sample_id"] for row in selected}
    output = {"scope": "format_only", "semantic_claims": "FORBIDDEN", "sample_count": len(selected), "modes": {}}
    for name, records in (("free", free_records), ("constrained", constrained_records)):
        if {item.get("sample_id") for item in records} != row_ids or len(records) != len(selected):
            raise Week7EvaluationError(f"{name} schema experiment identity mismatch")
        counts = Counter()
        latencies = []
        for record in records:
            row = next(item for item in selected if item["sample_id"] == record["sample_id"])
            _, json_valid, schema_valid, _ = strict_parse_output(root, row["scenario"], str(record.get("raw_output") or ""))
            counts["json_valid"] += json_valid
            counts["schema_valid"] += schema_valid
            counts["fallback_failure"] += bool(record.get("failed") or record.get("fallback_failed"))
            latencies.append(float(record.get("latency_ms", 0.0)))
        output["modes"][name] = {
            "json_compliance": counts["json_valid"] / len(selected),
            "schema_coverage": counts["schema_valid"] / len(selected),
            "fallback_failure_rate": counts["fallback_failure"] / len(selected),
            "latency_ms_mean": statistics.fmean(latencies),
        }
    free = output["modes"]["free"]
    constrained = output["modes"]["constrained"]
    output["deltas"] = {
        "json_compliance_absolute": constrained["json_compliance"] - free["json_compliance"],
        "schema_coverage_absolute": constrained["schema_coverage"] - free["schema_coverage"],
        "latency_ratio": constrained["latency_ms_mean"] / free["latency_ms_mean"] if free["latency_ms_mean"] else None,
        "fallback_failure_absolute": constrained["fallback_failure_rate"] - free["fallback_failure_rate"],
    }
    output["gate"] = {
        "latency": output["deltas"]["latency_ratio"] is not None and output["deltas"]["latency_ratio"] <= float(config["evaluation"]["schema_decoding"]["max_latency_ratio"]),
        "fallback": constrained["fallback_failure_rate"] <= float(config["evaluation"]["schema_decoding"]["max_fallback_failure_rate"]),
    }
    return output


def enforce_test_once(lock_root: Path, parameter_lock: Path, run_id: str) -> dict[str, Any]:
    """Atomically consume the final-test allowance after a complete parameter lock exists."""
    lock_root, parameter_lock = Path(lock_root), Path(parameter_lock)
    marker = lock_root / "test_consumed.json"
    if marker.exists():
        raise Week7EvaluationError("Week 7 test has already been consumed")
    payload = json.loads(parameter_lock.read_text(encoding="utf-8"))
    required = {"status", "config_sha256", "dataset_lock_sha256", "selected_checkpoint", "selected_checkpoint_sha256"}
    if payload.get("status") != "LOCKED" or not required <= set(payload):
        raise Week7EvaluationError("complete parameter lock is required before test")
    marker_payload = {
        "status": "CONSUMED", "run_id": run_id,
        "parameter_lock_sha256": sha256_file(parameter_lock),
        "test_file_sha256": sha256_file(lock_root / "test.jsonl"),
    }
    marker.write_text(json.dumps(marker_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return marker_payload

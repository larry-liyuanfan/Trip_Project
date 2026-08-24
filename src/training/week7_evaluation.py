"""Week 7 raw-output scoring, development evaluation, and one-shot test gate."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from src.evaluation.metrics import (
    WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL,
    aggregate_scenario_scores,
    load_metric_aliases,
    score_sample,
    score_sample_with_gold_evaluable_support,
)
from src.evaluation.schema_validation import SchemaValidationError, load_output_schema, validate_output
from src.training.week7_data import (
    ALIGNED_DIALOGUE_CONSTRUCTION_VERSIONS,
    CORE_SCENARIOS,
    Week7DataError,
    canonical_sha256,
    iter_jsonl,
    sha256_file,
)


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


def valid_check_constraints_tool_call(raw_output: str) -> bool:
    compact = str(raw_output or "").strip()
    if not compact.startswith("<tool_call>") or not compact.endswith("</tool_call>"):
        return False
    payload = compact[len("<tool_call>"):-len("</tool_call>")]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(parsed, dict)
        and parsed.get("name") == "check_constraints"
        and parsed.get("arguments") == {"scope": "conversation"}
    )


FREE_EVIDENCE_FIELDS = frozenset({
    "observed_evidence",
    "source_evidence",
    "historical_image_reference",
})


def _without_free_evidence(value: Any) -> Any:
    """Remove free-text evidence leaves while preserving task decisions and structure."""
    if isinstance(value, dict):
        return {
            key: _without_free_evidence(child)
            for key, child in value.items()
            if key not in FREE_EVIDENCE_FIELDS
        }
    if isinstance(value, list):
        return [_without_free_evidence(child) for child in value]
    return value


def _leaf_items(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any]]:
    """Flatten structured targets so one nested mismatch cannot zero an entire object."""
    if isinstance(value, dict):
        items: list[tuple[tuple[Any, ...], Any]] = []
        for key, child in value.items():
            items.extend(_leaf_items(child, path + (key,)))
        return items or [(path, {})]
    if isinstance(value, list):
        items = []
        for index, child in enumerate(value):
            items.extend(_leaf_items(child, path + (index,)))
        return items or [(path, [])]
    return [(path, value)]


def _value_at_path(value: Any, path: tuple[Any, ...]) -> tuple[bool, Any]:
    current = value
    for key in path:
        if isinstance(key, int):
            if not isinstance(current, list) or key >= len(current):
                return False, None
            current = current[key]
        else:
            if not isinstance(current, dict) or key not in current:
                return False, None
            current = current[key]
    return True, current


def _leaf_value_accuracy(observed: Any, expected: Any) -> float:
    leaves = _leaf_items(expected)
    if not leaves:
        return 0.0
    matches = 0
    for path, expected_value in leaves:
        present, observed_value = _value_at_path(observed, path)
        matches += bool(present and observed_value == expected_value)
    return matches / len(leaves)


def _stable_task_value_accuracy(observed: Any, expected: Any) -> float:
    if not isinstance(observed, dict) or not isinstance(expected, dict) or not expected:
        return 0.0
    return _leaf_value_accuracy(
        _without_free_evidence(observed),
        _without_free_evidence(expected),
    )


def _initial_task(turn_outputs: Any) -> dict[str, Any] | None:
    if not isinstance(turn_outputs, list) or not turn_outputs:
        return None
    first = turn_outputs[0]
    if not isinstance(first, dict):
        return None
    try:
        parsed = json.loads(str(first.get("raw_output") or ""))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _sequential_turn_score(
    row: dict[str, Any],
    turn_outputs: Any,
    *,
    expected_context: dict[str, Any] | None = None,
    scoring_protocol: str = "gold_exact_v1",
) -> tuple[float, float, float, int, int]:
    expected_assistants = [
        (index, message.get("content"))
        for index, message in enumerate(row.get("messages", []))
        if message.get("role") == "assistant"
    ]
    if not isinstance(turn_outputs, list) or not expected_assistants:
        return 0.0, 0.0, 1.0, len(expected_assistants), 0
    protocol_scores = []
    semantic_scores = []
    failures = 0
    for turn_index, (message_index, expected_output) in enumerate(expected_assistants):
        if turn_index >= len(turn_outputs) or not isinstance(turn_outputs[turn_index], dict):
            protocol_scores.append(0.0)
            semantic_scores.append(0.0)
            failures += 1
            continue
        observed = turn_outputs[turn_index]
        raw = str(observed.get("raw_output") or "")
        if (
            observed.get("assistant_turn_index") != turn_index
            or observed.get("message_index") != message_index
        ):
            protocol_scores.append(0.0)
            semantic_scores.append(0.0)
            failures += 1
            continue
        failures += bool(observed.get("failed"))
        if observed.get("protocol_valid") is False:
            failures += 1
        expected_text = str(expected_output or "")
        if "<tool_call>" in expected_text:
            valid = float(valid_check_constraints_tool_call(raw))
            protocol_scores.append(valid)
            semantic_scores.append(valid)
            continue
        try:
            expected_json = json.loads(expected_text)
            observed_json = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            protocol_scores.append(float(bool(raw.strip())))
            context_for_semantics = (
                _without_free_evidence(expected_context or {})
                if scoring_protocol == "gate_aligned_v2"
                else expected_context or row.get("context_expectations", {})
            )
            required = {
                term for term in _dialogue_terms(
                    context_for_semantics
                )
                if term in expected_text.casefold()
            }
            if "unknown" in expected_text.casefold():
                required.add("unknown")
            semantic_scores.append(
                sum(term in raw.casefold() for term in required) / len(required)
                if required else float(bool(raw.strip()))
            )
        else:
            if isinstance(expected_json, dict) and isinstance(observed_json, dict):
                protocol_scores.append(1.0)
                semantic_scores.append(
                    _stable_task_value_accuracy(observed_json, expected_json)
                    if scoring_protocol == "gate_aligned_v2"
                    else (
                        sum(observed_json.get(key) == value for key, value in expected_json.items())
                        / len(expected_json)
                        if expected_json else 1.0
                    )
                )
            else:
                exact = float(observed_json == expected_json)
                protocol_scores.append(exact)
                semantic_scores.append(exact)
    observed_count = min(len(turn_outputs), len(expected_assistants))
    if len(turn_outputs) > len(expected_assistants):
        failures += len(turn_outputs) - len(expected_assistants)
    return (
        statistics.fmean(protocol_scores) if protocol_scores else 0.0,
        statistics.fmean(semantic_scores) if semantic_scores else 0.0,
        min(1.0, failures / len(expected_assistants)),
        len(expected_assistants),
        observed_count,
    )


def score_dialogue_record(
    row: dict[str, Any],
    raw_output: str,
    latency_ms: float,
    failed: bool,
    turn_outputs: Any = None,
    *,
    scoring_protocol: str = "gold_exact_v1",
) -> dict[str, Any]:
    expectations = row.get("context_expectations")
    if not isinstance(expectations, dict):
        raise Week7EvaluationError(f"dialogue context expectations are missing: {row.get('sample_id')}")
    if scoring_protocol not in {"gold_exact_v1", "gold_plus_anchor_v1", "gate_aligned_v2"}:
        raise Week7EvaluationError("unsupported dialogue scoring protocol")
    initial_task = _initial_task(turn_outputs)
    expected = _dialogue_terms(
        _without_free_evidence(expectations)
        if scoring_protocol == "gate_aligned_v2"
        else expectations
    )
    image_terms = _dialogue_terms(
        expectations.get("historical_image_reference")
    )
    retained_terms = _dialogue_terms(expectations.get("retained_hard_constraints"))
    updated_requirement = str(expectations.get("updated_requirement") or "").casefold()
    normalized = raw_output.casefold()
    recalled = sum(term in normalized for term in expected)
    parsed = None
    json_valid = False
    try:
        parsed = json.loads(raw_output)
        json_valid = isinstance(parsed, dict)
    except json.JSONDecodeError:
        pass
    automatic_eligible = row.get("construction_version") in ALIGNED_DIALOGUE_CONSTRUCTION_VERSIONS
    expected_target = row.get("target") if isinstance(row.get("target"), dict) else {}
    expected_context = expectations
    gold_task = expected_target.get("task_result", {})
    expected_task = gold_task
    parsed_context = parsed.get("context_state", {}) if isinstance(parsed, dict) else {}
    parsed_task = parsed.get("task_result", {}) if isinstance(parsed, dict) else {}
    context_keys = set(expected_context)
    task_keys = set(expected_task) if isinstance(expected_task, dict) else set()
    context_key_coverage = (
        len(context_keys & set(parsed_context)) / len(context_keys)
        if context_keys and isinstance(parsed_context, dict)
        else 0.0
    )
    context_value_accuracy = (
        (
            _leaf_value_accuracy(
                _without_free_evidence(parsed_context),
                _without_free_evidence(expected_context),
            )
            if scoring_protocol == "gate_aligned_v2"
            else sum(parsed_context.get(key) == value for key, value in expected_context.items())
            / len(expected_context)
        )
        if expected_context and isinstance(parsed_context, dict)
        else 0.0
    )
    task_key_coverage = (
        len(task_keys & set(parsed_task)) / len(task_keys)
        if task_keys and isinstance(parsed_task, dict)
        else 0.0
    )
    task_value_accuracy = (
        (
            _stable_task_value_accuracy(parsed_task, expected_task)
            if scoring_protocol == "gate_aligned_v2"
            else sum(parsed_task.get(key) == value for key, value in expected_task.items())
            / len(expected_task)
        )
        if expected_task and isinstance(parsed_task, dict)
        else 0.0
    )
    expected_tool_indices = [
        index
        for index, message in enumerate(row.get("messages", []))
        if message.get("role") == "assistant"
        and "<tool_call>" in str(message.get("content") or "")
    ]
    observed_by_message = {
        turn.get("message_index"): turn
        for turn in turn_outputs or []
        if isinstance(turn, dict)
    }
    tool_protocol_compliance = float(all(
        index in observed_by_message
        and observed_by_message[index].get("protocol_valid") is not False
        and valid_check_constraints_tool_call(
            str(observed_by_message[index].get("raw_output") or "")
        )
        for index in expected_tool_indices
    )) if expected_tool_indices else None
    initial_stable_value_accuracy = _stable_task_value_accuracy(initial_task, gold_task)
    anchor_terms = _dialogue_terms(
        initial_task.get("observed_evidence") if initial_task else None
    )
    anchor_retention = (
        sum(term in normalized for term in anchor_terms) / len(anchor_terms)
        if anchor_terms else 0.0
    )
    sequential_coverage, sequential_semantic_accuracy, sequential_failure_rate, expected_turns, observed_turns = (
        _sequential_turn_score(
            row,
            turn_outputs,
            expected_context=expected_context,
            scoring_protocol=scoring_protocol,
        )
        if automatic_eligible else (0.0, 0.0, 0.0, 0, 0)
    )
    automatic_composite = None
    if automatic_eligible:
        automatic_composite = statistics.fmean((
            float(json_valid),
            recalled / len(expected) if expected else 1.0,
            context_key_coverage,
            context_value_accuracy,
            task_key_coverage,
            task_value_accuracy,
            sequential_semantic_accuracy,
        ))
    return {
        "sample_id": row["sample_id"], "scenario": "dialogue", "latency_ms": latency_ms,
        "failed": failed, "format_compliance": float(json_valid),
        "context_recall": recalled / len(expected) if expected else 1.0,
        "historical_image_reference": float(bool(image_terms) and any(term in normalized for term in image_terms)),
        "requirement_update": float(bool(updated_requirement) and updated_requirement in normalized),
        "context_carryover": float(not retained_terms or all(term in normalized for term in retained_terms)),
        "logical_consistency": context_value_accuracy if automatic_eligible else None,
        "context_state_key_coverage": context_key_coverage,
        "context_state_value_accuracy": context_value_accuracy,
        "task_result_key_coverage": task_key_coverage,
        "task_result_value_accuracy": task_value_accuracy,
        "initial_task_stable_value_accuracy": initial_stable_value_accuracy,
        "anchor_retention": anchor_retention,
        "tool_protocol_compliance": tool_protocol_compliance,
        "sequential_turn_coverage": sequential_coverage,
        "sequential_protocol_coverage": sequential_coverage,
        "sequential_semantic_accuracy": sequential_semantic_accuracy,
        "sequential_turn_failure_rate": sequential_failure_rate,
        "sequential_turn_count_expected": expected_turns,
        "sequential_turn_count_observed": observed_turns,
        "final_target_exact_match": float(parsed == expected_target),
        "automatic_semantic_gate_eligible": automatic_eligible,
        "automatic_composite": automatic_composite,
        "dialogue_scoring_protocol": scoring_protocol,
        "parsed_output": parsed,
        "human_required": not automatic_eligible,
    }


def _dialogue_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    automatic = all(item["automatic_semantic_gate_eligible"] for item in scores)
    result = {
        "sample_count": len(scores),
        "format_compliance": statistics.fmean(item["format_compliance"] for item in scores),
        "context_recall": statistics.fmean(item["context_recall"] for item in scores),
        "human_dimensions_status": (
            "NOT_REQUIRED_AUTOMATIC_V4" if automatic else "PENDING_REAL_HUMAN_INPUT"
        ),
        "scores": scores,
    }
    if automatic:
        protocols = {item["dialogue_scoring_protocol"] for item in scores}
        if len(protocols) != 1:
            raise Week7EvaluationError("dialogue scoring protocols differ within one summary")
        result.update({
            "dialogue_scoring_protocol": protocols.pop(),
            "context_state_key_coverage": statistics.fmean(
                item["context_state_key_coverage"] for item in scores
            ),
            "context_state_value_accuracy": statistics.fmean(
                item["context_state_value_accuracy"] for item in scores
            ),
            "task_result_key_coverage": statistics.fmean(
                item["task_result_key_coverage"] for item in scores
            ),
            "task_result_value_accuracy": statistics.fmean(
                item["task_result_value_accuracy"] for item in scores
            ),
            "initial_task_stable_value_accuracy": statistics.fmean(
                item["initial_task_stable_value_accuracy"] for item in scores
            ),
            "anchor_retention": statistics.fmean(
                item["anchor_retention"] for item in scores
            ),
            "sequential_turn_coverage": statistics.fmean(
                item["sequential_turn_coverage"] for item in scores
            ),
            "sequential_protocol_coverage": statistics.fmean(
                item["sequential_protocol_coverage"] for item in scores
            ),
            "sequential_semantic_accuracy": statistics.fmean(
                item["sequential_semantic_accuracy"] for item in scores
            ),
            "sequential_turn_failure_rate": statistics.fmean(
                item["sequential_turn_failure_rate"] for item in scores
            ),
            "final_target_exact_match": statistics.fmean(
                item["final_target_exact_match"] for item in scores
            ),
            "automatic_composite": statistics.fmean(
                item["automatic_composite"] for item in scores
            ),
        })
        tool_scores = [
            item["tool_protocol_compliance"] for item in scores
            if item["tool_protocol_compliance"] is not None
        ]
        result["tool_protocol_compliance"] = (
            statistics.fmean(tool_scores) if tool_scores else None
        )
        result["tool_protocol_support_count"] = len(tool_scores)
    return result


def evaluate_dialogue_automatic_gate(
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the same preregistered dialogue gates during training and selection."""
    declared = config["evaluation"].get("dialogue_automatic_gate")
    dialogue = metrics.get("dialogue")
    if (
        not isinstance(declared, dict)
        or declared.get("enabled") is not True
        or declared.get("human_input_required") is not False
        or not isinstance(dialogue, dict)
        or dialogue.get("human_dimensions_status") != "NOT_REQUIRED_AUTOMATIC_V4"
    ):
        raise Week7EvaluationError("automatic dialogue gate identity mismatch")
    scores = dialogue.get("scores")
    if not isinstance(scores, list) or not scores or len(scores) != int(dialogue.get("sample_count", -1)):
        raise Week7EvaluationError("automatic dialogue gate score coverage mismatch")
    expected_protocol = config["evaluation"].get("dialogue_scoring_protocol", "gold_exact_v1")
    if dialogue.get("dialogue_scoring_protocol", "gold_exact_v1") != expected_protocol:
        raise Week7EvaluationError("automatic dialogue scoring protocol mismatch")

    minimums = {
        "format_compliance": "minimum_format_compliance",
        "context_recall": "minimum_context_recall",
        "context_state_value_accuracy": "minimum_context_state_value_accuracy",
        "task_result_key_coverage": "minimum_task_result_key_coverage",
        "task_result_value_accuracy": "minimum_task_result_value_accuracy",
        "automatic_composite": "minimum_automatic_composite",
        "initial_task_stable_value_accuracy": "minimum_initial_task_stable_value_accuracy",
        "anchor_retention": "minimum_anchor_retention",
        "tool_protocol_compliance": "minimum_tool_protocol_compliance",
        "sequential_turn_coverage": "minimum_sequential_turn_coverage",
        "sequential_protocol_coverage": "minimum_sequential_protocol_coverage",
        "sequential_semantic_accuracy": "minimum_sequential_semantic_accuracy",
    }
    maximums = {
        "sequential_turn_failure_rate": "maximum_sequential_turn_failure_rate",
        "dialogue_failure_rate": "maximum_failure_rate",
        "overall_failure_rate": "maximum_overall_failure_rate",
    }
    values: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    directions: dict[str, str] = {}
    for metric_name, threshold_name in minimums.items():
        if threshold_name not in declared:
            continue
        if metric_name == "tool_protocol_compliance":
            support = sum(score.get(metric_name) is not None for score in scores)
            if support <= 0 or int(dialogue.get("tool_protocol_support_count", -1)) != support:
                raise Week7EvaluationError("tool protocol support identity mismatch")
        value = float(dialogue.get(metric_name))
        threshold = float(declared[threshold_name])
        values[metric_name] = value
        thresholds[metric_name] = threshold
        directions[metric_name] = "minimum"
    derived = {
        "dialogue_failure_rate": sum(bool(score.get("failed")) for score in scores) / len(scores),
        "overall_failure_rate": float(metrics.get("failure_rate")),
    }
    for metric_name, threshold_name in maximums.items():
        if metric_name == "overall_failure_rate":
            threshold = float(
                declared.get(
                    threshold_name,
                    config["evaluation"]["non_regression"]["max_failure_rate"],
                )
            )
        elif threshold_name not in declared:
            continue
        else:
            threshold = float(declared[threshold_name])
        value = (
            float(dialogue.get(metric_name))
            if metric_name == "sequential_turn_failure_rate"
            else derived[metric_name]
        )
        values[metric_name] = value
        thresholds[metric_name] = threshold
        directions[metric_name] = "maximum"
    for name, value in values.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise Week7EvaluationError(f"automatic gate metric outside [0, 1]: {name}")
    for name, threshold in thresholds.items():
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise Week7EvaluationError(f"automatic gate threshold outside [0, 1]: {name}")
    checks = {
        name: values[name] >= thresholds[name]
        if directions[name] == "minimum"
        else values[name] <= thresholds[name]
        for name in values
    }
    return {
        "values": values,
        "thresholds": thresholds,
        "directions": directions,
        "checks": checks,
        "passed": all(checks.values()),
    }


def gate_first_selection_score(
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Encode hard-gate feasibility before the ordinary weighted composite."""
    gate = evaluate_dialogue_automatic_gate(config, metrics)
    weighted = float(metrics["weighted_composite"])
    if not math.isfinite(weighted) or not 0.0 <= weighted <= 1.0:
        raise Week7EvaluationError("weighted composite is outside [0, 1]")
    if gate["passed"]:
        return 1.0 + weighted, gate
    progress = []
    for name, value in gate["values"].items():
        threshold = gate["thresholds"][name]
        if gate["directions"][name] == "minimum":
            ratio = 1.0 if threshold <= 0.0 else min(1.0, value / threshold)
        else:
            ratio = 1.0 if value <= threshold else max(
                0.0,
                1.0 - (value - threshold) / max(1e-12, 1.0 - threshold),
            )
        progress.append(ratio)
    bottleneck = min(progress)
    mean_progress = statistics.fmean(progress)
    return 0.80 * bottleneck + 0.19 * mean_progress + 0.009 * weighted, gate


def summarize_dialogue_raw_records(
    rows: Iterable[dict[str, Any]],
    records: Iterable[dict[str, Any]],
    *,
    scoring_protocol: str = "gold_exact_v1",
) -> dict[str, Any]:
    """Summarize an exact dialogue-only development run without fabricating task metrics."""
    rows_by_id = {row["sample_id"]: row for row in rows}
    if not rows_by_id or any(row.get("scenario") != "dialogue" for row in rows_by_id.values()):
        raise Week7EvaluationError("dialogue summary requires dialogue-only rows")
    records_list = list(records)
    if (
        len(records_list) != len(rows_by_id)
        or {record.get("sample_id") for record in records_list} != set(rows_by_id)
    ):
        raise Week7EvaluationError("raw records do not exactly cover the dialogue rows")
    scores = []
    latencies = []
    failures = 0
    for record in records_list:
        row = rows_by_id[record["sample_id"]]
        raw = record.get("raw_output")
        if not isinstance(raw, str):
            raw = ""
        latency = float(record.get("latency_ms", 0.0))
        failed = bool(record.get("failed"))
        scores.append(score_dialogue_record(
            row,
            raw,
            latency,
            failed,
            record.get("turn_outputs"),
            scoring_protocol=scoring_protocol,
        ))
        latencies.append(latency)
        failures += failed
    return {
        "sample_count": len(records_list),
        "weighted_composite": None,
        "scenarios": {},
        "dialogue": _dialogue_summary(scores),
        "latency_ms_mean": statistics.fmean(latencies),
        "latency_ms_median": statistics.median(latencies),
        "failure_count": failures,
        "failure_rate": failures / len(records_list),
    }


def summarize_raw_records(
    root: Path,
    config: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    records: Iterable[dict[str, Any]],
    *,
    metric_support_protocol: str | None = None,
) -> dict[str, Any]:
    rows_by_id = {row["sample_id"]: row for row in rows}
    records_list = list(records)
    if len(records_list) != len(rows_by_id) or {record.get("sample_id") for record in records_list} != set(rows_by_id):
        raise Week7EvaluationError("raw records do not exactly cover the selected rows")
    aliases = load_metric_aliases(root / "configs/evaluation/metric_aliases_v1.json")
    sample_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dialogue_scores = []
    failures = 0
    latencies = []
    if metric_support_protocol not in {None, WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL}:
        raise Week7EvaluationError("unsupported metric-support protocol")
    sample_scorer = (
        score_sample_with_gold_evaluable_support
        if metric_support_protocol == WEEK7_GOLD_EVALUABLE_SUPPORT_PROTOCOL
        else score_sample
    )
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
            dialogue_scores.append(score_dialogue_record(
                row,
                raw,
                latency,
                failed,
                record.get("turn_outputs"),
                scoring_protocol=config["evaluation"].get(
                    "dialogue_scoring_protocol", "gold_exact_v1"
                ),
            ))
            continue
        parsed, json_valid, schema_valid, error = strict_parse_output(root, row["scenario"], raw)
        result = {
            "run_id": record.get("run_id"), "sample_id": row["sample_id"], "scenario": row["scenario"],
            "model_name": record.get("model_name"), "prompt_version": "week7_locked_v1",
            "raw_output": raw, "parsed_output": parsed, "json_valid": json_valid,
            "schema_valid": schema_valid, "parse_or_schema_error": error, "latency_ms": latency,
        }
        sample_scores[row["scenario"]].append(
            sample_scorer(result, _annotation(row), aliases)
        )
    present_scenarios = [scenario for scenario in CORE_SCENARIOS if sample_scores[scenario]]
    if not present_scenarios:
        raise Week7EvaluationError("at least one core scenario is required for a business summary")
    scenario_results = {}
    scenario_composites = {}
    for scenario in present_scenarios:
        aggregate = aggregate_scenario_scores(sample_scores[scenario])
        composite, support = _weighted_metric(aggregate, config["evaluation"]["metric_weights"][scenario])
        scenario_results[scenario] = {"aggregate": aggregate, "composite": composite, "metric_support": support}
        scenario_composites[scenario] = composite
    selected_weight = sum(float(config["evaluation"]["scenario_weights"][scenario]) for scenario in present_scenarios)
    weighted = sum(float(config["evaluation"]["scenario_weights"][scenario]) * scenario_composites[scenario] for scenario in present_scenarios) / selected_weight
    dialogue_summary = None
    if dialogue_scores:
        dialogue_summary = _dialogue_summary(dialogue_scores)
    core_weighted = weighted
    automatic_gate = config["evaluation"].get("dialogue_automatic_gate", {})
    if automatic_gate.get("enabled") is True:
        if not dialogue_summary or "automatic_composite" not in dialogue_summary:
            raise Week7EvaluationError("v4 automatic dialogue evidence is missing")
        dialogue_weight = float(automatic_gate["selection_weight"])
        if not 0.0 < dialogue_weight < 1.0:
            raise Week7EvaluationError("v4 dialogue selection weight is invalid")
        weighted = (
            (1.0 - dialogue_weight) * core_weighted
            + dialogue_weight * float(dialogue_summary["automatic_composite"])
        )
    result = {
        "sample_count": len(records_list), "weighted_composite": weighted,
        "core_weighted_composite": core_weighted,
        "scenarios": scenario_results, "dialogue": dialogue_summary,
        "latency_ms_mean": statistics.fmean(latencies), "latency_ms_median": statistics.median(latencies),
        "failure_count": failures, "failure_rate": failures / len(records_list),
    }
    if metric_support_protocol is not None:
        result["metric_support_protocol"] = metric_support_protocol
    return result


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
            raw_output = record.get("raw_output")
            if name == "constrained":
                raw_output = record.get("primary_constrained_raw_output", raw_output)
            _, json_valid, schema_valid, _ = strict_parse_output(root, row["scenario"], str(raw_output or ""))
            counts["json_valid"] += json_valid
            counts["schema_valid"] += schema_valid
            counts["primary_failure"] += bool(record.get("constrained_error")) if name == "constrained" else bool(record.get("failed"))
            counts["operational_failure"] += bool(record.get("failed"))
            counts["fallback_used"] += bool(record.get("fallback_used"))
            counts["fallback_failure"] += bool(record.get("fallback_failed"))
            latencies.append(float(record.get("latency_ms", 0.0)))
        fallback_count = int(counts["fallback_used"])
        output["modes"][name] = {
            "json_compliance": counts["json_valid"] / len(selected),
            "schema_coverage": counts["schema_valid"] / len(selected),
            "request_count": len(selected),
            "primary_failure_count": int(counts["primary_failure"]),
            "primary_failure_rate": counts["primary_failure"] / len(selected),
            "operational_failure_count": int(counts["operational_failure"]),
            "operational_failure_rate": counts["operational_failure"] / len(selected),
            "fallback_request_count": fallback_count,
            "fallback_failure_count": int(counts["fallback_failure"]),
            "fallback_failure_rate": (
                counts["fallback_failure"] / fallback_count if fallback_count else 0.0
            ),
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
        "free_request": free["primary_failure_rate"]
        <= float(config["evaluation"]["non_regression"]["max_failure_rate"]),
        "constrained_request": constrained["primary_failure_rate"]
        <= float(config["evaluation"]["non_regression"]["max_failure_rate"]),
        "fallback": constrained["fallback_failure_rate"] <= float(config["evaluation"]["schema_decoding"]["max_fallback_failure_rate"]),
    }
    return output


def enforce_test_once(lock_root: Path, parameter_lock: Path, run_id: str) -> dict[str, Any]:
    """Reject the obsolete marker-only path, which could bypass the final suite."""
    raise Week7EvaluationError(
        "marker-only test consumption is disabled; use the final suite"
    )

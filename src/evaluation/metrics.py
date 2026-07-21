"""Deterministic retrieval and Week 3 structured-output metrics."""

import csv
import json
import math
import statistics
import unicodedata
from pathlib import Path
from typing import Any, Iterable


SCENARIO_METRIC_NAMES = {
    "image_product_search": (
        "business_category_accuracy",
        "price_range_accuracy",
        "style_precision",
        "style_recall",
        "style_f1",
        "facility_precision",
        "facility_recall",
        "facility_f1",
        "label_completeness",
    ),
    "after_sales": (
        "issue_type_accuracy",
        "severity_accuracy",
        "key_information_precision",
        "key_information_recall",
        "key_information_f1",
        "ocr_recall",
        "ocr_exact_match",
    ),
    "itinerary_planning": (
        "constraint_recognition_accuracy",
        "hard_constraint_precision",
        "hard_constraint_recall",
        "hard_constraint_f1",
        "soft_constraint_precision",
        "soft_constraint_recall",
        "soft_constraint_f1",
        "itinerary_element_precision",
        "itinerary_element_recall",
        "itinerary_element_completeness",
        "itinerary_element_f1",
        "constraint_check_coverage",
        "constraint_violation_rate",
    ),
}

MULTILABEL_PREFIXES = {
    "style_tags": "style",
    "visible_facilities": "facility",
    "key_information": "key_information",
    "hard_constraints": "hard_constraint",
    "soft_constraints": "soft_constraint",
    "required_itinerary_elements": "itinerary_element",
}


class MetricConfigurationError(ValueError):
    """Raised when the metric alias contract is malformed or ambiguous."""


def load_result_records(path: Path) -> list[dict[str, Any]]:
    """Load persisted result JSONL with strict JSON and record validation."""
    from src.evaluation.results import validate_result_record

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped, parse_constant=_reject_json_constant)
                records.append(validate_result_record(payload))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid result on line {line_number}: {exc}") from exc
    return records


def build_annotation_index(
    manifests: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Index completed annotations and reject cross-manifest sample ambiguity."""
    index: dict[str, dict[str, Any]] = {}
    for configured_scenario, records in manifests.items():
        for record in records:
            if record.get("annotation_status") != "completed":
                continue
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("completed annotation requires a sample_id")
            if sample_id in index:
                raise ValueError(f"duplicate annotation sample_id: {sample_id}")
            if record.get("scenario") != configured_scenario:
                raise ValueError(f"annotation scenario mismatch: {sample_id}")
            annotation = record.get("annotation")
            if not isinstance(annotation, dict):
                raise ValueError(f"completed sample has no annotation: {sample_id}")
            index[sample_id] = {
                "scenario": configured_scenario,
                "annotation": annotation,
            }
    return index


def normalize_text(value: Any) -> str:
    """Normalize comparable text without fuzzy matching or token rewriting."""
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def load_metric_aliases(path: Path) -> dict[str, dict[str, str]]:
    """Load and validate a versioned, one-hop field-specific alias table."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricConfigurationError(f"cannot load metric aliases: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("version"), str):
        raise MetricConfigurationError("metric aliases require a string version")
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise MetricConfigurationError("metric aliases require a fields object")

    validated: dict[str, dict[str, str]] = {}
    for field, mappings in fields.items():
        if not isinstance(field, str) or not field or not isinstance(mappings, dict):
            raise MetricConfigurationError("alias fields must map names to objects")
        normalized_mappings: dict[str, str] = {}
        for alias, canonical in mappings.items():
            alias_value = normalize_text(alias)
            canonical_value = normalize_text(canonical)
            if not alias_value or not canonical_value:
                raise MetricConfigurationError(f"{field} aliases must be non-empty text")
            previous = normalized_mappings.get(alias_value)
            if previous is not None and previous != canonical_value:
                raise MetricConfigurationError(f"conflicting alias in {field}: {alias_value}")
            normalized_mappings[alias_value] = canonical_value
        for canonical in normalized_mappings.values():
            if (
                canonical in normalized_mappings
                and normalized_mappings[canonical] != canonical
            ):
                raise MetricConfigurationError(
                    f"alias chains are not allowed in {field}: {canonical}"
                )
        validated[field] = normalized_mappings
    return validated


def normalize_value(
    field: str,
    value: Any,
    aliases: dict[str, dict[str, str]],
) -> str:
    """Normalize one value and apply at most one explicit field alias."""
    normalized = normalize_text(value)
    return aliases.get(field, {}).get(normalized, normalized)


def set_metric_counts(
    expected: Iterable[Any] | None,
    predicted: Iterable[Any] | None,
    field: str,
    aliases: dict[str, dict[str, str]],
) -> dict[str, int | float]:
    """Return set counts and precision/recall/F1 under explicit empty-set rules."""
    expected_set = _normalized_set(expected, field, aliases)
    predicted_set = _normalized_set(predicted, field, aliases)
    true_positive = len(expected_set & predicted_set)
    false_positive = len(predicted_set - expected_set)
    false_negative = len(expected_set - predicted_set)
    return _metrics_from_counts(true_positive, false_positive, false_negative)


def score_sample(
    result: dict[str, Any],
    annotation: dict[str, Any],
    aliases: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Score one persisted result against its scenario annotation."""
    scenario = result.get("scenario")
    if scenario not in SCENARIO_METRIC_NAMES:
        raise ValueError(f"unsupported scoring scenario: {scenario}")
    if not isinstance(annotation, dict):
        raise ValueError("annotation must be an object")

    structured_valid = (
        result.get("json_valid") is True and result.get("schema_valid") is True
    )
    score: dict[str, Any] = {
        "run_id": result.get("run_id"),
        "sample_id": result.get("sample_id"),
        "scenario": scenario,
        "model_name": result.get("model_name"),
        "prompt_version": result.get("prompt_version"),
        "json_compliance": float(result.get("json_valid") is True),
        "schema_pass": float(result.get("schema_valid") is True),
        "structured_valid": structured_valid,
        "latency_ms": result.get("latency_ms"),
        "multilabel_counts": {},
    }
    if not structured_valid:
        _score_unstructured_result(score, result, scenario, annotation, aliases)
        return score

    output = result.get("parsed_output")
    if not isinstance(output, dict):
        _score_unstructured_result(score, result, scenario, annotation, aliases)
        return score

    if scenario == "image_product_search":
        _score_product(score, output, annotation, aliases)
    elif scenario == "after_sales":
        _score_after_sales(score, output, annotation, aliases)
    else:
        _score_itinerary(score, output, annotation, aliases)
    return score


def _score_unstructured_result(
    score: dict[str, Any],
    result: dict[str, Any],
    scenario: str,
    annotation: dict[str, Any],
    aliases: dict[str, dict[str, str]],
) -> None:
    """Keep minimal-baseline semantics pending while retaining strict standardized zeros."""
    if result.get("prompt_version") == "baseline_minimal_v1":
        score.update({name: None for name in SCENARIO_METRIC_NAMES[scenario]})
        score["multilabel_counts"] = {}
        score["scoring_track"] = "format_only_unparsed_baseline"
        score["semantic_metrics_status"] = "pending"
        return
    strict_metrics = _zero_structured_metrics(scenario)
    if scenario == "image_product_search":
        if normalize_value(
            "business_category", annotation.get("business_category"), aliases
        ) == "unknown":
            strict_metrics["business_category_accuracy"] = None
        if normalize_value("price_range", annotation.get("price_range"), aliases) == "unknown":
            strict_metrics["price_range_accuracy"] = None
    elif scenario == "after_sales":
        if normalize_value("issue_type", annotation.get("issue_type"), aliases) == "unknown":
            strict_metrics["issue_type_accuracy"] = None
        if normalize_value("severity", annotation.get("severity"), aliases) == "unknown":
            strict_metrics["severity_accuracy"] = None
        if annotation.get("ocr_ground_truth") is None:
            strict_metrics["ocr_recall"] = None
            strict_metrics["ocr_exact_match"] = None
    score.update(strict_metrics)
    score["multilabel_counts"] = _invalid_multilabel_counts(
        scenario, annotation, aliases
    )
    score["scoring_track"] = "strict_structured_business"
    score["semantic_metrics_status"] = "scored"


def aggregate_scenario_scores(sample_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one scenario with macro, micro, format, and latency statistics."""
    if not sample_scores:
        raise ValueError("cannot aggregate an empty score collection")
    scenarios = {score.get("scenario") for score in sample_scores}
    if len(scenarios) != 1:
        raise ValueError("all sample scores must belong to one scenario")
    scenario = next(iter(scenarios))
    aggregate: dict[str, Any] = {
        "scenario": scenario,
        "sample_count": len(sample_scores),
    }

    excluded = {
        "run_id", "sample_id", "scenario", "model_name", "prompt_version",
        "latency_ms", "multilabel_counts", "structured_valid", "scoring_track",
        "format_structured_valid", "semantic_metrics_status",
    }
    metric_names = sorted(
        {
            key
            for score in sample_scores
            for key, value in score.items()
            if key not in excluded and (value is None or _is_number(value))
        }
    )
    multilabel_macro_names = {
        f"{prefix}_{suffix}"
        for prefix in MULTILABEL_PREFIXES.values()
        for suffix in ("precision", "recall", "f1")
    }
    for metric_name in metric_names:
        output_name = (
            f"{metric_name}_macro"
            if metric_name in multilabel_macro_names
            else metric_name
        )
        values = [
            float(score[metric_name])
            for score in sample_scores
            if _is_number(score.get(metric_name))
        ]
        if any(score.get(metric_name) is None for score in sample_scores):
            aggregate[f"{output_name}_support_count"] = len(values)
        if not values:
            aggregate[output_name] = None
            continue
        aggregate[output_name] = sum(values) / len(values)

    all_count_fields = sorted(
        {
            field
            for score in sample_scores
            for field in score.get("multilabel_counts", {})
        }
    )
    for field in all_count_fields:
        totals = {"tp": 0, "fp": 0, "fn": 0}
        for score in sample_scores:
            counts = score.get("multilabel_counts", {}).get(field, {})
            for key in totals:
                totals[key] += int(counts.get(key, 0))
        metrics = _metrics_from_counts(totals["tp"], totals["fp"], totals["fn"])
        if (
            totals["tp"] + totals["fp"] + totals["fn"] == 0
            and any(score.get("structured_valid") is False for score in sample_scores)
        ):
            metrics.update({"precision": 0.0, "recall": 0.0, "f1": 0.0})
        prefix = MULTILABEL_PREFIXES.get(field, field)
        aggregate[f"{prefix}_precision_micro"] = metrics["precision"]
        aggregate[f"{prefix}_recall_micro"] = metrics["recall"]
        aggregate[f"{prefix}_f1_micro"] = metrics["f1"]
        aggregate[f"{prefix}_tp"] = totals["tp"]
        aggregate[f"{prefix}_fp"] = totals["fp"]
        aggregate[f"{prefix}_fn"] = totals["fn"]

    latencies = sorted(
        float(score["latency_ms"])
        for score in sample_scores
        if _is_number(score.get("latency_ms")) and float(score["latency_ms"]) >= 0
    )
    aggregate["latency_count"] = len(latencies)
    if latencies:
        aggregate.update(
            {
                "latency_min_ms": latencies[0],
                "latency_mean_ms": sum(latencies) / len(latencies),
                "latency_median_ms": statistics.median(latencies),
                "latency_p95_ms": latencies[math.ceil(0.95 * len(latencies)) - 1],
                "latency_max_ms": latencies[-1],
            }
        )
    else:
        for name in ("min", "mean", "median", "p95", "max"):
            aggregate[f"latency_{name}_ms"] = None
    return aggregate


def score_records(
    results: Iterable[dict[str, Any]],
    annotations_by_sample_id: dict[str, dict[str, Any]],
    aliases: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Join results to annotations, score them, aggregate by scenario, and classify errors."""
    from src.evaluation.error_analysis import build_error_case, classify_result_error

    scores: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        sample_id = result.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("result sample_id must be non-empty text")
        if sample_id in seen:
            raise ValueError(f"duplicate result sample_id: {sample_id}")
        seen.add(sample_id)
        annotation_record = annotations_by_sample_id.get(sample_id)
        if not isinstance(annotation_record, dict):
            raise ValueError(f"missing annotation for sample_id: {sample_id}")
        if annotation_record.get("scenario") != result.get("scenario"):
            raise ValueError(f"scenario mismatch for sample_id: {sample_id}")
        annotation = annotation_record.get("annotation")
        score = score_sample(result, annotation, aliases)
        scores.append(score)
        if classify_result_error(result) != "valid":
            errors.append(build_error_case(result, score))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for score in scores:
        grouped.setdefault(score["scenario"], []).append(score)
    aggregates = {
        scenario: aggregate_scenario_scores(scenario_scores)
        for scenario, scenario_scores in sorted(grouped.items())
    }
    return scores, aggregates, errors


def export_score_artifacts(
    output_dir: Path,
    sample_scores: list[dict[str, Any]],
    aggregate_scores: dict[str, dict[str, Any]],
    error_cases: list[dict[str, Any]],
) -> dict[str, Path]:
    """Write immutable, strict-JSON sample/error JSONL and aggregate CSV artifacts."""
    for record in sample_scores:
        _strict_json_dumps(record)
    for record in error_cases:
        _strict_json_dumps(record)
    rows = [dict(row) for _, row in sorted(aggregate_scores.items())]
    for row in rows:
        _strict_json_dumps(row)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    sample_path = output_path / "sample_scores.jsonl"
    aggregate_path = output_path / "aggregate_scores.csv"
    errors_path = output_path / "error_cases.jsonl"
    _write_jsonl(sample_path, sample_scores)
    _write_jsonl(errors_path, error_cases)

    fieldnames = sorted({key for row in rows for key in row})
    if "scenario" in fieldnames:
        fieldnames.remove("scenario")
        fieldnames.insert(0, "scenario")
    with aggregate_path.open("x", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
    return {
        "sample_scores": sample_path,
        "aggregate_scores": aggregate_path,
        "error_cases": errors_path,
    }


def recall_at_k(expected_ids: set[str], ranked_ids: list[str], k: int) -> float:
    """Compute the fraction of expected IDs present in the first k results."""
    if not expected_ids:
        return 0.0
    hits = expected_ids & set(ranked_ids[:k])
    return len(hits) / len(expected_ids)


def top_k_hit(expected_id: str, ranked_ids: list[str], k: int) -> int:
    """Return one when the expected ID appears in the first k results."""
    return int(expected_id in ranked_ids[:k])


def _score_product(
    score: dict[str, Any],
    output: dict[str, Any],
    annotation: dict[str, Any],
    aliases: dict[str, dict[str, str]],
) -> None:
    expected_category = normalize_value(
        "business_category", annotation.get("business_category"), aliases
    )
    expected_price = normalize_value(
        "price_range", annotation.get("price_range"), aliases
    )
    category_accuracy = (
        None
        if expected_category == "unknown"
        else float(
            normalize_value("business_category", output.get("business_category"), aliases)
            == expected_category
        )
    )
    price_accuracy = (
        None
        if expected_price == "unknown"
        else float(
            normalize_value("price_range", output.get("price_range"), aliases)
            == expected_price
        )
    )
    style = set_metric_counts(
        annotation.get("style_tags"), output.get("style_tags"), "style_tags", aliases
    )
    facility = set_metric_counts(
        annotation.get("visible_facilities"),
        output.get("visible_facilities"),
        "visible_facilities",
        aliases,
    )
    score.update(
        {
            "business_category_accuracy": category_accuracy,
            "price_range_accuracy": price_accuracy,
            "style_precision": style["precision"],
            "style_recall": style["recall"],
            "style_f1": style["f1"],
            "facility_precision": facility["precision"],
            "facility_recall": facility["recall"],
            "facility_f1": facility["f1"],
        }
    )
    denominator = (
        int(category_accuracy is not None)
        + int(price_accuracy is not None)
        + len(_normalized_set(annotation.get("style_tags"), "style_tags", aliases))
        + len(
            _normalized_set(
                annotation.get("visible_facilities"), "visible_facilities", aliases
            )
        )
    )
    numerator = (
        (category_accuracy or 0.0)
        + (price_accuracy or 0.0)
        + int(style["tp"])
        + int(facility["tp"])
    )
    score["label_completeness"] = numerator / denominator if denominator else None
    score["multilabel_counts"] = {
        "style_tags": _count_only(style),
        "visible_facilities": _count_only(facility),
    }


def _score_after_sales(
    score: dict[str, Any],
    output: dict[str, Any],
    annotation: dict[str, Any],
    aliases: dict[str, dict[str, str]],
) -> None:
    key_information = set_metric_counts(
        annotation.get("key_information"),
        output.get("key_information"),
        "key_information",
        aliases,
    )
    expected_issue = normalize_value("issue_type", annotation.get("issue_type"), aliases)
    expected_severity = normalize_value("severity", annotation.get("severity"), aliases)
    score.update(
        {
            "issue_type_accuracy": None
            if expected_issue == "unknown"
            else float(
                normalize_value("issue_type", output.get("issue_type"), aliases)
                == expected_issue
            ),
            "severity_accuracy": None
            if expected_severity == "unknown"
            else float(
                normalize_value("severity", output.get("severity"), aliases)
                == expected_severity
            ),
            "key_information_precision": key_information["precision"],
            "key_information_recall": key_information["recall"],
            "key_information_f1": key_information["f1"],
        }
    )
    expected_ocr = annotation.get("ocr_ground_truth")
    if expected_ocr is None:
        score["ocr_recall"] = None
        score["ocr_exact_match"] = None
    else:
        predicted_ocr = output.get("ocr_text") or []
        ocr = set_metric_counts(expected_ocr, predicted_ocr, "ocr_text", aliases)
        score["ocr_recall"] = ocr["recall"]
        score["ocr_exact_match"] = float(
            _normalized_list(expected_ocr, "ocr_text", aliases)
            == _normalized_list(predicted_ocr, "ocr_text", aliases)
        )
    score["multilabel_counts"] = {
        "key_information": _count_only(key_information),
    }


def _score_itinerary(
    score: dict[str, Any],
    output: dict[str, Any],
    annotation: dict[str, Any],
    aliases: dict[str, dict[str, str]],
) -> None:
    hard = set_metric_counts(
        annotation.get("hard_constraints"),
        output.get("hard_constraints"),
        "hard_constraints",
        aliases,
    )
    soft = set_metric_counts(
        annotation.get("soft_constraints"),
        output.get("soft_constraints"),
        "soft_constraints",
        aliases,
    )
    elements = set_metric_counts(
        annotation.get("required_itinerary_elements"),
        output.get("required_itinerary_elements"),
        "required_itinerary_elements",
        aliases,
    )
    expected_constraints = _typed_constraints(annotation, aliases)
    predicted_constraints = _typed_constraints(output, aliases)
    union = expected_constraints | predicted_constraints
    recognition_accuracy = (
        len(expected_constraints & predicted_constraints) / len(union) if union else 1.0
    )

    checks: dict[tuple[str, str], set[str]] = {}
    for item in output.get("constraint_check") or []:
        if not isinstance(item, dict):
            continue
        constraint_type = normalize_text(item.get("constraint_type"))
        field = f"{constraint_type}_constraints"
        key = (
            constraint_type,
            normalize_value(field, item.get("constraint"), aliases),
        )
        if key[0] in {"hard", "soft"} and key[1]:
            checks.setdefault(key, set()).add(
                normalize_value("constraint_status", item.get("status"), aliases)
            )
    expected_count = len(expected_constraints)
    covered = sum(key in checks for key in expected_constraints)
    violated = sum("violated" in checks.get(key, set()) for key in expected_constraints)
    score.update(
        {
            "constraint_recognition_accuracy": recognition_accuracy,
            "hard_constraint_precision": hard["precision"],
            "hard_constraint_recall": hard["recall"],
            "hard_constraint_f1": hard["f1"],
            "soft_constraint_precision": soft["precision"],
            "soft_constraint_recall": soft["recall"],
            "soft_constraint_f1": soft["f1"],
            "itinerary_element_precision": elements["precision"],
            "itinerary_element_recall": elements["recall"],
            "itinerary_element_completeness": elements["recall"],
            "itinerary_element_f1": elements["f1"],
            "constraint_check_coverage": covered / expected_count if expected_count else 1.0,
            "constraint_violation_rate": violated / expected_count if expected_count else 0.0,
        }
    )
    score["multilabel_counts"] = {
        "hard_constraints": _count_only(hard),
        "soft_constraints": _count_only(soft),
        "required_itinerary_elements": _count_only(elements),
    }


def _zero_structured_metrics(scenario: str) -> dict[str, Any]:
    return {metric_name: 0.0 for metric_name in SCENARIO_METRIC_NAMES[scenario]}


def _invalid_multilabel_counts(
    scenario: str,
    annotation: dict[str, Any],
    aliases: dict[str, dict[str, str]],
) -> dict[str, dict[str, int]]:
    fields = {
        "image_product_search": ("style_tags", "visible_facilities"),
        "after_sales": ("key_information",),
        "itinerary_planning": (
            "hard_constraints", "soft_constraints", "required_itinerary_elements"
        ),
    }[scenario]
    return {
        field: _count_only(set_metric_counts(annotation.get(field), [], field, aliases))
        for field in fields
    }


def _typed_constraints(
    value: dict[str, Any], aliases: dict[str, dict[str, str]]
) -> set[tuple[str, str]]:
    return {
        (constraint_type, normalized)
        for constraint_type in ("hard", "soft")
        for normalized in _normalized_set(
            value.get(f"{constraint_type}_constraints"),
            f"{constraint_type}_constraints",
            aliases,
        )
    }


def _normalized_set(
    values: Iterable[Any] | None,
    field: str,
    aliases: dict[str, dict[str, str]],
) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {
        normalized
        for value in values
        if (normalized := normalize_value(field, value, aliases))
    }


def _normalized_list(
    values: Iterable[Any] | None,
    field: str,
    aliases: dict[str, dict[str, str]],
) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [
        normalized
        for value in values
        if (normalized := normalize_value(field, value, aliases))
    ]


def _metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    predicted_count = tp + fp
    expected_count = tp + fn
    if predicted_count == 0 and expected_count == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / expected_count if expected_count else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _count_only(metrics: dict[str, int | float]) -> dict[str, int]:
    return {key: int(metrics[key]) for key in ("tp", "fp", "fn")}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _strict_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_strict_json_dumps(record) + "\n")


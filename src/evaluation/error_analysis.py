"""Aggregate experiment failure cases into reviewable error counts."""

from collections import Counter


def summarize_failure_types(failure_cases: list[dict[str, str]]) -> dict[str, int]:
    """Count failure records by error type, using `unknown` when absent."""
    return dict(Counter(case.get("error_type", "unknown") for case in failure_cases))


def classify_result_error(result: dict[str, object]) -> str:
    """Classify runner, JSON, and Schema failures without conflating their causes."""
    error = result.get("error")
    if result.get("json_valid") is True and result.get("schema_valid") is True:
        return "valid"
    if isinstance(error, str):
        for error_type in (
            "dry_run",
            "mock_fixture_missing",
            "model_request_error",
            "json_parse_error",
            "schema_validation_error",
        ):
            if error == error_type or error.startswith(f"{error_type}:"):
                return error_type
        return "unknown_error"
    if result.get("raw_output") is None:
        return "missing_output"
    if result.get("json_valid") is not True:
        return "json_parse_error"
    if result.get("schema_valid") is not True:
        return "schema_validation_error"
    return "unknown_error"


def build_error_case(
    result: dict[str, object], sample_metrics: dict[str, object]
) -> dict[str, object]:
    """Build a reviewable error record with raw output and scoring context."""
    return {
        "run_id": result.get("run_id"),
        "sample_id": result.get("sample_id"),
        "scenario": result.get("scenario"),
        "mode": result.get("mode"),
        "model_name": result.get("model_name"),
        "prompt_version": result.get("prompt_version"),
        "error_type": classify_result_error(result),
        "error": result.get("error"),
        "raw_output": result.get("raw_output"),
        "parsed_output": result.get("parsed_output"),
        "json_valid": result.get("json_valid"),
        "schema_valid": result.get("schema_valid"),
        "sample_metrics": sample_metrics,
    }


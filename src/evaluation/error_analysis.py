"""Aggregate experiment failure cases into reviewable error counts."""

from collections import Counter


def summarize_failure_types(failure_cases: list[dict[str, str]]) -> dict[str, int]:
    """Count failure records by error type, using `unknown` when absent."""
    return dict(Counter(case.get("error_type", "unknown") for case in failure_cases))


from collections import Counter


def summarize_failure_types(failure_cases: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(case.get("error_type", "unknown") for case in failure_cases))


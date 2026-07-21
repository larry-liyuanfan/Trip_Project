"""Paired baseline-versus-standardized comparison for Week 3 score artifacts."""

import csv
import json
import math
import random
from pathlib import Path
from typing import Any


LOWER_IS_BETTER = {"latency_ms", "constraint_violation_rate"}
NON_METRIC_FIELDS = {
    "run_id",
    "sample_id",
    "scenario",
    "model_name",
    "prompt_version",
    "scoring_track",
    "structured_valid",
    "format_structured_valid",
    "multilabel_counts",
    "semantic_metrics_status",
}


class ComparisonValidationError(ValueError):
    """Raised when two runs cannot support an attributable paired comparison."""


def validate_comparable_runs(
    baseline: dict[str, Any],
    standardized: dict[str, Any],
) -> None:
    """Require identical data/model contracts and the expected prompt pair."""
    for name, metadata, prompt_version in (
        ("baseline", baseline, "baseline_minimal_v1"),
        ("standardized", standardized, "standardized_v1"),
    ):
        if metadata.get("status") != "completed" or metadata.get("mode") != "live":
            raise ComparisonValidationError(f"{name} run must be completed live inference")
        if metadata.get("run_scope") != "full":
            raise ComparisonValidationError(f"{name} run must be a full-scope evaluation")
        if metadata.get("prompt_version") != prompt_version:
            raise ComparisonValidationError(
                f"{name} run must use {prompt_version}"
            )
    if baseline.get("dataset_version") != standardized.get("dataset_version"):
        raise ComparisonValidationError("runs use different dataset versions")
    if baseline.get("selected_sample_ids_sha256") != standardized.get(
        "selected_sample_ids_sha256"
    ):
        raise ComparisonValidationError("runs do not use the same sample set")
    for field in ("selected_count", "record_count", "model_name", "model_config"):
        if baseline.get(field) != standardized.get(field):
            raise ComparisonValidationError(f"runs differ in {field}")
    baseline_assets = _non_prompt_artifacts(baseline.get("artifact_hashes"))
    standardized_assets = _non_prompt_artifacts(standardized.get("artifact_hashes"))
    if baseline_assets != standardized_assets:
        raise ComparisonValidationError("runs differ in non-Prompt evaluation artifacts")


def compare_score_records(
    baseline_scores: list[dict[str, Any]],
    standardized_scores: list[dict[str, Any]],
    *,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20260714,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return sample deltas and scenario/metric paired aggregates."""
    if bootstrap_iterations <= 0:
        raise ComparisonValidationError("bootstrap_iterations must be positive")
    baseline = _index_scores(baseline_scores, "baseline")
    standardized = _index_scores(standardized_scores, "standardized")
    if set(baseline) != set(standardized):
        raise ComparisonValidationError("baseline and standardized sample IDs differ")
    if not baseline:
        raise ComparisonValidationError("cannot compare empty score sets")

    sample_rows: list[dict[str, Any]] = []
    grouped_deltas: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for sample_id in sorted(baseline):
        baseline_row = baseline[sample_id]
        standardized_row = standardized[sample_id]
        scenario = baseline_row.get("scenario")
        if scenario != standardized_row.get("scenario"):
            raise ComparisonValidationError(f"scenario mismatch for {sample_id}")
        common_metrics = sorted(
            field
            for field in set(baseline_row) & set(standardized_row)
            if field not in NON_METRIC_FIELDS
            and _is_number(baseline_row[field])
            and _is_number(standardized_row[field])
        )
        metrics: dict[str, dict[str, float]] = {}
        for metric in common_metrics:
            before = float(baseline_row[metric])
            after = float(standardized_row[metric])
            delta = after - before
            metrics[metric] = {
                "baseline": before,
                "standardized": after,
                "delta": delta,
            }
            grouped_deltas.setdefault((scenario, metric), []).append(
                (before, after, delta)
            )
        sample_rows.append(
            {"sample_id": sample_id, "scenario": scenario, "metrics": metrics}
        )

    aggregate_rows: list[dict[str, Any]] = []
    for (scenario, metric), values in sorted(grouped_deltas.items()):
        baseline_values = [value[0] for value in values]
        standardized_values = [value[1] for value in values]
        deltas = [value[2] for value in values]
        baseline_mean = sum(baseline_values) / len(values)
        standardized_mean = sum(standardized_values) / len(values)
        absolute_delta = standardized_mean - baseline_mean
        direction = -1 if metric in LOWER_IS_BETTER else 1
        standardized_wins = sum(direction * delta > 0 for delta in deltas)
        baseline_wins = sum(direction * delta < 0 for delta in deltas)
        ties = len(deltas) - standardized_wins - baseline_wins
        ci_low, ci_high = _paired_bootstrap_ci(
            deltas,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + _stable_seed_offset(scenario, metric),
        )
        aggregate_rows.append(
            {
                "scenario": scenario,
                "metric": metric,
                "paired_count": len(values),
                "baseline_mean": baseline_mean,
                "standardized_mean": standardized_mean,
                "absolute_delta": absolute_delta,
                "relative_delta": (
                    absolute_delta / abs(baseline_mean)
                    if baseline_mean != 0
                    else None
                ),
                "baseline_wins": baseline_wins,
                "standardized_wins": standardized_wins,
                "ties": ties,
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
            }
        )
    return sample_rows, aggregate_rows


def export_comparison_artifacts(
    output_dir: Path,
    metadata: dict[str, Any],
    sample_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    """Write immutable strict JSON/CSV comparison artifacts."""
    _strict_json(metadata)
    for row in sample_rows:
        _strict_json(row)
    for row in aggregate_rows:
        _strict_json(row)
    output = Path(output_dir)
    try:
        output.mkdir(parents=True)
    except FileExistsError as exc:
        raise ComparisonValidationError(
            f"comparison directory already exists: {output}"
        ) from exc
    metadata_path = output / "metadata.json"
    samples_path = output / "sample_deltas.jsonl"
    aggregates_path = output / "aggregate_deltas.csv"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    with samples_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in sample_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
    fieldnames = [
        "scenario",
        "metric",
        "paired_count",
        "baseline_mean",
        "standardized_mean",
        "absolute_delta",
        "relative_delta",
        "baseline_wins",
        "standardized_wins",
        "ties",
        "delta_ci95_low",
        "delta_ci95_high",
    ]
    with aggregates_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return {
        "metadata": metadata_path,
        "sample_deltas": samples_path,
        "aggregate_deltas": aggregates_path,
    }


def select_representative_cases(
    sample_rows: list[dict[str, Any]],
    *,
    per_direction: int = 3,
) -> list[dict[str, Any]]:
    """Select largest paired improvements/regressions by a fixed auditable rule."""
    if per_direction <= 0:
        raise ComparisonValidationError("per_direction must be positive")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in sample_rows:
        scenario = row.get("scenario")
        sample_id = row.get("sample_id")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for metric, values in metrics.items():
            if not isinstance(values, dict) or not _is_number(values.get("delta")):
                continue
            delta = float(values["delta"])
            directed_delta = -delta if metric in LOWER_IS_BETTER else delta
            if directed_delta == 0:
                continue
            direction = "improvement" if directed_delta > 0 else "regression"
            grouped.setdefault((scenario, metric, direction), []).append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "sample_id": sample_id,
                    "direction": direction,
                    "baseline": values.get("baseline"),
                    "standardized": values.get("standardized"),
                    "delta": delta,
                    "selection_rule": "largest absolute paired delta",
                }
            )
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = sorted(
            grouped[key],
            key=lambda row: (-abs(float(row["delta"])), str(row["sample_id"])),
        )
        selected.extend(rows[:per_direction])
    return selected


def generate_comparison_report(
    metadata: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    representative_cases: list[dict[str, Any]],
) -> str:
    """Render a report only from persisted paired-comparison artifacts."""
    lines = [
        "# Week 3 Baseline vs Standardized Prompt Comparison",
        "",
        "## Run identity",
        "",
        f"- Comparison ID: `{metadata['comparison_id']}`",
        f"- Baseline run: `{metadata['baseline_run_id']}`",
        f"- Standardized run: `{metadata['standardized_run_id']}`",
        f"- Dataset version: `{metadata['dataset_version']}`",
        f"- Paired sample count: {metadata['paired_sample_count']}",
        f"- Scoring track: `{metadata.get('scoring_track', 'strict_business')}`",
        "",
        "## Paired metric deltas",
        "",
        "| Scenario | Metric | N | Baseline | Standardized | Delta | 95% CI | W/T/L |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {scenario} | {metric} | {paired_count} | {baseline_mean:.4f} | "
            "{standardized_mean:.4f} | {absolute_delta:+.4f} | "
            "[{delta_ci95_low:+.4f}, {delta_ci95_high:+.4f}] | {standardized_wins}/{ties}/{baseline_wins} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "W/T/L is standardized wins / ties / baseline wins. For latency and constraint violation rate, lower is better.",
            "The minimal baseline intentionally has no JSON-format instruction. Its JSON/Schema compliance and latency are comparable, but semantic task metrics remain PENDING when the natural-language output is not deterministically parsed. Standardized outputs continue to use strict structured-business scoring.",
            "",
            "## Deterministically selected cases",
            "",
            "Cases are selected by the largest absolute paired delta within each scenario, metric, and direction; they are not manually cherry-picked.",
            "",
            "| Scenario | Metric | Direction | Sample ID | Baseline | Standardized | Delta |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for case in representative_cases:
        lines.append(
            "| {scenario} | {metric} | {direction} | `{sample_id}` | "
            "{baseline:.4f} | {standardized:.4f} | {delta:+.4f} |".format(**case)
        )
    if not representative_cases:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    lines.extend(
        [
            "",
            "Interpretations and optimization priorities must be written only after inspecting the persisted raw outputs for these sample IDs.",
            "",
        ]
    )
    return "\n".join(lines)


def _index_scores(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id") if isinstance(row, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ComparisonValidationError(f"{label} score requires sample_id")
        if sample_id in indexed:
            raise ComparisonValidationError(f"duplicate {label} score: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _non_prompt_artifacts(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ComparisonValidationError("run artifact_hashes must be an object")
    return {
        path: digest
        for path, digest in value.items()
        if "/prompts/" not in path.replace("\\", "/")
    }


def _paired_bootstrap_ci(
    deltas: list[float],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    if len(deltas) == 1:
        return deltas[0], deltas[0]
    randomizer = random.Random(seed)
    means = sorted(
        sum(randomizer.choice(deltas) for _ in deltas) / len(deltas)
        for _ in range(iterations)
    )
    low_index = max(math.floor(0.025 * (len(means) - 1)), 0)
    high_index = min(math.ceil(0.975 * (len(means) - 1)), len(means) - 1)
    return means[low_index], means[high_index]


def _stable_seed_offset(scenario: str, metric: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(f"{scenario}|{metric}"))


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _strict_json(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ComparisonValidationError(f"comparison contains invalid JSON value: {exc}") from exc

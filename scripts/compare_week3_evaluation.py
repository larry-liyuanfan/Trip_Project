"""Compare completed baseline and standardized runs on the same validated samples."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.comparison import (
    ComparisonValidationError,
    compare_score_records,
    export_comparison_artifacts,
    generate_comparison_report,
    select_representative_cases,
    validate_comparable_runs,
)
from src.evaluation.config import load_evaluation_config
from src.evaluation.manifests import read_jsonl_objects
from src.evaluation.provenance import build_artifact_hashes, verify_artifact_hashes
from src.evaluation.results import load_run_metadata


COMPARISON_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def compare_runs(
    *,
    root: Path,
    config_path: Path,
    comparison_id: str,
    baseline_run_id: str,
    standardized_run_id: str,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    """Validate two runs, compare paired scores, and write generated evidence."""
    if COMPARISON_ID_PATTERN.fullmatch(comparison_id) is None:
        raise ComparisonValidationError("comparison_id contains unsupported characters")
    project_root = Path(root)
    resolved_config = config_path if config_path.is_absolute() else project_root / config_path
    config = load_evaluation_config(resolved_config)
    paths = config.get("paths", {})
    for name in (
        "runs_dir",
        "scores_dir",
        "comparisons_dir",
        "generated_reports_dir",
    ):
        if not isinstance(paths.get(name), str):
            raise ComparisonValidationError(f"evaluation config requires paths.{name}")

    baseline_metadata = load_run_metadata(
        project_root / paths["runs_dir"] / baseline_run_id / "metadata.json"
    )
    standardized_metadata = load_run_metadata(
        project_root / paths["runs_dir"] / standardized_run_id / "metadata.json"
    )
    if baseline_metadata["run_id"] != baseline_run_id:
        raise ComparisonValidationError("baseline metadata run_id mismatch")
    if standardized_metadata["run_id"] != standardized_run_id:
        raise ComparisonValidationError("standardized metadata run_id mismatch")
    verify_artifact_hashes(project_root, baseline_metadata["artifact_hashes"])
    verify_artifact_hashes(project_root, standardized_metadata["artifact_hashes"])
    validate_comparable_runs(baseline_metadata, standardized_metadata)

    baseline_score_path = (
        project_root
        / paths["scores_dir"]
        / baseline_run_id
        / "sample_scores.jsonl"
    )
    standardized_score_path = (
        project_root
        / paths["scores_dir"]
        / standardized_run_id
        / "sample_scores.jsonl"
    )
    baseline_scores = read_jsonl_objects(baseline_score_path)
    standardized_scores = read_jsonl_objects(standardized_score_path)
    sample_rows, aggregate_rows = compare_score_records(
        baseline_scores,
        standardized_scores,
        bootstrap_iterations=bootstrap_iterations,
    )
    representative_cases = select_representative_cases(sample_rows)
    metadata = {
        "comparison_id": comparison_id,
        "baseline_run_id": baseline_run_id,
        "standardized_run_id": standardized_run_id,
        "dataset_version": baseline_metadata["dataset_version"],
        "selected_sample_ids_sha256": baseline_metadata[
            "selected_sample_ids_sha256"
        ],
        "paired_sample_count": len(sample_rows),
        "scoring_track": "strict_business",
        "bootstrap_iterations": bootstrap_iterations,
        "score_artifact_hashes": build_artifact_hashes(
            project_root,
            [baseline_score_path, standardized_score_path],
        ),
    }
    report = generate_comparison_report(
        metadata,
        aggregate_rows,
        representative_cases,
    )
    comparison_dir = project_root / paths["comparisons_dir"] / comparison_id
    report_path = (
        project_root / paths["generated_reports_dir"] / comparison_id / "report.md"
    )
    if comparison_dir.exists():
        raise ComparisonValidationError(
            f"comparison directory already exists: {comparison_dir}"
        )
    if report_path.exists():
        raise ComparisonValidationError(f"generated report already exists: {report_path}")

    output_paths = export_comparison_artifacts(
        comparison_dir,
        metadata,
        sample_rows,
        aggregate_rows,
    )
    cases_path = comparison_dir / "representative_cases.jsonl"
    with cases_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in representative_cases:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    return {
        **metadata,
        "artifacts": {
            **{name: str(path) for name, path in output_paths.items()},
            "representative_cases": str(cases_path),
            "report": str(report_path),
        },
    }


def run_cli(argv: list[str] | None = None, *, root: Path | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week3.yaml")
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--standardized-run-id", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args(argv)
    summary = compare_runs(
        root=Path(root) if root is not None else Path.cwd(),
        config_path=Path(args.config),
        comparison_id=args.comparison_id,
        baseline_run_id=args.baseline_run_id,
        standardized_run_id=args.standardized_run_id,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()

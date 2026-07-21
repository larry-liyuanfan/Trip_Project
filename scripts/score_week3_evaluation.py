"""Score one persisted Week 3 run without invoking a model or changing the run."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.config import load_evaluation_config
from src.evaluation.error_analysis import summarize_failure_types
from src.evaluation.manifests import load_configured_manifests
from src.evaluation.metrics import (
    build_annotation_index,
    export_score_artifacts,
    load_metric_aliases,
    load_result_records,
    score_records,
)
from src.evaluation.results import RUN_ID_PATTERN, load_run_metadata
from src.evaluation.provenance import canonical_sha256, verify_artifact_hashes


def score_run(
    *,
    root: Path,
    config_path: Path,
    run_id: str,
) -> dict[str, Any]:
    """Read an immutable run and export reproducible scores to a separate directory."""
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id contains unsupported characters")
    project_root = Path(root)
    resolved_config = config_path if config_path.is_absolute() else project_root / config_path
    config = load_evaluation_config(resolved_config)
    scores_dir = config.get("paths", {}).get("scores_dir")
    metrics = config.get("metrics")
    if not isinstance(scores_dir, str) or not isinstance(metrics, dict):
        raise ValueError("evaluation config must declare paths.scores_dir and metrics")
    aliases_path = metrics.get("aliases_path")
    if not isinstance(aliases_path, str):
        raise ValueError("evaluation config must declare metrics.aliases_path")

    run_dir = project_root / config["paths"]["runs_dir"] / run_id
    metadata = load_run_metadata(run_dir / "metadata.json")
    if metadata["run_id"] != run_id:
        raise ValueError("metadata.run_id does not match --run-id")
    verify_artifact_hashes(project_root, metadata["artifact_hashes"])
    results = load_result_records(run_dir / "results.jsonl")
    if metadata["record_count"] != len(results):
        raise ValueError(
            "metadata.record_count does not match results.jsonl record count"
        )
    if (
        metadata["status"] == "completed"
        and metadata["selected_count"] != metadata["record_count"]
    ):
        raise ValueError(
            "metadata.selected_count must equal metadata.record_count for completed run"
        )
    if metadata["status"] == "failed":
        raise ValueError(f"failed run is not scoreable: {metadata['error']}")
    for result in results:
        if result["run_id"] != run_id:
            raise ValueError(f"result run_id mismatch: {result['sample_id']}")
    selected_sample_ids_sha256 = canonical_sha256(
        [result["sample_id"] for result in results]
    )
    if selected_sample_ids_sha256 != metadata["selected_sample_ids_sha256"]:
        raise ValueError("metadata selected sample IDs do not match results.jsonl")

    manifests = load_configured_manifests(config, root=project_root)
    annotations = build_annotation_index(manifests)
    aliases = load_metric_aliases(project_root / aliases_path)
    sample_scores, aggregates, error_cases = score_records(
        results,
        annotations,
        aliases,
    )
    score_dir = project_root / scores_dir / run_id
    paths = export_score_artifacts(
        score_dir,
        sample_scores,
        aggregates,
        error_cases,
    )
    summary = {
        "run_id": run_id,
        "run_status": metadata["status"],
        "selected_count": metadata["selected_count"],
        "record_count": metadata["record_count"],
        "sample_count": len(sample_scores),
        "scenario_counts": {
            scenario: aggregate["sample_count"]
            for scenario, aggregate in aggregates.items()
        },
        "error_counts": summarize_failure_types(error_cases),
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    summary_path = score_dir / "score_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    return summary


def run_cli(argv: list[str] | None = None, *, root: Path | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week3.yaml")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    summary = score_run(
        root=Path(root) if root is not None else Path.cwd(),
        config_path=Path(args.config),
        run_id=args.run_id,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()

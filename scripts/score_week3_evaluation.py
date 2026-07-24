"""Score one persisted Week 3 run without invoking a model or changing the run."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.config import load_evaluation_config
from src.evaluation.error_analysis import summarize_failure_types
from src.evaluation.baseline_semantics import BaselineSemanticCoder
from src.evaluation.manifests import load_configured_manifests
from src.evaluation.metrics import (
    aggregate_scenario_scores,
    build_annotation_index,
    export_score_artifacts,
    load_metric_aliases,
    load_result_records,
    score_semantic_prediction,
    score_records,
)
from src.evaluation.results import RUN_ID_PATTERN, load_run_metadata
from src.evaluation.provenance import canonical_sha256, verify_artifact_hashes


def score_run(
    *,
    root: Path,
    config_path: Path,
    run_id: str,
    semantic_coding_config: Path | None = None,
    score_id: str | None = None,
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

    aliases = load_metric_aliases(project_root / aliases_path)
    coding_metadata: dict[str, Any] = {}
    if semantic_coding_config is None:
        if score_id not in (None, run_id):
            raise ValueError("score_id may differ from run_id only for semantic coding")
        resolved_score_id = run_id
        manifests = load_configured_manifests(config, root=project_root)
        annotations = build_annotation_index(manifests)
        sample_scores, aggregates, error_cases = score_records(
            results,
            annotations,
            aliases,
        )
    else:
        if not isinstance(score_id, str) or RUN_ID_PATTERN.fullmatch(score_id) is None:
            raise ValueError("semantic coding requires a valid --score-id")
        resolved_semantic_config = (
            semantic_coding_config
            if semantic_coding_config.is_absolute()
            else project_root / semantic_coding_config
        )
        coder = BaselineSemanticCoder.from_path(resolved_semantic_config)
        if metadata.get("prompt_version") != coder.allowed_prompt_version:
            raise ValueError(
                "semantic coding config is restricted to baseline_minimal_v1"
            )

        # Stage A is intentionally completed before any manifest or gold is loaded.
        predictions = [
            coder.encode(
                scenario=result["scenario"],
                raw_output=result["raw_output"],
            )
            for result in results
        ]
        manifests = load_configured_manifests(config, root=project_root)
        annotations = build_annotation_index(manifests)
        sample_scores, aggregates, error_cases = _score_semantic_records(
            results,
            predictions,
            annotations,
            aliases,
            coding_version=coder.version,
            codebook_sha256=coder.codebook_sha256,
        )
        resolved_score_id = score_id
        coding_metadata = {
            "scoring_track": "baseline_semantic_coding_v1",
            "coding_version": coder.version,
            "codebook_sha256": coder.codebook_sha256,
            "semantic_coding_config": str(
                resolved_semantic_config.relative_to(project_root)
            ).replace("\\", "/"),
        }

    score_dir = project_root / scores_dir / resolved_score_id
    paths = export_score_artifacts(
        score_dir,
        sample_scores,
        aggregates,
        error_cases,
    )
    summary = {
        "score_id": resolved_score_id,
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
        **coding_metadata,
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
    parser.add_argument("--semantic-coding-config")
    parser.add_argument("--score-id")
    args = parser.parse_args(argv)
    summary = score_run(
        root=Path(root) if root is not None else Path.cwd(),
        config_path=Path(args.config),
        run_id=args.run_id,
        semantic_coding_config=(
            Path(args.semantic_coding_config)
            if args.semantic_coding_config is not None
            else None
        ),
        score_id=args.score_id,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> None:
    run_cli()


def _score_semantic_records(
    results: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    annotations_by_sample_id: dict[str, dict[str, Any]],
    aliases: dict[str, dict[str, str]],
    *,
    coding_version: str,
    codebook_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Join frozen predictions to gold only after the coding stage is complete."""
    from src.evaluation.error_analysis import build_error_case, classify_result_error

    if len(results) != len(predictions):
        raise ValueError("semantic predictions must align one-to-one with results")
    sample_scores: list[dict[str, Any]] = []
    error_cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result, prediction in zip(results, predictions):
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
        sample_score = score_semantic_prediction(
            result,
            annotation_record["annotation"],
            prediction,
            aliases,
            coding_version=coding_version,
            codebook_sha256=codebook_sha256,
        )
        sample_scores.append(sample_score)
        if classify_result_error(result) != "valid":
            error_cases.append(build_error_case(result, sample_score))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample_score in sample_scores:
        grouped.setdefault(sample_score["scenario"], []).append(sample_score)
    aggregates = {
        scenario: aggregate_scenario_scores(rows)
        for scenario, rows in sorted(grouped.items())
    }
    return sample_scores, aggregates, error_cases


if __name__ == "__main__":
    main()

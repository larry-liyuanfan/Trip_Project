"""Week 3 baseline 与 Week 4 winner 的共同确定性语义评分轨道。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.evaluation.baseline_semantics import BaselineSemanticCoder
from src.evaluation.comparison import (
    compare_score_records,
    export_comparison_artifacts,
)
from src.evaluation.config import load_evaluation_config
from src.evaluation.error_analysis import build_error_case, classify_result_error
from src.evaluation.manifests import load_configured_manifests
from src.evaluation.metrics import (
    aggregate_scenario_scores,
    build_annotation_index,
    export_score_artifacts,
    load_metric_aliases,
    load_result_records,
    score_common_semantic_prediction,
)
from src.evaluation.provenance import canonical_sha256
from src.evaluation.results import load_run_metadata


SCORING_TRACK = "week4_common_semantic_coding_v1"
BASELINE_PROMPT = "baseline_minimal_v1"
WINNER_PROMPTS = frozenset({"standardized_v2"})


def compare_common_semantics(
    *,
    root: Path,
    week3_config_path: Path,
    baseline_run_id: str,
    winner_run_id: str,
    winner_runs_dir: Path,
    semantic_coding_config: Path,
    output_dir: Path,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20260726,
) -> dict[str, Any]:
    """冻结两组原始输出，先统一编码，再连接同一金标评分并成对比较。"""
    project_root = Path(root)
    resolved_week3_config = _resolve(project_root, week3_config_path)
    week3 = load_evaluation_config(resolved_week3_config)
    baseline_dir = project_root / week3["paths"]["runs_dir"] / baseline_run_id
    winner_root = _resolve(project_root, winner_runs_dir)
    winner_dir = winner_root / winner_run_id
    baseline_metadata = load_run_metadata(baseline_dir / "metadata.json")
    winner_metadata = load_run_metadata(winner_dir / "metadata.json")
    _validate_run_pair(
        baseline_metadata,
        winner_metadata,
        baseline_run_id=baseline_run_id,
        winner_run_id=winner_run_id,
    )
    baseline_results = load_result_records(baseline_dir / "results.jsonl")
    winner_results = load_result_records(winner_dir / "results.jsonl")
    _validate_result_pair(
        baseline_results,
        winner_results,
        baseline_run_id=baseline_run_id,
        winner_run_id=winner_run_id,
    )

    coder = BaselineSemanticCoder.from_path(
        _resolve(project_root, semantic_coding_config)
    )
    # 两组预测均在加载 manifest/人工金标前完成，避免金标进入编码阶段。
    baseline_predictions = _encode_results(coder, baseline_results)
    winner_predictions = _encode_results(coder, winner_results)

    manifests = load_configured_manifests(week3, root=project_root)
    annotations = build_annotation_index(manifests)
    aliases = load_metric_aliases(
        project_root / week3["metrics"]["aliases_path"]
    )
    baseline_scores, baseline_aggregates, baseline_errors = _score_predictions(
        baseline_results,
        baseline_predictions,
        annotations,
        aliases,
        coding_version=coder.version,
        codebook_sha256=coder.codebook_sha256,
    )
    winner_scores, winner_aggregates, winner_errors = _score_predictions(
        winner_results,
        winner_predictions,
        annotations,
        aliases,
        coding_version=coder.version,
        codebook_sha256=coder.codebook_sha256,
    )
    sample_deltas, aggregate_deltas = compare_score_records(
        baseline_scores,
        winner_scores,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )

    resolved_output = _resolve(project_root, output_dir)
    if resolved_output.exists():
        raise FileExistsError(
            f"common semantic comparison output already exists: {resolved_output}"
        )
    resolved_output.mkdir(parents=True)
    _write_predictions(
        resolved_output / "baseline_canonical_predictions.jsonl",
        baseline_results,
        baseline_predictions,
    )
    _write_predictions(
        resolved_output / "winner_canonical_predictions.jsonl",
        winner_results,
        winner_predictions,
    )
    export_score_artifacts(
        resolved_output / "baseline_score",
        baseline_scores,
        baseline_aggregates,
        baseline_errors,
    )
    export_score_artifacts(
        resolved_output / "winner_score",
        winner_scores,
        winner_aggregates,
        winner_errors,
    )
    metadata = {
        "comparison_id": resolved_output.name,
        "scoring_track": SCORING_TRACK,
        "coding_version": coder.version,
        "codebook_sha256": coder.codebook_sha256,
        "baseline_run_id": baseline_run_id,
        "winner_run_id": winner_run_id,
        "dataset_version": baseline_metadata["dataset_version"],
        "model_name": baseline_metadata["model_name"],
        "selected_sample_ids_sha256": baseline_metadata[
            "selected_sample_ids_sha256"
        ],
        "paired_sample_count": len(baseline_scores),
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "prediction_input_fields": ["scenario", "raw_output"],
        "gold_joined_after_prediction": True,
        "baseline_results_sha256": _file_sha256(
            baseline_dir / "results.jsonl"
        ),
        "winner_results_sha256": _file_sha256(winner_dir / "results.jsonl"),
        "interpretation": (
            "same deterministic lexical encoder and metric functions; "
            "JSON/Schema and latency remain direct raw-run measurements"
        ),
    }
    export_comparison_artifacts(
        resolved_output / "paired_comparison",
        metadata,
        sample_deltas,
        aggregate_deltas,
    )
    summary = {
        **metadata,
        "scenario_counts": {
            scenario: aggregate["sample_count"]
            for scenario, aggregate in sorted(baseline_aggregates.items())
        },
        "aggregate_metric_count": len(aggregate_deltas),
        "artifact_paths": {
            "baseline_predictions": "baseline_canonical_predictions.jsonl",
            "winner_predictions": "winner_canonical_predictions.jsonl",
            "baseline_score": "baseline_score",
            "winner_score": "winner_score",
            "paired_comparison": "paired_comparison",
        },
    }
    _write_json(resolved_output / "summary.json", summary)
    return summary


def _validate_run_pair(
    baseline: dict[str, Any],
    winner: dict[str, Any],
    *,
    baseline_run_id: str,
    winner_run_id: str,
) -> None:
    for label, metadata, run_id in (
        ("baseline", baseline, baseline_run_id),
        ("winner", winner, winner_run_id),
    ):
        if metadata.get("run_id") != run_id:
            raise ValueError(f"{label} metadata run_id mismatch")
        if (
            metadata.get("status") != "completed"
            or metadata.get("mode") != "live"
            or metadata.get("run_scope") != "full"
        ):
            raise ValueError(f"{label} must be a completed live full run")
        if metadata.get("selected_count") != 450 or metadata.get(
            "record_count"
        ) != 450:
            raise ValueError(f"{label} run must contain 450 records")
    if baseline.get("prompt_version") != BASELINE_PROMPT:
        raise ValueError("baseline run must use baseline_minimal_v1")
    winner_prompts = winner.get("prompt_versions_by_scenario")
    if (
        not isinstance(winner_prompts, dict)
        or set(winner_prompts.values()) - WINNER_PROMPTS
    ):
        raise ValueError("winner run contains an unsupported Prompt version")
    for field in (
        "dataset_version",
        "model_name",
        "model_config",
        "selected_sample_ids_sha256",
        "selected_count",
        "record_count",
    ):
        if baseline.get(field) != winner.get(field):
            raise ValueError(f"run pair differs in {field}")


def _validate_result_pair(
    baseline: list[dict[str, Any]],
    winner: list[dict[str, Any]],
    *,
    baseline_run_id: str,
    winner_run_id: str,
) -> None:
    if len(baseline) != 450 or len(winner) != 450:
        raise ValueError("both result sets must contain 450 records")
    baseline_ids = [row["sample_id"] for row in baseline]
    winner_ids = [row["sample_id"] for row in winner]
    if baseline_ids != winner_ids:
        raise ValueError("result sets must have identical ordered sample IDs")
    if canonical_sha256(baseline_ids) != canonical_sha256(winner_ids):
        raise ValueError("result sample hashes differ")
    if any(row["run_id"] != baseline_run_id for row in baseline):
        raise ValueError("baseline result run_id mismatch")
    if any(row["run_id"] != winner_run_id for row in winner):
        raise ValueError("winner result run_id mismatch")
    for before, after in zip(baseline, winner):
        if before["scenario"] != after["scenario"]:
            raise ValueError(
                f"scenario mismatch for sample_id: {before['sample_id']}"
            )


def _encode_results(
    coder: BaselineSemanticCoder,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        coder.encode(
            scenario=result["scenario"],
            raw_output=result["raw_output"],
        )
        for result in results
    ]


def _score_predictions(
    results: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    aliases: dict[str, dict[str, str]],
    *,
    coding_version: str,
    codebook_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    sample_scores: list[dict[str, Any]] = []
    error_cases: list[dict[str, Any]] = []
    for result, prediction in zip(results, predictions):
        annotation_record = annotations.get(result["sample_id"])
        if not isinstance(annotation_record, dict):
            raise ValueError(
                f"missing annotation for sample_id: {result['sample_id']}"
            )
        score = score_common_semantic_prediction(
            result,
            annotation_record["annotation"],
            prediction,
            aliases,
            coding_version=coding_version,
            codebook_sha256=codebook_sha256,
            scoring_track=SCORING_TRACK,
        )
        sample_scores.append(score)
        if classify_result_error(result) != "valid":
            error_cases.append(build_error_case(result, score))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for score in sample_scores:
        grouped.setdefault(score["scenario"], []).append(score)
    aggregates = {
        scenario: aggregate_scenario_scores(rows)
        for scenario, rows in sorted(grouped.items())
    }
    return sample_scores, aggregates, error_cases


def _write_predictions(
    path: Path,
    results: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for result, prediction in zip(results, predictions):
            row = {
                "run_id": result["run_id"],
                "sample_id": result["sample_id"],
                "scenario": result["scenario"],
                "coding_version": "baseline_semantic_coding_v1",
                "scoring_track": SCORING_TRACK,
                "prediction": prediction,
            }
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path

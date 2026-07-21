"""Validate Week 3 manifests, five-count summaries, and exclusion registry."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.config import load_evaluation_config
from src.evaluation.manifests import (
    load_configured_manifests,
    summarize_counts,
    validate_exclusion_manifest,
)
from src.evaluation.provenance import verify_artifact_hashes
from src.evaluation.results import load_run_metadata
from src.evaluation.metrics import load_result_records


def load_tested_sample_ids(
    config: dict[str, Any],
    *,
    root: Path,
    run_id: str,
) -> dict[str, set[str]]:
    """Load tested IDs only from one complete, immutable, full live run."""
    runs_dir = config.get("paths", {}).get("runs_dir")
    if not isinstance(runs_dir, str):
        raise ValueError("evaluation config requires paths.runs_dir")
    run_dir = root / runs_dir / run_id
    metadata = load_run_metadata(run_dir / "metadata.json")
    if metadata["run_id"] != run_id:
        raise ValueError("metadata.run_id does not match --run-id")
    if (
        metadata["status"] != "completed"
        or metadata["mode"] != "live"
        or metadata.get("run_scope") != "full"
    ):
        raise ValueError("tested counts require a completed full live run")
    verify_artifact_hashes(root, metadata["artifact_hashes"])
    results = load_result_records(run_dir / "results.jsonl")
    if metadata["record_count"] != len(results) or metadata["selected_count"] != len(
        results
    ):
        raise ValueError("run metadata counts do not match results.jsonl")
    tested = {scenario: set() for scenario in config.get("scenarios", {})}
    for result in results:
        if result["run_id"] != run_id:
            raise ValueError(f"result run_id mismatch: {result['sample_id']}")
        scenario = result["scenario"]
        if scenario not in tested:
            raise ValueError(f"result has unconfigured scenario: {scenario}")
        if result["sample_id"] in tested[scenario]:
            raise ValueError(f"duplicate tested sample_id: {result['sample_id']}")
        tested[scenario].add(result["sample_id"])
    return tested


def validate_configured_manifests(
    config: dict[str, Any],
    *,
    root: Path,
    tested_sample_ids: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Validate every configured scenario and ensure the registry is current."""
    all_records: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    tested_by_scenario = tested_sample_ids or {}
    configured_records = load_configured_manifests(config, root=root)
    for scenario, settings in config["scenarios"].items():
        records = configured_records[scenario]
        counts[scenario] = summarize_counts(
            records,
            target_count=settings["target_count"],
            tested_sample_ids=tested_by_scenario.get(scenario, set()),
        )
        all_records.extend(records)

    exclusion_path = root / config["paths"]["exclusion_manifest"]
    actual_exclusions = validate_exclusion_manifest(all_records, exclusion_path)
    return {
        "status": "ok",
        "dataset_version": config["dataset_version"],
        "counts": counts,
        "exclusion_count": len(actual_exclusions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week3.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = load_evaluation_config(args.config)
    root = Path.cwd()
    tested = (
        load_tested_sample_ids(config, root=root, run_id=args.run_id)
        if args.run_id
        else None
    )
    result = validate_configured_manifests(
        config,
        root=root,
        tested_sample_ids=tested,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

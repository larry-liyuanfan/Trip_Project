"""Prepare deterministic Week 3 candidates or rebuild the exclusion registry."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.config import load_evaluation_config
from src.evaluation.manifests import (
    ManifestValidationError,
    build_exclusion_rows,
    load_manifest,
    read_jsonl_objects,
    write_jsonl,
)
from src.evaluation.sampling import stratified_sample


def run_candidate_sampling(
    config: dict[str, Any],
    *,
    scenario: str,
    candidates_path: Path,
    output_path: Path,
    log_path: Path,
    root: Path,
) -> dict[str, Any]:
    """Sample a candidate JSONL into a pending evaluation manifest and audit log."""
    scenarios = config.get("scenarios", {})
    if scenario not in scenarios:
        raise ManifestValidationError(f"unknown configured scenario: {scenario}")
    candidates = read_jsonl_objects(candidates_path)
    sampling = scenarios[scenario]["sampling"]
    records, sampling_log = stratified_sample(
        candidates,
        scenario=scenario,
        dataset_version=config["dataset_version"],
        seed=sampling["seed"],
        stratum_field=sampling["stratum_field"],
        quotas=sampling["quotas"],
        root=root,
    )
    write_jsonl(output_path, records)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(sampling_log, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sampling_log


def rebuild_exclusion_registry(
    config: dict[str, Any], *, root: Path
) -> dict[str, Any]:
    """Rebuild the exclusion manifest from every configured evaluation candidate."""
    records: list[dict[str, Any]] = []
    for scenario, settings in config["scenarios"].items():
        scenario_records = load_manifest(root / settings["manifest_path"], root=root)
        for record in scenario_records:
            if record["scenario"] != scenario:
                raise ManifestValidationError(
                    f"{settings['manifest_path']} contains scenario {record['scenario']!r}, expected {scenario!r}"
                )
        records.extend(scenario_records)
    exclusions = build_exclusion_rows(records)
    output_path = root / config["paths"]["exclusion_manifest"]
    write_jsonl(output_path, exclusions)
    return {"path": str(output_path), "exclusion_count": len(exclusions)}


def initialize_evaluation_workspace(
    config: dict[str, Any], *, root: Path
) -> dict[str, Any]:
    """Create missing local manifests and registry without overwriting data."""
    root = Path(root)
    directories = {
        root / config["paths"]["images_dir"],
        root / config["paths"]["sampling_logs_dir"],
    }
    files = [
        root / settings["manifest_path"]
        for settings in config["scenarios"].values()
    ] + [root / config["paths"]["exclusion_manifest"]]
    directories.update(path.parent for path in files)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    existing: list[str] = []
    for path in files:
        if path.exists():
            existing.append(str(path))
            continue
        path.touch()
        created.append(str(path))
    return {
        "status": "initialized",
        "created": sorted(created),
        "existing": sorted(existing),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week3.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create missing local manifests and registry")

    sample = subparsers.add_parser("sample", help="Create a pending candidate manifest")
    sample.add_argument("--scenario", required=True)
    sample.add_argument("--candidates", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--log", type=Path, required=True)

    subparsers.add_parser("build-exclusion", help="Rebuild the evaluation exclusion manifest")
    args = parser.parse_args()
    config = load_evaluation_config(args.config)
    root = Path.cwd()
    if args.command == "init":
        result = initialize_evaluation_workspace(config, root=root)
    elif args.command == "sample":
        result = run_candidate_sampling(
            config,
            scenario=args.scenario,
            candidates_path=args.candidates,
            output_path=args.output,
            log_path=args.log,
            root=root,
        )
    else:
        result = rebuild_exclusion_registry(config, root=root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

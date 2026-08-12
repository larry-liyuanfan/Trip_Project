"""Spartan migration packaging and deterministic Week 5 result consolidation."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.data.week5_dataset import (
    SCENARIOS,
    Week5DataError,
    candidate_payload_sha256,
    iter_jsonl,
    write_jsonl_new,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise Week5DataError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _relative_to_root(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise Week5DataError(f"migration paths must stay inside repository root: {path}") from exc


def _validated_migration_id(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value) is None:
        raise Week5DataError("migration_id contains unsupported characters")
    return value


def _load_successes(path: Path) -> dict[str, dict[str, Any]]:
    successes: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return successes
    for row in iter_jsonl(path):
        if row.get("status") != "completed" or row.get("schema_valid") is not True:
            raise Week5DataError(f"results file contains a non-success row: {row.get('sample_id')}")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise Week5DataError("result row is missing sample_id")
        previous = successes.get(sample_id)
        if previous is not None and previous != row:
            raise Week5DataError(f"conflicting duplicate success result: {sample_id}")
        successes[sample_id] = row
    return successes


def _candidate_inventory(
    root: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    pool_root = root / config["paths"]["output_dir"] / "pools"
    for scenario in SCENARIOS:
        scenario_count = 0
        for candidate in iter_jsonl(pool_root / f"{scenario}.jsonl"):
            if candidate.get("scenario") != scenario:
                raise Week5DataError(f"candidate scenario mismatch: {candidate.get('sample_id')}")
            sample_id = candidate.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in by_id:
                raise Week5DataError(f"duplicate or invalid candidate sample_id: {sample_id}")
            by_id[sample_id] = candidate
            ordered.append(candidate)
            scenario_count += 1
        expected = int(config["targets"][scenario])
        if scenario_count != expected:
            raise Week5DataError(
                f"candidate pool count mismatch for {scenario}: {scenario_count}/{expected}"
            )
        counts[scenario] = scenario_count
    return ordered, by_id, counts


def _write_subset(
    root: Path,
    base_config: dict[str, Any],
    subset_root: Path,
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    grouped = {scenario: [] for scenario in SCENARIOS}
    for row in rows:
        grouped[row["scenario"]].append(row)
    subset_root.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for scenario in SCENARIOS:
        path = subset_root / "pools" / f"{scenario}.jsonl"
        counts[scenario] = write_jsonl_new(path, grouped[scenario])
        hashes[scenario] = _sha256_file(path)
    config = copy.deepcopy(base_config)
    config["paths"]["output_dir"] = _relative_to_root(root, subset_root)
    config["targets"] = {**counts, "dialogues": 0}
    config["final_minimums"] = {**counts, "dialogues": 0}
    config["runtime"]["base_url"] = "http://127.0.0.1:8001/v1"
    config_path = subset_root / "config.json"
    _write_json_new(config_path, config)
    return {
        "path": _relative_to_root(root, subset_root),
        "config": _relative_to_root(root, config_path),
        "counts": counts,
        "candidate_manifest_sha256": hashes,
    }


def prepare_spartan_migration(
    root: Path,
    config: dict[str, Any],
    *,
    source_run_dir: Path,
    output_dir: Path,
    migration_id: str,
    shard_count: int = 4,
    benchmark_count: int = 100,
) -> dict[str, Any]:
    """Create non-overlapping benchmark/shard inputs without modifying the source run."""
    migration_id = _validated_migration_id(migration_id)
    if not 1 <= shard_count <= 16:
        raise Week5DataError("shard_count must be between 1 and 16")
    if benchmark_count < 0:
        raise Week5DataError("benchmark_count cannot be negative")
    output_dir = output_dir.resolve()
    _relative_to_root(root, output_dir)
    if output_dir.exists():
        raise Week5DataError(f"refusing to overwrite migration directory: {output_dir}")

    ordered, candidates, candidate_counts = _candidate_inventory(root, config)
    source_results = source_run_dir / "results.jsonl"
    base_successes = _load_successes(source_results)
    unknown = sorted(set(base_successes) - set(candidates))
    if unknown:
        raise Week5DataError(f"source results contain unknown sample_id: {unknown[0]}")
    for sample_id, row in base_successes.items():
        expected_hash = candidate_payload_sha256(candidates[sample_id])
        actual_hash = row.get("candidate_sha256")
        if actual_hash is not None and actual_hash != expected_hash:
            raise Week5DataError(f"source result candidate hash mismatch: {sample_id}")

    pending = [row for row in ordered if row["sample_id"] not in base_successes]
    benchmark_count = min(benchmark_count, len(pending))
    benchmark = sorted(
        pending,
        key=lambda row: hashlib.sha256(
            f"{migration_id}\0benchmark\0{row['sample_id']}".encode("utf-8")
        ).hexdigest(),
    )[:benchmark_count]
    benchmark_ids = {row["sample_id"] for row in benchmark}
    remaining = [row for row in pending if row["sample_id"] not in benchmark_ids]
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for row in remaining:
        shard_index = int(
            hashlib.sha256(
                f"{migration_id}\0shard\0{row['sample_id']}".encode("utf-8")
            ).hexdigest(),
            16,
        ) % shard_count
        shards[shard_index].append(row)

    output_dir.mkdir(parents=True, exist_ok=False)
    benchmark_meta = _write_subset(root, config, output_dir / "benchmark", benchmark)
    shard_meta = [
        {
            "index": index,
            **_write_subset(root, config, output_dir / "shards" / f"{index:03d}", rows),
        }
        for index, rows in enumerate(shards)
    ]
    source_files = {}
    for name in ("run_manifest.json", "checkpoint.json", "results.jsonl", "failures.jsonl"):
        path = source_run_dir / name
        if path.is_file():
            source_files[name] = {
                "path": _relative_to_root(root, path),
                "sha256": _sha256_file(path),
            }
    manifest = {
        "schema_version": "week5_spartan_migration_v1",
        "migration_id": migration_id,
        "source_run": {
            "path": _relative_to_root(root, source_run_dir),
            "files": source_files,
            "verified_success_count": len(base_successes),
            "success_by_scenario": dict(Counter(row["scenario"] for row in base_successes.values())),
        },
        "candidate_counts": candidate_counts,
        "candidate_total": len(ordered),
        "benchmark": benchmark_meta,
        "shards": shard_meta,
        "coverage": {
            "base_success": len(base_successes),
            "benchmark": len(benchmark),
            "shards": len(remaining),
            "total": len(base_successes) + len(benchmark) + len(remaining),
        },
        "run_ids": {
            "benchmark": f"{migration_id}_benchmark",
            "shards": [f"{migration_id}_shard_{index:03d}" for index in range(shard_count)],
        },
        "single_writer_required": True,
        "human_completion": False,
    }
    if manifest["coverage"]["total"] != len(ordered):
        raise Week5DataError("migration coverage does not equal candidate inventory")
    _write_json_new(output_dir / "migration_manifest.json", manifest)
    return manifest


def _migration_runs(root: Path, migration_dir: Path, manifest: dict[str, Any]) -> list[tuple[str, Path]]:
    runs = [
        (
            "benchmark",
            root
            / manifest["benchmark"]["path"]
            / "runs"
            / manifest["run_ids"]["benchmark"],
        )
    ]
    for shard, run_id in zip(manifest["shards"], manifest["run_ids"]["shards"]):
        runs.append((f"shard_{shard['index']:03d}", root / shard["path"] / "runs" / run_id))
    return runs


def spartan_migration_status(root: Path, migration_dir: Path) -> dict[str, Any]:
    manifest = json.loads((migration_dir / "migration_manifest.json").read_text(encoding="utf-8"))
    success_ids = set(
        _load_successes(root / manifest["source_run"]["path"] / "results.jsonl")
    )
    runs: dict[str, Any] = {}
    for name, run_dir in _migration_runs(root, migration_dir, manifest):
        successes = _load_successes(run_dir / "results.jsonl")
        success_ids.update(successes)
        unresolved = 0
        if (run_dir / "failures.jsonl").exists():
            unresolved = sum(
                row.get("sample_id") not in successes for row in iter_jsonl(run_dir / "failures.jsonl")
            )
        runs[name] = {
            "exists": run_dir.exists(),
            "success": len(successes),
            "unresolved_failures": unresolved,
        }
    return {
        "migration_id": manifest["migration_id"],
        "candidate_total": manifest["candidate_total"],
        "unique_success": len(success_ids),
        "remaining": manifest["candidate_total"] - len(success_ids),
        "runs": runs,
    }


def merge_spartan_migration(
    root: Path, migration_dir: Path, destination: Path
) -> dict[str, Any]:
    """Consolidate source and Spartan runs into a new, immutable result directory."""
    if destination.exists():
        raise Week5DataError(f"refusing to overwrite merged result: {destination}")
    manifest = json.loads((migration_dir / "migration_manifest.json").read_text(encoding="utf-8"))
    _, candidates, candidate_counts = _candidate_inventory(
        root, _load_base_config_from_manifest(root, manifest)
    )
    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}

    sources = [("source", root / manifest["source_run"]["path"])] + _migration_runs(
        root, migration_dir, manifest
    )
    for source_name, run_dir in sources:
        for sample_id, row in _load_successes(run_dir / "results.jsonl").items():
            candidate = candidates.get(sample_id)
            if candidate is None:
                raise Week5DataError(f"merged result references unknown sample_id: {sample_id}")
            expected_hash = candidate_payload_sha256(candidate)
            if row.get("candidate_sha256") not in (None, expected_hash):
                raise Week5DataError(f"merged result candidate hash mismatch: {sample_id}")
            if sample_id in successes:
                raise Week5DataError(f"migration produced duplicate success: {sample_id}")
            successes[sample_id] = {**row, "migration_source": source_name}
        failure_path = run_dir / "failures.jsonl"
        if failure_path.exists():
            for row in iter_jsonl(failure_path):
                if isinstance(row.get("sample_id"), str):
                    failures[row["sample_id"]] = {**row, "migration_source": source_name}

    unresolved = [row for sample_id, row in failures.items() if sample_id not in successes]
    order = {sample_id: index for index, sample_id in enumerate(candidates)}
    destination.mkdir(parents=True, exist_ok=False)
    write_jsonl_new(
        destination / "results.jsonl",
        sorted(successes.values(), key=lambda row: order[row["sample_id"]]),
    )
    write_jsonl_new(
        destination / "failures.jsonl",
        sorted(unresolved, key=lambda row: order.get(row["sample_id"], len(order))),
    )
    by_scenario = {
        scenario: sum(row.get("scenario") == scenario for row in successes.values())
        for scenario in SCENARIOS
    }
    summary = {
        "schema_version": "week5_spartan_merge_v1",
        "migration_id": manifest["migration_id"],
        "status": "completed" if len(successes) == manifest["candidate_total"] else "partial",
        "candidate_counts": candidate_counts,
        "success_by_scenario": by_scenario,
        "unique_success": len(successes),
        "unresolved_failures": len(unresolved),
        "remaining": manifest["candidate_total"] - len(successes),
        "human_completion": False,
    }
    _write_json_new(destination / "summary.json", summary)
    return summary


def _load_base_config_from_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Recover the canonical pool location from the source run manifest."""
    source_manifest_path = root / manifest["source_run"]["path"] / "run_manifest.json"
    if not source_manifest_path.is_file():
        raise Week5DataError("source run manifest is required to merge migration results")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    candidate_manifests = source_manifest.get("identity", {}).get("candidate_manifests", {})
    if set(candidate_manifests) != set(SCENARIOS):
        raise Week5DataError("source run manifest has incomplete candidate manifests")
    paths = {Path(item["path"]).parent.as_posix() for item in candidate_manifests.values()}
    if len(paths) != 1:
        raise Week5DataError("source candidate manifests do not share one pool directory")
    pool_dir = Path(paths.pop())
    return {
        "paths": {"output_dir": pool_dir.parent.as_posix()},
        "targets": manifest["candidate_counts"],
    }

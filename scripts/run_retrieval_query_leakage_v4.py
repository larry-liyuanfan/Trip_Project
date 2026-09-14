"""Audit all locked v4 query bytes against the 1,000-image formal retrieval index."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_retrieval_query_leakage_v2 import _read_metadata, audit
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--retrieval-archive", type=Path, required=True)
    parser.add_argument("--index-overlay-root", type=Path, required=True)
    parser.add_argument("--query-pool-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    if args.registry_output.exists() or args.report_output.exists():
        raise FileExistsError("v4 leakage evidence outputs must not already exist")
    report, registry = build_evidence(
        config_path=args.config,
        retrieval_archive=args.retrieval_archive,
        index_overlay_root=args.index_overlay_root,
        query_pool_dir=args.query_pool_dir,
        source_snapshot_sha256=args.source_snapshot_sha256,
        implementation_commit=args.implementation_commit,
    )
    _write_jsonl(args.registry_output, registry)
    report = finalize_report(report, args.registry_output)
    _write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def build_evidence(
    *,
    config_path: Path,
    retrieval_archive: Path,
    index_overlay_root: Path,
    query_pool_dir: Path,
    source_snapshot_sha256: str,
    implementation_commit: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_sha(source_snapshot_sha256, "source snapshot")
    _validate_commit(implementation_commit)
    config = _load_object(config_path)
    formal = config["formal_release_read_only"]
    if file_sha256(retrieval_archive) != formal["retrieval_archive_sha256"]:
        raise ValueError("formal retrieval archive SHA-256 mismatch")
    metadata = _read_metadata(retrieval_archive, formal["metadata_member"])
    if len(metadata) != int(formal["expected_index_support"]):
        raise ValueError("formal retrieval metadata support mismatch")
    metadata_sha = _archive_member_sha256(retrieval_archive, formal["metadata_member"])
    if metadata_sha != formal["metadata_sha256"]:
        raise ValueError("formal retrieval metadata SHA-256 mismatch")

    overlay_manifest_path = index_overlay_root / "week8_retrieval_photo_overlay.json"
    overlay_manifest = _load_object(overlay_manifest_path)
    overlay_config = config["index_byte_source"]
    if file_sha256(overlay_manifest_path) != overlay_config["overlay_manifest_sha256"]:
        raise ValueError("index overlay manifest SHA-256 mismatch")
    expected_overlay = {
        "schema_version": overlay_config["overlay_schema_version"],
        "status": "COMPLETED",
        "metadata_sha256": formal["metadata_sha256"],
        "photos_zip_size": overlay_config["official_zip_size"],
        "requested_photo_count": overlay_config["expected_extracted_photo_count"],
        "extracted_photo_count": overlay_config["expected_extracted_photo_count"],
        "image_identity_sha256": overlay_config["image_identity_sha256"],
    }
    for key, expected in expected_overlay.items():
        if overlay_manifest.get(key) != expected:
            raise ValueError(f"index overlay manifest mismatch: {key}")

    lock, query_paths, query_support, query_image_hashes = _validate_query_pool(
        config_path, config, query_pool_dir
    )
    comparison_root = index_overlay_root / "data" / "yelp" / "raw" / "photos"
    report, registry = audit(
        metadata,
        index_overlay_root,
        comparison_root,
        query_paths,
        int(formal["expected_index_support"]),
    )
    image_identity = _index_image_identity(registry)
    if image_identity != overlay_config["image_identity_sha256"]:
        raise ValueError("index image byte identity differs from the locked overlay")
    if report["query_support"] != query_support:
        raise ValueError("audited query support mismatch")
    if report["status"] == "PASS_COMPLETE_NO_QUERY_INDEX_COLLISION":
        acceptance_status = "PASS"
    else:
        acceptance_status = "FAIL"
    acceptance = config["acceptance"]
    checks = {
        "primary_index_byte_coverage": report["primary_source_coverage"]["coverage"]
        >= float(acceptance["required_primary_index_byte_coverage"]),
        "query_image_byte_coverage": 1.0 >= float(acceptance["required_query_image_byte_coverage"]),
        "byte_collision_support": report["collision_check"]["byte_collision_support"]
        <= int(acceptance["maximum_byte_collision_support"]),
        "source_identity_collision_support": report["collision_check"]["source_identity_collision_support"]
        <= int(acceptance["maximum_source_identity_collision_support"]),
    }
    if acceptance_status == "PASS" and not all(checks.values()):
        raise ValueError("audit status contradicts fixed acceptance checks")
    report.update(
        {
            "schema_version": "retrieval_query_leakage_audit_v4",
            "configuration": {
                "config_sha256": file_sha256(config_path),
                "source_snapshot_sha256": source_snapshot_sha256,
                "implementation_commit_sha": implementation_commit,
                "retrieval_archive_sha256": formal["retrieval_archive_sha256"],
                "retrieval_metadata_sha256": metadata_sha,
                "query_pool_lock_file_sha256": file_sha256(
                    _resolve_repo_path(config_path, config["query_pool"]["committed_lock"])
                ),
                "query_pool_lock_canonical_sha256": canonical_json_sha256(lock),
                "index_overlay_manifest_sha256": file_sha256(overlay_manifest_path),
                "index_image_identity_sha256": overlay_config["image_identity_sha256"],
            },
            "query_byte_coverage": {
                "hashed_support": len(query_image_hashes),
                "expected_support": query_support,
                "coverage": len(query_image_hashes) / query_support,
                "status": "PASS",
            },
            "fixed_acceptance": {
                "thresholds": acceptance,
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            },
            "hardware": {
                "node": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            },
            "scope": {
                "query_splits": config["query_pool"]["splits"],
                "query_labels_or_rankings_read": False,
                "search_scoring_or_threshold_tuning_run": False,
                "fresh_test_used": False,
                "formal_release_changed": False,
                "human_annotation_support": 0,
            },
        }
    )
    return report, registry


def finalize_report(report: dict[str, Any], registry_path: Path) -> dict[str, Any]:
    finalized = dict(report)
    finalized["registry_file_sha256"] = file_sha256(registry_path)
    finalized["artifact_sha256"] = canonical_json_sha256(finalized)
    return finalized


def _validate_query_pool(
    config_path: Path,
    config: dict[str, Any],
    query_pool_dir: Path,
) -> tuple[dict[str, Any], list[Path], int, set[str]]:
    pool_config = config["query_pool"]
    lock_path = _resolve_repo_path(config_path, pool_config["committed_lock"])
    lock = _load_object(lock_path)
    if canonical_json_sha256(lock) != pool_config["committed_lock_canonical_sha256"]:
        raise ValueError("committed v4 pool lock canonical SHA-256 mismatch")
    if _load_object(query_pool_dir / "bundle_lock.json") != lock:
        raise ValueError("generated v4 pool lock mismatch")
    query_paths: list[Path] = []
    record_ids: set[str] = set()
    source_ids: set[str] = set()
    image_hashes: set[str] = set()
    support = 0
    for split in pool_config["splits"]:
        path = query_pool_dir / f"search_{split}_manifest.jsonl"
        rows = load_jsonl(path)
        expected = lock["search"][split]
        if (
            len(rows) != int(pool_config["expected_query_support_per_split"])
            or len(rows) != expected["query_support"]
            or canonical_json_sha256(rows) != expected["query_manifest_canonical_sha256"]
            or file_sha256(path) != expected["query_manifest_file_sha256"]
        ):
            raise ValueError(f"v4 query manifest lock mismatch: {split}")
        for row in rows:
            image = row.get("image", {})
            relative_path = image.get("relative_path")
            image_sha = image.get("sha256")
            source = row.get("source", {})
            query_id = row.get("query_id")
            source_id = source.get("source_id")
            if not all(isinstance(value, str) and value for value in (relative_path, image_sha, query_id, source_id)):
                raise ValueError(f"v4 query identity is incomplete: {split}")
            image_path = (query_pool_dir / relative_path).resolve()
            pool_root = query_pool_dir.resolve()
            if pool_root not in image_path.parents or file_sha256(image_path) != image_sha:
                raise ValueError(f"v4 query image byte mismatch: {query_id}")
            if query_id in record_ids or source_id in source_ids or image_sha in image_hashes:
                raise ValueError("v4 query identities are not disjoint across splits")
            record_ids.add(query_id)
            source_ids.add(source_id)
            image_hashes.add(image_sha)
        support += len(rows)
        query_paths.append(path)
    return lock, query_paths, support, image_hashes


def _archive_member_sha256(path: Path, member: str) -> str:
    import hashlib
    import tarfile

    digest = hashlib.sha256()
    with tarfile.open(path, "r:gz") as archive:
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"retrieval archive misses {member}")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_image_identity(registry: list[dict[str, Any]]) -> str:
    import hashlib

    if any(not isinstance(row.get("primary_sha256"), str) for row in registry):
        raise ValueError("index image byte identity is incomplete")
    payload = "".join(
        f"{row['image_id']}:{row['primary_sha256']}\n"
        for row in sorted(registry, key=lambda item: item["image_id"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_repo_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.resolve().parents[2] / path


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _validate_sha(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} is not a lowercase SHA-256")


def _validate_commit(value: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("implementation commit is not a full lowercase Git SHA")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

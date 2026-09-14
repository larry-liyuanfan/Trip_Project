"""Hash the locked retrieval source images in place and audit query/index leakage."""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--retrieval-archive", type=Path, required=True)
    parser.add_argument("--primary-index-root", type=Path, required=True)
    parser.add_argument("--comparison-photos-root", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, action="append", default=[])
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    if args.registry_output.exists() or args.report_output.exists():
        raise FileExistsError("leakage audit outputs must not already exist")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_sha = config["formal_release_read_only"]["retrieval_archive_sha256"]
    if file_sha256(args.retrieval_archive) != expected_sha:
        raise ValueError("formal retrieval archive SHA-256 mismatch")
    metadata = _read_metadata(
        args.retrieval_archive, config["formal_release_read_only"]["metadata_member"]
    )
    expected_support = int(config["formal_release_read_only"]["expected_index_support"])
    if len(metadata) != expected_support:
        raise ValueError(f"metadata support {len(metadata)} != expected {expected_support}")
    report, registry = audit(
        metadata,
        args.primary_index_root,
        args.comparison_photos_root,
        args.query_manifest,
        expected_support,
    )
    _write_jsonl(args.registry_output, registry)
    report["registry_file_sha256"] = file_sha256(args.registry_output)
    report["registry_canonical_sha256"] = canonical_json_sha256(registry)
    _write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def audit(
    metadata: list[dict[str, Any]],
    primary_index_root: Path,
    comparison_photos_root: Path,
    query_manifests: list[Path],
    expected_support: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    primary_root = primary_index_root.resolve()
    comparison_root = comparison_photos_root.resolve()
    registry: list[dict[str, Any]] = []
    primary_missing: list[str] = []
    comparison_missing: list[str] = []
    replica_mismatches: list[str] = []
    index_hash_to_ids: dict[str, list[str]] = defaultdict(list)
    image_ids: set[str] = set()
    source_paths: set[str] = set()

    for row in metadata:
        image_id = _required_text(row, "image_id")
        source_path = _required_text(row, "source_image_path")
        if image_id in image_ids:
            raise ValueError(f"duplicate metadata image_id: {image_id}")
        if source_path in source_paths:
            raise ValueError(f"duplicate metadata source_image_path: {source_path}")
        image_ids.add(image_id)
        source_paths.add(source_path)
        if Path(source_path).stem != image_id:
            raise ValueError(f"metadata source identity mismatch: {image_id} != {source_path}")
        primary_path = (primary_root / source_path).resolve()
        if primary_root != primary_path and primary_root not in primary_path.parents:
            raise ValueError(f"metadata path escapes primary root: {source_path}")
        comparison_path = (comparison_root / f"{image_id}.jpg").resolve()
        if comparison_root != comparison_path and comparison_root not in comparison_path.parents:
            raise ValueError(f"comparison path escapes photos root: {image_id}")
        primary_sha = file_sha256(primary_path) if primary_path.is_file() else None
        replica_sha = file_sha256(comparison_path) if comparison_path.is_file() else None
        if primary_sha is None:
            primary_missing.append(image_id)
        else:
            index_hash_to_ids[primary_sha].append(image_id)
        if replica_sha is None:
            comparison_missing.append(image_id)
        if primary_sha is not None and replica_sha is not None and primary_sha != replica_sha:
            replica_mismatches.append(image_id)
        registry.append({
            "image_id": image_id,
            "business_id": row.get("business_id"),
            "source_image_path": source_path,
            "primary_sha256": primary_sha,
            "comparison_replica_relative_path": f"{image_id}.jpg",
            "comparison_replica_sha256": replica_sha,
            "replicas_byte_identical": (
                primary_sha == replica_sha if primary_sha is not None and replica_sha is not None else None
            ),
            "contains_image_bytes": False,
        })

    query_rows: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    for manifest in query_manifests:
        rows = load_jsonl(manifest)
        manifest_records.append({
            "path_name": manifest.name,
            "support": len(rows),
            "file_sha256": file_sha256(manifest),
            "canonical_sha256": canonical_json_sha256(rows),
        })
        for row in rows:
            image = row.get("image") if isinstance(row.get("image"), dict) else {}
            image_sha = image.get("sha256") or row.get("image_sha256")
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            source_id = source.get("source_id") or row.get("source_id")
            record_id = row.get("query_id") or row.get("sample_id")
            if image_sha:
                query_rows.append({
                    "record_id": record_id,
                    "source_id": source_id,
                    "image_sha256": image_sha,
                    "manifest": manifest.name,
                })

    index_hashes = set(index_hash_to_ids)
    byte_collisions = [row for row in query_rows if row["image_sha256"] in index_hashes]
    identity_collisions = [
        row for row in query_rows
        if _identity_tokens(row.get("source_id")) & image_ids
    ]
    complete_primary = len(primary_missing) == 0 and len(registry) == expected_support
    complete_comparison = len(comparison_missing) == 0 and len(registry) == expected_support
    if replica_mismatches or byte_collisions or identity_collisions:
        status = "FAIL_COLLISION_OR_REPLICA_MISMATCH"
        collision_status = "FAIL"
    elif not complete_primary:
        status = "UNKNOWN_INCOMPLETE_FORMAL_INDEX_BYTE_COVERAGE"
        collision_status = "UNKNOWN"
    else:
        status = "PASS_COMPLETE_NO_QUERY_INDEX_COLLISION"
        collision_status = "PASS"
    duplicate_index_groups = [
        {"sha256": sha, "image_ids": ids}
        for sha, ids in sorted(index_hash_to_ids.items()) if len(ids) > 1
    ]
    return ({
        "schema_version": "retrieval_query_leakage_audit_v2",
        "status": status,
        "evidence_class": "byte_identity_and_source_identity_audit_not_semantic_independence",
        "formal_index": {
            "expected_support": expected_support,
            "metadata_support": len(metadata),
            "unique_image_id_support": len(image_ids),
            "unique_source_path_support": len(source_paths),
        },
        "primary_source_coverage": {
            "hashed_support": expected_support - len(primary_missing),
            "expected_support": expected_support,
            "coverage": (expected_support - len(primary_missing)) / expected_support,
            "missing_image_ids": primary_missing,
            "status": "PASS" if complete_primary else "UNKNOWN_INCOMPLETE",
        },
        "comparison_replica_coverage": {
            "hashed_support": expected_support - len(comparison_missing),
            "expected_support": expected_support,
            "coverage": (expected_support - len(comparison_missing)) / expected_support,
            "missing_image_ids": comparison_missing,
            "byte_mismatch_image_ids": replica_mismatches,
            "status": (
                "FAIL_REPLICA_MISMATCH" if replica_mismatches
                else "PASS" if complete_comparison else "UNKNOWN_INCOMPLETE"
            ),
            "required_for_formal_index_collision_denominator": False,
            "interpretation": "overlap_crosscheck_only; incomplete replica coverage is never reported as PASS",
        },
        "query_support": len(query_rows),
        "query_manifests": manifest_records,
        "collision_check": {
            "status": collision_status,
            "byte_collision_support": len(byte_collisions),
            "byte_collisions": byte_collisions,
            "source_identity_collision_support": len(identity_collisions),
            "source_identity_collisions": identity_collisions,
        },
        "index_internal_duplicate_bytes": {
            "group_support": len(duplicate_index_groups),
            "groups": duplicate_index_groups,
            "interpretation": "reported_only_not_query_leakage",
        },
        "image_bytes_copied_or_written": False,
        "promotion_eligible_as_human_ground_truth": False,
    }, registry)


def _read_metadata(path: Path, member: str) -> list[dict[str, Any]]:
    with tarfile.open(path, "r:gz") as archive:
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"retrieval archive misses {member}")
        payload = handle.read().decode("utf-8")
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def _identity_tokens(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    tokens = {value}
    for separator in (":", "/", "\\"):
        expanded: set[str] = set()
        for token in tokens:
            expanded.update(token.split(separator))
        tokens.update(expanded)
    return {token.removesuffix(".jpg") for token in tokens if token}


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"metadata {key} must be non-empty text")
    return value


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

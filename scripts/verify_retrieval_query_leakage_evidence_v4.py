"""Independently recompute and verify v4 query/index byte-leakage evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_retrieval_query_leakage_v4 import build_evidence, finalize_report
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--retrieval-archive", type=Path, required=True)
    parser.add_argument("--index-overlay-root", type=Path, required=True)
    parser.add_argument("--query-pool-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-job-id", required=True)
    parser.add_argument("--expected-source-snapshot-sha256", required=True)
    parser.add_argument("--expected-implementation-commit", required=True)
    args = parser.parse_args()
    result = verify_evidence(
        config_path=args.config,
        retrieval_archive=args.retrieval_archive,
        index_overlay_root=args.index_overlay_root,
        query_pool_dir=args.query_pool_dir,
        registry_path=args.registry,
        report_path=args.report,
        expected_job_id=args.expected_job_id,
        expected_source_snapshot_sha256=args.expected_source_snapshot_sha256,
        expected_implementation_commit=args.expected_implementation_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def verify_evidence(
    *,
    config_path: Path,
    retrieval_archive: Path,
    index_overlay_root: Path,
    query_pool_dir: Path,
    registry_path: Path,
    report_path: Path,
    expected_job_id: str,
    expected_source_snapshot_sha256: str,
    expected_implementation_commit: str,
) -> dict[str, Any]:
    saved_report = _load_object(report_path)
    if str(saved_report.get("hardware", {}).get("slurm_job_id")) != str(expected_job_id):
        raise ValueError("v4 leakage Slurm job mismatch")
    configuration = saved_report.get("configuration", {})
    if configuration.get("source_snapshot_sha256") != expected_source_snapshot_sha256:
        raise ValueError("v4 leakage source snapshot mismatch")
    if configuration.get("implementation_commit_sha") != expected_implementation_commit:
        raise ValueError("v4 leakage implementation commit mismatch")

    recomputed_report, recomputed_registry = build_evidence(
        config_path=config_path,
        retrieval_archive=retrieval_archive,
        index_overlay_root=index_overlay_root,
        query_pool_dir=query_pool_dir,
        source_snapshot_sha256=expected_source_snapshot_sha256,
        implementation_commit=expected_implementation_commit,
    )
    saved_registry = load_jsonl(registry_path)
    if canonical_json_sha256(saved_registry) != canonical_json_sha256(recomputed_registry):
        raise ValueError("v4 leakage registry differs from recomputed image bytes")
    recomputed_report["hardware"] = saved_report["hardware"]
    finalized = finalize_report(recomputed_report, registry_path)
    if canonical_json_sha256(saved_report) != canonical_json_sha256(finalized):
        raise ValueError("v4 leakage report differs from independent recomputation")
    return {
        "schema_version": "retrieval_query_leakage_evidence_verification_v4",
        "status": "PASS",
        "slurm_job_id": str(expected_job_id),
        "implementation_commit_sha": expected_implementation_commit,
        "source_snapshot_sha256": expected_source_snapshot_sha256,
        "formal_index_image_support": saved_report["formal_index"]["expected_support"],
        "query_image_support": saved_report["query_support"],
        "byte_collision_support": saved_report["collision_check"]["byte_collision_support"],
        "source_identity_collision_support": saved_report["collision_check"][
            "source_identity_collision_support"
        ],
        "fixed_acceptance_status": saved_report["fixed_acceptance"]["status"],
        "registry_file_sha256": file_sha256(registry_path),
        "report_file_sha256": file_sha256(report_path),
        "fresh_test_used": False,
        "query_labels_or_rankings_read": False,
    }


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


if __name__ == "__main__":
    main()

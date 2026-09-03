"""Verify the completed two-node Milvus HTTP evidence bundle without rerunning it."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_http_milvus_service_benchmark_v4 import (
    STAGES,
    _performance_gates,
    _summarize_role,
    _validate_external_cluster_identity,
)
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-job-id", required=True)
    parser.add_argument("--expected-source-snapshot-sha256", required=True)
    parser.add_argument("--expected-implementation-commit", required=True)
    parser.add_argument("--expected-baseline-adapter-sha256", required=True)
    parser.add_argument("--expected-candidate-adapter-sha256", required=True)
    args = parser.parse_args()
    report = verify_evidence_bundle(
        config_path=args.config,
        output_dir=args.output_dir,
        expected_job_id=args.expected_job_id,
        expected_source_snapshot_sha256=args.expected_source_snapshot_sha256,
        expected_implementation_commit=args.expected_implementation_commit,
        expected_baseline_adapter_sha256=args.expected_baseline_adapter_sha256,
        expected_candidate_adapter_sha256=args.expected_candidate_adapter_sha256,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def verify_evidence_bundle(
    *,
    config_path: Path,
    output_dir: Path,
    expected_job_id: str,
    expected_source_snapshot_sha256: str,
    expected_implementation_commit: str,
    expected_baseline_adapter_sha256: str,
    expected_candidate_adapter_sha256: str,
) -> dict[str, Any]:
    config = _load_object(config_path)
    identity_path = output_dir / "cluster_identity.json"
    summary_path = output_dir / "summary.json"
    raw_path = output_dir / "raw.jsonl"
    chain_path = output_dir / "chain_summary.json"
    identity = _load_object(identity_path)
    summary = _load_object(summary_path)
    raw_rows = load_jsonl(raw_path)
    chain = _load_object(chain_path)
    performance = config["performance"]

    _validate_identity(
        identity,
        performance,
        expected_job_id,
        expected_source_snapshot_sha256,
    )
    _validate_summary(
        summary,
        raw_rows,
        identity,
        config,
        config_path,
        identity_path,
        expected_job_id,
        expected_source_snapshot_sha256,
        expected_baseline_adapter_sha256,
        expected_candidate_adapter_sha256,
    )
    _validate_chain(
        chain,
        identity_path,
        summary_path,
        raw_path,
        expected_job_id,
        summary["fixed_gates"]["status"],
    )
    return {
        "schema_version": "distributed_milvus_http_evidence_verification_v6",
        "status": "PASS",
        "slurm_job_id": expected_job_id,
        "implementation_commit_sha": expected_implementation_commit,
        "source_snapshot_sha256": expected_source_snapshot_sha256,
        "node_support": 2,
        "steady_raw_row_support": len(raw_rows),
        "fixed_gate_status": summary["fixed_gates"]["status"],
        "artifact_files": {
            path.name: file_sha256(path)
            for path in (identity_path, summary_path, raw_path, chain_path)
        },
        "fresh_test_used": False,
        "development_or_final_consumed": False,
    }


def _validate_identity(
    identity: dict[str, Any],
    performance: dict[str, Any],
    expected_job_id: str,
    expected_source_snapshot_sha256: str,
) -> None:
    _validate_external_cluster_identity(identity, performance)
    if str(identity.get("slurm_job_id")) != str(expected_job_id):
        raise ValueError("cluster identity Slurm job mismatch")
    if identity.get("source_snapshot_sha256") != expected_source_snapshot_sha256:
        raise ValueError("cluster identity source snapshot mismatch")
    if identity.get("fresh_test_used") is not False:
        raise ValueError("cluster identity must keep Fresh Test unused")
    if identity.get("development_or_final_consumed") is not False:
        raise ValueError("cluster identity must not consume development/final data")
    if not _positive_number(identity.get("startup_cold_ms")):
        raise ValueError("cluster identity lacks a positive cold startup duration")
    dependencies = identity.get("dependencies", {})
    expected_dependencies = performance["dependencies"]
    if dependencies.get("etcd") != {
        "version": expected_dependencies["etcd"]["version"],
        "sha256": expected_dependencies["etcd"]["binary_sha256"],
    }:
        raise ValueError("cluster identity etcd dependency mismatch")
    if dependencies.get("minio") != {
        "version": expected_dependencies["minio"]["version"],
        "sha256": expected_dependencies["minio"]["sha256"],
    }:
        raise ValueError("cluster identity MinIO dependency mismatch")
    if dependencies.get("message_queue") != expected_dependencies["message_queue"]:
        raise ValueError("cluster identity message queue mismatch")
    quality_hash = str(performance["quality_gate_source"]).rsplit("_", 1)[-1]
    if identity.get("quality_selection_file_sha256") != quality_hash:
        raise ValueError("cluster identity quality-selection SHA mismatch")


def _validate_summary(
    summary: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    identity: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    identity_path: Path,
    expected_job_id: str,
    expected_source_snapshot_sha256: str,
    expected_baseline_adapter_sha256: str,
    expected_candidate_adapter_sha256: str,
) -> None:
    performance = config["performance"]
    if summary.get("schema_version") != "http_milvus_service_benchmark_v6":
        raise ValueError("unsupported distributed benchmark schema")
    scope = summary.get("scope", {})
    expected_scope = {
        "http": "real_loopback_HTTP_FastAPI_Uvicorn",
        "milvus": "external_multi_node_distributed_Milvus_cluster",
        "distributed_cluster": True,
        "production_sla_supported": False,
        "fresh_test_used": False,
        "development_or_final_consumed": False,
    }
    if scope != expected_scope:
        raise ValueError("distributed benchmark scope mismatch")
    expected_concurrency = [int(value) for value in performance["concurrency"]]
    expected_steady = sum(expected_concurrency) * int(performance["steady_repetitions"])
    expected_denominators = {
        "cold_requests_per_role": int(performance["cold_repetitions"]),
        "warmup_requests_per_role": int(performance["warmup_repetitions"]),
        "steady_batches_per_concurrency_per_role": int(performance["steady_repetitions"]),
        "steady_requests_per_role": expected_steady,
    }
    if summary.get("denominators") != expected_denominators:
        raise ValueError("distributed benchmark denominators mismatch")
    fixed = summary.get("fixed_input", {})
    if fixed.get("split") != "training" or fixed.get("query_id") != performance["fixed_input"]["query_id"]:
        raise ValueError("performance input is not the locked training query")
    if fixed.get("label_provenance") != "synthetic_training_query_for_performance_only":
        raise ValueError("performance input provenance is not synthetic training-only")
    for key in (
        "query_sha256", "source_record_sha256", "image_sha256",
        "manifest_file_sha256", "request_lock_sha256",
    ):
        _require_sha256(fixed.get(key), f"fixed input {key}")

    configuration = summary.get("configuration", {})
    if configuration.get("config_sha256") != file_sha256(config_path):
        raise ValueError("benchmark configuration SHA mismatch")
    if configuration.get("source_snapshot_sha256") != expected_source_snapshot_sha256:
        raise ValueError("benchmark source snapshot mismatch")
    if configuration.get("base_model") != config["vlm"]["base_model"]:
        raise ValueError("benchmark base model mismatch")
    if configuration.get("base_revision") != config["vlm"]["base_revision"]:
        raise ValueError("benchmark base revision mismatch")
    if configuration.get("clip_model") != config["search"]["embedding_model"]:
        raise ValueError("benchmark CLIP model mismatch")
    if configuration.get("milvus_server") != identity["milvus_server"]:
        raise ValueError("benchmark Milvus server identity mismatch")
    if configuration.get("external_cluster_identity_file_sha256") != file_sha256(identity_path):
        raise ValueError("benchmark external cluster identity SHA mismatch")
    if configuration.get("milvus_rpm_sha256") != performance["milvus_server"]["package_sha256"]:
        raise ValueError("benchmark Milvus package SHA mismatch")
    if configuration.get("retrieval_archive_sha256") != config["formal_release_read_only"]["retrieval_archive_sha256"]:
        raise ValueError("benchmark retrieval archive SHA mismatch")
    if configuration.get("concurrency") != expected_concurrency:
        raise ValueError("benchmark concurrency mismatch")
    if configuration.get("vlm_max_new_tokens") != int(performance["vlm_max_new_tokens"]):
        raise ValueError("benchmark VLM generation limit mismatch")
    collection = configuration.get("milvus_collection", {})
    if collection.get("visible_entities") != int(config["formal_release_read_only"]["expected_index_support"]):
        raise ValueError("benchmark Milvus entity support mismatch")
    if not collection.get("index_names") or not _is_sha256(collection.get("index_details_sha256")):
        raise ValueError("benchmark Milvus index identity is incomplete")
    for key in ("milvus_service_startup_cold_ms", "milvus_collection_build_and_load_ms"):
        if not _positive_number(configuration.get(key)):
            raise ValueError(f"benchmark configuration lacks positive {key}")

    roles = summary.get("roles", {})
    baseline_role = performance["baseline_role"]
    candidate_role = performance["candidate_role"]
    if set(roles) != {baseline_role, candidate_role}:
        raise ValueError("benchmark roles differ from the fixed comparison")
    expected_adapter_hashes = {
        baseline_role: expected_baseline_adapter_sha256,
        candidate_role: expected_candidate_adapter_sha256,
    }
    if len(raw_rows) != 2 * expected_steady:
        raise ValueError("raw steady result row count mismatch")
    observed_keys: set[tuple[Any, ...]] = set()
    for row in raw_rows:
        role = row.get("role")
        concurrency = row.get("concurrency")
        key = (role, concurrency, row.get("batch"), row.get("request_index"))
        if key in observed_keys:
            raise ValueError("duplicate raw benchmark request identity")
        observed_keys.add(key)
        if role not in roles or row.get("phase") != "steady" or concurrency not in expected_concurrency:
            raise ValueError("raw benchmark row falls outside the fixed scope")
        if not isinstance(row.get("success"), bool) or not _positive_number(row.get("group_wall_seconds")):
            raise ValueError("raw benchmark row has invalid outcome or duration")
        if row["success"]:
            for stage in STAGES:
                if not _nonnegative_number(row.get(stage)):
                    raise ValueError(f"successful raw row lacks stage timing: {stage}")
            peak = row.get("gpu_memory", {}).get("peak_allocated_mib")
            if not _nonnegative_number(peak):
                raise ValueError("successful raw row lacks peak VRAM")
    for role, expected_adapter_hash in expected_adapter_hashes.items():
        role_summary = roles[role]
        if role_summary.get("adapter_model_sha256") != expected_adapter_hash:
            raise ValueError(f"{role} adapter SHA mismatch")
        if not _positive_number(role_summary.get("service_startup_cold_ms")):
            raise ValueError(f"{role} service cold-start duration is missing")
        cold = role_summary.get("first_request_cold", {})
        if cold.get("phase") != "cold" or cold.get("success") is not True:
            raise ValueError(f"{role} first cold request did not succeed")
        role_rows = [row for row in raw_rows if row["role"] == role]
        recomputed = _summarize_role(role_rows, expected_concurrency)
        if role_summary.get("steady") != recomputed:
            raise ValueError(f"{role} steady summary does not match raw rows")
        for concurrency in expected_concurrency:
            group = role_summary["steady"][str(concurrency)]
            expected_support = concurrency * int(performance["steady_repetitions"])
            if group.get("request_support") != expected_support:
                raise ValueError(f"{role} concurrency {concurrency} support mismatch")
            if not _positive_number(group.get("throughput_requests_per_second")):
                raise ValueError(f"{role} concurrency {concurrency} throughput is missing")
            if not _nonnegative_number(group.get("peak_vram_mib")):
                raise ValueError(f"{role} concurrency {concurrency} peak VRAM is missing")
            for stage in STAGES:
                stats = group.get("stage_latency_ms", {}).get(stage, {})
                if not _nonnegative_number(stats.get("p50")) or not _nonnegative_number(stats.get("p95")):
                    raise ValueError(f"{role} concurrency {concurrency} lacks {stage} P50/P95")

    recomputed_gates = _performance_gates(roles, performance)
    if summary.get("fixed_gates") != recomputed_gates:
        raise ValueError("fixed performance gates do not match raw-derived summaries")
    expected_status = "COMPLETED" if recomputed_gates["status"] == "PASS" else "NEGATIVE_EXPERIMENT_GATE_FAILED"
    if summary.get("status") != expected_status:
        raise ValueError("benchmark status does not match its fixed gate")
    hardware = summary.get("hardware", {})
    if str(hardware.get("slurm_job_id")) != str(expected_job_id):
        raise ValueError("benchmark hardware Slurm job mismatch")
    for key in ("node", "platform", "python", "cpu_count", "torch", "cuda", "gpu", "gpu_count_visible"):
        if hardware.get(key) in (None, "", 0):
            raise ValueError(f"benchmark hardware field is missing: {key}")
    artifact_sha = summary.get("artifact_sha256")
    unhashed = dict(summary)
    unhashed.pop("artifact_sha256", None)
    if artifact_sha != canonical_json_sha256(unhashed):
        raise ValueError("benchmark artifact canonical SHA mismatch")
    if summary.get("raw_result_sha256") != canonical_json_sha256(raw_rows):
        raise ValueError("benchmark raw canonical SHA mismatch")


def _validate_chain(
    chain: dict[str, Any],
    identity_path: Path,
    summary_path: Path,
    raw_path: Path,
    expected_job_id: str,
    fixed_gate_status: str,
) -> None:
    if chain.get("schema_version") != "distributed_milvus_http_chain_v6":
        raise ValueError("unsupported distributed evidence chain schema")
    if chain.get("status") != "COMPLETED" or chain.get("node_support") != 2:
        raise ValueError("distributed evidence chain is incomplete")
    if str(chain.get("slurm_job_id")) != str(expected_job_id):
        raise ValueError("distributed evidence chain Slurm job mismatch")
    expected_files = {
        "cluster_identity_file_sha256": file_sha256(identity_path),
        "summary_file_sha256": file_sha256(summary_path),
        "raw_file_sha256": file_sha256(raw_path),
    }
    for key, expected in expected_files.items():
        if chain.get(key) != expected:
            raise ValueError(f"distributed evidence chain mismatch: {key}")
    if chain.get("fixed_gate_status") != fixed_gate_status:
        raise ValueError("distributed evidence chain gate mismatch")
    if chain.get("fresh_test_used") is not False:
        raise ValueError("distributed evidence chain must keep Fresh Test unused")
    if chain.get("development_or_final_consumed") is not False:
        raise ValueError("distributed evidence chain must not consume development/final data")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _require_sha256(value: Any, label: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{label} is not a lowercase SHA-256")


def _nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _positive_number(value: Any) -> bool:
    return _nonnegative_number(value) and value > 0


if __name__ == "__main__":
    main()

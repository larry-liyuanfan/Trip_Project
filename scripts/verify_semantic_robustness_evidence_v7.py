"""Verify a completed semantic-robustness bundle without model inference."""

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
)
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl
from src.evaluation.semantic_robustness_v7 import (
    apply_semantic_robustness_v7_gates,
    score_semantic_robustness_v7,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-job-id", required=True)
    parser.add_argument("--expected-source-snapshot-sha256", required=True)
    parser.add_argument("--expected-implementation-commit", required=True)
    parser.add_argument("--expected-baseline-adapter-sha256", required=True)
    args = parser.parse_args()
    report = verify_evidence_bundle(
        config_path=args.config,
        output_dir=args.output_dir,
        expected_job_id=args.expected_job_id,
        expected_source_snapshot_sha256=args.expected_source_snapshot_sha256,
        expected_implementation_commit=args.expected_implementation_commit,
        expected_baseline_adapter_sha256=args.expected_baseline_adapter_sha256,
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
) -> dict[str, Any]:
    config = _load_object(config_path)
    cycle_id = str(config.get("cycle_id", "v7"))
    if cycle_id not in {"v7", "v9"}:
        raise ValueError(f"unsupported semantic-robustness cycle: {cycle_id}")
    lock_path = Path(config["pool"]["committed_lock"])
    if not lock_path.is_absolute():
        lock_path = config_path.resolve().parents[2] / lock_path
    lock = _load_object(lock_path)
    training_dir = output_dir / "training"
    development_dir = output_dir / "vlm-development"
    service_dir = output_dir / "service"
    chain_path = output_dir / "chain_summary.json"
    identity_path = training_dir / "run_identity.json"
    training_summary_path = training_dir / "run_summary.json"
    adapter_path = training_dir / "adapter" / "adapter_model.safetensors"
    adapter_config_path = training_dir / "adapter" / "adapter_config.json"
    selection_path = development_dir / "selection.json"

    identity = _load_object(identity_path)
    training_summary = _load_object(training_summary_path)
    selection = _load_object(selection_path)
    chain = _load_object(chain_path)
    candidate_sha = file_sha256(adapter_path)
    _validate_sha(expected_source_snapshot_sha256, "expected source snapshot")
    _validate_commit(expected_implementation_commit)
    _validate_training(
        identity,
        training_summary,
        config,
        lock,
        config_path,
        adapter_config_path,
        candidate_sha,
        expected_job_id,
        expected_implementation_commit,
        expected_baseline_adapter_sha256,
        cycle_id,
    )

    roles = config["vlm"]["development_roles"]
    raw_paths = [development_dir / f"{role}.jsonl" for role in roles]
    role_summary_paths = [path.with_suffix(".summary.json") for path in raw_paths]
    rows_by_role = {role: load_jsonl(path) for role, path in zip(roles, raw_paths)}
    _validate_development_rows(
        rows_by_role,
        role_summary_paths,
        config,
        lock,
        candidate_sha,
        expected_baseline_adapter_sha256,
        expected_job_id,
        cycle_id,
    )
    rows = [row for role in roles for row in rows_by_role[role]]
    recomputed = score_semantic_robustness_v7(
        rows,
        cycle_id=cycle_id,
        primary_factor=str(config.get("primary_factor", "robustness_training_data_only")),
    )
    recomputed_gate = apply_semantic_robustness_v7_gates(
        recomputed,
        config["vlm"]["exploration_gates"],
        config["vlm"]["selection_objective"],
        candidate=config["vlm"]["candidate_variant"],
        baseline=config["vlm"]["baseline_variant"],
    )
    _validate_selection(selection, recomputed, recomputed_gate, raw_paths, rows, config, cycle_id)

    performance_status = "NOT_RUN_DEVELOPMENT_GATE_FAILED"
    service_files: list[Path] = []
    if recomputed_gate["status"] == "PASS":
        service_summary_path = service_dir / "summary.json"
        service_raw_path = service_dir / "raw.jsonl"
        service_summary = _load_object(service_summary_path)
        service_rows = load_jsonl(service_raw_path)
        _validate_service(
            service_summary,
            service_rows,
            config,
            config_path,
            expected_job_id,
            expected_source_snapshot_sha256,
            expected_baseline_adapter_sha256,
            candidate_sha,
            cycle_id,
        )
        performance_status = service_summary["fixed_gates"]["status"]
        service_files = [service_summary_path, service_raw_path]
    elif any(service_dir.iterdir()):
        raise ValueError("service evidence exists despite a failed development gate")

    _validate_chain(
        chain,
        output_dir,
        expected_job_id,
        recomputed_gate["status"],
        performance_status,
        candidate_sha,
        cycle_id,
    )
    artifact_paths = [
        identity_path,
        training_summary_path,
        adapter_path,
        adapter_config_path,
        *raw_paths,
        *role_summary_paths,
        selection_path,
        *service_files,
        chain_path,
    ]
    return {
        "schema_version": f"semantic_robustness_evidence_verification_{cycle_id}",
        "status": "PASS",
        "slurm_job_id": expected_job_id,
        "implementation_commit_sha": expected_implementation_commit,
        "source_snapshot_sha256": expected_source_snapshot_sha256,
        "candidate_adapter_model_sha256": candidate_sha,
        "development_raw_row_support": len(rows),
        "development_gate_status": recomputed_gate["status"],
        "performance_gate_status": performance_status,
        "artifact_files": {
            path.relative_to(output_dir).as_posix(): file_sha256(path)
            for path in artifact_paths
        },
        "fresh_test_used": False,
        "final_defined_or_consumed": False,
    }


def _validate_training(
    identity: dict[str, Any],
    summary: dict[str, Any],
    config: dict[str, Any],
    lock: dict[str, Any],
    config_path: Path,
    adapter_config_path: Path,
    candidate_sha: str,
    expected_job_id: str,
    expected_commit: str,
    expected_baseline_sha: str,
    cycle_id: str = "v7",
) -> None:
    expected_common = {
        "run_id": config["training"]["run_id"],
        "git_commit": expected_commit,
        "config_sha256": file_sha256(config_path),
        "pool_lock_sha256": canonical_json_sha256(lock),
        "training_manifest_sha256": lock["vlm"]["training"]["manifest_file_sha256"],
        "initial_adapter_model_sha256": expected_baseline_sha,
        "development_or_final_opened": False,
    }
    if identity.get("schema_version") != f"targeted_exploration_training_identity_{cycle_id}":
        raise ValueError(f"{cycle_id} training identity schema mismatch")
    for key, expected in expected_common.items():
        if identity.get(key) != expected:
            raise ValueError(f"{cycle_id} training identity mismatch: {key}")
    if summary.get("schema_version") != f"targeted_exploration_training_summary_{cycle_id}":
        raise ValueError(f"{cycle_id} training summary schema mismatch")
    for key, expected in expected_common.items():
        if summary.get(key) != expected:
            raise ValueError(f"{cycle_id} training summary mismatch: {key}")
    training_lock = lock["vlm"]["training"]
    expected_support = {
        "training_support": training_lock["sample_support"],
        "product_support": training_lock["product_support"],
        "dialogue_support": training_lock["dialogue_support"],
    }
    for key, expected in expected_support.items():
        if summary.get(key) != expected:
            raise ValueError(f"{cycle_id} training support mismatch: {key}")
    if summary.get("status") != "COMPLETED" or summary.get("adapter_only") is not True:
        raise ValueError(f"{cycle_id} training did not complete as adapter-only SFT")
    if str(summary.get("slurm_job_id")) != str(expected_job_id):
        raise ValueError(f"{cycle_id} training Slurm job mismatch")
    if summary.get("adapter_model_sha256") != candidate_sha:
        raise ValueError(f"{cycle_id} candidate adapter SHA mismatch")
    if summary.get("adapter_config_sha256") != file_sha256(adapter_config_path):
        raise ValueError(f"{cycle_id} adapter config SHA mismatch")
    for key in ("global_step", "duration_seconds", "peak_gpu_memory_allocated_bytes"):
        if not _positive_number(summary.get(key)):
            raise ValueError(f"{cycle_id} training lacks positive {key}")


def _validate_development_rows(
    rows_by_role: dict[str, list[dict[str, Any]]],
    summary_paths: list[Path],
    config: dict[str, Any],
    lock: dict[str, Any],
    candidate_sha: str,
    baseline_sha: str,
    expected_job_id: str,
    cycle_id: str = "v7",
) -> None:
    development = lock["vlm"]["development"]
    expected_adapter = {
        config["vlm"]["baseline_variant"]: baseline_sha,
        config["vlm"]["candidate_variant"]: candidate_sha,
    }
    expected_samples: set[str] | None = None
    for (role, rows), summary_path in zip(rows_by_role.items(), summary_paths):
        if len(rows) != development["sample_support"]:
            raise ValueError(f"{role} development support mismatch")
        identities = [(row.get("variant"), row.get("sample_id")) for row in rows]
        if len(set(identities)) != len(rows) or any(item[0] != role for item in identities):
            raise ValueError(f"{role} development row identity mismatch")
        samples = {str(row["sample_id"]) for row in rows}
        if expected_samples is None:
            expected_samples = samples
        elif samples != expected_samples:
            raise ValueError(f"{cycle_id} roles do not use the exact same development samples")
        if sum(row.get("scenario") == "product" for row in rows) != development["product_support"]:
            raise ValueError(f"{role} product support mismatch")
        if sum(row.get("scenario") == "dialogue" for row in rows) != development["dialogue_support"]:
            raise ValueError(f"{role} dialogue support mismatch")
        data_locks = {row.get("data_lock_sha256") for row in rows}
        if data_locks != {development["manifest_canonical_sha256"]}:
            raise ValueError(f"{role} development data lock mismatch")
        for row in rows:
            if row.get("base_model") != config["vlm"]["base_model"]:
                raise ValueError(f"{role} base model mismatch")
            if row.get("base_revision") != config["vlm"]["base_revision"]:
                raise ValueError(f"{role} base revision mismatch")
            if row.get("adapter_model_sha256") != expected_adapter[role]:
                raise ValueError(f"{role} adapter SHA mismatch")
            provenance = row.get("label_provenance")
            if not (
                isinstance(provenance, str)
                and provenance.startswith("synthetic_")
                and provenance.endswith("_no_human_annotation")
            ):
                raise ValueError(f"{role} label provenance is not synthetic")
            if not isinstance(row.get("first_attempt_json_valid"), bool):
                raise ValueError(f"{role} lacks first-attempt JSON outcome")
            if not isinstance(row.get("correction_triggered"), bool):
                raise ValueError(f"{role} lacks correction outcome")
            if not _positive_number(row.get("latency_ms")) or not _nonnegative_number(row.get("peak_vram_mib")):
                raise ValueError(f"{role} runtime evidence is invalid")
        _validate_role_summary(
            _load_object(summary_path), rows, role, development, expected_job_id, cycle_id
        )


def _validate_role_summary(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    role: str,
    development: dict[str, Any],
    expected_job_id: str,
    cycle_id: str = "v7",
) -> None:
    expected = {
        "schema_version": f"vlm_semantic_role_evidence_{cycle_id}",
        "status": "COMPLETED",
        "split": "development",
        "role": role,
        "support": len(rows),
        "product_support": development["product_support"],
        "dialogue_support": development["dialogue_support"],
        "data_lock_sha256": development["manifest_canonical_sha256"],
        "result_sha256": canonical_json_sha256(rows),
        "first_attempt_json_compliance": sum(row["first_attempt_json_valid"] for row in rows) / len(rows),
        "correction_trigger_rate": sum(row["correction_triggered"] for row in rows) / len(rows),
        "mean_latency_ms": sum(float(row["latency_ms"]) for row in rows) / len(rows),
        "peak_vram_mib": max(float(row["peak_vram_mib"]) for row in rows),
        "final_consumed_once": False,
        "fresh_test_used": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"{role} summary mismatch: {key}")
    if str(summary.get("slurm_job_id")) != str(expected_job_id):
        raise ValueError(f"{role} summary Slurm job mismatch")
    for key in ("prompt_sha256", "generation_config_sha256"):
        values = {row.get(key) for row in rows}
        if len(values) != 1 or summary.get(key) not in values:
            raise ValueError(f"{role} summary mismatch: {key}")


def _validate_selection(
    selection: dict[str, Any],
    recomputed: dict[str, Any],
    recomputed_gate: dict[str, Any],
    raw_paths: list[Path],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    cycle_id: str = "v7",
) -> None:
    for key, value in recomputed.items():
        if canonical_json_sha256(selection.get(key)) != canonical_json_sha256(value):
            raise ValueError(f"{cycle_id} selection metrics differ from raw rows: {key}")
    expected = {
        "status": "COMPLETED",
        "split": "development",
        "candidate_variant": config["vlm"]["candidate_variant"],
        "fixed_gates": recomputed_gate,
        "gate_class": config["gate_class"],
        "human_annotation_support": 0,
        "final_defined_or_consumed": False,
        "fresh_test_used": False,
        "raw_result_support": len(rows),
        "raw_result_canonical_sha256": canonical_json_sha256(rows),
        "raw_result_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in raw_paths
        ],
        "promotion_eligible_as_human_ground_truth": False,
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"{cycle_id} selection mismatch: {key}")


def _validate_service(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    config_path: Path,
    expected_job_id: str,
    expected_source_sha: str,
    baseline_sha: str,
    candidate_sha: str,
    cycle_id: str = "v7",
) -> None:
    performance = config["performance"]
    if summary.get("schema_version") != f"http_milvus_service_benchmark_{cycle_id}":
        raise ValueError(f"{cycle_id} service schema mismatch")
    if summary.get("scope") != {
        "http": "real_loopback_HTTP_FastAPI_Uvicorn",
        "milvus": "external_server_process_single_node_standalone_not_Milvus_Lite",
        "distributed_cluster": "NOT_RUN_MULTI_NODE_CLUSTER",
        "production_sla_supported": False,
        "fresh_test_used": False,
        "development_or_final_consumed": False,
    }:
        raise ValueError(f"{cycle_id} service scope mismatch")
    concurrency = [int(value) for value in performance["concurrency"]]
    expected_steady = sum(concurrency) * int(performance["steady_repetitions"])
    if len(rows) != 2 * expected_steady:
        raise ValueError(f"{cycle_id} service raw support mismatch")
    expected_denominators = {
        "cold_requests_per_role": int(performance["cold_repetitions"]),
        "warmup_requests_per_role": int(performance["warmup_repetitions"]),
        "steady_batches_per_concurrency_per_role": int(performance["steady_repetitions"]),
        "steady_requests_per_role": expected_steady,
    }
    if summary.get("denominators") != expected_denominators:
        raise ValueError(f"{cycle_id} service denominators mismatch")
    fixed = summary.get("fixed_input", {})
    if fixed.get("split") != "training" or fixed.get("query_id") != performance["fixed_input"]["query_id"]:
        raise ValueError(f"{cycle_id} service input is not the fixed training query")
    if fixed.get("label_provenance") != "synthetic_training_query_for_performance_only":
        raise ValueError(f"{cycle_id} service input provenance mismatch")
    configuration = summary.get("configuration", {})
    expected_configuration = {
        "config_sha256": file_sha256(config_path),
        "source_snapshot_sha256": expected_source_sha,
        "base_model": config["vlm"]["base_model"],
        "base_revision": config["vlm"]["base_revision"],
        "clip_model": config["search"]["embedding_model"],
        "milvus_rpm_sha256": performance["milvus_server"]["package_sha256"],
        "retrieval_archive_sha256": config["formal_release_read_only"]["retrieval_archive_sha256"],
        "concurrency": performance["concurrency"],
        "vlm_max_new_tokens": performance["vlm_max_new_tokens"],
    }
    for key, value in expected_configuration.items():
        if configuration.get(key) != value:
            raise ValueError(f"{cycle_id} service configuration mismatch: {key}")
    for key in ("milvus_service_startup_cold_ms", "milvus_collection_build_and_load_ms"):
        if not _positive_number(configuration.get(key)):
            raise ValueError(f"{cycle_id} service lacks positive {key}")
    collection = configuration.get("milvus_collection", {})
    if collection.get("visible_entities") != config["formal_release_read_only"]["expected_index_support"]:
        raise ValueError(f"{cycle_id} service Milvus entity support mismatch")
    if not collection.get("index_names"):
        raise ValueError(f"{cycle_id} service Milvus index identity is missing")
    roles = summary.get("roles", {})
    expected_hashes = {
        performance["baseline_role"]: baseline_sha,
        performance["candidate_role"]: candidate_sha,
    }
    if set(roles) != set(expected_hashes):
        raise ValueError(f"{cycle_id} service roles mismatch")
    observed: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (row.get("role"), row.get("concurrency"), row.get("batch"), row.get("request_index"))
        if key in observed:
            raise ValueError(f"duplicate {cycle_id} service request identity")
        observed.add(key)
        if row.get("role") not in roles or row.get("phase") != "steady" or row.get("concurrency") not in concurrency:
            raise ValueError(f"{cycle_id} service row outside fixed scope")
        if not isinstance(row.get("success"), bool) or not _positive_number(row.get("group_wall_seconds")):
            raise ValueError(f"invalid {cycle_id} service outcome")
        if row["success"]:
            for stage in STAGES:
                if not _nonnegative_number(row.get(stage)):
                    raise ValueError(f"{cycle_id} service row lacks {stage}")
    for role, adapter_sha in expected_hashes.items():
        role_rows = [row for row in rows if row["role"] == role]
        expected_identities = {
            (role, concurrency_level, batch, request_index)
            for concurrency_level in concurrency
            for batch in range(int(performance["steady_repetitions"]))
            for request_index in range(concurrency_level)
        }
        actual_identities = {
            (row["role"], row["concurrency"], row["batch"], row["request_index"])
            for row in role_rows
        }
        if actual_identities != expected_identities:
            raise ValueError(f"{cycle_id} service {role} request denominators mismatch")
        if roles[role].get("adapter_model_sha256") != adapter_sha:
            raise ValueError(f"{cycle_id} service {role} adapter SHA mismatch")
        if not _positive_number(roles[role].get("service_startup_cold_ms")):
            raise ValueError(f"{cycle_id} service {role} cold startup is missing")
        cold = roles[role].get("first_request_cold", {})
        if cold.get("phase") != "cold" or cold.get("success") is not True:
            raise ValueError(f"{cycle_id} service {role} first cold request failed")
        if roles[role].get("steady") != _summarize_role(role_rows, concurrency):
            raise ValueError(f"{cycle_id} service {role} summary differs from raw rows")
    gates = _performance_gates(roles, performance)
    if summary.get("fixed_gates") != gates:
        raise ValueError(f"{cycle_id} service gates differ from raw rows")
    expected_status = "COMPLETED" if gates["status"] == "PASS" else "NEGATIVE_EXPERIMENT_GATE_FAILED"
    if summary.get("status") != expected_status:
        raise ValueError(f"{cycle_id} service status differs from fixed gate")
    if summary.get("raw_result_sha256") != canonical_json_sha256(rows):
        raise ValueError(f"{cycle_id} service raw canonical SHA mismatch")
    unhashed = dict(summary)
    artifact_sha = unhashed.pop("artifact_sha256", None)
    if artifact_sha != canonical_json_sha256(unhashed):
        raise ValueError(f"{cycle_id} service artifact canonical SHA mismatch")
    hardware = summary.get("hardware", {})
    if str(hardware.get("slurm_job_id")) != str(expected_job_id):
        raise ValueError(f"{cycle_id} service Slurm job mismatch")
    for key in ("node", "platform", "python", "cpu_count", "torch", "cuda", "gpu", "gpu_count_visible"):
        if hardware.get(key) in (None, "", 0):
            raise ValueError(f"{cycle_id} service hardware field is missing: {key}")


def _validate_chain(
    chain: dict[str, Any],
    output_dir: Path,
    expected_job_id: str,
    development_status: str,
    performance_status: str,
    candidate_sha: str,
    cycle_id: str = "v7",
) -> None:
    expected = {
        "schema_version": f"semantic_robustness_chain_{cycle_id}",
        "status": "COMPLETED",
        "development_gate_status": development_status,
        "performance_gate_status": performance_status,
        "candidate_adapter_model_sha256": candidate_sha,
        "final_defined_or_consumed": False,
        "fresh_test_used": False,
    }
    for key, value in expected.items():
        if chain.get(key) != value:
            raise ValueError(f"{cycle_id} chain mismatch: {key}")
    if str(chain.get("slurm_job_id")) != str(expected_job_id):
        raise ValueError(f"{cycle_id} chain Slurm job mismatch")
    artifacts = chain.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"{cycle_id} chain artifact list is missing")
    recorded: dict[str, tuple[str, int]] = {}
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"{cycle_id} chain artifact entry is invalid")
        path = item["path"]
        if path in recorded:
            raise ValueError(f"duplicate {cycle_id} chain artifact path")
        recorded[path] = (item.get("sha256"), item.get("size_bytes"))
    actual = {
        path.relative_to(output_dir).as_posix(): (file_sha256(path), path.stat().st_size)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "chain_summary.json"
    }
    if recorded != actual:
        raise ValueError(f"{cycle_id} chain artifacts differ from files on disk")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _validate_sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} is not a lowercase SHA-256")


def _validate_commit(value: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("implementation commit is not a full lowercase Git SHA")


def _nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _positive_number(value: Any) -> bool:
    return _nonnegative_number(value) and value > 0


if __name__ == "__main__":
    main()

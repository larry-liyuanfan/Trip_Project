"""Tests for the independent distributed Milvus evidence verifier."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_http_milvus_service_benchmark_v4 import _performance_gates, _summarize_role
from scripts.verify_distributed_milvus_evidence_v6 import verify_evidence_bundle
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "evaluation" / "automated_evidence_v6_distributed.json"
SOURCE_SHA = "b" * 64
BASELINE_SHA = "c" * 64
CANDIDATE_SHA = "d" * 64
JOB_ID = "30004826"


class VerifyDistributedMilvusEvidenceV6Tests(unittest.TestCase):
    def test_valid_bundle_passes_and_tampered_raw_row_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _write_valid_bundle(output_dir)
            report = _verify(output_dir)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["steady_raw_row_support"], 112)

            raw_path = output_dir / "raw.jsonl"
            rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["milvus_query_ms"] = 99.0
            _write_jsonl(raw_path, rows)
            with self.assertRaisesRegex(ValueError, "steady summary does not match raw rows"):
                _verify(output_dir)

    def test_cross_node_role_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _write_valid_bundle(output_dir)
            identity_path = output_dir / "cluster_identity.json"
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            identity["roles"]["datanode"] = "node-a"
            _write_json(identity_path, identity)
            with self.assertRaisesRegex(ValueError, "worker roles"):
                _verify(output_dir)


def _verify(output_dir: Path) -> dict[str, object]:
    return verify_evidence_bundle(
        config_path=CONFIG,
        output_dir=output_dir,
        expected_job_id=JOB_ID,
        expected_source_snapshot_sha256=SOURCE_SHA,
        expected_implementation_commit="e" * 40,
        expected_baseline_adapter_sha256=BASELINE_SHA,
        expected_candidate_adapter_sha256=CANDIDATE_SHA,
    )


def _write_valid_bundle(output_dir: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    performance = config["performance"]
    identity = {
        "schema_version": "distributed_milvus_cluster_identity_v6",
        "status": "READY",
        "slurm_job_id": JOB_ID,
        "nodes": ["node-a", "node-b"],
        "roles": {
            "mixcoord": "node-a",
            "proxy": "node-a",
            "querynode": "node-b",
            "datanode": "node-b",
            "streamingnode": "node-b",
        },
        "runtime_port_base": 28000,
        "startup_cold_ms": 1000.0,
        "milvus_server": performance["milvus_server"],
        "dependencies": {
            "etcd": {
                "version": performance["dependencies"]["etcd"]["version"],
                "sha256": performance["dependencies"]["etcd"]["binary_sha256"],
            },
            "minio": {
                "version": performance["dependencies"]["minio"]["version"],
                "sha256": performance["dependencies"]["minio"]["sha256"],
            },
            "message_queue": performance["dependencies"]["message_queue"],
        },
        "quality_selection_file_sha256": performance["quality_gate_source"].rsplit("_", 1)[-1],
        "source_snapshot_sha256": SOURCE_SHA,
        "credentials": "random_job_local_not_persisted",
        "fresh_test_used": False,
        "development_or_final_consumed": False,
    }
    identity_path = output_dir / "cluster_identity.json"
    _write_json(identity_path, identity)

    roles = {}
    all_rows = []
    adapter_hashes = {
        performance["baseline_role"]: BASELINE_SHA,
        performance["candidate_role"]: CANDIDATE_SHA,
    }
    for role_index, (role, adapter_sha) in enumerate(adapter_hashes.items()):
        role_rows = []
        for concurrency in performance["concurrency"]:
            for batch in range(performance["steady_repetitions"]):
                for request_index in range(concurrency):
                    latency = 100.0 + 10.0 * role_index
                    row = {
                        "role": role,
                        "phase": "steady",
                        "concurrency": concurrency,
                        "batch": batch,
                        "request_index": request_index,
                        "success": True,
                        "http_status": 200,
                        "http_e2e_ms": latency,
                        "queue_wait_ms": 1.0,
                        "clip_encode_ms": 2.0,
                        "milvus_query_ms": 3.0,
                        "rerank_ms": 0.1,
                        "vlm_inference_ms": latency - 10.0,
                        "service_total_ms": latency - 1.0,
                        "transport_overhead_ms": 1.0,
                        "output_tokens": 16,
                        "gpu_memory": {
                            "allocated_mib": 100.0,
                            "reserved_mib": 200.0,
                            "peak_allocated_mib": 150.0,
                        },
                        "error": None,
                        "group_wall_seconds": 8.0,
                    }
                    role_rows.append(row)
        all_rows.extend(role_rows)
        roles[role] = {
            "adapter_model_sha256": adapter_sha,
            "service_startup_cold_ms": 500.0,
            "first_request_cold": {
                "role": role,
                "phase": "cold",
                "success": True,
                "http_e2e_ms": 150.0,
            },
            "health": {"status": "ready"},
            "steady": _summarize_role(role_rows, performance["concurrency"]),
        }
    raw_path = output_dir / "raw.jsonl"
    _write_jsonl(raw_path, all_rows)
    fixed_gates = _performance_gates(roles, performance)
    summary = {
        "schema_version": "http_milvus_service_benchmark_v6",
        "status": "COMPLETED",
        "evidence_class": config["evidence_class"],
        "gate_class": config["gate_class"],
        "scope": {
            "http": "real_loopback_HTTP_FastAPI_Uvicorn",
            "milvus": "external_multi_node_distributed_Milvus_cluster",
            "distributed_cluster": True,
            "production_sla_supported": False,
            "fresh_test_used": False,
            "development_or_final_consumed": False,
        },
        "denominators": {
            "cold_requests_per_role": 1,
            "warmup_requests_per_role": 2,
            "steady_batches_per_concurrency_per_role": 8,
            "steady_requests_per_role": 56,
        },
        "fixed_input": {
            "split": "training",
            "query_id": performance["fixed_input"]["query_id"],
            "query_sha256": "1" * 64,
            "source_id": "source-1",
            "source_record_sha256": "2" * 64,
            "image_sha256": "3" * 64,
            "manifest_file_sha256": "4" * 64,
            "request_lock_sha256": "5" * 64,
            "label_provenance": "synthetic_training_query_for_performance_only",
        },
        "configuration": {
            "config_sha256": file_sha256(CONFIG),
            "source_snapshot_sha256": SOURCE_SHA,
            "base_model": config["vlm"]["base_model"],
            "base_revision": config["vlm"]["base_revision"],
            "clip_model": config["search"]["embedding_model"],
            "milvus_server": identity["milvus_server"],
            "external_cluster_identity_file_sha256": file_sha256(identity_path),
            "milvus_rpm_sha256": performance["milvus_server"]["package_sha256"],
            "retrieval_archive_sha256": config["formal_release_read_only"]["retrieval_archive_sha256"],
            "milvus_service_startup_cold_ms": 1000.0,
            "milvus_collection_build_and_load_ms": 2000.0,
            "milvus_collection": {
                "name": "ota_business_image_vector",
                "visible_entities": config["formal_release_read_only"]["expected_index_support"],
                "index_names": ["multimodal_vector"],
                "index_details_sha256": "6" * 64,
            },
            "concurrency": performance["concurrency"],
            "vlm_max_new_tokens": performance["vlm_max_new_tokens"],
        },
        "hardware": {
            "node": "node-a",
            "platform": "Linux",
            "python": "3.11.3",
            "cpu_count": 4,
            "torch": "2.6.0",
            "cuda": "12.4",
            "gpu": "NVIDIA A100-SXM4-80GB",
            "gpu_count_visible": 1,
            "slurm_job_id": JOB_ID,
            "slurm_job_partition": "gpu-a100-preempt",
            "slurm_cpus_per_task": "4",
            "slurm_mem_per_node": "65536",
        },
        "roles": roles,
        "fixed_gates": fixed_gates,
        "raw_result_sha256": canonical_json_sha256(all_rows),
    }
    summary["artifact_sha256"] = canonical_json_sha256(summary)
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    chain = {
        "schema_version": "distributed_milvus_http_chain_v6",
        "status": "COMPLETED",
        "slurm_job_id": JOB_ID,
        "node_support": 2,
        "cluster_identity_file_sha256": file_sha256(identity_path),
        "summary_file_sha256": file_sha256(summary_path),
        "raw_file_sha256": file_sha256(raw_path),
        "fixed_gate_status": fixed_gates["status"],
        "fresh_test_used": False,
        "development_or_final_consumed": False,
    }
    _write_json(output_dir / "chain_summary.json", chain)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()

"""Tests for the independent v7 semantic-robustness evidence verifier."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_semantic_robustness_pool_v7 import build_pool
from scripts.run_http_milvus_service_benchmark_v4 import _performance_gates, _summarize_role
from scripts.run_vlm_semantic_evidence import _dialogue_metrics
from scripts.verify_semantic_robustness_evidence_v7 import _validate_service, verify_evidence_bundle
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl
from src.evaluation.semantic_robustness_v7 import (
    apply_semantic_robustness_v7_gates,
    score_semantic_robustness_v7,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "evaluation" / "automated_evidence_v7.json"
V5_LOCK = ROOT / "configs" / "evaluation" / "evidence_enhancement" / "context_focus_pool_lock_v5.json"
JOB_ID = "30005386"
SOURCE_SHA = "a" * 64
COMMIT = "b" * 40


class VerifySemanticRobustnessEvidenceV7Tests(unittest.TestCase):
    def test_valid_negative_bundle_passes_and_raw_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "run"
            _write_valid_bundle(output, root / "pool")
            report = _verify(output)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["development_raw_row_support"], 192)
            self.assertEqual(report["development_gate_status"], "FAIL")

            raw_path = output / "vlm-development" / "semantic_robustness_adapter_v7.jsonl"
            rows = load_jsonl(raw_path)
            rows[0]["latency_ms"] = 99.0
            _write_jsonl(raw_path, rows)
            with self.assertRaisesRegex(ValueError, "summary mismatch"):
                _verify(output)

    def test_service_verifier_covers_concurrency_two_and_four(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        performance = config["performance"]
        candidate_sha = "e" * 64
        baseline_sha = config["training"]["initial_adapter_model_sha256"]
        roles = {}
        all_rows = []
        for role_index, (role, adapter_sha) in enumerate((
            (performance["baseline_role"], baseline_sha),
            (performance["candidate_role"], candidate_sha),
        )):
            role_rows = []
            for concurrency in performance["concurrency"]:
                for batch in range(performance["steady_repetitions"]):
                    for request_index in range(concurrency):
                        latency = 100.0 + role_index * 5.0
                        role_rows.append({
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
                        })
            all_rows.extend(role_rows)
            roles[role] = {
                "adapter_model_sha256": adapter_sha,
                "service_startup_cold_ms": 500.0,
                "first_request_cold": {"phase": "cold", "success": True},
                "health": {"status": "ready"},
                "steady": _summarize_role(role_rows, performance["concurrency"]),
            }
        gates = _performance_gates(roles, performance)
        summary = {
            "schema_version": "http_milvus_service_benchmark_v7",
            "status": "COMPLETED",
            "scope": {
                "http": "real_loopback_HTTP_FastAPI_Uvicorn",
                "milvus": "external_server_process_single_node_standalone_not_Milvus_Lite",
                "distributed_cluster": "NOT_RUN_MULTI_NODE_CLUSTER",
                "production_sla_supported": False,
                "fresh_test_used": False,
                "development_or_final_consumed": False,
            },
            "denominators": {
                "cold_requests_per_role": performance["cold_repetitions"],
                "warmup_requests_per_role": performance["warmup_repetitions"],
                "steady_batches_per_concurrency_per_role": performance["steady_repetitions"],
                "steady_requests_per_role": 56,
            },
            "fixed_input": {
                "split": "training",
                "query_id": performance["fixed_input"]["query_id"],
                "label_provenance": "synthetic_training_query_for_performance_only",
            },
            "configuration": {
                "config_sha256": file_sha256(CONFIG),
                "source_snapshot_sha256": SOURCE_SHA,
                "base_model": config["vlm"]["base_model"],
                "base_revision": config["vlm"]["base_revision"],
                "clip_model": config["search"]["embedding_model"],
                "milvus_rpm_sha256": performance["milvus_server"]["package_sha256"],
                "retrieval_archive_sha256": config["formal_release_read_only"]["retrieval_archive_sha256"],
                "concurrency": performance["concurrency"],
                "vlm_max_new_tokens": performance["vlm_max_new_tokens"],
                "milvus_service_startup_cold_ms": 1000.0,
                "milvus_collection_build_and_load_ms": 2000.0,
                "milvus_collection": {
                    "visible_entities": config["formal_release_read_only"]["expected_index_support"],
                    "index_names": ["multimodal_vector"],
                },
            },
            "hardware": {
                "node": "node-a",
                "platform": "Linux",
                "python": "3.11.3",
                "cpu_count": 8,
                "torch": "2.6.0",
                "cuda": "12.4",
                "gpu": "NVIDIA L40S",
                "gpu_count_visible": 1,
                "slurm_job_id": JOB_ID,
            },
            "roles": roles,
            "fixed_gates": gates,
            "raw_result_sha256": canonical_json_sha256(all_rows),
        }
        summary["artifact_sha256"] = canonical_json_sha256(summary)
        _validate_service(
            summary,
            all_rows,
            config,
            CONFIG,
            JOB_ID,
            SOURCE_SHA,
            baseline_sha,
            candidate_sha,
        )
        all_rows[-1]["http_e2e_ms"] = 999.0
        with self.assertRaisesRegex(ValueError, "summary differs from raw rows"):
            _validate_service(
                summary,
                all_rows,
                config,
                CONFIG,
                JOB_ID,
                SOURCE_SHA,
                baseline_sha,
                candidate_sha,
            )


def _verify(output: Path) -> dict[str, object]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return verify_evidence_bundle(
        config_path=CONFIG,
        output_dir=output,
        expected_job_id=JOB_ID,
        expected_source_snapshot_sha256=SOURCE_SHA,
        expected_implementation_commit=COMMIT,
        expected_baseline_adapter_sha256=config["training"]["initial_adapter_model_sha256"],
    )


def _write_valid_bundle(output: Path, pool: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    lock = build_pool(pool, V5_LOCK)
    training = output / "training"
    development = output / "vlm-development"
    service = output / "service"
    adapter = training / "adapter"
    adapter.mkdir(parents=True)
    development.mkdir(parents=True)
    service.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"candidate-v7")
    _write_json(adapter / "adapter_config.json", {"base_model": config["vlm"]["base_model"]})
    candidate_sha = file_sha256(adapter / "adapter_model.safetensors")
    common = {
        "run_id": config["training"]["run_id"],
        "git_commit": COMMIT,
        "config_sha256": file_sha256(CONFIG),
        "pool_lock_sha256": canonical_json_sha256(lock),
        "training_manifest_sha256": lock["vlm"]["training"]["manifest_file_sha256"],
        "initial_adapter_model_sha256": config["training"]["initial_adapter_model_sha256"],
        "development_or_final_opened": False,
    }
    _write_json(training / "run_identity.json", {
        "schema_version": "targeted_exploration_training_identity_v7",
        **common,
    })
    _write_json(training / "run_summary.json", {
        "schema_version": "targeted_exploration_training_summary_v7",
        **common,
        "status": "COMPLETED",
        "training_support": lock["vlm"]["training"]["sample_support"],
        "product_support": lock["vlm"]["training"]["product_support"],
        "dialogue_support": lock["vlm"]["training"]["dialogue_support"],
        "global_step": 32,
        "training_metrics": {"train_loss": 0.1},
        "duration_seconds": 120.0,
        "peak_gpu_memory_allocated_bytes": 1024,
        "peak_gpu_memory_reserved_bytes": 2048,
        "adapter_model_sha256": candidate_sha,
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
        "adapter_only": True,
        "slurm_job_id": JOB_ID,
    })

    manifest = load_jsonl(pool / "vlm_development_manifest.jsonl")
    data_lock = canonical_json_sha256(manifest)
    prompt_sha = "c" * 64
    generation_sha = "d" * 64
    roles = config["vlm"]["development_roles"]
    adapter_hashes = {
        config["vlm"]["baseline_variant"]: config["training"]["initial_adapter_model_sha256"],
        config["vlm"]["candidate_variant"]: candidate_sha,
    }
    raw_paths = []
    all_rows = []
    for role in roles:
        rows = []
        for index, record in enumerate(manifest):
            prediction = {
                key: value for key, value in record["gold"].items() if key != "unknown_fields"
            }
            row = {
                "variant": role,
                "sample_id": record["sample_id"],
                "scenario": record["scenario"],
                "data_lock_sha256": data_lock,
                "base_model": config["vlm"]["base_model"],
                "base_revision": config["vlm"]["base_revision"],
                "prompt_sha256": prompt_sha,
                "generation_config_sha256": generation_sha,
                "adapter_model_sha256": adapter_hashes[role],
                "label_provenance": record["label_provenance"],
                "slices": record["slices"],
                "gold": record["gold"],
                "prediction": prediction,
                "first_attempt_json_valid": True,
                "correction_triggered": False,
                "first_attempt_raw": "{}",
                "correction_raw": None,
                "latency_ms": 10.0 + index / 1000,
                "peak_vram_mib": 100.0,
            }
            if record["scenario"] == "dialogue":
                row.update(_dialogue_metrics(record["gold"], prediction))
            rows.append(row)
        raw_path = development / f"{role}.jsonl"
        _write_jsonl(raw_path, rows)
        raw_paths.append(raw_path)
        all_rows.extend(rows)
        _write_json(raw_path.with_suffix(".summary.json"), {
            "schema_version": "vlm_semantic_role_evidence_v7",
            "status": "COMPLETED",
            "split": "development",
            "role": role,
            "support": len(rows),
            "product_support": lock["vlm"]["development"]["product_support"],
            "dialogue_support": lock["vlm"]["development"]["dialogue_support"],
            "data_lock_sha256": data_lock,
            "prompt_sha256": prompt_sha,
            "generation_config_sha256": generation_sha,
            "result_sha256": canonical_json_sha256(rows),
            "first_attempt_json_compliance": 1.0,
            "correction_trigger_rate": 0.0,
            "mean_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows),
            "peak_vram_mib": 100.0,
            "final_consumed_once": False,
            "fresh_test_used": False,
            "slurm_job_id": JOB_ID,
        })
    metrics = score_semantic_robustness_v7(all_rows)
    gate = apply_semantic_robustness_v7_gates(
        metrics,
        config["vlm"]["exploration_gates"],
        config["vlm"]["selection_objective"],
        candidate=config["vlm"]["candidate_variant"],
        baseline=config["vlm"]["baseline_variant"],
    )
    selection = {
        **metrics,
        "status": "COMPLETED",
        "split": "development",
        "candidate_variant": config["vlm"]["candidate_variant"],
        "fixed_gates": gate,
        "gate_class": config["gate_class"],
        "human_annotation_support": 0,
        "final_defined_or_consumed": False,
        "fresh_test_used": False,
        "raw_result_support": len(all_rows),
        "raw_result_canonical_sha256": canonical_json_sha256(all_rows),
        "raw_result_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in raw_paths
        ],
        "promotion_eligible_as_human_ground_truth": False,
    }
    _write_json(development / "selection.json", selection)
    artifacts = [
        {
            "path": path.relative_to(output).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    _write_json(output / "chain_summary.json", {
        "schema_version": "semantic_robustness_chain_v7",
        "status": "COMPLETED",
        "development_gate_status": gate["status"],
        "performance_gate_status": "NOT_RUN_DEVELOPMENT_GATE_FAILED",
        "candidate_adapter_model_sha256": candidate_sha,
        "final_defined_or_consumed": False,
        "fresh_test_used": False,
        "slurm_job_id": JOB_ID,
        "artifacts": artifacts,
    })


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Smoke-test the distributed Milvus runtime without consuming evaluation data."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_distributed_milvus_runtime_v6 import prepare_runtime
from scripts.run_distributed_milvus_cluster_v6 import (
    build_minio_environment,
    build_minio_server_command,
    file_sha256,
    find_local_port_base,
    load_json,
    redact_logs,
    require_processes_alive,
    terminate_steps,
    wait_for_cluster,
    wait_for_text,
    wait_http,
    wait_tcp,
    write_json_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--milvus-rpm", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_smoke(args)


def run_smoke(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite smoke evidence: {args.output_dir}")
    config = load_json(args.config)
    validate_dependencies(args, config)
    args.output_dir.mkdir(parents=True)
    logs = args.output_dir / "service-logs"
    logs.mkdir()

    job_id = os.environ["SLURM_JOB_ID"]
    node = socket.gethostname().split(".")[0]
    local_root = Path("/tmp") / f"trip-distributed-milvus-smoke-{job_id}"
    expected_local_root = Path("/tmp") / f"trip-distributed-milvus-smoke-{job_id}"
    port_base = find_local_port_base(job_id, (0, 20, 40))
    access_key = "trip" + secrets.token_hex(8)
    secret_key = secrets.token_urlsafe(36)
    secret_env = os.environ.copy()
    secret_env["TRIP_MINIO_ACCESS_KEY"] = access_key
    secret_env["TRIP_MINIO_SECRET_KEY"] = secret_key
    secret_env["CUDA_VISIBLE_DEVICES"] = ""
    processes: list[subprocess.Popen[str]] = []
    started = time.perf_counter()
    passed = False
    try:
        query_streaming_port_base = port_base + 20
        data_port_base = port_base + 40
        prepare_runtime(
            rpm=args.milvus_rpm,
            output_dir=local_root / "runtime-control",
            control_node=node,
            port_base=port_base,
            expected_rpm_sha256=config["performance"]["milvus_server"]["package_sha256"],
            access_key=access_key,
            secret_key=secret_key,
            component_port_base=port_base,
        )
        prepare_runtime(
            rpm=args.milvus_rpm,
            output_dir=local_root / "runtime-query-streaming",
            control_node=node,
            port_base=port_base,
            expected_rpm_sha256=config["performance"]["milvus_server"]["package_sha256"],
            access_key=access_key,
            secret_key=secret_key,
            component_port_base=query_streaming_port_base,
        )
        prepare_runtime(
            rpm=args.milvus_rpm,
            output_dir=local_root / "runtime-data",
            control_node=node,
            port_base=port_base,
            expected_rpm_sha256=config["performance"]["milvus_server"]["package_sha256"],
            access_key=access_key,
            secret_key=secret_key,
            component_port_base=data_port_base,
        )
        dependencies = config["performance"]["dependencies"]
        etcd = args.dependency_dir / dependencies["etcd"]["archive"].replace(".tar.gz", "") / "etcd"
        minio = args.dependency_dir / dependencies["minio"]["binary"]
        processes.append(
            start_local(
                [
                    str(etcd),
                    "--name", f"trip-smoke-{job_id}",
                    "--data-dir", str(local_root / "etcd-data"),
                    "--listen-client-urls", f"http://0.0.0.0:{port_base}",
                    "--advertise-client-urls", f"http://{node}:{port_base}",
                    "--listen-peer-urls", f"http://0.0.0.0:{port_base + 12}",
                    "--initial-advertise-peer-urls", f"http://{node}:{port_base + 12}",
                    "--initial-cluster", f"trip-smoke-{job_id}=http://{node}:{port_base + 12}",
                ],
                logs / "etcd.log",
                secret_env,
            )
        )
        minio_env = build_minio_environment(
            secret_env,
            access_key=access_key,
            secret_key=secret_key,
        )
        processes.append(
            start_local(
                build_minio_server_command(
                    minio,
                    data_dir=local_root / "minio-data",
                    certs_dir=local_root / "minio-certs",
                    address=f":{port_base + 1}",
                    console_address=f":{port_base + 2}",
                ),
                logs / "minio.log",
                minio_env,
            )
        )
        wait_tcp(node, port_base, processes, 60)
        wait_http(f"http://{node}:{port_base + 1}/minio/health/live", processes, 60)
        for placement, metrics_port in (
            ("control", port_base + 13),
            ("query-streaming", query_streaming_port_base + 13),
            ("data", data_port_base + 13),
        ):
            processes.append(
                start_local(
                    [
                        sys.executable,
                        str(Path(__file__).resolve().with_name("run_distributed_milvus_cluster_v6.py")),
                        "serve-milvus",
                        "--runtime-dir", str(local_root / f"runtime-{placement}"),
                        "--placement", placement,
                        "--metrics-port", str(metrics_port),
                    ],
                    logs / f"milvus-{placement}.log",
                    secret_env,
                )
            )
        proxy_uri = f"http://{node}:{port_base + 4}"
        wait_for_cluster(proxy_uri, processes, 240)
        wait_for_text(
            logs / "milvus-control.log",
            ["mixcoord", "proxy"],
            processes,
            30,
        )
        wait_for_text(
            logs / "milvus-query-streaming.log",
            ["querynode", "streamingnode"],
            processes,
            30,
        )
        wait_for_text(
            logs / "milvus-data.log",
            ["datanode"],
            processes,
            30,
        )
        operation_metrics = exercise_collection(proxy_uri, job_id)
        result = {
            "schema_version": "distributed_milvus_runtime_smoke_v6",
            "status": "PASS",
            "scope": "one_node_engineering_smoke_not_distributed_performance_evidence",
            "slurm_job_id": job_id,
            "node_support": 1,
            "process_topology": "control_plus_query_streaming_plus_data_mixtures_on_one_node",
            "runtime_port_base": port_base,
            "roles": {
                "control": ["mixcoord", "proxy"],
                "query_streaming": ["querynode", "streamingnode"],
                "data": ["datanode"],
            },
            "startup_and_crud_ms": (time.perf_counter() - started) * 1000,
            "operations": operation_metrics,
            "milvus_server": config["performance"]["milvus_server"],
            "dependencies": dependency_identity(args, config),
            "source_snapshot_sha256": args.source_snapshot_sha256,
            "credentials": "random_job_local_not_persisted",
            "query": "fixed_four_dimensional_synthetic_vector_only",
            "fresh_test_used": False,
            "development_or_final_consumed": False,
        }
        write_json_exclusive(args.output_dir / "smoke_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        passed = True
    except Exception as exc:
        failure = {
            "schema_version": "distributed_milvus_runtime_smoke_failure_v6",
            "status": "FAIL",
            "scope": "engineering_failure_not_quality_or_performance_evidence",
            "slurm_job_id": job_id,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "source_snapshot_sha256": args.source_snapshot_sha256,
            "fresh_test_used": False,
            "development_or_final_consumed": False,
        }
        write_json_exclusive(args.output_dir / "smoke_failure.json", failure)
        raise
    finally:
        terminate_steps(processes)
        redact_logs(logs, (access_key, secret_key))
        if local_root.resolve() != expected_local_root:
            raise RuntimeError("refusing to clean an unexpected smoke path")
        shutil.rmtree(local_root, ignore_errors=True)
        if not passed:
            print("distributed Milvus runtime smoke test did not complete", file=sys.stderr)


def start_local(command: list[str], log: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    handle.close()
    return process


def exercise_collection(uri: str, job_id: str) -> dict[str, Any]:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=uri, timeout=15)
    collection = f"trip_runtime_smoke_{job_id.replace('-', '_')}"
    timings: dict[str, float] = {}
    result: dict[str, Any] = {}
    collection_created = False
    try:
        started = time.perf_counter()
        client.create_collection(
            collection_name=collection,
            dimension=4,
            metric_type="COSINE",
            consistency_level="Strong",
        )
        collection_created = True
        timings["create_collection_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        insert_result = client.insert(
            collection_name=collection,
            data=[
                {"id": 1, "vector": [1.0, 0.0, 0.0, 0.0]},
                {"id": 2, "vector": [0.0, 1.0, 0.0, 0.0]},
                {"id": 3, "vector": [-1.0, 0.0, 0.0, 0.0]},
            ],
        )
        client.flush(collection_name=collection)
        timings["insert_and_flush_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        search_result = client.search(
            collection_name=collection,
            data=[[1.0, 0.0, 0.0, 0.0]],
            limit=2,
        )
        timings["search_ms"] = (time.perf_counter() - started) * 1000
        if not search_result or not search_result[0]:
            raise RuntimeError("Milvus smoke search returned no hits")
        top_hit = search_result[0][0]
        top_id = int(top_hit["id"])
        if top_id != 1:
            raise RuntimeError(f"Milvus smoke search returned unexpected top id: {top_id}")
        insert_count = int(insert_result.get("insert_count", 0))
        if insert_count != 3:
            raise RuntimeError(f"Milvus smoke insert count differs from 3: {insert_count}")
        result = {
            "insert_count": insert_count,
            "search_limit": 2,
            "top_id": top_id,
            "top_id_expected": 1,
        }
    finally:
        started = time.perf_counter()
        if collection_created:
            client.drop_collection(collection_name=collection)
        timings["drop_collection_ms"] = (time.perf_counter() - started) * 1000
        client.close()
    return {**timings, **result}


def validate_dependencies(args: argparse.Namespace, config: dict[str, Any]) -> None:
    performance = config["performance"]
    server = performance["milvus_server"]
    if server.get("multi_node_distributed_cluster") is not True:
        raise ValueError("configuration is not the locked distributed Milvus configuration")
    dependencies = performance["dependencies"]
    checks = (
        (args.milvus_rpm, server["package_sha256"], "Milvus RPM"),
        (args.dependency_dir / dependencies["etcd"]["archive"], dependencies["etcd"]["sha256"], "etcd archive"),
        (
            args.dependency_dir / dependencies["etcd"]["archive"].replace(".tar.gz", "") / "etcd",
            dependencies["etcd"]["binary_sha256"],
            "etcd binary",
        ),
        (args.dependency_dir / dependencies["minio"]["binary"], dependencies["minio"]["sha256"], "MinIO binary"),
    )
    for path, expected, label in checks:
        if file_sha256(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")


def dependency_identity(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    dependencies = config["performance"]["dependencies"]
    etcd = args.dependency_dir / dependencies["etcd"]["archive"].replace(".tar.gz", "") / "etcd"
    minio = args.dependency_dir / dependencies["minio"]["binary"]
    return {
        "etcd": {"version": dependencies["etcd"]["version"], "sha256": file_sha256(etcd)},
        "minio": {"version": dependencies["minio"]["version"], "sha256": file_sha256(minio)},
        "message_queue": dependencies["message_queue"],
    }


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a fail-closed two-node Milvus cluster and the locked HTTP benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_distributed_milvus_runtime_v6 import prepare_runtime


REQUIRED_ROLES = {
    "mixcoord": "control",
    "proxy": "control",
    "querynode": "worker",
    "datanode": "worker",
    "streamingnode": "worker",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    for name in (
        "config",
        "bundle-dir",
        "retrieval-archive",
        "milvus-rpm",
        "dependency-dir",
        "baseline-adapter",
        "candidate-adapter",
        "quality-selection",
        "output-dir",
    ):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--baseline-adapter-sha256", required=True)
    run.add_argument("--candidate-adapter-sha256", required=True)
    run.add_argument("--source-snapshot-sha256", required=True)

    prepare = subparsers.add_parser("prepare-node")
    prepare.add_argument("--rpm", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--control-node", required=True)
    prepare.add_argument("--port-base", type=int, required=True)
    prepare.add_argument("--expected-rpm-sha256", required=True)

    ports = subparsers.add_parser("check-ports")
    ports.add_argument("--port-base", type=int, required=True)

    serve = subparsers.add_parser("serve-milvus")
    serve.add_argument("--runtime-dir", type=Path, required=True)
    serve.add_argument("--placement", choices=("control", "worker", "all"), required=True)
    args = parser.parse_args()
    if args.command == "prepare-node":
        prepare_runtime(
            rpm=args.rpm,
            output_dir=args.output_dir,
            control_node=args.control_node,
            port_base=args.port_base,
            expected_rpm_sha256=args.expected_rpm_sha256,
            access_key=os.environ.get("TRIP_MINIO_ACCESS_KEY", ""),
            secret_key=os.environ.get("TRIP_MINIO_SECRET_KEY", ""),
        )
    elif args.command == "check-ports":
        check_ports(args.port_base)
    elif args.command == "serve-milvus":
        serve_milvus(args.runtime_dir, args.placement)
    else:
        run_cluster(args)


def run_cluster(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite distributed evidence: {args.output_dir}")
    config = load_json(args.config)
    performance = config["performance"]
    validate_inputs(args, config)
    nodes = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
        text=True,
    ).splitlines()
    if len(nodes) != 2 or len(set(nodes)) != 2:
        raise RuntimeError(f"distributed benchmark requires exactly two unique nodes, got {nodes}")
    control, worker = nodes
    if socket.gethostname().split(".")[0] != control.split(".")[0]:
        raise RuntimeError("batch orchestrator is not running on the first allocated node")

    args.output_dir.mkdir(parents=True)
    logs = args.output_dir / "cluster-logs"
    logs.mkdir()
    job_id = os.environ["SLURM_JOB_ID"]
    local_root = Path("/tmp") / f"trip-distributed-milvus-{job_id}"
    port_base = 26000 + (int(job_id) % 400) * 20
    access_key = "trip" + secrets.token_hex(8)
    secret_key = secrets.token_urlsafe(36)
    secret_env = os.environ.copy()
    secret_env["TRIP_MINIO_ACCESS_KEY"] = access_key
    secret_env["TRIP_MINIO_SECRET_KEY"] = secret_key
    secret_env["CUDA_VISIBLE_DEVICES"] = ""
    processes: list[subprocess.Popen[str]] = []
    startup_started = time.perf_counter()
    success = False
    try:
        for node in nodes:
            run_step(
                node,
                [sys.executable, str(Path(__file__).resolve()), "check-ports", "--port-base", str(port_base)],
                cpus=1,
            )
            run_step(
                node,
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "prepare-node",
                    "--rpm",
                    str(args.milvus_rpm),
                    "--output-dir",
                    str(local_root / "runtime"),
                    "--control-node",
                    control,
                    "--port-base",
                    str(port_base),
                    "--expected-rpm-sha256",
                    performance["milvus_server"]["package_sha256"],
                ],
                cpus=1,
                env=secret_env,
            )

        dependencies = performance["dependencies"]
        etcd = args.dependency_dir / "etcd-v3.5.25-linux-amd64" / "etcd"
        minio = args.dependency_dir / "minio.RELEASE.2024-12-18T13-15-44Z"
        etcd_log = logs / "etcd.log"
        minio_log = logs / "minio.log"
        processes.append(
            start_step(
                control,
                [
                    str(etcd),
                    "--name",
                    f"trip-{job_id}",
                    "--data-dir",
                    str(local_root / "etcd-data"),
                    "--listen-client-urls",
                    f"http://0.0.0.0:{port_base}",
                    "--advertise-client-urls",
                    f"http://{control}:{port_base}",
                    "--listen-peer-urls",
                    f"http://0.0.0.0:{port_base + 12}",
                    "--initial-advertise-peer-urls",
                    f"http://{control}:{port_base + 12}",
                    "--initial-cluster",
                    f"trip-{job_id}=http://{control}:{port_base + 12}",
                    "--auto-compaction-mode",
                    "revision",
                    "--auto-compaction-retention",
                    "1000",
                    "--quota-backend-bytes",
                    "4294967296",
                ],
                etcd_log,
                cpus=2,
                env=secret_env,
            )
        )
        minio_env = secret_env.copy()
        minio_env.update(
            {
                "MINIO_ACCESS_KEY": access_key,
                "MINIO_SECRET_KEY": secret_key,
                "MINIO_ROOT_USER": access_key,
                "MINIO_ROOT_PASSWORD": secret_key,
            }
        )
        processes.append(
            start_step(
                control,
                [
                    str(minio),
                    "server",
                    str(local_root / "minio-data"),
                    "--address",
                    f":{port_base + 1}",
                    "--console-address",
                    f":{port_base + 2}",
                ],
                minio_log,
                cpus=2,
                env=minio_env,
            )
        )
        wait_tcp(control, port_base, processes, 60)
        wait_http(f"http://{control}:{port_base + 1}/minio/health/live", processes, 60)

        control_log = logs / "milvus-control.log"
        worker_log = logs / "milvus-worker.log"
        for node, placement, log in (
            (control, "control", control_log),
            (worker, "worker", worker_log),
        ):
            processes.append(
                start_step(
                    node,
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "serve-milvus",
                        "--runtime-dir",
                        str(local_root / "runtime"),
                        "--placement",
                        placement,
                    ],
                    log,
                    cpus=4,
                    env=secret_env,
                )
            )
        proxy_uri = f"http://{control}:{port_base + 4}"
        wait_for_cluster(proxy_uri, processes, 240)
        wait_for_text(control_log, ["mixcoord", "proxy"], processes, 30)
        wait_for_text(worker_log, ["querynode", "datanode", "streamingnode"], processes, 30)
        startup_ms = (time.perf_counter() - startup_started) * 1000

        identity = {
            "schema_version": "distributed_milvus_cluster_identity_v6",
            "status": "READY",
            "slurm_job_id": job_id,
            "nodes": nodes,
            "roles": {role: control if placement == "control" else worker for role, placement in REQUIRED_ROLES.items()},
            "startup_cold_ms": startup_ms,
            "milvus_server": performance["milvus_server"],
            "dependencies": {
                "etcd": {"version": dependencies["etcd"]["version"], "sha256": file_sha256(etcd)},
                "minio": {"version": dependencies["minio"]["version"], "sha256": file_sha256(minio)},
                "message_queue": dependencies["message_queue"],
            },
            "quality_selection_file_sha256": file_sha256(args.quality_selection),
            "source_snapshot_sha256": args.source_snapshot_sha256,
            "credentials": "random_job_local_not_persisted",
            "fresh_test_used": False,
            "development_or_final_consumed": False,
        }
        identity_path = args.output_dir / "cluster_identity.json"
        write_json_exclusive(identity_path, identity)
        benchmark_output = args.output_dir / "summary.json"
        raw_output = args.output_dir / "raw.jsonl"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().with_name("run_http_milvus_service_benchmark_v4.py")),
                "run",
                "--config",
                str(args.config),
                "--bundle-dir",
                str(args.bundle_dir),
                "--retrieval-archive",
                str(args.retrieval_archive),
                "--milvus-rpm",
                str(args.milvus_rpm),
                "--jobfs",
                os.environ.get("JOBFS", "/tmp"),
                "--baseline-adapter",
                str(args.baseline_adapter),
                "--baseline-adapter-sha256",
                args.baseline_adapter_sha256,
                "--candidate-adapter",
                str(args.candidate_adapter),
                "--candidate-adapter-sha256",
                args.candidate_adapter_sha256,
                "--source-snapshot-sha256",
                args.source_snapshot_sha256,
                "--external-milvus-uri",
                proxy_uri,
                "--external-milvus-identity",
                str(identity_path),
                "--output",
                str(benchmark_output),
                "--raw-output",
                str(raw_output),
            ],
            check=True,
        )
        report = load_json(benchmark_output)
        if report["scope"]["distributed_cluster"] is not True:
            raise RuntimeError("benchmark did not retain distributed cluster scope")
        chain = {
            "schema_version": "distributed_milvus_http_chain_v6",
            "status": "COMPLETED",
            "slurm_job_id": job_id,
            "node_support": len(nodes),
            "cluster_identity_file_sha256": file_sha256(identity_path),
            "summary_file_sha256": file_sha256(benchmark_output),
            "raw_file_sha256": file_sha256(raw_output),
            "fixed_gate_status": report["fixed_gates"]["status"],
            "fresh_test_used": False,
            "development_or_final_consumed": False,
        }
        write_json_exclusive(args.output_dir / "chain_summary.json", chain)
        success = True
        print(json.dumps(chain, indent=2, sort_keys=True))
    finally:
        terminate_steps(processes)
        redact_logs(logs, (access_key, secret_key))
        for node in nodes:
            cleanup_step(node, local_root)
        if not success:
            print("distributed Milvus benchmark did not complete", file=sys.stderr)


def validate_inputs(args: argparse.Namespace, config: dict[str, Any]) -> None:
    performance = config["performance"]
    server = performance["milvus_server"]
    if server.get("multi_node_distributed_cluster") is not True:
        raise ValueError("configuration is not locked to multi-node distributed Milvus")
    if set(server.get("required_roles", [])) != set(REQUIRED_ROLES):
        raise ValueError("configuration does not lock the exact required Milvus roles")
    checks = (
        (args.milvus_rpm, server["package_sha256"], "Milvus RPM"),
        (args.retrieval_archive, config["formal_release_read_only"]["retrieval_archive_sha256"], "retrieval"),
        (args.baseline_adapter / "adapter_model.safetensors", args.baseline_adapter_sha256, "baseline adapter"),
        (args.candidate_adapter / "adapter_model.safetensors", args.candidate_adapter_sha256, "candidate adapter"),
    )
    for path, expected, label in checks:
        if file_sha256(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
    selection_sha = performance["quality_gate_source"].rsplit("_sha256_", 1)[-1]
    if file_sha256(args.quality_selection) != selection_sha:
        raise ValueError("v5 quality selection SHA-256 mismatch")
    selection = load_json(args.quality_selection)
    if selection.get("fixed_gates", {}).get("status") != "PASS":
        raise ValueError("v5 quality gate did not pass")
    dependencies = performance["dependencies"]
    etcd_archive = args.dependency_dir / dependencies["etcd"]["archive"]
    etcd = args.dependency_dir / dependencies["etcd"]["archive"].replace(".tar.gz", "") / "etcd"
    minio = args.dependency_dir / dependencies["minio"]["binary"]
    for path, expected, label in (
        (etcd_archive, dependencies["etcd"]["sha256"], "etcd archive"),
        (etcd, dependencies["etcd"]["binary_sha256"], "etcd binary"),
        (minio, dependencies["minio"]["sha256"], "MinIO binary"),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")


def check_ports(port_base: int) -> None:
    ports = list(range(port_base, port_base + 13))
    sockets = []
    try:
        for port in ports:
            current = socket.socket()
            current.bind(("0.0.0.0", port))
            sockets.append(current)
    finally:
        for current in sockets:
            current.close()


def serve_milvus(runtime_dir: Path, placement: str) -> None:
    binary = runtime_dir / "usr" / "bin" / "milvus"
    env = os.environ.copy()
    env["MILVUSCONF"] = str(runtime_dir / "etc" / "milvus" / "configs")
    env["LD_LIBRARY_PATH"] = str(runtime_dir / "usr" / "lib" / "milvus")
    env["DEPLOY_MODE"] = "CLUSTER"
    if placement == "control":
        command = [
            str(binary), "run", "mixture", "-rootcoord=true", "-querycoord=true",
            "-datacoord=true", "-proxy=true", "-alias=trip-control",
        ]
    elif placement == "worker":
        command = [
            str(binary), "run", "mixture", "-querynode=true", "-datanode=true",
            "-streamingnode=true", "-alias=trip-worker",
        ]
    else:
        command = [
            str(binary), "run", "mixture", "-rootcoord=true", "-querycoord=true",
            "-datacoord=true", "-querynode=true", "-datanode=true", "-proxy=true",
            "-streamingnode=true", "-alias=trip-smoke",
        ]
    os.execve(str(binary), command, env)


def run_step(node: str, command: list[str], *, cpus: int, env: dict[str, str] | None = None) -> None:
    subprocess.run(srun_prefix(node, cpus) + command, check=True, env=env)


def start_step(
    node: str,
    command: list[str],
    log: Path,
    *,
    cpus: int,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        srun_prefix(node, cpus) + command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    handle.close()
    return process


def srun_prefix(node: str, cpus: int) -> list[str]:
    return ["srun", "--overlap", "--nodes=1", "--ntasks=1", f"--cpus-per-task={cpus}", "--nodelist", node]


def wait_tcp(host: str, port: int, processes: list[subprocess.Popen[str]], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require_processes_alive(processes)
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"TCP endpoint did not become ready: {host}:{port}")


def wait_http(url: str, processes: list[subprocess.Popen[str]], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require_processes_alive(processes)
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"HTTP endpoint did not become ready: {url}")


def wait_for_cluster(uri: str, processes: list[subprocess.Popen[str]], timeout: float) -> None:
    from pymilvus import MilvusClient

    deadline = time.monotonic() + timeout
    error = "not attempted"
    while time.monotonic() < deadline:
        require_processes_alive(processes)
        try:
            client = MilvusClient(uri=uri, timeout=5)
            client.list_collections()
            client.close()
            return
        except Exception as exc:
            error = str(exc)
            time.sleep(2)
    raise TimeoutError(f"distributed Milvus proxy did not become ready: {error}")


def wait_for_text(
    path: Path,
    needles: list[str],
    processes: list[subprocess.Popen[str]],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require_processes_alive(processes)
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        lowered = text.lower()
        if "all components are ready" in lowered and all(needle in lowered for needle in needles):
            return
        time.sleep(1)
    raise TimeoutError(f"component readiness evidence missing from {path}")


def require_processes_alive(processes: list[subprocess.Popen[str]]) -> None:
    failed = [process.returncode for process in processes if process.poll() is not None]
    if failed:
        raise RuntimeError(f"distributed service step exited early: {failed}")


def terminate_steps(processes: list[subprocess.Popen[str]]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def cleanup_step(node: str, local_root: Path) -> None:
    resolved = str(local_root.resolve())
    expected = f"/tmp/trip-distributed-milvus-{os.environ['SLURM_JOB_ID']}"
    if resolved != expected:
        raise RuntimeError("refusing to clean an unexpected node-local path")
    subprocess.run(
        srun_prefix(node, 1) + [sys.executable, "-c", "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)", resolved],
        check=False,
    )


def redact_logs(log_dir: Path, secrets_to_redact: tuple[str, ...]) -> None:
    for path in log_dir.glob("*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for value in secrets_to_redact:
            if value:
                text = text.replace(value, "[REDACTED_JOB_SECRET]")
        path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

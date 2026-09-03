"""Benchmark a real HTTP pipeline backed by an external Milvus standalone server.

The benchmark reads one locked training query only. Development, final, and the frozen
120-sample Fresh Test are outside this process. It compares adapters sequentially in
one Slurm allocation and reports queue, CLIP, Milvus, rerank, VLM, service, and client
HTTP latency separately.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import platform
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_relevance_evidence import _read_retrieval_archive
from scripts.run_vlm_semantic_evidence import _load_model
from src.evaluation.relevance_evidence import (
    canonical_json_sha256,
    file_sha256,
    load_jsonl,
)
from src.retrieval.milvus_vectors import FILTER_FIELDS, OTAMilvusVectorStore


COLLECTION = "ota_business_image_vector"
STAGES = (
    "queue_wait_ms",
    "clip_encode_ms",
    "milvus_query_ms",
    "rerank_ms",
    "vlm_inference_ms",
    "service_total_ms",
    "http_e2e_ms",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--bundle-dir", type=Path, required=True)
    run.add_argument("--retrieval-archive", type=Path, required=True)
    run.add_argument("--milvus-rpm", type=Path, required=True)
    run.add_argument("--jobfs", type=Path, required=True)
    run.add_argument("--baseline-adapter", type=Path, required=True)
    run.add_argument("--baseline-adapter-sha256", required=True)
    run.add_argument("--candidate-adapter", type=Path, required=True)
    run.add_argument("--candidate-adapter-sha256", required=True)
    run.add_argument("--source-snapshot-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--raw-output", type=Path, required=True)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--base-model", required=True)
    serve.add_argument("--base-revision", required=True)
    serve.add_argument("--adapter-path", type=Path, required=True)
    serve.add_argument("--adapter-sha256", required=True)
    serve.add_argument("--clip-model", required=True)
    serve.add_argument("--milvus-uri", required=True)
    serve.add_argument("--milvus-config-json", required=True)
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--max-new-tokens", type=int, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        serve_http(args)
    else:
        run_benchmark(args)


def run_benchmark(args: argparse.Namespace) -> None:
    for path in (args.output, args.raw_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite benchmark evidence: {path}")
    config = _load_json(args.config)
    performance = config["performance"]
    _verify_file(args.milvus_rpm, performance["milvus_server"]["package_sha256"], "Milvus RPM")
    _verify_file(
        args.retrieval_archive,
        config["formal_release_read_only"]["retrieval_archive_sha256"],
        "formal retrieval archive",
    )
    adapters = {
        "current_system_repair_checkpoint_87": (
            args.baseline_adapter,
            args.baseline_adapter_sha256,
        ),
        "targeted_exploration_adapter_v4": (
            args.candidate_adapter,
            args.candidate_adapter_sha256,
        ),
    }
    for role, (path, expected_sha) in adapters.items():
        _verify_file(path / "adapter_model.safetensors", expected_sha, f"{role} adapter")

    locked_request = _load_locked_training_request(config, args.bundle_dir)
    archive = _read_retrieval_archive(args.retrieval_archive)
    if len(archive["vectors"]) != config["formal_release_read_only"]["expected_index_support"]:
        raise ValueError("formal vector support differs from the release lock")
    job_root = (args.jobfs / f"trip-service-benchmark-{os.getenv('SLURM_JOB_ID', 'local')}").resolve()
    if job_root.exists():
        raise FileExistsError(f"job-local benchmark root already exists: {job_root}")
    job_root.mkdir(parents=True)
    port_base = _choose_port_block()
    milvus_port, http_port = port_base + 2, port_base + 20
    milvus_process: subprocess.Popen[str] | None = None
    raw_rows: list[dict[str, Any]] = []
    role_summaries: dict[str, Any] = {}
    try:
        milvus_runtime = _prepare_milvus_runtime(args.milvus_rpm, job_root, port_base)
        milvus_startup_started = time.perf_counter()
        milvus_process = _start_milvus(milvus_runtime)
        milvus_uri = f"http://127.0.0.1:{milvus_port}"
        _wait_for_milvus(milvus_uri, milvus_process, timeout_seconds=180)
        milvus_startup_ms = (time.perf_counter() - milvus_startup_started) * 1000
        store_config = _store_config(config, milvus_uri)
        index_build_started = time.perf_counter()
        store = _populate_external_milvus(store_config, archive)
        index_build_ms = (time.perf_counter() - index_build_started) * 1000
        index_evidence = _index_evidence(store)

        for role, (adapter_path, adapter_sha) in adapters.items():
            server_log = job_root / f"{role}.http_server.log"
            server_command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "serve",
                "--base-model", config["vlm"]["base_model"],
                "--base-revision", config["vlm"]["base_revision"],
                "--adapter-path", str(adapter_path.resolve()),
                "--adapter-sha256", adapter_sha,
                "--clip-model", config["search"]["embedding_model"],
                "--milvus-uri", milvus_uri,
                "--milvus-config-json", json.dumps(store_config, sort_keys=True),
                "--port", str(http_port),
                "--max-new-tokens", str(performance["vlm_max_new_tokens"]),
            ]
            startup_started = time.perf_counter()
            with server_log.open("w", encoding="utf-8") as log_handle:
                server = subprocess.Popen(
                    server_command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                try:
                    health = _wait_for_http(http_port, server, timeout_seconds=300)
                    startup_ms = (time.perf_counter() - startup_started) * 1000
                    role_rows, cold = _benchmark_role(
                        role,
                        http_port,
                        locked_request,
                        performance,
                    )
                    raw_rows.extend(role_rows)
                    role_summaries[role] = {
                        "adapter_model_sha256": adapter_sha,
                        "service_startup_cold_ms": startup_ms,
                        "first_request_cold": cold,
                        "health": health,
                        "steady": _summarize_role(role_rows, performance["concurrency"]),
                    }
                finally:
                    _terminate_process_group(server)

        gates = _performance_gates(role_summaries, performance)
        report = {
            "schema_version": "http_milvus_service_benchmark_v4",
            "status": "COMPLETED" if gates["status"] == "PASS" else "NEGATIVE_EXPERIMENT_GATE_FAILED",
            "evidence_class": config["evidence_class"],
            "gate_class": config["gate_class"],
            "scope": {
                "http": "real_loopback_HTTP_FastAPI_Uvicorn",
                "milvus": "external_server_process_single_node_standalone_not_Milvus_Lite",
                "distributed_cluster": "NOT_RUN_MULTI_NODE_CLUSTER",
                "production_sla_supported": False,
                "fresh_test_used": False,
                "development_or_final_consumed": False,
            },
            "denominators": {
                "cold_requests_per_role": int(performance["cold_repetitions"]),
                "warmup_requests_per_role": int(performance["warmup_repetitions"]),
                "steady_batches_per_concurrency_per_role": int(performance["steady_repetitions"]),
                "steady_requests_per_role": sum(
                    int(level) * int(performance["steady_repetitions"])
                    for level in performance["concurrency"]
                ),
            },
            "fixed_input": locked_request["provenance"],
            "configuration": {
                "config_sha256": file_sha256(args.config),
                "source_snapshot_sha256": args.source_snapshot_sha256,
                "base_model": config["vlm"]["base_model"],
                "base_revision": config["vlm"]["base_revision"],
                "clip_model": config["search"]["embedding_model"],
                "milvus_server": performance["milvus_server"],
                "milvus_rpm_sha256": file_sha256(args.milvus_rpm),
                "retrieval_archive_sha256": file_sha256(args.retrieval_archive),
                "milvus_service_startup_cold_ms": milvus_startup_ms,
                "milvus_collection_build_and_load_ms": index_build_ms,
                "milvus_collection": index_evidence,
                "concurrency": performance["concurrency"],
                "vlm_max_new_tokens": performance["vlm_max_new_tokens"],
            },
            "hardware": _hardware(),
            "roles": role_summaries,
            "fixed_gates": gates,
            "raw_result_sha256": canonical_json_sha256(raw_rows),
        }
        report["artifact_sha256"] = canonical_json_sha256(report)
        _write_jsonl_exclusive(args.raw_output, raw_rows)
        _write_json_exclusive(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        if milvus_process is not None:
            _terminate_process_group(milvus_process)
        _persist_log_tails(job_root, args.output.with_suffix(".logs.json"))
        jobfs_root = args.jobfs.resolve()
        job_root.relative_to(jobfs_root)
        shutil.rmtree(job_root, ignore_errors=True)


def serve_http(args: argparse.Namespace) -> None:
    _verify_file(args.adapter_path / "adapter_model.safetensors", args.adapter_sha256, "adapter")
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from PIL import Image
    from transformers import AutoProcessor, CLIPModel

    store_config = json.loads(args.milvus_config_json)
    store = OTAMilvusVectorStore(store_config)
    ready, message = store.ready()
    if not ready:
        raise RuntimeError(message)
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_processor = AutoProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(clip_device).eval()
    vlm_args = SimpleNamespace(
        base_model=args.base_model,
        base_revision=args.base_revision,
        adapter_path=args.adapter_path,
    )
    vlm_model, vlm_processor, _ = _load_model(vlm_args)
    inference_lock = threading.Lock()
    app = FastAPI(title="Trip OTA service benchmark v4")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ready",
            "pid": os.getpid(),
            "milvus_uri": args.milvus_uri,
            "milvus_server_mode": True,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    @app.post("/admin/reset-peak")
    def reset_peak() -> dict[str, bool]:
        with inference_lock:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        return {"reset": True}

    @app.post("/v1/search-understand")
    def search_understand(payload: dict[str, Any]) -> dict[str, Any]:
        request_started = time.perf_counter()
        wait_started = time.perf_counter()
        with inference_lock:
            queue_wait_ms = (time.perf_counter() - wait_started) * 1000
            try:
                raw_image = base64.b64decode(str(payload["image_base64"]), validate=True)
                image = Image.open(io.BytesIO(raw_image)).convert("RGB")
                clip_started = time.perf_counter()
                clip_inputs = clip_processor(images=image, return_tensors="pt")
                clip_inputs = {key: value.to(clip_device) for key, value in clip_inputs.items()}
                with torch.inference_mode():
                    embedding = clip_model.get_image_features(**clip_inputs)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                _cuda_sync(torch)
                clip_ms = (time.perf_counter() - clip_started) * 1000

                milvus_started = time.perf_counter()
                result = store.search(
                    embedding[0].detach().float().cpu().tolist(),
                    top_k=20,
                    filters=dict(payload.get("filters") or {}),
                )
                milvus_ms = (time.perf_counter() - milvus_started) * 1000

                rerank_started = time.perf_counter()
                hits = _flatten_hits(result)
                reranked = sorted(
                    hits,
                    key=lambda hit: -(
                        float(hit.get("similarity", -1.0))
                        + 0.02 * float(hit.get("star_rating", 0.0)) / 5.0
                    ),
                )[:10]
                rerank_ms = (time.perf_counter() - rerank_started) * 1000

                vlm_started = time.perf_counter()
                vlm_text, output_tokens = _generate_vlm(
                    vlm_model,
                    vlm_processor,
                    torch,
                    image,
                    str(payload["vlm_prompt"]),
                    args.max_new_tokens,
                )
                _cuda_sync(torch)
                vlm_ms = (time.perf_counter() - vlm_started) * 1000
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"pipeline failure: {type(exc).__name__}: {exc}") from exc
            return {
                "request_lock_sha256": payload.get("request_lock_sha256"),
                "top_image_ids": [hit.get("image_id") for hit in reranked],
                "vlm_output": vlm_text,
                "output_tokens": output_tokens,
                "timing": {
                    "queue_wait_ms": queue_wait_ms,
                    "clip_encode_ms": clip_ms,
                    "milvus_query_ms": milvus_ms,
                    "rerank_ms": rerank_ms,
                    "vlm_inference_ms": vlm_ms,
                    "service_total_ms": (time.perf_counter() - request_started) * 1000,
                },
                "gpu_memory": {
                    "allocated_mib": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
                    "reserved_mib": torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
                },
            }

    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, log_level="warning")


def _load_locked_training_request(config: dict[str, Any], bundle_dir: Path) -> dict[str, Any]:
    expected_lock = _load_json(Path(config["pool"]["committed_lock"]))
    actual_lock = _load_json(bundle_dir / "bundle_lock.json")
    if actual_lock != expected_lock:
        raise ValueError("generated bundle lock differs from committed lock")
    manifest_path = bundle_dir / "search_training_manifest.jsonl"
    records = load_jsonl(manifest_path)
    split_lock = expected_lock["search"]["training"]
    if (
        len(records) != split_lock["query_support"]
        or canonical_json_sha256(records) != split_lock["query_manifest_canonical_sha256"]
        or file_sha256(manifest_path) != split_lock["query_manifest_file_sha256"]
    ):
        raise ValueError("training search manifest differs from committed lock")
    fixed = config["performance"]["fixed_input"]
    matches = [row for row in records if row["query_id"] == fixed["query_id"]]
    if len(matches) != 1 or matches[0].get("split") != "training":
        raise ValueError("fixed performance query is not exactly one training record")
    record = matches[0]
    image_path = bundle_dir / record["image"]["relative_path"]
    _verify_file(image_path, record["image"]["sha256"], "fixed training image")
    prompt = (
        "Inspect this image and emit one compact JSON object describing the visible "
        "business category and evidence. Do not invent hidden facts."
    )
    payload = {
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        "filters": record["requested_filters"],
        "vlm_prompt": prompt,
    }
    payload["request_lock_sha256"] = canonical_json_sha256(payload)
    return {
        "payload": payload,
        "provenance": {
            "split": "training",
            "query_id": record["query_id"],
            "query_sha256": record["query_sha256"],
            "source_id": record["source"]["source_id"],
            "source_record_sha256": record["source_record_sha256"],
            "image_sha256": record["image"]["sha256"],
            "manifest_file_sha256": file_sha256(manifest_path),
            "request_lock_sha256": payload["request_lock_sha256"],
            "label_provenance": "synthetic_training_query_for_performance_only",
        },
    }


def _benchmark_role(
    role: str,
    port: int,
    locked_request: dict[str, Any],
    performance: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import httpx

    url = f"http://127.0.0.1:{port}/v1/search-understand"
    reset_url = f"http://127.0.0.1:{port}/admin/reset-peak"
    payload = locked_request["payload"]
    timeout = httpx.Timeout(300.0)
    with httpx.Client(timeout=timeout) as client:
        cold = _one_http_request(client, url, payload, role, 1, "cold", 0, 0)
        for warmup in range(int(performance["warmup_repetitions"])):
            _one_http_request(client, url, payload, role, 1, "warmup", warmup, 0)
        rows = []
        for concurrency in [int(value) for value in performance["concurrency"]]:
            reset_response = client.post(reset_url)
            reset_response.raise_for_status()
            group_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for batch in range(int(performance["steady_repetitions"])):
                    futures = [
                        pool.submit(
                            _one_http_request,
                            client,
                            url,
                            payload,
                            role,
                            concurrency,
                            "steady",
                            batch,
                            request_index,
                        )
                        for request_index in range(concurrency)
                    ]
                    rows.extend(future.result() for future in as_completed(futures))
            group_seconds = time.perf_counter() - group_started
            for row in rows:
                if row["concurrency"] == concurrency:
                    row["group_wall_seconds"] = group_seconds
        return rows, cold


def _one_http_request(
    client: Any,
    url: str,
    payload: dict[str, Any],
    role: str,
    concurrency: int,
    phase: str,
    batch: int,
    request_index: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.post(url, json=payload)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        body = response.json()
        if body.get("request_lock_sha256") != payload["request_lock_sha256"]:
            raise ValueError("service response request lock mismatch")
        return {
            "role": role,
            "phase": phase,
            "concurrency": concurrency,
            "batch": batch,
            "request_index": request_index,
            "success": True,
            "http_status": response.status_code,
            "http_e2e_ms": elapsed_ms,
            **body["timing"],
            "transport_overhead_ms": max(0.0, elapsed_ms - float(body["timing"]["service_total_ms"])),
            "output_tokens": body["output_tokens"],
            "gpu_memory": body["gpu_memory"],
            "error": None,
        }
    except Exception as exc:
        return {
            "role": role,
            "phase": phase,
            "concurrency": concurrency,
            "batch": batch,
            "request_index": request_index,
            "success": False,
            "http_status": getattr(locals().get("response"), "status_code", None),
            "http_e2e_ms": (time.perf_counter() - started) * 1000,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _summarize_role(rows: list[dict[str, Any]], concurrency_levels: list[int]) -> dict[str, Any]:
    output = {}
    for concurrency in [int(value) for value in concurrency_levels]:
        selected = [row for row in rows if row["concurrency"] == concurrency]
        successful = [row for row in selected if row["success"]]
        group_seconds = max((float(row["group_wall_seconds"]) for row in selected), default=0.0)
        output[str(concurrency)] = {
            "request_support": len(selected),
            "success_support": len(successful),
            "failure_support": len(selected) - len(successful),
            "failure_rate": (len(selected) - len(successful)) / len(selected) if selected else None,
            "throughput_requests_per_second": len(successful) / group_seconds if group_seconds else None,
            "stage_latency_ms": {
                stage: _stats([float(row[stage]) for row in successful])
                for stage in STAGES
            } if successful else {},
            "peak_vram_mib": max(
                (float(row["gpu_memory"]["peak_allocated_mib"]) for row in successful),
                default=0.0,
            ),
        }
    return output


def _performance_gates(role_summaries: dict[str, Any], performance: dict[str, Any]) -> dict[str, Any]:
    baseline = role_summaries["current_system_repair_checkpoint_87"]["steady"]["1"]
    candidate = role_summaries["targeted_exploration_adapter_v4"]["steady"]["1"]
    baseline_p95 = baseline["stage_latency_ms"].get("http_e2e_ms", {}).get("p95")
    candidate_p95 = candidate["stage_latency_ms"].get("http_e2e_ms", {}).get("p95")
    ratio = candidate_p95 / baseline_p95 if baseline_p95 and candidate_p95 is not None else None
    failure_rates = [
        details["failure_rate"]
        for role in role_summaries.values()
        for details in role["steady"].values()
        if details["failure_rate"] is not None
    ]
    failure_gate = bool(failure_rates) and max(failure_rates) <= float(performance["failure_rate_max"])
    latency_gate = ratio is not None and ratio <= float(
        performance["candidate_to_checkpoint_87_concurrency_1_p95_ratio_max"]
    )
    return {
        "status": "PASS" if failure_gate and latency_gate else "FAIL",
        "failure_rate": {
            "pass": failure_gate,
            "observed_max": max(failure_rates) if failure_rates else None,
            "maximum": performance["failure_rate_max"],
        },
        "candidate_to_baseline_concurrency_1_http_p95": {
            "pass": latency_gate,
            "baseline_p95_ms": baseline_p95,
            "candidate_p95_ms": candidate_p95,
            "observed_ratio": ratio,
            "maximum_ratio": performance["candidate_to_checkpoint_87_concurrency_1_p95_ratio_max"],
        },
        "quality_gate": "SEPARATE_DEVELOPMENT_SELECTION_RECORD_REQUIRED_FOR_PROMOTION",
    }


def _prepare_milvus_runtime(rpm_path: Path, job_root: Path, port_base: int) -> dict[str, Path | int]:
    rootfs = job_root / "milvus-rootfs"
    rootfs.mkdir()
    rpm = subprocess.Popen(["rpm2cpio", str(rpm_path)], stdout=subprocess.PIPE)
    if rpm.stdout is None:
        raise RuntimeError("rpm2cpio did not expose stdout")
    unpack = subprocess.run(
        ["cpio", "-idm", "--quiet"],
        stdin=rpm.stdout,
        cwd=rootfs,
        check=False,
    )
    rpm.stdout.close()
    rpm_code = rpm.wait()
    if rpm_code or unpack.returncode:
        raise RuntimeError(f"Milvus RPM extraction failed: rpm2cpio={rpm_code}, cpio={unpack.returncode}")
    config_dir = rootfs / "etc" / "milvus" / "configs"
    storage = job_root / "milvus-data"
    storage.mkdir()
    config_path = config_dir / "milvus.yaml"
    embed_path = config_dir / "embedEtcd.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    replacements = {
        "localhost:2379": f"localhost:{port_base}",
        "/var/lib/milvus/etcd": str(storage / "etcd"),
        "/etc/milvus/configs/embedEtcd.yaml": str(embed_path),
        "/var/lib/milvus/data/": f"{storage / 'data'}/",
        "/var/lib/milvus/rdb_data": str(storage / "rdb_data"),
        "/tmp/milvus_access": str(storage / "access"),
        "  port: 22125 # TCP port of rootCoord": f"  port: {port_base + 1} # TCP port of rootCoord",
        "  port: 19530 # TCP port of proxy": f"  port: {port_base + 2} # TCP port of proxy",
        "  internalPort: 19529": f"  internalPort: {port_base + 3}",
        "  port: 19531 # TCP port of queryCoord": f"  port: {port_base + 4} # TCP port of queryCoord",
        "  port: 21123 # TCP port of queryNode": f"  port: {port_base + 5} # TCP port of queryNode",
        "  port: 13333 # TCP port of dataCoord": f"  port: {port_base + 6} # TCP port of dataCoord",
        "  port: 21124 # TCP port of dataNode": f"  port: {port_base + 7} # TCP port of dataNode",
        "  port: 22222 # TCP port of streamingNode": f"  port: {port_base + 8} # TCP port of streamingNode",
        "  minSegmentSizeToEnableIndex: 1024": "  minSegmentSizeToEnableIndex: 0",
        "    enabled: true # Whether to enable the http server": "    enabled: false # Whether to enable the http server",
    }
    for old, new in replacements.items():
        if old not in config_text:
            raise ValueError(f"Milvus config template misses expected token: {old}")
        config_text = config_text.replace(old, new, 1)
    config_path.write_text(config_text, encoding="utf-8")
    embed = embed_path.read_text(encoding="utf-8")
    embed = embed.replace("http://0.0.0.0:2379", f"http://127.0.0.1:{port_base}")
    embed_path.write_text(embed, encoding="utf-8")
    return {
        "binary": rootfs / "usr" / "bin" / "milvus",
        "library": rootfs / "usr" / "lib" / "milvus",
        "config_dir": config_dir,
        "log": job_root / "milvus.log",
        "proxy_port": port_base + 2,
    }


def _start_milvus(runtime: dict[str, Path | int]) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["MILVUSCONF"] = str(runtime["config_dir"])
    env["DEPLOY_MODE"] = "STANDALONE"
    env["LD_LIBRARY_PATH"] = str(runtime["library"])
    log_handle = Path(runtime["log"]).open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(runtime["binary"]), "run", "standalone"],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    log_handle.close()
    return process


def _wait_for_milvus(uri: str, process: subprocess.Popen[str], timeout_seconds: int) -> None:
    from pymilvus import MilvusClient

    deadline = time.monotonic() + timeout_seconds
    error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Milvus exited before readiness with code {process.returncode}")
        try:
            client = MilvusClient(uri=uri, timeout=5)
            client.list_collections()
            client.close()
            return
        except Exception as exc:
            error = str(exc)
            time.sleep(1)
    raise TimeoutError(f"Milvus did not become ready: {error}")


def _populate_external_milvus(config: dict[str, Any], archive: dict[str, Any]) -> OTAMilvusVectorStore:
    store = OTAMilvusVectorStore(config)
    store.create_collection()
    rows = []
    for index, metadata in enumerate(archive["metadata"]):
        row = {key: metadata[key] for key in FILTER_FIELDS}
        row["multimodal_vector"] = archive["vectors"][index].astype(float).tolist()
        rows.append(row)
    store.batch_insert(rows)
    store.client.flush(collection_name=store.collection)
    store.build_indexes()
    store.client.load_collection(collection_name=store.collection)
    if store.count_visible_entities() != len(rows):
        raise RuntimeError("external Milvus visible support differs from formal vectors")
    return store


def _index_evidence(store: OTAMilvusVectorStore) -> dict[str, Any]:
    names = store.client.list_indexes(collection_name=store.collection)
    details = {
        name: store.client.describe_index(collection_name=store.collection, index_name=name)
        for name in names
    }
    return {
        "name": store.collection,
        "visible_entities": store.count_visible_entities(),
        "index_names": names,
        "index_details_sha256": canonical_json_sha256(details),
    }


def _store_config(config: dict[str, Any], uri: str) -> dict[str, Any]:
    index = config["search"]["milvus"]
    return {
        "connection": {"uri": uri, "timeout_seconds": 120},
        "collection": {
            "name": COLLECTION,
            "vector_dimension": 512,
            "embedding_model": config["search"]["embedding_model"],
            "consistency_level": "Strong",
        },
        "index": {**index, "scalar_fields": sorted(FILTER_FIELDS)},
    }


def _generate_vlm(
    model: Any,
    processor: Any,
    torch: Any,
    image: Any,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, int]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs["input_ids"], generated)]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text, int(trimmed[0].numel())


def _flatten_hits(result: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {**hit.get("entity", {}), "similarity": float(hit.get("distance", -1.0))}
        for hit in (result[0] if result else [])
    ]


def _wait_for_http(port: int, process: subprocess.Popen[str], timeout_seconds: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_seconds
    error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"HTTP service exited before readiness with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
            if body.get("status") == "ready":
                return body
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"HTTP service did not become ready: {error}")


def _choose_port_block() -> int:
    job_id = int(os.getenv("SLURM_JOB_ID", "0") or 0)
    candidate = 20000 + (job_id % 1500) * 20
    if candidate > 50000:
        candidate = 20000 + (job_id % 1000) * 20
    for offset in range(0, 1000, 40):
        base = candidate + offset
        if base + 20 < 65535 and all(_port_available(base + item) for item in range(0, 21)):
            return base
    raise RuntimeError("could not reserve a free local port block")


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def _cuda_sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"support": 0, "p50": None, "p95": None, "min": None, "max": None}
    p95_index = max(0, min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1))
    return {
        "support": len(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _hardware() -> dict[str, Any]:
    import torch

    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count_visible": torch.cuda.device_count(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_job_partition": os.getenv("SLURM_JOB_PARTITION"),
        "slurm_cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.getenv("SLURM_MEM_PER_NODE"),
    }


def _persist_log_tails(job_root: Path, output: Path, line_limit: int = 200) -> None:
    logs = {}
    if job_root.is_dir():
        for path in sorted(job_root.glob("*.log")):
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            logs[path.name] = lines[-line_limit:]
    if logs and not output.exists():
        _write_json_exclusive(output, {
            "schema_version": "http_milvus_service_benchmark_v4_log_tails",
            "line_limit_per_file": line_limit,
            "logs": logs,
        })


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: {actual}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

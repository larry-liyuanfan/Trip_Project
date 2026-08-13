"""Resumable Week 5 model preannotation, human correction, QC, and dialogue work."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

from src.data.week5_dataset import (
    SCENARIOS,
    Week5DataError,
    append_jsonl,
    candidate_manifest_sha256,
    candidate_payload_sha256,
    iter_jsonl,
    load_pools,
    qc_audit_selected,
    qc_cross_review_selected,
    read_jsonl,
    validate_dialogue,
    validate_dialogue_v2,
    validate_human_annotation,
    write_jsonl_new,
)
from src.evaluation.config import load_evaluation_config
from src.evaluation.manifests import load_configured_manifests
from src.evaluation.prompting import render_standard_prompt
from src.evaluation.results import parse_and_validate_output
from src.evaluation.runner import _build_chat_payload, chat_completions_url, post_chat_completion
from src.evaluation.week4_prompting import load_week4_selection, render_week4_request


_FEWSHOT_PREFIX_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _api_key_available() -> bool:
    direct = os.getenv("MODEL_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    key_file = os.getenv("MODEL_API_KEY_FILE") or os.getenv("DASHSCOPE_API_KEY_FILE")
    return bool(direct or (key_file and Path(key_file).is_file()))


def _endpoint_allows_anonymous_access(base_url: str) -> bool:
    """仅本机回环推理端点允许无密钥访问。"""
    parsed = urlparse(base_url)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def _require_model_access(config: dict[str, Any]) -> None:
    base_url = str(config["runtime"]["base_url"])
    if not _api_key_available() and not _endpoint_allows_anonymous_access(base_url):
        raise Week5DataError(
            "MODEL_API_KEY or MODEL_API_KEY_FILE is required for a non-loopback model endpoint"
        )


def _runtime(root: Path, config: dict[str, Any], scenario: str) -> dict[str, Any]:
    from src.data.yelp_paths import parse_simple_yaml

    runtime_config = config["runtime"]
    inference_path = runtime_config["itinerary_inference_config"] if scenario == "itinerary_planning" else runtime_config["inference_config"]
    model = parse_simple_yaml((root / runtime_config["model_config"]).read_text(encoding="utf-8"))
    inference = parse_simple_yaml((root / inference_path).read_text(encoding="utf-8"))
    generation = {
        name: inference[name]
        for name in (
            "temperature", "top_p", "max_tokens", "repetition_penalty",
            "enable_thinking",
        )
        if name in inference
    }
    live_base_url = os.getenv(
        "WEEK5_MODEL_BASE_URL_OVERRIDE", runtime_config["base_url"]
    ).strip()
    if not _endpoint_allows_anonymous_access(live_base_url) and not _api_key_available():
        raise Week5DataError(
            "MODEL_API_KEY or MODEL_API_KEY_FILE is required for a non-loopback runtime override"
        )
    return {
        "model_name": model["served_model_name"],
        "served_model_name": model["served_model_name"],
        "model_config": model,
        "generation": generation,
        # 部署端点属于运行环境，不参与已冻结 run 的配置身份；只允许显式环境变量覆盖。
        "live_base_url": live_base_url,
        "timeout_seconds": runtime_config["timeout_seconds"],
    }


def _fewshot_context(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    selection = load_week4_selection(root / "configs/evaluation/week4_prompt_selection_v2.json")
    dev_config = load_evaluation_config(root / "configs/evaluation_week4_demo_dev_v1.yaml")
    configured = load_configured_manifests(dev_config, root=root)
    records = {row["sample_id"]: row for scenario in SCENARIOS for row in configured[scenario]}
    return selection, records


def _render_preannotation(
    root: Path, config: dict[str, Any], candidate: dict[str, Any],
    selection: dict[str, Any], examples: dict[str, dict[str, Any]],
    *, prompt_version: str | None = None,
) -> dict[str, Any]:
    scenario = candidate["scenario"]
    version = prompt_version or config["prompt_versions"][scenario]
    if version in {"standardized_v2", "standardized_v4"}:
        return render_standard_prompt(root, scenario, candidate["input"], version=version)
    cache_key = (scenario, version)
    cached = _FEWSHOT_PREFIX_CACHE.get(cache_key)
    if cached is None:
        first = render_week4_request(
            root, scenario, candidate["input"], prompt_version=version,
            selection=selection, records_by_id=examples,
        )
        cached = {
            "messages": copy.deepcopy(first["messages"][:-1]),
            "example_ids": copy.deepcopy(first.get("example_ids", [])),
            "example_count": first.get("example_count"),
            "example_collage_path": first.get("example_collage_path"),
            "example_collage_sha256": first.get("example_collage_sha256"),
        }
        _FEWSHOT_PREFIX_CACHE[cache_key] = cached
    rendered = render_standard_prompt(
        root, scenario, candidate["input"], version="week4_optimized_v2"
    )
    rendered["messages"] = copy.deepcopy(cached["messages"]) + [
        copy.deepcopy(rendered["messages"][-1])
    ]
    rendered["prompt_version"] = version
    for name in (
        "example_ids", "example_count", "example_collage_path",
        "example_collage_sha256",
    ):
        rendered[name] = copy.deepcopy(cached[name])
    return rendered


def _parse_and_validate_week5_output(
    root: Path,
    scenario: str,
    raw_output: str,
    schema_version: str,
) -> dict[str, Any]:
    """Apply only lossless Week 5 type normalization before Schema validation."""
    parsed = parse_and_validate_output(root, scenario, raw_output, schema_version)
    output = parsed.get("parsed_output")
    if not parsed["json_valid"] or not isinstance(output, dict):
        return parsed

    ocr_text = output.get("ocr_text")
    if scenario != "after_sales" or not isinstance(ocr_text, (str, dict)):
        return parsed

    # OCR 偶尔被压成字符串或键值对象；仅做可逆包装，不猜测或丢弃内容。
    normalized = dict(output)
    normalized["ocr_text"] = [
        ocr_text
        if isinstance(ocr_text, str)
        else json.dumps(
            ocr_text, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    ]
    return parse_and_validate_output(
        root,
        scenario,
        json.dumps(normalized, ensure_ascii=False),
        schema_version,
    )


def run_preannotation(
    root: Path, config: dict[str, Any], scenario: str, *, limit: int | None = None,
    retry_failures: bool = False,
) -> dict[str, int]:
    """Append one durable result per attempted sample and skip completed IDs on resume."""
    if scenario not in SCENARIOS:
        raise Week5DataError(f"unsupported scenario: {scenario}")
    _require_model_access(config)
    output = root / config["paths"]["output_dir"] / "preannotations" / f"{scenario}.jsonl"
    existing = read_jsonl(output)
    completed = {row["sample_id"] for row in existing if row.get("status") == "completed"}
    failed = {row["sample_id"] for row in existing if row.get("status") == "failed"}
    selection, examples = _fewshot_context(root)
    runtime = _runtime(root, config, scenario)
    attempted = succeeded = failed_count = skipped = 0
    pending: list[dict[str, Any]] = []
    for candidate in load_pools(root, config)[scenario]:
        sample_id = candidate["sample_id"]
        if sample_id in completed or (sample_id in failed and not retry_failures):
            skipped += 1
            continue
        if limit is not None and attempted >= limit:
            break
        pending.append(candidate)
        attempted += 1
    if pending:
        # 主线程先构建固定 Few-Shot 拼图和前缀，避免并发首次写入竞态。
        _render_preannotation(root, config, pending[0], selection, examples)

    def execute(candidate: dict[str, Any]) -> dict[str, Any]:
        sample_id = candidate["sample_id"]
        started = time.perf_counter()
        rendered: dict[str, Any] | None = None
        raw_output: str | None = None
        try:
            rendered = _render_preannotation(root, config, candidate, selection, examples)
            payload = _build_chat_payload(root, rendered, runtime)
            response = post_chat_completion(chat_completions_url(runtime["live_base_url"]), payload, runtime["timeout_seconds"])
            raw_output = response["choices"][0]["message"]["content"]
            if not isinstance(raw_output, str):
                raise Week5DataError("model response content must be text")
            parsed = _parse_and_validate_week5_output(
                root,
                scenario,
                raw_output,
                "v2" if scenario == "itinerary_planning" else "v1",
            )
            status = "completed" if parsed["schema_valid"] else "failed"
            error = parsed["error"]
        except Exception as exc:
            parsed = {"parsed_output": None, "json_valid": False, "schema_valid": False}
            status = "failed"
            error = f"model_request_error: {type(exc).__name__}: {exc}"
        return {
            "sample_id": sample_id, "scenario": scenario, "status": status,
            "attempt": 1 + sum(row.get("sample_id") == sample_id for row in existing),
            "model_name": runtime["model_name"], "prompt_version": config["prompt_versions"][scenario],
            "request_sha256": hashlib.sha256(json.dumps(rendered or {}, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "raw_output": raw_output, "parsed_output": parsed["parsed_output"],
            "json_valid": parsed["json_valid"], "schema_valid": parsed["schema_valid"],
            "latency_ms": (time.perf_counter() - started) * 1000,
            "error": error, "timestamp": _now(), "human_completed": False,
        }

    concurrency = int(config["runtime"].get("concurrency", 1))
    if concurrency < 1:
        raise Week5DataError("preannotation concurrency must be positive")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for result in executor.map(execute, pending):
            append_jsonl(output, result)
            if result["status"] == "completed":
                succeeded += 1
            else:
                failed_count += 1
    return {"attempted": attempted, "completed": succeeded, "failed": failed_count, "skipped": skipped}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _safe_run_directory(root: Path, config: dict[str, Any], run_id: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", run_id) is None:
        raise Week5DataError("run_id must contain only letters, numbers, dot, underscore, or dash")
    return root / config["paths"]["output_dir"] / "runs" / run_id


def _pilot_run_identity(
    root: Path,
    config: dict[str, Any],
    run_id: str,
    candidates: list[dict[str, Any]],
    prompt_versions: list[str],
) -> dict[str, Any]:
    runtime = _runtime(root, config, "itinerary_planning")
    pool_path = (
        root
        / config["paths"]["output_dir"]
        / "pools"
        / "itinerary_planning.jsonl"
    )
    return {
        "schema_version": "week5_preannotation_run_v2",
        "run_id": run_id,
        "run_kind": "itinerary_paired_prompt_pilot",
        "dataset_version": config["dataset_version"],
        "scenario": "itinerary_planning",
        "model_name": runtime["model_name"],
        "prompt_versions": prompt_versions,
        "config_sha256": _canonical_sha256(config),
        "candidate_manifest_path": pool_path.relative_to(root).as_posix(),
        "candidate_manifest_sha256": candidate_manifest_sha256(
            root, config, "itinerary_planning"
        ),
        "selected_sample_ids_sha256": _canonical_sha256(
            [candidate["sample_id"] for candidate in candidates]
        ),
        "selected_candidate_payloads_sha256": _canonical_sha256(
            [candidate_payload_sha256(candidate) for candidate in candidates]
        ),
        "selected_count": len(candidates),
        "shard": {"index": 0, "count": 1, "strategy": "ordered_fixed_pair_v1"},
    }


def _prepare_audited_run(
    run_dir: Path,
    identity: dict[str, Any],
    limits: dict[str, Any],
    *,
    resume: bool,
) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists() and not resume:
        raise Week5DataError(f"run already exists; use explicit resume: {run_dir.name}")
    if resume:
        if not manifest_path.is_file():
            raise Week5DataError("resume requires an existing run manifest")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity or existing.get("limits") != limits:
            raise Week5DataError("resume metadata hash or limits do not match the existing run")
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "raw").mkdir()
    for name in ("attempts.jsonl", "results.jsonl", "failures.jsonl"):
        (run_dir / name).touch(exist_ok=False)
    manifest_path.write_text(
        json.dumps(
            {"identity": identity, "limits": limits, "created_at": _now()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
        newline="\n",
    )


def _attempt_error_type(status: str, error: str | None) -> str | None:
    if status == "completed":
        return None
    if error and error.startswith("model_request_error:"):
        return "request_error"
    if error and error.startswith("input_validation_error:"):
        return "input_error"
    if error and ("JSON" in error or "json" in error):
        return "json_parse_error"
    return "schema_error"


def _validate_candidate_images(root: Path, candidate: dict[str, Any]) -> None:
    """在请求模型前拒绝缺失、越界或不可解码的候选图片。"""
    resolved_root = root.resolve()
    images = candidate.get("input", {}).get("images")
    if not isinstance(images, list) or not images:
        raise Week5DataError("candidate requires at least one image")
    for image in images:
        relative = image.get("path") if isinstance(image, dict) else None
        if not isinstance(relative, str) or not relative.strip():
            raise Week5DataError("candidate image path is missing")
        path = (resolved_root / relative).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise Week5DataError(f"candidate image escapes project root: {relative}") from exc
        if not path.is_file():
            raise Week5DataError(f"candidate image is missing: {relative}")
        try:
            with Image.open(path) as opened:
                opened.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise Week5DataError(f"candidate image is unreadable: {relative}") from exc


def _full_run_identity(
    root: Path, config: dict[str, Any], run_id: str,
) -> dict[str, Any]:
    runtimes = {scenario: _runtime(root, config, scenario) for scenario in SCENARIOS}
    return {
        "schema_version": "week5_preannotation_run_v2",
        "run_id": run_id,
        "run_kind": "full_preannotation",
        "dataset_version": config["dataset_version"],
        "model_names": {
            scenario: runtimes[scenario]["model_name"] for scenario in SCENARIOS
        },
        "prompt_versions": {
            scenario: config["prompt_versions"][scenario] for scenario in SCENARIOS
        },
        "config_sha256": _canonical_sha256(config),
        "candidate_manifests": {
            scenario: {
                "path": (
                    Path(config["paths"]["output_dir"])
                    / "pools"
                    / f"{scenario}.jsonl"
                ).as_posix(),
                "sha256": candidate_manifest_sha256(root, config, scenario),
            }
            for scenario in SCENARIOS
        },
        "sharding": {"strategy": "ordered_fixed_size_v1"},
    }


def run_full_preannotation(
    root: Path,
    config: dict[str, Any],
    run_id: str,
    *,
    resume: bool = False,
    retry_failures: bool = False,
) -> dict[str, Any]:
    """执行可审计、不可覆盖并可安全恢复的三场景全量预标注。"""
    _require_model_access(config)
    full = config.get("full_preannotation", {})
    shard_size = int(full.get("shard_size", 500))
    max_retries = int(full.get("max_retries", 2))
    max_consecutive_request_failures = int(
        full.get("max_consecutive_request_failures", 20)
    )
    if (
        shard_size < 1
        or max_retries < 0
        or max_consecutive_request_failures < 1
    ):
        raise Week5DataError("full preannotation shard_size/max_retries is invalid")
    identity = _full_run_identity(root, config, run_id)
    limits = {
        "shard_size": shard_size,
        "max_retries": max_retries,
        "max_consecutive_request_failures": max_consecutive_request_failures,
        "concurrency": int(config["runtime"].get("concurrency", 1)),
    }
    if limits["concurrency"] < 1:
        raise Week5DataError("preannotation concurrency must be positive")
    run_dir = _safe_run_directory(root, config, run_id)
    _prepare_audited_run(run_dir, identity, limits, resume=resume)
    attempts_path = run_dir / "attempts.jsonl"
    results_path = run_dir / "results.jsonl"
    failures_path = run_dir / "failures.jsonl"
    prior_attempts = read_jsonl(attempts_path)
    successful = {row["sample_id"] for row in read_jsonl(results_path)}
    failed_terminal = {row["sample_id"] for row in read_jsonl(failures_path)}
    prior_counts: dict[str, int] = {}
    for row in prior_attempts:
        prior_counts[row["sample_id"]] = max(
            prior_counts.get(row["sample_id"], 0), int(row.get("attempt", 0))
        )
    selection, examples = _fewshot_context(root)
    process_started = time.monotonic()
    process_attempts = process_completed = process_failed = process_skipped = 0
    consecutive_request_failures = 0

    for scenario in SCENARIOS:
        runtime = _runtime(root, config, scenario)
        pool_path = root / config["paths"]["output_dir"] / "pools" / f"{scenario}.jsonl"
        pending: list[tuple[int, dict[str, Any]]] = []
        scenario_seen = scenario_completed = scenario_failed = scenario_skipped = 0

        def execute(item: tuple[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            index, candidate = item
            sample_id = candidate["sample_id"]
            records: list[dict[str, Any]] = []
            final: dict[str, Any] | None = None
            start_attempt = prior_counts.get(sample_id, 0) + 1
            for retry_index in range(max_retries + 1):
                attempt_number = start_attempt + retry_index
                while (
                    run_dir / "raw" / scenario / sample_id
                    / f"attempt_{attempt_number:03d}.txt"
                ).exists():
                    # 进程可能在写 raw 后、追加 attempt 前中断；保留孤立文件并换新编号。
                    attempt_number += 1
                started = time.perf_counter()
                rendered: dict[str, Any] | None = None
                raw_output: str | None = None
                usage: dict[str, Any] | None = None
                try:
                    _validate_candidate_images(root, candidate)
                    rendered = _render_preannotation(
                        root, config, candidate, selection, examples
                    )
                    payload = _build_chat_payload(root, rendered, runtime)
                    response = post_chat_completion(
                        chat_completions_url(runtime["live_base_url"]),
                        payload,
                        runtime["timeout_seconds"],
                    )
                    raw_output = response["choices"][0]["message"]["content"]
                    usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
                    if not isinstance(raw_output, str):
                        raise Week5DataError("model response content must be text")
                    parsed = _parse_and_validate_week5_output(
                        root, scenario, raw_output,
                        "v2" if scenario == "itinerary_planning" else "v1",
                    )
                    status = "completed" if parsed["schema_valid"] else "failed"
                    error = parsed["error"]
                except Exception as exc:
                    parsed = {
                        "parsed_output": None,
                        "json_valid": False,
                        "schema_valid": False,
                    }
                    status = "failed"
                    prefix = (
                        "input_validation_error"
                        if isinstance(exc, Week5DataError)
                        and str(exc).startswith("candidate ")
                        else "model_request_error"
                    )
                    error = f"{prefix}: {type(exc).__name__}: {exc}"
                raw_relative: str | None = None
                if raw_output is not None:
                    raw_path = (
                        run_dir / "raw" / scenario / sample_id
                        / f"attempt_{attempt_number:03d}.txt"
                    )
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    with raw_path.open("x", encoding="utf-8", newline="\n") as handle:
                        handle.write(raw_output)
                    raw_relative = raw_path.relative_to(run_dir).as_posix()
                record = {
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "scenario": scenario,
                    "pool_index": index,
                    "shard_index": index // shard_size,
                    "prompt_version": config["prompt_versions"][scenario],
                    "attempt": attempt_number,
                    "retry_count": attempt_number - 1,
                    "status": status,
                    "error_type": _attempt_error_type(status, error),
                    "error": error,
                    "input_sha256": _canonical_sha256(candidate["input"]),
                    "candidate_sha256": candidate_payload_sha256(candidate),
                    "request_sha256": _canonical_sha256(rendered or {}),
                    "raw_output_path": raw_relative,
                    "parsed_output": parsed["parsed_output"],
                    "json_valid": parsed["json_valid"],
                    "schema_valid": parsed["schema_valid"],
                    "usage": usage,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "timestamp": _now(),
                    "human_completed": False,
                }
                records.append(record)
                final = record
                if status == "completed" or record["error_type"] == "input_error":
                    break
            assert final is not None
            return records, final

        def flush(batch: list[tuple[int, dict[str, Any]]]) -> None:
            nonlocal process_attempts, process_completed, process_failed
            nonlocal scenario_completed, scenario_failed
            nonlocal consecutive_request_failures
            if not batch:
                return
            concurrency = limits["concurrency"]
            for offset in range(0, len(batch), concurrency):
                window = batch[offset:offset + concurrency]
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    completed_window = list(executor.map(execute, window))
                for records, final in completed_window:
                    for record in records:
                        append_jsonl(attempts_path, record)
                        process_attempts += 1
                        if record.get("error_type") == "request_error":
                            consecutive_request_failures += 1
                        else:
                            consecutive_request_failures = 0
                    if final["status"] == "completed":
                        append_jsonl(results_path, final)
                        successful.add(final["sample_id"])
                        process_completed += 1
                        scenario_completed += 1
                    else:
                        append_jsonl(failures_path, final)
                        failed_terminal.add(final["sample_id"])
                        process_failed += 1
                        scenario_failed += 1
                    _atomic_write_json(
                        run_dir / "checkpoint.json",
                        {
                            "run_id": run_id,
                            "scenario": scenario,
                            "last_sample_id": final["sample_id"],
                            "last_pool_index": final["pool_index"],
                            "last_shard_index": final["shard_index"],
                            "process_attempts": process_attempts,
                            "process_completed": process_completed,
                            "process_failed": process_failed,
                            "consecutive_request_failures": consecutive_request_failures,
                            "updated_at": _now(),
                        },
                    )
                    if consecutive_request_failures >= max_consecutive_request_failures:
                        raise Week5DataError(
                            "full preannotation stopped after consecutive model request failures"
                        )

        for index, candidate in enumerate(iter_jsonl(pool_path)):
            scenario_seen += 1
            sample_id = candidate["sample_id"]
            if sample_id in successful or (
                sample_id in failed_terminal and not retry_failures
            ):
                process_skipped += 1
                scenario_skipped += 1
                continue
            pending.append((index, candidate))
            if len(pending) >= shard_size:
                flush(pending)
                pending = []
        flush(pending)
        _atomic_write_json(
            run_dir / f"summary_{scenario}.json",
            {
                "scenario": scenario,
                "pool_records": scenario_seen,
                "completed_this_process": scenario_completed,
                "failed_this_process": scenario_failed,
                "skipped_this_process": scenario_skipped,
                "updated_at": _now(),
            },
        )

    all_results = read_jsonl(results_path)
    latest_failures = {
        row["sample_id"]: row for row in read_jsonl(failures_path)
        if row["sample_id"] not in {item["sample_id"] for item in all_results}
    }
    by_scenario = {
        scenario: {
            "completed": sum(row["scenario"] == scenario for row in all_results),
            "failed": sum(row["scenario"] == scenario for row in latest_failures.values()),
        }
        for scenario in SCENARIOS
    }
    summary = {
        "run_id": run_id,
        "status": "completed" if sum(v["completed"] for v in by_scenario.values()) == sum(config["targets"][s] for s in SCENARIOS) else "partial",
        "by_scenario": by_scenario,
        "attempts_this_process": process_attempts,
        "completed_this_process": process_completed,
        "failed_this_process": process_failed,
        "skipped_this_process": process_skipped,
        "elapsed_seconds_this_process": time.monotonic() - process_started,
        "updated_at": _now(),
    }
    _atomic_write_json(run_dir / "summary.json", summary)
    return summary


def _pilot_summary(
    attempts: list[dict[str, Any]],
    prompt_versions: list[str],
    selected_count: int,
    stop_reason: str | None,
) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    attempt_keys = {(row["sample_id"], row["prompt_version"]) for row in attempts}
    paired = sum(
        all((sample_id, prompt) in attempt_keys for prompt in prompt_versions)
        for sample_id in {row["sample_id"] for row in attempts}
    )
    for prompt in prompt_versions:
        rows = [row for row in attempts if row["prompt_version"] == prompt]
        token_values = [
            row.get("usage", {}).get("total_tokens")
            for row in rows
            if isinstance(row.get("usage"), dict)
            and isinstance(row.get("usage", {}).get("total_tokens"), (int, float))
        ]
        aggregates[prompt] = {
            "requests": len(rows),
            "request_failures": sum(row.get("error_type") == "request_error" for row in rows),
            "json_valid": sum(row.get("json_valid") is True for row in rows),
            "schema_valid": sum(row.get("schema_valid") is True for row in rows),
            "schema_rate": sum(row.get("schema_valid") is True for row in rows) / len(rows) if rows else 0.0,
            "request_failure_rate": sum(row.get("error_type") == "request_error" for row in rows) / len(rows) if rows else 1.0,
            "mean_total_tokens": sum(token_values) / len(token_values) if token_values else None,
            "mean_latency_ms": sum(float(row["latency_ms"]) for row in rows) / len(rows) if rows else None,
            "business_quality": None,
        }
    complete = paired == selected_count and all(
        aggregates[prompt]["requests"] == selected_count for prompt in prompt_versions
    )
    selected = "fewshot_4_v2"
    selection_status = "default_incomplete"
    if complete and stop_reason is None:
        def rank(prompt: str) -> tuple[float, float, float, float, int]:
            item = aggregates[prompt]
            return (
                -float(item["schema_rate"]),
                float(item["request_failure_rate"]),
                float(item["mean_total_tokens"]) if item["mean_total_tokens"] is not None else float("inf"),
                float(item["mean_latency_ms"]) if item["mean_latency_ms"] is not None else float("inf"),
                prompt_versions.index(prompt),
            )
        selected = min(prompt_versions, key=rank)
        selection_status = "selected_by_structural_tiebreak"
    return {
        "selected_prompt": selected,
        "selection_status": selection_status,
        "selection_rule": [
            "business_quality_unavailable_tied",
            "schema_rate_desc",
            "request_failure_rate_asc",
            "mean_total_tokens_asc",
            "mean_latency_ms_asc",
        ],
        "paired_samples": paired,
        "selected_samples": selected_count,
        "stop_reason": stop_reason,
        "prompt_metrics": aggregates,
    }


def run_itinerary_paired_prompt_pilot(
    root: Path,
    config: dict[str, Any],
    run_id: str,
    *,
    limit: int = 30,
    resume: bool = False,
) -> dict[str, Any]:
    """执行 Project Control 授权的唯一 4B 双 Prompt pilot。"""
    _require_model_access(config)
    pilot = config.get("pilot", {})
    prompt_versions = list(pilot.get("itinerary_prompt_versions", []))
    if prompt_versions != ["fewshot_4_v2", "standardized_v4"]:
        raise Week5DataError("paired pilot prompt versions must be fixed and ordered")
    max_samples = int(pilot.get("max_unique_samples", 0))
    max_requests = int(pilot.get("max_total_requests", 0))
    if limit < 1 or limit > max_samples or max_requests > 60 or limit * 2 > max_requests:
        raise Week5DataError("paired pilot exceeds the approved sample or request limit")
    if float(pilot.get("max_gpu_hours", 0)) > 1.0 or float(pilot.get("max_cost_cny", 0)) > 20.0:
        raise Week5DataError("paired pilot exceeds the approved time or cost limit")
    pool_path = (
        root / config["paths"]["output_dir"] / "pools" / "itinerary_planning.jsonl"
    )
    candidates = list(islice(iter_jsonl(pool_path), limit))
    if len(candidates) != limit:
        raise Week5DataError("itinerary candidate pool is shorter than the pilot limit")
    identity = _pilot_run_identity(root, config, run_id, candidates, prompt_versions)
    limits = {
        "max_unique_samples": limit,
        "max_total_requests": max_requests,
        "max_gpu_hours": float(pilot["max_gpu_hours"]),
        "estimated_hourly_cost_cny": float(pilot["estimated_hourly_cost_cny"]),
        "max_cost_cny": float(pilot["max_cost_cny"]),
        "early_stop_pairs": int(pilot["early_stop_pairs"]),
        "max_failure_rate_after_early_stop": float(pilot["max_failure_rate_after_early_stop"]),
    }
    run_dir = _safe_run_directory(root, config, run_id)
    _prepare_audited_run(run_dir, identity, limits, resume=resume)
    attempts_path = run_dir / "attempts.jsonl"
    attempts = read_jsonl(attempts_path)
    attempted_keys = {(row["sample_id"], row["prompt_version"]) for row in attempts}
    selection, examples = _fewshot_context(root)
    runtime = _runtime(root, config, "itinerary_planning")
    started = time.monotonic()
    stop_reason: str | None = None

    for candidate in candidates:
        for prompt_version in prompt_versions:
            key = (candidate["sample_id"], prompt_version)
            if key in attempted_keys:
                continue
            elapsed = time.monotonic() - started
            estimated_cost = elapsed / 3600 * limits["estimated_hourly_cost_cny"]
            if len(attempts) >= limits["max_total_requests"]:
                stop_reason = "request_limit_reached"
                break
            if elapsed >= limits["max_gpu_hours"] * 3600:
                stop_reason = "gpu_time_limit_reached"
                break
            if estimated_cost >= limits["max_cost_cny"]:
                stop_reason = "cost_limit_reached"
                break
            request_started = time.perf_counter()
            rendered: dict[str, Any] | None = None
            raw_output: str | None = None
            usage: dict[str, Any] | None = None
            try:
                rendered = _render_preannotation(
                    root, config, candidate, selection, examples,
                    prompt_version=prompt_version,
                )
                payload = _build_chat_payload(root, rendered, runtime)
                response = post_chat_completion(
                    chat_completions_url(runtime["live_base_url"]),
                    payload,
                    runtime["timeout_seconds"],
                )
                raw_output = response["choices"][0]["message"]["content"]
                usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
                if not isinstance(raw_output, str):
                    raise Week5DataError("model response content must be text")
                parsed = _parse_and_validate_week5_output(
                    root, "itinerary_planning", raw_output, "v2"
                )
                status = "completed" if parsed["schema_valid"] else "failed"
                error = parsed["error"]
            except Exception as exc:
                parsed = {"parsed_output": None, "json_valid": False, "schema_valid": False}
                status = "failed"
                error = f"model_request_error: {type(exc).__name__}: {exc}"
            attempt_number = 1 + sum(
                row["sample_id"] == candidate["sample_id"]
                and row["prompt_version"] == prompt_version
                for row in attempts
            )
            raw_relative: str | None = None
            if raw_output is not None:
                raw_path = (
                    run_dir / "raw" / candidate["sample_id"] / prompt_version
                    / f"attempt_{attempt_number:03d}.txt"
                )
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                with raw_path.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(raw_output)
                raw_relative = raw_path.relative_to(run_dir).as_posix()
            attempt = {
                "run_id": run_id,
                "sample_id": candidate["sample_id"],
                "scenario": "itinerary_planning",
                "prompt_version": prompt_version,
                "attempt": attempt_number,
                "retry_count": attempt_number - 1,
                "status": status,
                "error_type": _attempt_error_type(status, error),
                "error": error,
                "input_sha256": _canonical_sha256(candidate["input"]),
                "candidate_sha256": candidate_payload_sha256(candidate),
                "request_sha256": _canonical_sha256(rendered or {}),
                "raw_output_path": raw_relative,
                "parsed_output": parsed["parsed_output"],
                "json_valid": parsed["json_valid"],
                "schema_valid": parsed["schema_valid"],
                "usage": usage,
                "latency_ms": (time.perf_counter() - request_started) * 1000,
                "timestamp": _now(),
            }
            append_jsonl(attempts_path, attempt)
            append_jsonl(
                run_dir / ("results.jsonl" if status == "completed" else "failures.jsonl"),
                attempt,
            )
            attempts.append(attempt)
            attempted_keys.add(key)
            _atomic_write_json(
                run_dir / "checkpoint.json",
                {
                    "run_id": run_id,
                    "attempted_requests": len(attempts),
                    "completed": sum(row["status"] == "completed" for row in attempts),
                    "failed": sum(row["status"] == "failed" for row in attempts),
                    "last_sample_id": candidate["sample_id"],
                    "last_prompt_version": prompt_version,
                    "updated_at": _now(),
                },
            )
        if stop_reason:
            break
        early_pairs = limits["early_stop_pairs"]
        if len(attempts) >= early_pairs * len(prompt_versions):
            first_ids = {candidate["sample_id"] for candidate in candidates[:early_pairs]}
            early = [row for row in attempts if row["sample_id"] in first_ids]
            if len(early) >= early_pairs * len(prompt_versions):
                request_failures = sum(row.get("error_type") == "request_error" for row in early)
                if request_failures / len(early) > limits["max_failure_rate_after_early_stop"]:
                    stop_reason = "early_request_failure_rate_exceeded"
                    break

    summary = _pilot_summary(attempts, prompt_versions, limit, stop_reason)
    summary.update({
        "run_id": run_id,
        "total_requests": len(attempts),
        "elapsed_seconds_this_process": time.monotonic() - started,
        "estimated_compute_cost_cny_this_process": (
            (time.monotonic() - started) / 3600 * limits["estimated_hourly_cost_cny"]
        ),
    })
    _atomic_write_json(run_dir / "summary.json", summary)
    return summary


def _load_audited_pilot_winner(
    root: Path, config: dict[str, Any], run_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """读取已完成 pilot 的胜出预标注，并重新验证候选绑定。"""
    run_dir = _safe_run_directory(root, config, run_id)
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise Week5DataError("audited pilot manifest or summary is missing")
    identity = json.loads(manifest_path.read_text(encoding="utf-8")).get("identity", {})
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        identity.get("run_id") != run_id
        or identity.get("scenario") != "itinerary_planning"
        or identity.get("candidate_manifest_sha256")
        != candidate_manifest_sha256(root, config, "itinerary_planning")
    ):
        raise Week5DataError("audited pilot identity or candidate manifest mismatch")
    selected_prompt = summary.get("selected_prompt")
    if (
        summary.get("stop_reason") is not None
        or summary.get("paired_samples") != identity.get("selected_count")
        or selected_prompt not in identity.get("prompt_versions", [])
    ):
        raise Week5DataError("audited pilot is incomplete or has no valid winner")
    rows = [
        row for row in read_jsonl(run_dir / "attempts.jsonl")
        if row.get("prompt_version") == selected_prompt
        and row.get("status") == "completed"
        and row.get("schema_valid") is True
    ]
    if len(rows) != identity.get("selected_count"):
        raise Week5DataError("selected pilot winner lacks a valid output for every sample")
    return {"identity": identity, "summary": summary}, rows


def export_audited_pilot_annotation_packet(
    root: Path, config: dict[str, Any], run_id: str, output: Path,
) -> dict[str, Any]:
    """导出仅等待真实人工填写的胜出 pilot 任务包。"""
    if not output.is_absolute():
        output = root / output
    metadata, attempts = _load_audited_pilot_winner(root, config, run_id)
    candidates = {
        row["sample_id"]: row
        for row in load_pools(root, config)["itinerary_planning"]
    }

    def tasks() -> Any:
        for attempt in attempts:
            candidate = candidates.get(attempt["sample_id"])
            if candidate is None or candidate_payload_sha256(candidate) != attempt["candidate_sha256"]:
                raise Week5DataError("pilot task candidate hash mismatch")
            yield {
                "schema_version": "week5_human_annotation_task_v2",
                "sample_id": candidate["sample_id"],
                "scenario": "itinerary_planning",
                "candidate_sha256": attempt["candidate_sha256"],
                "candidate_manifest_sha256": metadata["identity"]["candidate_manifest_sha256"],
                "input": candidate["input"],
                "sampling_metadata": candidate.get("sampling_metadata", {}),
                "isolation": candidate.get("isolation", {}),
                "model_preannotation": attempt["parsed_output"],
                "model_preannotation_status": "completed",
                "model_run_id": run_id,
                "model_prompt_version": metadata["summary"]["selected_prompt"],
                "workflow_status": "awaiting_human_annotation",
                "annotator": None,
                "human_annotation": None,
                "corrected_at": None,
                "revision_history": [],
                "self_review_confirmed": False,
                "review_session_id": None,
            }

    count = write_jsonl_new(output, tasks())
    return {
        "exported": count,
        "run_id": run_id,
        "prompt_version": metadata["summary"]["selected_prompt"],
        "workflow_status": "awaiting_human_annotation",
        "output": output.relative_to(root).as_posix() if output.is_relative_to(root) else str(output),
    }


def apply_human_corrections(
    root: Path, config: dict[str, Any], scenario: str, input_path: Path,
) -> dict[str, int]:
    """Append a real human correction and its explicit inline self-review."""
    if scenario not in SCENARIOS:
        raise Week5DataError(f"unsupported scenario: {scenario}")
    candidates = {row["sample_id"]: row for row in load_pools(root, config)[scenario]}
    preannotation_path = root / config["paths"]["output_dir"] / "preannotations" / f"{scenario}.jsonl"
    preannotated = {
        row["sample_id"] for row in read_jsonl(preannotation_path)
        if row.get("status") == "completed" and row.get("schema_valid") is True
    }
    submitted = read_jsonl(input_path)
    output = root / config["paths"]["output_dir"] / "annotations" / f"{scenario}.jsonl"
    existing = read_jsonl(output)
    revisions: dict[str, int] = {}
    for row in existing:
        revisions[row["sample_id"]] = max(revisions.get(row["sample_id"], 0), int(row.get("revision", 1)))
    seen: set[str] = set()
    validated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    audited_runs: dict[str, set[str]] = {}
    for row in submitted:
        sample_id = row.get("sample_id")
        if sample_id in seen or sample_id not in candidates:
            raise Week5DataError(f"duplicate or unknown human sample: {sample_id}")
        model_run_id = row.get("model_run_id")
        if isinstance(model_run_id, str) and model_run_id:
            if model_run_id not in audited_runs:
                _, audited = _load_audited_pilot_winner(root, config, model_run_id)
                audited_runs[model_run_id] = {item["sample_id"] for item in audited}
            has_preannotation = sample_id in audited_runs[model_run_id]
        else:
            has_preannotation = sample_id in preannotated
        if not has_preannotation:
            raise Week5DataError(f"human correction requires completed model preannotation: {sample_id}")
        seen.add(sample_id)
        annotator = row.get("annotator")
        corrected_at = row.get("corrected_at")
        review_session_id = row.get("review_session_id")
        if not isinstance(annotator, str) or not annotator.strip() or not isinstance(corrected_at, str) or not corrected_at.strip():
            raise Week5DataError("human correction requires annotator and corrected_at")
        if row.get("self_review_confirmed") is not True:
            raise Week5DataError("human correction requires explicit inline self-review confirmation")
        if not isinstance(review_session_id, str) or not review_session_id.strip():
            raise Week5DataError("human correction requires review_session_id")
        validate_human_annotation(root, scenario, row.get("human_annotation"))
        revision = revisions.get(sample_id, 0) + 1
        annotation_record = {
            "sample_id": sample_id, "scenario": scenario, "annotator": annotator.strip(),
            "human_annotation": copy.deepcopy(row["human_annotation"]),
            "corrected_at": corrected_at, "revision": revision,
            "source": "human_correction", "model_run_id": model_run_id,
            "qc_reset": True, "review_session_id": review_session_id.strip(),
        }
        self_review_record = {
            "sample_id": sample_id, "scenario": scenario,
            "annotation_revision": revision, "stage": "self_review",
            "decision": "pass", "reviewer": annotator.strip(), "issues": [],
            "notes": row.get("self_review_notes"), "reviewed_at": corrected_at,
            "review_session_id": review_session_id.strip(),
            "review_mode": "inline_human_confirmation",
        }
        validated.append((annotation_record, self_review_record))
    quality_output = root / config["paths"]["output_dir"] / "quality" / f"{scenario}.jsonl"
    for annotation_record, self_review_record in validated:
        append_jsonl(output, annotation_record)
        append_jsonl(quality_output, self_review_record)
    return {"applied": len(validated)}


def export_quality_packet(
    root: Path,
    config: dict[str, Any],
    scenario: str,
    stage: str,
    output: Path,
) -> dict[str, int | str]:
    """Export only deterministically selected, ready, unfinished QC work."""
    if scenario not in SCENARIOS:
        raise Week5DataError(f"unsupported scenario: {scenario}")
    if stage not in {"cross_review", "core_audit"}:
        raise Week5DataError("single-operator QC export supports cross_review or core_audit")
    candidates = {row["sample_id"]: row for row in load_pools(root, config)[scenario]}
    annotations = {
        row["sample_id"]: row
        for row in read_jsonl(
            root / config["paths"]["output_dir"] / "annotations" / f"{scenario}.jsonl"
        )
    }
    quality = read_jsonl(
        root / config["paths"]["output_dir"] / "quality" / f"{scenario}.jsonl"
    )
    passed: set[tuple[str, int, str]] = {
        (str(row.get("sample_id")), int(row.get("annotation_revision", 0)), str(row.get("stage")))
        for row in quality
        if row.get("decision") == "pass"
    }
    recorded: set[tuple[str, int, str]] = {
        (str(row.get("sample_id")), int(row.get("annotation_revision", 0)), str(row.get("stage")))
        for row in quality
    }

    def tasks() -> Any:
        for sample_id in sorted(annotations):
            annotation = annotations[sample_id]
            revision = int(annotation["revision"])
            key = (sample_id, revision, stage)
            if key in recorded:
                continue
            self_key = (sample_id, revision, "self_review")
            cross_key = (sample_id, revision, "cross_review")
            if self_key not in passed:
                continue
            if stage == "cross_review":
                if not qc_cross_review_selected(sample_id, scenario, config):
                    continue
            elif not qc_audit_selected(sample_id, scenario, config) or cross_key not in passed:
                continue
            candidate = candidates.get(sample_id)
            if candidate is None:
                raise Week5DataError(f"QC sample is missing from candidate pool: {sample_id}")
            yield {
                "sample_id": sample_id,
                "scenario": scenario,
                "annotation_revision": revision,
                "image": copy.deepcopy(candidate.get("image")),
                "input": copy.deepcopy(candidate.get("input")),
                "human_annotation": copy.deepcopy(annotation["human_annotation"]),
                "stage": stage,
                "decision": None,
                "reviewer": None,
                "issues": [],
                "notes": None,
                "reviewed_at": None,
                "review_session_id": None,
            }

    exported = write_jsonl_new(output, tasks())
    return {"scenario": scenario, "stage": stage, "exported": exported}


def apply_quality_records(
    root: Path, config: dict[str, Any], scenario: str, input_path: Path,
) -> dict[str, int]:
    """Validate and append self-review, same-scenario cross-review, or core audit."""
    annotations_path = root / config["paths"]["output_dir"] / "annotations" / f"{scenario}.jsonl"
    annotations = {row["sample_id"]: row for row in read_jsonl(annotations_path)}
    rows = read_jsonl(input_path)
    output = root / config["paths"]["output_dir"] / "quality" / f"{scenario}.jsonl"
    existing = read_jsonl(output)
    allowed_issues = set(json.loads((root / "configs/week5/annotation_tool.json").read_text(encoding="utf-8"))["qc_issue_codes"])
    checked: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("sample_id")
        annotation = annotations.get(sample_id)
        if not annotation:
            raise Week5DataError(f"QC sample has no human correction: {sample_id}")
        stage = row.get("stage")
        decision = row.get("decision")
        reviewer = row.get("reviewer")
        review_session_id = row.get("review_session_id")
        issues = row.get("issues", [])
        if stage not in {"self_review", "cross_review", "core_audit"} or decision not in {"pass", "rework", "reject"}:
            raise Week5DataError("invalid QC stage or decision")
        if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(issues, list) or not set(issues) <= allowed_issues:
            raise Week5DataError("invalid QC reviewer or issue code")
        if not isinstance(review_session_id, str) or not review_session_id.strip():
            raise Week5DataError("QC record requires review_session_id")
        if stage == "self_review" and reviewer != annotation["annotator"]:
            raise Week5DataError("self-review must be recorded by the annotator")
        single_operator = config.get("quality", {}).get("mode") == "single_operator_minimal_review_v1"
        if stage == "cross_review" and not qc_cross_review_selected(sample_id, scenario, config):
            raise Week5DataError("sample was not deterministically selected for cross-review")
        if stage == "cross_review" and not single_operator and reviewer == annotation["annotator"]:
            raise Week5DataError("cross reviewer must differ from the annotator")
        if stage in {"cross_review", "core_audit"} and single_operator and reviewer != annotation["annotator"]:
            raise Week5DataError("single-operator QC must use the real annotator identity")
        if stage == "core_audit" and not qc_audit_selected(sample_id, scenario, config):
            raise Week5DataError("sample was not deterministically selected for core audit")
        if any(
            item.get("sample_id") == sample_id
            and item.get("annotation_revision") == annotation["revision"]
            and item.get("stage") == stage
            for item in existing + checked
        ):
            raise Week5DataError("QC stage already recorded for current revision")
        passed_for_revision = {
            item.get("stage") for item in existing + checked
            if item.get("sample_id") == sample_id
            and item.get("annotation_revision") == annotation["revision"]
            and item.get("decision") == "pass"
        }
        if stage == "cross_review" and "self_review" not in passed_for_revision:
            raise Week5DataError("cross-review requires passed self-review for current revision")
        if stage == "core_audit" and "cross_review" not in passed_for_revision:
            raise Week5DataError("core audit requires passed cross-review for current revision")
        used_sessions = {
            str(item.get("review_session_id")) for item in existing + checked
            if item.get("sample_id") == sample_id
            and item.get("annotation_revision") == annotation["revision"]
        }
        used_sessions.add(str(annotation.get("review_session_id")))
        if stage in {"cross_review", "core_audit"} and review_session_id.strip() in used_sessions:
            raise Week5DataError("later QC stages require a distinct review_session_id")
        checked.append({
            "sample_id": sample_id, "scenario": scenario, "annotation_revision": annotation["revision"],
            "stage": stage, "decision": decision, "reviewer": reviewer.strip(),
            "issues": list(issues), "notes": row.get("notes"), "reviewed_at": row.get("reviewed_at") or _now(),
            "review_session_id": review_session_id.strip(),
            "review_mode": "same_operator_blind_second_pass" if single_operator else "multi_operator",
        })
    for row in checked:
        append_jsonl(output, row)
    return {"applied": len(checked)}


def _qualified_sample_ids(root: Path, config: dict[str, Any]) -> dict[str, list[str]]:
    output = root / config["paths"]["output_dir"]
    qualified: dict[str, list[str]] = {}
    for scenario in SCENARIOS:
        annotations = {row["sample_id"]: row for row in read_jsonl(output / "annotations" / f"{scenario}.jsonl")}
        passed = {stage: set() for stage in ("self_review", "cross_review", "core_audit")}
        for row in read_jsonl(output / "quality" / f"{scenario}.jsonl"):
            if row.get("decision") == "pass" and row.get("annotation_revision") == annotations.get(row.get("sample_id"), {}).get("revision"):
                passed[row["stage"]].add(row["sample_id"])
        qualified[scenario] = sorted(
            sample_id for sample_id in annotations
            if sample_id in passed["self_review"]
            and (not qc_cross_review_selected(sample_id, scenario, config) or sample_id in passed["cross_review"])
            and (not qc_audit_selected(sample_id, scenario, config) or sample_id in passed["core_audit"])
        )
    return qualified


def generate_dialogue_candidates(
    root: Path, config: dict[str, Any], *, run_id: str, limit: int | None = None,
) -> dict[str, int]:
    """Generate resumable model-assisted dialogues only from qualified single-turn data."""
    _require_model_access(config)
    qualified = _qualified_sample_ids(root, config)
    if not all(qualified.values()):
        raise Week5DataError("each scenario needs qualified human/QC single-turn samples before dialogue generation")
    pools = {scenario: {row["sample_id"]: row for row in rows} for scenario, rows in load_pools(root, config).items()}
    run_dir = _safe_run_directory(root, config, f"dialogue-{run_id}")
    if run_dir.exists():
        raise Week5DataError("dialogue run already exists; candidates are immutable")
    run_dir.mkdir(parents=True)
    output = run_dir / "candidates.jsonl"
    existing = read_jsonl(output)
    existing_ids = {row["dialogue_id"] for row in existing}
    target = config["targets"]["dialogues"] if limit is None else min(limit, config["targets"]["dialogues"])
    generated = failed = 0
    dialogue_scenarios = {
        "image_product_search": "image_search",
        "after_sales": "after_sales",
        "itinerary_planning": "itinerary",
    }
    for index in range(target):
        source_scenario = SCENARIOS[index % 3]
        sample_id = qualified[source_scenario][(index // 3) % len(qualified[source_scenario])]
        dialogue_id = f"week5-dialogue-{index:05d}-{hashlib.sha256(sample_id.encode()).hexdigest()[:8]}"
        if dialogue_id in existing_ids:
            continue
        candidate = pools[source_scenario][sample_id]
        turns = 4 + (index % 3)
        normalized_images = [
            {"image_id": f"img_{image_index}", "path": image["path"], "sha256": image["sha256"]}
            for image_index, image in enumerate(candidate["input"]["images"], start=1)
        ]
        prompt = (
            "基于给定 OTA 单轮样本生成一段真实多轮对话。只返回 JSON 对象，字段必须为 scenario、turns。"
            f"对话共 {turns} 轮（每轮含 user 和 assistant 两条消息），用户口语化，助手专业友好；覆盖上传图片、补充条件、历史追问、约束修改和历史图片指代中的至少三项。"
            "不得编造图片不可见事实、退款承诺、价格保证或安全结论。turns 中 role 必须从 user 开始严格交替；image_refs 只能引用给定 image_resources 中的 image_id。"
            f"\ndialogue_id={dialogue_id}\nscenario={dialogue_scenarios[source_scenario]}\n"
            f"image_resources={json.dumps(normalized_images, ensure_ascii=False)}\n"
            f"原始约束={candidate['input'].get('text_constraints')}"
        )
        runtime = _runtime(root, config, source_scenario)
        dialogue_runtime = copy.deepcopy(runtime)
        dialogue_runtime["generation"] = {
            "temperature": 0.1, "top_p": 0.9, "max_tokens": 2560,
            "enable_thinking": False,
        }
        payload = _build_chat_payload(root, {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"file://{candidate['input']['images'][0]['path']}"}},
            ]}],
            "response_format": {"type": "json_object"},
        }, dialogue_runtime)
        try:
            response = post_chat_completion(chat_completions_url(runtime["live_base_url"]), payload, runtime["timeout_seconds"])
            raw = response["choices"][0]["message"]["content"]
            generated_payload = json.loads(raw)
            dialogue = {
                "schema_version": "multimodal_dialogue_v2",
                "dialogue_id": dialogue_id,
                "scenario": generated_payload.get("scenario"),
                "image_resources": normalized_images,
                "turns": generated_payload.get("turns"),
                "source_sample_ids": [sample_id],
                "generation": {
                    "run_id": run_id,
                    "model_name": runtime["model_name"],
                    "prompt_version": "week5_dialogue_v2",
                },
                "human_review": {
                    "status": "awaiting_human_annotation",
                    "reviewer": None,
                    "reviewed_at": None,
                    "checks": {},
                },
                "qc": {
                    "status": "partial",
                    "reviewer": None,
                    "reviewed_at": None,
                    "issues": [],
                },
            }
            validate_dialogue_v2(root, dialogue)
            if dialogue["scenario"] != dialogue_scenarios[source_scenario]:
                raise Week5DataError("generated dialogue scenario changed")
            if len(dialogue["turns"]) != turns * 2:
                raise Week5DataError("generated dialogue did not preserve requested turn count")
            append_jsonl(output, dialogue)
            existing_ids.add(dialogue_id)
            generated += 1
        except Exception as exc:
            append_jsonl(output.with_name("failures.jsonl"), {"dialogue_id": dialogue_id, "sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}", "timestamp": _now(), "run_id": run_id})
            failed += 1
    return {"generated": generated, "failed": failed, "existing": len(existing)}


def apply_dialogue_validation(root: Path, config: dict[str, Any], input_path: Path, *, run_id: str) -> dict[str, int]:
    output_dir = _safe_run_directory(root, config, f"dialogue-{run_id}")
    candidates = {row["dialogue_id"]: row for row in read_jsonl(output_dir / "candidates.jsonl")}
    existing = {row.get("dialogue_id") for row in read_jsonl(output_dir / "human_validation.jsonl")}
    checked: list[dict[str, Any]] = []
    for row in read_jsonl(input_path):
        dialogue_id = row.get("dialogue_id")
        if dialogue_id not in candidates or dialogue_id in existing:
            raise Week5DataError(f"unknown or already validated dialogue: {dialogue_id}")
        validate_dialogue_v2(root, candidates[dialogue_id])
        if row.get("decision") not in {"pass", "rework", "reject"} or not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip():
            raise Week5DataError("dialogue validation requires reviewer and valid decision")
        checks = row.get("checks")
        required = {"logic", "context", "image_reference", "business_compliance", "ota_tone"}
        if not isinstance(checks, dict) or set(checks) != required or any(value not in {"pass", "fail"} for value in checks.values()):
            raise Week5DataError("dialogue validation checks are incomplete")
        if row["decision"] == "pass" and "fail" in checks.values():
            raise Week5DataError("dialogue with failed checks cannot pass")
        checked.append({**row, "validated_at": row.get("validated_at") or _now()})
    for row in checked:
        append_jsonl(output_dir / "human_validation.jsonl", row)
    return {"applied": len(checked)}

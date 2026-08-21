"""Build and evaluate corrected Week 7 development dialogues for real human review."""

from __future__ import annotations

import gc
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.training.week6_qlora import environment_report
from src.training.week7_data import (
    DIALOGUE_DIMENSIONS,
    Week7DataError,
    canonical_sha256,
    iter_jsonl,
    load_week7_config,
    sha256_file,
    write_jsonl_new,
)
from src.training.week7_evaluation import summarize_dialogue_raw_records
from src.training.week7_latency_protocol import (
    _default_model_loader,
    _default_record_generator,
    _release_model,
)
from src.training.week7_qlora import Week7TrainingError
from src.training.week7_runtime import LATENCY_PROTOCOL_V5_VERSION


REVIEW_CONFIG_SCHEMA = "week7_dialogue_review_config_v2"
REVIEW_LOCK_SCHEMA = "week7_dialogue_review_lock_v2"
REVIEW_RUN_SCHEMA = "week7_dialogue_review_run_v2"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week7TrainingError(f"invalid dialogue-review JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Week7TrainingError(f"dialogue-review JSON must be an object: {path}")
    return value


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _verify_output_parent_writable(output_dir: Path) -> None:
    """Fail before loading the 8B model when quota/inodes cannot hold new evidence."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    probe = output_dir.parent / f".{output_dir.name}.write-probe-{os.getpid()}"
    try:
        with probe.open("x", encoding="utf-8") as handle:
            handle.write("week7-dialogue-review-v2\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise Week7TrainingError(
            f"dialogue-review output parent is not writable: {output_dir.parent}"
        ) from exc
    finally:
        probe.unlink(missing_ok=True)


def _config(root: Path, config_path: Path) -> dict[str, Any]:
    value = _read_json(config_path)
    if value.get("schema_version") != REVIEW_CONFIG_SCHEMA:
        raise Week7TrainingError("unsupported dialogue-review config schema")
    if value.get("construction_version") != "aligned_concrete_turns_v2":
        raise Week7TrainingError("dialogue-review construction identity changed")
    if value.get("scope") != {
        "split": "development",
        "test_allowed": False,
        "training_allowed": False,
        "may_change_final_test_claims": False,
    }:
        raise Week7TrainingError("dialogue-review scope must remain development-only")
    base = root / value["base_config"]["path"]
    if not base.is_file() or sha256_file(base) != value["base_config"]["sha256"]:
        raise Week7TrainingError("dialogue-review base config identity mismatch")
    if tuple(value["human_review"]["dimensions"]) != DIALOGUE_DIMENSIONS:
        raise Week7TrainingError("dialogue-review dimensions changed")
    return value


def _text_list(values: list[Any], *, fallback: str) -> str:
    clean = [str(value).strip() for value in values if str(value).strip()]
    return "、".join(clean) if clean else fallback


def _assistant_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _corrected_messages(source: dict[str, Any], rounds: int) -> list[dict[str, Any]]:
    original = source.get("messages")
    target = source.get("target")
    if (
        not isinstance(original, list)
        or len(original) < 2
        or original[0].get("role") != "system"
        or original[1].get("role") != "user"
        or not isinstance(target, dict)
        or not isinstance(target.get("task_result"), dict)
        or not isinstance(target.get("context_state"), dict)
    ):
        raise Week7TrainingError("source dialogue cannot be repaired deterministically")
    base_target = target["task_result"]
    context = target["context_state"]
    evidence = _text_list(
        list(context.get("historical_image_reference") or []), fallback="没有可确认的图片证据",
    )
    constraints = _text_list(
        list(context.get("retained_hard_constraints") or []), fallback="没有额外硬约束",
    )
    updated = str(context.get("updated_requirement") or "预算优先")
    messages = [deepcopy(original[0]), deepcopy(original[1])]
    messages.append({"role": "assistant", "content": _assistant_json(base_target)})

    mandatory = [
        (
            "请明确引用刚才那张图片中的证据。",
            f"刚才图片中可确认的证据是：{evidence}。图片未支持的字段继续标记为 unknown。",
        ),
        (
            "把偏好调整为更重视预算，但保留硬约束。",
            f"已将当前偏好更新为“{updated}”；继续保留的硬约束是：{constraints}。",
        ),
        (
            "总结目前累计的限制并检查是否存在逻辑冲突，不要遗漏。",
            f"当前图片证据：{evidence}；最新需求：{updated}；保留硬约束：{constraints}。"
            "这些已确认信息之间未发现冲突，未确认字段仍保持 unknown。",
        ),
    ]
    optional = [
        (
            "再次确认没有引入图片外事实。",
            f"已复核：只使用图片证据“{evidence}”和对话中明确给出的需求与约束，没有补充图片外事实。",
        ),
        (
            "说明本轮相对最初需求的变化。",
            f"相对最初需求，本轮新增“{updated}”偏好；原有硬约束“{constraints}”未被删除。",
        ),
        (
            "如果信息仍不足，应如何处理？",
            "信息不足的字段继续输出 unknown，不根据常识或图片外信息进行猜测。",
        ),
    ]
    pairs = mandatory + optional[: rounds - 5]
    for user_text, assistant_text in pairs:
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        messages.append({"role": "assistant", "content": assistant_text})

    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": "给出包含当前上下文状态的最终结构化结果。"}],
    })
    if bool(source.get("contains_tool_call")):
        messages.append({
            "role": "assistant",
            "content": "<tool_call>{\"name\":\"check_constraints\",\"arguments\":{\"scope\":\"conversation\"}}</tool_call>",
        })
        messages.append({"role": "tool", "content": "{\"status\":\"ok\"}"})
    messages.append({"role": "assistant", "content": _assistant_json(target)})
    return messages


def _validate_corrected_row(row: dict[str, Any]) -> None:
    messages = row["messages"]
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise Week7TrainingError("corrected dialogue must start with system and image user")
    user_count = sum(message.get("role") == "user" for message in messages)
    if user_count != int(row["dialogue_rounds"]) or not 5 <= user_count <= 8:
        raise Week7TrainingError("corrected dialogue round count is not 5-8")
    image_parts = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                image_parts.append(message_index)
    if image_parts != [1]:
        raise Week7TrainingError("corrected dialogue image must appear only in first user turn")
    for index, message in enumerate(messages[:-1]):
        following = messages[index + 1]
        if message.get("role") == "assistant" and following.get("role") == "user":
            # A completed response followed by a new request is valid; the v3 defect used canned
            # answers written for that following request, which this construction never reuses.
            if str(message.get("content")) in {
                "我会继续只引用首次用户轮的图片证据。",
                "已更新当前需求，历史硬约束保持不变。",
                "已承接图片证据、预算调整和原有硬约束。",
                "上下文逻辑一致；不确定信息仍标记 unknown。",
            }:
                raise Week7TrainingError("legacy anticipatory assistant reply entered corrected row")
    if messages[-1].get("role") != "assistant":
        raise Week7TrainingError("corrected dialogue must end with assistant target")
    if json.loads(str(messages[-1]["content"])) != row["target"]:
        raise Week7TrainingError("corrected dialogue final target changed")


def build_dialogue_review_v2(
    root: Path,
    config_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Create a new development-only dialogue identity; never mutate the v3 lock."""
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _config(root, config_path)
    source_dir = root / config["source"]["dataset_dir"]
    source_lock_path = source_dir / "dataset_lock.json"
    source_development = source_dir / "development.jsonl"
    source_queue = source_dir / "dialogue_human_review_queue.jsonl"
    source_lock = _read_json(source_lock_path)
    if (
        source_lock.get("dataset_version") != config["source"]["dataset_version"]
        or source_lock.get("lock_sha256") != config["source"]["dataset_lock_sha256"]
        or sha256_file(source_development) != config["source"]["development_sha256"]
        or sha256_file(source_queue) != config["source"]["queue_sha256"]
    ):
        raise Week7TrainingError("dialogue-review v3 source evidence changed")
    source_rows = {
        str(row["sample_id"]): row
        for row in iter_jsonl(source_development)
        if row.get("scenario") == "dialogue"
    }
    queue = list(iter_jsonl(source_queue))
    if len(queue) != int(config["source"]["sample_count"]):
        raise Week7TrainingError("dialogue-review source queue count changed")
    target_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / config["output_root"] / config["dataset_version"]
    )
    if target_dir.exists():
        raise Week7TrainingError("refusing to overwrite dialogue-review v2 identity")
    target_dir.mkdir(parents=True, exist_ok=False)
    corrected = []
    corrected_queue = []
    for index, item in enumerate(queue):
        source = source_rows.get(str(item.get("sample_id")))
        if source is None:
            raise Week7TrainingError("dialogue-review queue source is missing")
        rounds = 5 + index % 4
        sample_id = f"week7-development-dialogue-review-v2-{index:04d}"
        row = deepcopy(source)
        row.update({
            "sample_id": sample_id,
            "source_dialogue_sample_id": source["sample_id"],
            "constraint_template_id": f"week7-dialogue-review-v2:{source['parent_scenario']}:{index:04d}",
            "dialogue_rounds": rounds,
            "messages": _corrected_messages(source, rounds),
            "context_expectations": deepcopy(source["target"]["context_state"]),
            "construction_version": config["construction_version"],
            "split": "development",
            "label_source": "programmatic_silver",
        })
        _validate_corrected_row(row)
        corrected.append(row)
        corrected_queue.append({
            "queue_id": f"week7-dialogue-review-v2-{index:03d}",
            "sample_id": sample_id,
            "source_dialogue_sample_id": source["sample_id"],
            "required_dimensions": list(DIALOGUE_DIMENSIONS),
            "decision": "PENDING_REAL_HUMAN_INPUT",
            "human_reviewer": None,
            "human_scores": None,
        })
    development_path = target_dir / "development.jsonl"
    queue_path = target_dir / "dialogue_human_review_queue.jsonl"
    write_jsonl_new(development_path, corrected)
    write_jsonl_new(queue_path, corrected_queue)
    lock = {
        "schema_version": REVIEW_LOCK_SCHEMA,
        "dataset_version": config["dataset_version"],
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "construction_version": config["construction_version"],
        "source": {
            "dataset_version": source_lock["dataset_version"],
            "dataset_lock_sha256": source_lock["lock_sha256"],
            "development_sha256": sha256_file(source_development),
            "queue_sha256": sha256_file(source_queue),
        },
        "files": {
            "development.jsonl": {"count": len(corrected), "sha256": sha256_file(development_path)},
            "dialogue_human_review_queue.jsonl": {"count": len(corrected_queue), "sha256": sha256_file(queue_path)},
        },
        "scope": config["scope"],
        "human_review_status": "PENDING_REAL_HUMAN_INPUT",
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    _write_json_new(target_dir / "dataset_lock.json", lock)
    return lock


def run_dialogue_review_v2(
    root: Path,
    base_config_path: Path,
    review_config_path: Path,
    dataset_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    processor: Any = None,
    model_loader: Any = None,
    record_generator: Any = None,
) -> dict[str, Any]:
    """Generate new checkpoint-151 raw output for the corrected development prompts."""
    root = Path(root).resolve()
    base_config_path = Path(base_config_path).resolve()
    review_config_path = Path(review_config_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    adapter_dir = Path(adapter_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise Week7TrainingError("refusing to overwrite dialogue-review inference evidence")
    _verify_output_parent_writable(output_dir)
    review_config = _config(root, review_config_path)
    if (
        base_config_path != (root / review_config["base_config"]["path"]).resolve()
        or sha256_file(base_config_path) != review_config["base_config"]["sha256"]
    ):
        raise Week7TrainingError("dialogue-review inference base config mismatch")
    lock = _read_json(dataset_dir / "dataset_lock.json")
    development_path = dataset_dir / "development.jsonl"
    if (
        lock.get("schema_version") != REVIEW_LOCK_SCHEMA
        or lock.get("dataset_version") != review_config["dataset_version"]
        or lock.get("config_sha256") != sha256_file(review_config_path)
        or lock.get("files", {}).get("development.jsonl", {}).get("sha256") != sha256_file(development_path)
        or lock.get("scope") != review_config["scope"]
    ):
        raise Week7TrainingError("dialogue-review inference dataset identity mismatch")
    adapter_model = adapter_dir / "adapter_model.safetensors"
    if (
        not adapter_model.is_file()
        or sha256_file(adapter_model) != review_config["selected_checkpoint"]["adapter_sha256"]
    ):
        raise Week7TrainingError("dialogue-review selected checkpoint mismatch")
    rows = list(iter_jsonl(development_path))
    if len(rows) != int(review_config["source"]["sample_count"]):
        raise Week7TrainingError("dialogue-review development count changed")
    for row in rows:
        _validate_corrected_row(row)

    report = environment_report(require_cuda=True) if model_loader is None else {"status": "injected-test-runtime"}
    if report["status"] not in {"ok", "injected-test-runtime"}:
        raise Week7TrainingError(f"dialogue-review environment is not ready: {report['status']}")
    if report["status"] == "ok" and not os.environ.get("SLURM_JOB_ID"):
        raise Week7TrainingError("dialogue-review inference must run inside Slurm")
    base_config = load_week7_config(base_config_path)
    inference = review_config["inference"]
    if model_loader is None:
        processor, model_loader = _default_model_loader(
            base_config, {"inference_precision": inference["precision"]},
        )
    if processor is None or model_loader is None:
        raise Week7TrainingError("dialogue-review model loader requires processor")
    runtime_options = {
        "latency_protocol": LATENCY_PROTOCOL_V5_VERSION,
        "cache_implementation": inference["cache_implementation"],
        "compile_config": inference["compile_config"],
        "warmup_max_new_tokens": int(inference["warmup_max_new_tokens"]),
    }
    record_generator = record_generator or (
        lambda model, proc, data, run_id, model_name, max_tokens: _default_record_generator(
            model, proc, data, run_id, model_name, max_tokens,
            runtime_options=runtime_options,
        )
    )
    started = time.time()
    model = model_loader(adapter_dir)
    try:
        records, warmup = record_generator(
            model, processor, rows, inference["run_id"], inference["model_name"],
            int(inference["max_new_tokens"]),
        )
    finally:
        del model
        gc.collect()
        _release_model()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "raw_outputs.jsonl"
    warmup_path = output_dir / "warmup.jsonl"
    write_jsonl_new(raw_path, records)
    write_jsonl_new(warmup_path, [warmup])
    metrics = summarize_dialogue_raw_records(rows, records)
    metrics_path = output_dir / "metrics.json"
    _write_json_new(metrics_path, metrics)
    summary = {
        "schema_version": REVIEW_RUN_SCHEMA,
        "status": "COMPLETED",
        "run_id": inference["run_id"],
        "model_name": inference["model_name"],
        "split": "development",
        "test_read": False,
        "review_config_sha256": sha256_file(review_config_path),
        "dataset_lock_sha256": lock["lock_sha256"],
        "development_sha256": sha256_file(development_path),
        "adapter_sha256": sha256_file(adapter_model),
        "sample_count": len(rows),
        "raw_outputs": {"path": str(raw_path), "sha256": sha256_file(raw_path), "count": len(records)},
        "warmup": {"path": str(warmup_path), "sha256": sha256_file(warmup_path), "count": 1},
        "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
        "runtime": inference,
        "elapsed_seconds": time.time() - started,
    }
    _write_json_new(output_dir / "run_summary.json", summary)
    return summary

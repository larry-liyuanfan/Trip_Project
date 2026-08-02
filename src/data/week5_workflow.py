"""Resumable Week 5 model preannotation, human correction, QC, and dialogue work."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.week5_dataset import (
    SCENARIOS,
    Week5DataError,
    append_jsonl,
    load_pools,
    qc_audit_selected,
    read_jsonl,
    validate_dialogue,
    validate_human_annotation,
)
from src.evaluation.config import load_evaluation_config
from src.evaluation.manifests import load_configured_manifests
from src.evaluation.prompting import render_standard_prompt
from src.evaluation.results import parse_and_validate_output
from src.evaluation.runner import _build_chat_payload, chat_completions_url, post_chat_completion
from src.evaluation.week4_prompting import load_week4_selection, render_week4_request


_FEWSHOT_PREFIX_CACHE: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _api_key_available() -> bool:
    direct = os.getenv("MODEL_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    key_file = os.getenv("MODEL_API_KEY_FILE") or os.getenv("DASHSCOPE_API_KEY_FILE")
    return bool(direct or (key_file and Path(key_file).is_file()))


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
    return {
        "model_name": model["served_model_name"],
        "served_model_name": model["served_model_name"],
        "model_config": model,
        "generation": generation,
        "live_base_url": runtime_config["base_url"],
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
) -> dict[str, Any]:
    scenario = candidate["scenario"]
    version = config["prompt_versions"][scenario]
    if version == "standardized_v4":
        return render_standard_prompt(root, scenario, candidate["input"], version="standardized_v4")
    cached = _FEWSHOT_PREFIX_CACHE.get(scenario)
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
        _FEWSHOT_PREFIX_CACHE[scenario] = cached
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


def run_preannotation(
    root: Path, config: dict[str, Any], scenario: str, *, limit: int | None = None,
    retry_failures: bool = False,
) -> dict[str, int]:
    """Append one durable result per attempted sample and skip completed IDs on resume."""
    if scenario not in SCENARIOS:
        raise Week5DataError(f"unsupported scenario: {scenario}")
    if not _api_key_available():
        raise Week5DataError("MODEL_API_KEY or MODEL_API_KEY_FILE is required for real Qwen3.7 preannotation")
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
            parsed = parse_and_validate_output(root, scenario, raw_output, "v2" if scenario == "itinerary_planning" else "v1")
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


def apply_human_corrections(
    root: Path, config: dict[str, Any], scenario: str, input_path: Path,
) -> dict[str, int]:
    """Append human corrections; a correction never implies any QC stage passed."""
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
    validated: list[dict[str, Any]] = []
    for row in submitted:
        sample_id = row.get("sample_id")
        if sample_id in seen or sample_id not in candidates:
            raise Week5DataError(f"duplicate or unknown human sample: {sample_id}")
        if sample_id not in preannotated:
            raise Week5DataError(f"human correction requires completed model preannotation: {sample_id}")
        seen.add(sample_id)
        annotator = row.get("annotator")
        corrected_at = row.get("corrected_at")
        if not isinstance(annotator, str) or not annotator.strip() or not isinstance(corrected_at, str) or not corrected_at.strip():
            raise Week5DataError("human correction requires annotator and corrected_at")
        validate_human_annotation(root, scenario, row.get("human_annotation"))
        validated.append({
            "sample_id": sample_id, "scenario": scenario, "annotator": annotator.strip(),
            "human_annotation": copy.deepcopy(row["human_annotation"]),
            "corrected_at": corrected_at, "revision": revisions.get(sample_id, 0) + 1,
            "source": "human_correction", "qc_reset": True,
        })
    for row in validated:
        append_jsonl(output, row)
    return {"applied": len(validated)}


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
        issues = row.get("issues", [])
        if stage not in {"self_review", "cross_review", "core_audit"} or decision not in {"pass", "rework", "reject"}:
            raise Week5DataError("invalid QC stage or decision")
        if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(issues, list) or not set(issues) <= allowed_issues:
            raise Week5DataError("invalid QC reviewer or issue code")
        if stage == "self_review" and reviewer != annotation["annotator"]:
            raise Week5DataError("self-review must be recorded by the annotator")
        if stage == "cross_review" and reviewer == annotation["annotator"]:
            raise Week5DataError("cross reviewer must differ from the annotator")
        if stage == "core_audit" and not qc_audit_selected(sample_id, scenario, config):
            raise Week5DataError("sample was not deterministically selected for core audit")
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
        checked.append({
            "sample_id": sample_id, "scenario": scenario, "annotation_revision": annotation["revision"],
            "stage": stage, "decision": decision, "reviewer": reviewer.strip(),
            "issues": list(issues), "notes": row.get("notes"), "reviewed_at": row.get("reviewed_at") or _now(),
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
            if sample_id in passed["self_review"] and sample_id in passed["cross_review"]
            and (not qc_audit_selected(sample_id, scenario, config) or sample_id in passed["core_audit"])
        )
    return qualified


def generate_dialogue_candidates(
    root: Path, config: dict[str, Any], *, limit: int | None = None,
) -> dict[str, int]:
    """Generate resumable model-assisted dialogues only from qualified single-turn data."""
    if not _api_key_available():
        raise Week5DataError("a real Qwen3.7 API key is required for dialogue generation")
    qualified = _qualified_sample_ids(root, config)
    if not all(qualified.values()):
        raise Week5DataError("each scenario needs qualified human/QC single-turn samples before dialogue generation")
    pools = {scenario: {row["sample_id"]: row for row in rows} for scenario, rows in load_pools(root, config).items()}
    output = root / config["paths"]["output_dir"] / "dialogues" / "candidates.jsonl"
    existing = read_jsonl(output)
    existing_ids = {row["dialogue_id"] for row in existing}
    target = config["targets"]["dialogues"] if limit is None else min(limit, config["targets"]["dialogues"])
    generated = failed = 0
    dialogue_scenarios = {
        "image_product_search": "image_search_consultation",
        "after_sales": "after_sales_negotiation",
        "itinerary_planning": "itinerary_iteration",
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
            "基于给定 OTA 单轮样本生成一段真实多轮对话。只返回 JSON 对象，字段必须为 dialogue_id、scenario、images、messages。"
            f"对话共 {turns} 轮（每轮含 user 和 assistant 两条消息），用户口语化，助手专业友好；覆盖上传图片、补充条件、历史追问、约束修改和历史图片指代中的至少三项。"
            "不得编造图片不可见事实、退款承诺、价格保证或安全结论。消息 role 必须从 user 开始严格交替；image_refs 只能引用 images 中的 image_id。"
            f"\ndialogue_id={dialogue_id}\nscenario={dialogue_scenarios[source_scenario]}\n"
            f"images={json.dumps(normalized_images, ensure_ascii=False)}\n"
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
            dialogue = json.loads(raw)
            dialogue["dialogue_id"] = dialogue_id
            dialogue["images"] = normalized_images
            validate_dialogue(dialogue)
            if dialogue["scenario"] != dialogue_scenarios[source_scenario]:
                raise Week5DataError("generated dialogue scenario changed")
            if len(dialogue["messages"]) != turns * 2:
                raise Week5DataError("generated dialogue did not preserve requested turn count")
            append_jsonl(output, dialogue)
            existing_ids.add(dialogue_id)
            generated += 1
        except Exception as exc:
            append_jsonl(output.with_name("failures.jsonl"), {"dialogue_id": dialogue_id, "sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}", "timestamp": _now()})
            failed += 1
    return {"generated": generated, "failed": failed, "existing": len(existing)}


def apply_dialogue_validation(root: Path, config: dict[str, Any], input_path: Path) -> dict[str, int]:
    output_dir = root / config["paths"]["output_dir"] / "dialogues"
    candidates = {row["dialogue_id"]: row for row in read_jsonl(output_dir / "candidates.jsonl")}
    existing = {row.get("dialogue_id") for row in read_jsonl(output_dir / "human_validation.jsonl")}
    checked: list[dict[str, Any]] = []
    for row in read_jsonl(input_path):
        dialogue_id = row.get("dialogue_id")
        if dialogue_id not in candidates or dialogue_id in existing:
            raise Week5DataError(f"unknown or already validated dialogue: {dialogue_id}")
        validate_dialogue(candidates[dialogue_id])
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

"""Versioned Week 5 repair and final release gates for system consolidation."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests
from PIL import Image

from src.data.week5_dataset import (
    PRICE_MAP,
    SCENARIOS,
    Week5DataError,
    _business_category,
    _load_businesses,
    _style_hint,
    load_exclusions,
    load_week5_config,
)
from src.evaluation.schema_validation import validate_output
from src.inference.schemas import TaskRequest
from src.inference.system_runtime import ModelGenerationError, ScenarioService
from src.training.week7_data import canonical_sha256, sha256_file


IDENTITY_FIELDS = (
    "sample_id",
    "source_id",
    "image_sha256",
    "group_id",
    "constraint_template_id",
)


class SystemRepairError(ValueError):
    """Raised when repair provenance, counts, or release gates are invalid."""


def load_repair_config(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "week5_preannotation_repair_v2":
        raise SystemRepairError("unsupported Week 5 repair config")
    expected = payload.get("expected", {})
    locked = {
        "candidate_count": 80000,
        "historical_schema_valid": 79936,
        "input_replacements": 44,
        "schema_retries": 19,
        "json_retries": 1,
        "repair_queue": 64,
    }
    if expected != locked:
        raise SystemRepairError("Week 5 repair counts changed")
    if payload.get("model", {}).get("base_model") != "Qwen/Qwen3-VL-8B-Instruct":
        raise SystemRepairError("Week 5 repair model changed")
    return payload


def build_week5_repair_v2(
    root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Replace unreadable inputs and create a 64-record immutable retry queue."""
    root = Path(root).resolve()
    repair = load_repair_config(config_path)
    historical = root / repair["historical_output_dir"]
    merged = historical / repair["historical_merged_run"]
    output = root / repair["output_dir"]
    if output.exists():
        raise SystemRepairError(f"repair output already exists: {output}")

    week5_config = load_week5_config(root, repair["week5_config"])
    failures = list(_iter_jsonl(merged / "failures.jsonl"))
    successful = list(_iter_jsonl(merged / "results.jsonl"))
    _validate_historical_failure_contract(failures, successful, repair["expected"])
    pools = {
        scenario: list(_iter_jsonl(historical / "pools" / f"{scenario}.jsonl"))
        for scenario in SCENARIOS
    }
    by_sample = {
        row["sample_id"]: row
        for rows in pools.values()
        for row in rows
    }
    failed_by_id = {row["sample_id"]: row for row in failures}
    if not set(failed_by_id).issubset(by_sample):
        raise SystemRepairError("historical failures are not all present in the pools")

    input_failure_ids = {
        row["sample_id"] for row in failures if row["error_type"] == "input_error"
    }
    replacements = _select_replacements(
        root,
        week5_config,
        [by_sample[sample_id] for sample_id in sorted(input_failure_ids)],
        pools,
    )
    replacement_by_old = {row["replaces_sample_id"]: row for row in replacements}
    output.mkdir(parents=True)
    repaired_pools: dict[str, list[dict[str, Any]]] = {}
    for scenario, rows in pools.items():
        repaired = [
            replacement_by_old.get(row["sample_id"], row)
            for row in rows
        ]
        repaired_pools[scenario] = repaired
        _write_jsonl_new(output / "pools" / f"{scenario}.jsonl", repaired)

    queue = []
    for failure in sorted(failures, key=lambda row: row["sample_id"]):
        original = by_sample[failure["sample_id"]]
        candidate = replacement_by_old.get(failure["sample_id"], original)
        queue.append(
            {
                "repair_id": repair["repair_id"],
                "sample_id": candidate["sample_id"],
                "scenario": candidate["scenario"],
                "candidate": candidate,
                "historical_failure": failure,
                "repair_action": (
                    "replace_unreadable_input"
                    if failure["error_type"] == "input_error"
                    else "regenerate_with_single_schema_correction"
                ),
                "label_source": "silver_model_preannotation",
                "human_completed": False,
            }
        )
    _write_jsonl_new(output / "repair_queue.jsonl", queue)
    replacement_records = [
        {
            "old_sample_id": row["replaces_sample_id"],
            "new_sample_id": row["sample_id"],
            "scenario": row["scenario"],
            "new_source_id": row["source_id"],
            "new_image_sha256": row["image_sha256"],
            "stratum": row["sampling_metadata"],
        }
        for row in replacements
    ]
    _write_jsonl_new(output / "replacement_manifest.jsonl", replacement_records)
    audit = _audit_repaired_pools(root, week5_config, repaired_pools)
    summary = {
        "status": "AWAITING_MODEL_REPAIR",
        "repair_id": repair["repair_id"],
        "historical_success": len(successful),
        "repair_queue": len(queue),
        "replacements": len(replacements),
        "candidate_counts": {
            scenario: len(rows) for scenario, rows in repaired_pools.items()
        },
        "failure_types": dict(sorted(Counter(row["error_type"] for row in failures).items())),
        "audit": audit,
        "artifact_sha256": {
            path.relative_to(output).as_posix(): sha256_file(path)
            for path in sorted(output.rglob("*.jsonl"))
        },
    }
    _write_json_new(output / "preparation_summary.json", summary)
    return summary


def run_week5_repair_queue(
    root: Path,
    config_path: Path,
    *,
    run_id: str,
    base_url: str | None = None,
    resume: bool = False,
    service: ScenarioService | None = None,
) -> dict[str, Any]:
    """Run the 64-record queue through packaged endpoints with resumable evidence."""
    root = Path(root).resolve()
    repair = load_repair_config(config_path)
    output = root / repair["output_dir"]
    queue_path = output / "repair_queue.jsonl"
    queue = list(_iter_jsonl(queue_path))
    run_dir = output / "runs" / run_id
    identity = {
        "run_id": run_id,
        "repair_id": repair["repair_id"],
        "config_sha256": sha256_file(config_path),
        "queue_sha256": sha256_file(queue_path),
        "base_model": repair["model"]["base_model"],
    }
    identity_path = run_dir / "run_identity.json"
    if run_dir.exists():
        if not resume or not identity_path.is_file():
            raise SystemRepairError(f"run directory already exists: {run_dir}")
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise SystemRepairError("resume identity differs from the original run")
    else:
        run_dir.mkdir(parents=True)
        _write_json_new(identity_path, identity)
    results_path = run_dir / "results.jsonl"
    failures_path = run_dir / "failures.jsonl"
    completed = {
        row["sample_id"] for row in _iter_jsonl(results_path)
    } if results_path.exists() else set()
    session = requests.Session()
    if service is None and not base_url:
        raise SystemRepairError("base_url is required without an in-process service")
    for item in queue:
        if item["sample_id"] in completed:
            continue
        candidate = item["candidate"]
        endpoint = {
            "image_product_search": "image-product-search",
            "after_sales": "after-sales",
            "itinerary_planning": "itinerary-planning",
        }[item["scenario"]]
        request_payload = {
            "image_urls": [image["path"] for image in candidate["input"]["images"]],
            "text_context": candidate["input"].get("text_constraints"),
        }
        started = time.perf_counter()
        response_payload = None
        error = None
        for network_attempt in range(1, int(repair["model"]["max_network_retries"]) + 2):
            try:
                if service is not None:
                    response_payload = service.run_task(
                        item["scenario"],
                        TaskRequest.model_validate(request_payload),
                    ).model_dump()
                else:
                    response = session.post(
                        f"{str(base_url).rstrip('/')}/v1/tasks/{endpoint}",
                        json=request_payload,
                        timeout=int(repair["model"]["timeout_seconds"]),
                    )
                    response.raise_for_status()
                    response_payload = response.json()
                if response_payload.get("schema_valid") is not True:
                    raise SystemRepairError("system endpoint returned non-valid output")
                break
            except ModelGenerationError as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if network_attempt > int(repair["model"]["max_network_retries"]):
                    break
        record = {
            "run_id": run_id,
            "repair_id": repair["repair_id"],
            "sample_id": item["sample_id"],
            "scenario": item["scenario"],
            "candidate_sha256": canonical_sha256(candidate),
            "status": "completed" if response_payload else "failed",
            "parsed_output": response_payload.get("result") if response_payload else None,
            "schema_valid": bool(response_payload),
            "model": response_payload.get("model") if response_payload else repair["model"]["base_model"],
            "adapter": response_payload.get("adapter") if response_payload else None,
            "release_id": response_payload.get("release_id") if response_payload else None,
            "model_attempts": response_payload.get("attempts") if response_payload else [],
            "request_latency_ms": (time.perf_counter() - started) * 1000,
            "error": error if not response_payload else None,
            "label_source": "silver_model_preannotation",
            "human_completed": False,
        }
        _append_jsonl(results_path if response_payload else failures_path, record)
    results = list(_iter_jsonl(results_path)) if results_path.exists() else []
    unresolved = {row["sample_id"] for row in queue} - {row["sample_id"] for row in results}
    summary = {
        "status": "COMPLETED" if not unresolved else "PARTIAL",
        "run_id": run_id,
        "repair_queue": len(queue),
        "schema_valid": len(results),
        "unresolved": len(unresolved),
        "results_sha256": sha256_file(results_path) if results_path.exists() else None,
    }
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary_path.unlink()
    _write_json_new(summary_path, summary)
    return summary


def merge_week5_repair_results(
    root: Path,
    config_path: Path,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Materialize 80,000 Schema-valid silver results without changing history."""
    root = Path(root).resolve()
    repair = load_repair_config(config_path)
    output = root / repair["output_dir"]
    merged_historical = (
        root / repair["historical_output_dir"] / repair["historical_merged_run"]
    )
    historical_results = {
        row["sample_id"]: row for row in _iter_jsonl(merged_historical / "results.jsonl")
    }
    repair_results = {
        row["sample_id"]: row
        for row in _iter_jsonl(output / "runs" / run_id / "results.jsonl")
    }
    queue_ids = {row["sample_id"] for row in _iter_jsonl(output / "repair_queue.jsonl")}
    if set(repair_results) != queue_ids:
        raise SystemRepairError("repair run does not cover the complete immutable queue")
    final_path = output / "schema_valid_silver_80000.jsonl"
    if final_path.exists():
        raise SystemRepairError("final Week 5 repair result already exists")
    counts = Counter()
    with final_path.open("x", encoding="utf-8", newline="\n") as handle:
        for scenario in SCENARIOS:
            for candidate in _iter_jsonl(output / "pools" / f"{scenario}.jsonl"):
                sample_id = candidate["sample_id"]
                source = repair_results.get(sample_id) or historical_results.get(sample_id)
                if source is None:
                    raise SystemRepairError(f"missing final result: {sample_id}")
                parsed = source.get("parsed_output")
                schema_version = "v2" if scenario == "itinerary_planning" else "v1"
                validate_output(root, scenario, parsed, schema_version)
                row = {
                    "repair_id": repair["repair_id"],
                    "sample_id": sample_id,
                    "scenario": scenario,
                    "candidate_sha256": canonical_sha256(candidate),
                    "parsed_output": parsed,
                    "schema_valid": True,
                    "label_source": "silver_model_preannotation",
                    "human_completed": False,
                    "provenance_run_id": source.get("run_id"),
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                counts[scenario] += 1
    total = sum(counts.values())
    if total != 80000:
        raise SystemRepairError(f"final Week 5 result count changed: {total}")
    summary = {
        "status": "COMPLETED",
        "repair_id": repair["repair_id"],
        "schema_valid": total,
        "unresolved": 0,
        "counts": dict(counts),
        "human_accepted_unchanged": {
            "image_product_search": 100,
            "after_sales": 100,
            "itinerary_planning": 100,
            "dialogue": 100,
        },
        "result_sha256": sha256_file(final_path),
    }
    _write_json_new(output / "final_summary.json", summary)
    return summary


def evaluate_system_release_gates(
    config: dict[str, Any],
    candidate: dict[str, Any],
    existing: dict[str, Any],
    zero_shot: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate predeclared core and DIALOGUE_BETA gates from observed metrics."""
    failures = []
    non_regression = config["evaluation"]["non_regression"]
    for scenario in SCENARIOS:
        current = candidate["scenarios"][scenario]
        best_baseline = max(
            float(existing["scenarios"][scenario]["composite"]),
            float(zero_shot["scenarios"][scenario]["composite"]),
        )
        if float(current["composite"]) < best_baseline:
            failures.append(f"{scenario}:composite_below_best_baseline")
        aggregate = current.get("aggregate", current)
        for metric in ("json_compliance", "schema_pass"):
            threshold = float(non_regression[f"minimum_{metric}"])
            if float(aggregate[metric]) < threshold:
                failures.append(f"{scenario}:{metric}_below_{threshold}")
        if int(aggregate.get("sample_count", 0)) != int(
            config["dataset"]["development_per_core_scenario"]
        ):
            failures.append(f"{scenario}:development_support_count_changed")
    if float(candidate["failure_rate"]) > float(non_regression["max_failure_rate"]):
        failures.append("failure_rate_above_limit")
    candidate_latency = float(
        candidate.get("latency_ms_mean", candidate.get("mean_latency_ms", 0.0))
    )
    existing_latency = float(
        existing.get("latency_ms_mean", existing.get("mean_latency_ms", 0.0))
    )
    if candidate_latency > existing_latency * 1.25:
        failures.append("latency_ratio_above_1.25")
    product = candidate["scenarios"]["image_product_search"]
    product_support = product.get("metric_support", product.get("support", {}))
    support_fields = {
        "style_tags": "style_f1",
        "visible_facilities": "facility_f1",
        "price_range": "price_range_accuracy",
    }
    for field, minimum in config["evaluation"]["product_minimum_development_support"].items():
        support_key = support_fields[field]
        support = product_support.get(support_key, product_support.get(field, 0))
        if int(support) < int(minimum):
            failures.append(f"image_product_search:{field}_support_below_{minimum}")

    dialogue = candidate["dialogue"]
    gate = config["evaluation"]["dialogue_automatic_gate"]
    dialogue_checks = {
        "automatic_composite": (">=", gate["minimum_automatic_composite"]),
        "format_compliance": (">=", gate["minimum_format_compliance"]),
        "context_recall": (">=", gate["minimum_context_recall"]),
        "context_state_value_accuracy": (">=", gate["minimum_context_state_value_accuracy"]),
        "task_result_key_coverage": (">=", gate["minimum_task_result_key_coverage"]),
        "task_result_value_accuracy": (">=", gate["minimum_task_result_value_accuracy"]),
        "sequential_protocol_coverage": (">=", gate["minimum_sequential_protocol_coverage"]),
        "sequential_semantic_accuracy": (">=", gate["minimum_sequential_semantic_accuracy"]),
        "tool_protocol_compliance": (">=", gate["minimum_tool_protocol_compliance"]),
        "failure_rate": ("<=", gate["maximum_failure_rate"]),
    }
    for name, (operator, threshold) in dialogue_checks.items():
        if name == "failure_rate" and name not in dialogue:
            scores = dialogue.get("scores", [])
            value = (
                sum(bool(item.get("failed")) for item in scores) / len(scores)
                if scores
                else 1.0
            )
        else:
            value = float(dialogue[name])
        passed = value >= float(threshold) if operator == ">=" else value <= float(threshold)
        if not passed:
            failures.append(f"dialogue:{name}_{operator}_{threshold}")
    for baseline_name, baseline in (("existing", existing), ("zero_shot", zero_shot)):
        if float(dialogue["automatic_composite"]) <= float(baseline["dialogue"]["automatic_composite"]):
            failures.append(f"dialogue:not_above_{baseline_name}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "test_consumption_allowed": not failures,
    }


def _select_replacements(
    root: Path,
    config: dict[str, Any],
    failed_candidates: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    exclusions = load_exclusions(root, config)
    used_sources = {row["source_id"] for rows in pools.values() for row in rows}
    used_hashes = {row["image_sha256"] for rows in pools.values() for row in rows}
    businesses = _load_businesses(root / config["paths"]["businesses"])
    ranked = []
    photos = parquet.ParquetFile(root / config["paths"]["photos"])
    for batch in photos.iter_batches(
        batch_size=8192,
        columns=["photo_id", "business_id", "image_path"],
    ):
        for photo in batch.to_pylist():
            business = businesses.get(photo["business_id"])
            if business is None:
                continue
            source_id = f"yelp-photo:{photo['photo_id']}"
            group_id = f"yelp-business:{photo['business_id']}"
            if source_id in used_sources or source_id in exclusions["source_id"]:
                continue
            if group_id in exclusions["group_id"]:
                continue
            rank = hashlib.sha256(
                f"20260824\0{photo['photo_id']}".encode("utf-8")
            ).hexdigest()
            ranked.append((rank, photo, business))
    ranked.sort(key=lambda item: item[0])
    replacements = []
    consumed_indexes: set[int] = set()
    for old in failed_candidates:
        selected = None
        for index, (_, photo, business) in enumerate(ranked):
            if index in consumed_indexes:
                continue
            if (
                old["scenario"] == "image_product_search"
                and business["ota_category"]
                != old["sampling_metadata"]["business_category"]
            ):
                continue
            image_path = str(photo["image_path"]).replace("\\", "/")
            path = root / image_path
            if not _readable_image(path):
                continue
            digest = sha256_file(path)
            if digest in used_hashes or digest in exclusions["image_sha256"]:
                continue
            selected = (index, photo, business, image_path, digest)
            break
        if selected is None:
            raise SystemRepairError(f"no replacement candidate for {old['sample_id']}")
        index, photo, business, image_path, digest = selected
        consumed_indexes.add(index)
        used_hashes.add(digest)
        used_sources.add(f"yelp-photo:{photo['photo_id']}")
        row = copy.deepcopy(old)
        row["replaces_sample_id"] = old["sample_id"]
        row["sample_id"] = f"week5r2-{old['scenario']}-{digest[:20]}"
        row["source_id"] = f"yelp-photo:{photo['photo_id']}"
        row["image_sha256"] = digest
        row["dataset_version"] = "week5_preannotation_repair_v2"
        row["input"]["images"] = [{"path": image_path, "sha256": digest}]
        row["provenance"]["group_id"] = f"yelp-business:{photo['business_id']}"
        row["workflow"] = {
            key: "pending" for key in row.get("workflow", {"model_preannotation": "pending"})
        }
        if old["scenario"] == "image_product_search":
            row["sampling_metadata"] = {
                "business_category": business["ota_category"],
                "style_hint": _style_hint(business),
                "price_hint": PRICE_MAP.get(
                    business.get("attr_RestaurantsPriceRange2"), "unknown"
                ),
                "city": business.get("city") or "unknown",
                "hints_are_gold": False,
            }
        else:
            metadata = row["sampling_metadata"]
            city = business.get("city") or "目的地待定"
            metadata["city"] = city
            row["input"]["text_constraints"] = (
                f"计划{metadata['trip_days']}天前往{city}，{metadata['travel_group']}出行，"
                f"预算档位为{metadata['budget_tier']}；偏好慢节奏，优先公共交通，"
                "每日包含用餐安排。"
            )
        replacements.append(row)
    return replacements


def _audit_repaired_pools(
    root: Path,
    config: dict[str, Any],
    pools: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = [row for values in pools.values() for row in values]
    exclusions = load_exclusions(root, config)
    dimensions: dict[str, set[str]] = {field: set() for field in IDENTITY_FIELDS}
    for row in rows:
        values = {
            "sample_id": row["sample_id"],
            "source_id": row["source_id"],
            "image_sha256": row["image_sha256"],
            "group_id": row["provenance"].get("group_id"),
            "constraint_template_id": row["provenance"].get("constraint_template_id"),
        }
        for field, value in values.items():
            if value and field in exclusions and value in exclusions[field]:
                raise SystemRepairError(f"evaluation {field} collision: {value}")
            if field in {"sample_id", "source_id", "image_sha256"} and value in dimensions[field]:
                raise SystemRepairError(f"duplicate {field}: {value}")
            if value:
                dimensions[field].add(value)
    if len(rows) != 80000:
        raise SystemRepairError(f"repaired pool count changed: {len(rows)}")
    return {
        "status": "PASS",
        "candidate_count": len(rows),
        "dimension_unique_counts": {
            field: len(values) for field, values in dimensions.items()
        },
        "evaluation_conflicts": {field: 0 for field in IDENTITY_FIELDS},
    }


def _validate_historical_failure_contract(
    failures: list[dict[str, Any]],
    successful: list[dict[str, Any]],
    expected: dict[str, int],
) -> None:
    failure_types = Counter(row.get("error_type") for row in failures)
    actual = {
        "input_replacements": failure_types["input_error"],
        "schema_retries": failure_types["schema_error"],
        "json_retries": failure_types["json_parse_error"],
        "repair_queue": len(failures),
        "historical_schema_valid": len(successful),
    }
    for field, value in actual.items():
        if value != expected[field]:
            raise SystemRepairError(
                f"historical {field} changed: expected {expected[field]}, got {value}"
            )
    if len({row["sample_id"] for row in failures}) != len(failures):
        raise SystemRepairError("historical failures contain duplicate sample IDs")


def _readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError):
        return False


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemRepairError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise SystemRepairError(f"JSONL row is not an object: {path}:{line_number}")
            yield value


def _write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

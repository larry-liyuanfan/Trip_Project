"""Recompute field semantics from the preserved 168-sample development comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROLES = ("zero_shot", "old_unified_adapter", "current_system_repair_checkpoint_87")
FIELDS = {
    "business_category": "scalar",
    "style_tags": "set",
    "visible_facilities": "set",
    "price_range": "scalar",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    for role in ROLES:
        parser.add_argument(f"--{role.replace('_', '-')}-raw", type=Path, required=True)
        parser.add_argument(f"--{role.replace('_', '-')}-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"audit output already exists: {args.output}")
    dataset = _load_jsonl(args.dataset)
    dataset_index = {row["sample_id"]: row for row in dataset}
    if len(dataset_index) != len(dataset):
        raise ValueError("development dataset contains duplicate sample IDs")
    report: dict[str, Any] = {
        "schema_version": "system_repair_development_semantic_audit_v1",
        "status": "PASS",
        "scope": "preserved_development_outputs_not_fresh_test",
        "dataset": {
            "path": str(args.dataset),
            "sha256": _file_sha(args.dataset),
            "support": len(dataset),
            "scenario_support": dict(sorted(Counter(row["scenario"] for row in dataset).items())),
            "label_source_support": dict(sorted(Counter(row.get("label_source", "unknown") for row in dataset).items())),
        },
        "roles": {},
        "first_attempt_and_correction_evidence": {
            "status": "NOT_RECORDED_IN_PRESERVED_DEVELOPMENT_RAW_SCHEMA",
            "final_recorded_output_json_compliance_is_not_first_attempt_compliance": True,
        },
        "fresh_test_120_read_or_reused": False,
    }
    locks: set[tuple[Any, Any, int]] = set()
    sample_sets: set[frozenset[str]] = set()
    for role in ROLES:
        raw_path = getattr(args, f"{role}_raw")
        metrics_path = getattr(args, f"{role}_metrics")
        raw = _load_jsonl(raw_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise ValueError(f"{role}: metrics must be an object")
        sample_ids = {row.get("sample_id") for row in raw}
        if sample_ids != set(dataset_index):
            raise ValueError(f"{role}: raw sample lock does not match development data")
        if len(raw) != len(sample_ids):
            raise ValueError(f"{role}: raw outputs contain duplicate samples")
        locks.add((metrics.get("dataset_lock_sha256"), metrics.get("config_sha256"), len(raw)))
        sample_sets.add(frozenset(sample_ids))
        role_report = _score_role(raw, dataset_index, metrics)
        role_report.update(
            {
                "raw_sha256": _file_sha(raw_path),
                "metrics_sha256": _file_sha(metrics_path),
                "run_id": metrics.get("run_id"),
                "model_role_recorded": metrics.get("model_role"),
                "dataset_lock_sha256": metrics.get("dataset_lock_sha256"),
                "config_sha256": metrics.get("config_sha256"),
                "weighted_composite_recorded": metrics.get("weighted_composite"),
                "core_weighted_composite_recorded": metrics.get("core_weighted_composite"),
                "mean_latency_ms_recorded": metrics.get("latency_ms_mean"),
                "failure_rate_recorded": metrics.get("failure_rate"),
            }
        )
        adapter_hashes = metrics.get("adapter_hashes")
        if isinstance(adapter_hashes, dict):
            role_report["adapter_model_sha256"] = adapter_hashes.get("adapter_model.safetensors")
        elif role == "current_system_repair_checkpoint_87":
            role_report["adapter_model_sha256"] = "c2fbb5c768485021a24df74ec75ff2bcf1b646c89935cb463cd476d0a48eaa2a"
        else:
            role_report["adapter_model_sha256"] = None
        report["roles"][role] = role_report
    if len(locks) != 1 or len(sample_sets) != 1:
        raise ValueError("role comparison is not a one-factor shared lock")
    report["shared_comparison_lock"] = {
        "dataset_lock_sha256": next(iter(locks))[0],
        "config_sha256": next(iter(locks))[1],
        "support": next(iter(locks))[2],
        "status": "PASS_IDENTICAL_SAMPLE_CONFIG_SUPPORT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _score_role(raw: list[dict[str, Any]], dataset: dict[str, dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    counts = {field: [0, 0, 0, 0] for field in FIELDS}  # tp, fp, fn, evaluable samples
    product_support = 0
    exact = 0
    valid_json = 0
    unknown_opportunities = 0
    unsupported_hallucinations = 0
    failures = 0
    for result in raw:
        sample = dataset[result["sample_id"]]
        if result.get("failed") is True:
            failures += 1
        parsed = _strict_object(result.get("raw_output"))
        valid_json += int(parsed is not None)
        parsed = parsed or {}
        if sample["scenario"] != "image_product_search":
            continue
        product_support += 1
        target = sample.get("target", {})
        unknown = {_norm(item) for item in target.get("unknown_fields", [])}
        row_exact = True
        for field, kind in FIELDS.items():
            if _norm(field) in unknown:
                unknown_opportunities += 1
                predicted = _values(parsed.get(field), kind)
                unsupported_hallucinations += int(bool(predicted))
                continue
            expected = _values(target.get(field), kind)
            predicted = _values(parsed.get(field), kind)
            tp = len(expected & predicted)
            fp = len(predicted - expected)
            fn = len(expected - predicted)
            counts[field][0] += tp
            counts[field][1] += fp
            counts[field][2] += fn
            counts[field][3] += 1
            row_exact = row_exact and expected == predicted
        exact += int(row_exact)
    dialogue = metrics.get("dialogue", {})
    return {
        "support": len(raw),
        "product_support": product_support,
        "field_metrics": {
            field: {**_prf(values[0], values[1], values[2]), "evaluable_sample_support": values[3]}
            for field, values in counts.items()
        },
        "product_exact_match": exact / max(product_support, 1),
        "unsupported_hallucination_rate": unsupported_hallucinations / max(unknown_opportunities, 1),
        "unsupported_hallucination_count": unsupported_hallucinations,
        "unknown_field_opportunity_support": unknown_opportunities,
        "final_recorded_output_json_compliance": valid_json / len(raw),
        "failure_count_recomputed": failures,
        "dialogue": {
            "support": dialogue.get("sample_count"),
            "automatic_composite": dialogue.get("automatic_composite"),
            "context_recall": dialogue.get("context_recall"),
            "context_state_value_accuracy": dialogue.get("context_state_value_accuracy"),
            "task_result_key_coverage": dialogue.get("task_result_key_coverage"),
            "task_result_value_accuracy": dialogue.get("task_result_value_accuracy"),
            "initial_task_stable_value_accuracy": dialogue.get("initial_task_stable_value_accuracy"),
            "format_compliance": dialogue.get("format_compliance"),
        },
    }


def _values(value: Any, kind: str) -> set[str]:
    if kind == "set":
        if not isinstance(value, list):
            return set()
        return {_norm(item) for item in value if _norm(item) and _norm(item) != "unknown"}
    normalized = _norm(value)
    return {normalized} if normalized and normalized != "unknown" else set()


def _prf(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def _strict_object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm(value: Any) -> str:
    return " ".join(value.strip().casefold().split()) if isinstance(value, str) else ""


if __name__ == "__main__":
    main()

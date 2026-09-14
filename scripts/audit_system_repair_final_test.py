"""Offline-only semantic audit of an already-consumed system-repair final test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.audit_system_repair_development import (
        FIELDS,
        _file_sha,
        _load_jsonl,
        _norm,
        _score_role,
        _strict_object,
        _values,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ rather than the repo root first.
    from audit_system_repair_development import (
        FIELDS,
        _file_sha,
        _load_jsonl,
        _norm,
        _score_role,
        _strict_object,
        _values,
    )


def audit_final_test(
    dataset_path: Path,
    raw_path: Path,
    metrics_path: Path,
    gate_path: Path,
    consumption_path: Path,
    *,
    implementation_commit_sha: str,
    run_source_snapshot_sha256: str,
) -> dict[str, Any]:
    dataset = _load_jsonl(dataset_path)
    raw = _load_jsonl(raw_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    consumption = json.loads(consumption_path.read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (metrics, gate, consumption)):
        raise ValueError("metrics, gate, and consumption records must be objects")
    dataset_index = {row.get("sample_id"): row for row in dataset}
    raw_ids = [row.get("sample_id") for row in raw]
    if len(dataset_index) != len(dataset) or len(set(raw_ids)) != len(raw):
        raise ValueError("final-test dataset or raw outputs contain duplicate sample IDs")
    if set(raw_ids) != set(dataset_index):
        raise ValueError("final-test raw sample lock does not match the dataset")
    if metrics.get("split") != "test" or metrics.get("status") != "COMPLETED":
        raise ValueError("metrics do not describe a completed test run")
    if metrics.get("sample_count") != len(raw):
        raise ValueError("metrics sample count does not match raw outputs")
    recorded_raw = metrics.get("raw_outputs", {})
    raw_sha256 = _file_sha(raw_path)
    if recorded_raw.get("count") != len(raw) or recorded_raw.get("sha256") != raw_sha256:
        raise ValueError("metrics raw-output identity mismatch")
    run_id = metrics.get("run_id")
    if not run_id or {row.get("run_id") for row in raw} != {run_id}:
        raise ValueError("raw outputs do not share the recorded final-test run ID")

    score = _score_role(raw, dataset_index, metrics)
    product_metrics = metrics.get("scenarios", {}).get("image_product_search", {})
    recorded_composite = product_metrics.get("composite")
    if not isinstance(recorded_composite, (int, float)):
        raise ValueError("recorded image-product composite is missing")
    return {
        "schema_version": "system_repair_final_test_offline_audit_v1",
        "status": "PASS_HISTORICAL_FINAL_TEST_OFFLINE_AUDIT_ONLY",
        "scope": "already_consumed_final_test_no_model_execution_no_tuning",
        "promotion_effect": "NONE_HISTORICAL_AUDIT_ONLY",
        "implementation_commit_sha": implementation_commit_sha,
        "run_source_snapshot_sha256": run_source_snapshot_sha256,
        "source_identity": {
            "dataset_sha256": _file_sha(dataset_path),
            "raw_outputs_sha256": raw_sha256,
            "metrics_sha256": _file_sha(metrics_path),
            "gate_sha256": _file_sha(gate_path),
            "consumption_sha256": _file_sha(consumption_path),
            "sample_support": len(raw),
            "run_id": run_id,
            "dataset_lock_sha256": metrics.get("dataset_lock_sha256"),
            "config_sha256": metrics.get("config_sha256"),
            "adapter_model_sha256": metrics.get("adapter_hashes", {}).get(
                "adapter_model.safetensors"
            ),
        },
        "historical_image_product_composite": {
            "recorded": recorded_composite,
            "reported_six_decimals": round(float(recorded_composite), 6),
            "status": "RECONCILED_TO_PRESERVED_METRICS_AND_RAW_IDENTITY",
        },
        "recomputed_semantics": score,
        "error_slices": _error_slices(raw, dataset_index),
        "first_attempt_and_correction": "NOT_RECORDED_IN_PRESERVED_FINAL_RAW_SCHEMA",
        "consumption_record_status": consumption.get("status"),
        "gate_record_status": gate.get("status"),
    }


def _error_slices(
    raw: list[dict[str, Any]],
    dataset: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    field_mismatch_ids = {field: [] for field in FIELDS}
    invalid_json_ids: list[str] = []
    failed_ids: list[str] = []
    unknown_hallucination_ids: list[str] = []
    for result in raw:
        sample_id = result["sample_id"]
        if result.get("failed") is True:
            failed_ids.append(sample_id)
        parsed = _strict_object(result.get("raw_output"))
        if parsed is None:
            invalid_json_ids.append(sample_id)
            parsed = {}
        sample = dataset[sample_id]
        if sample.get("scenario") != "image_product_search":
            continue
        target = sample.get("target", {})
        unknown = {_norm(item) for item in target.get("unknown_fields", [])}
        for field, kind in FIELDS.items():
            predicted = _values(parsed.get(field), kind)
            if _norm(field) in unknown:
                if predicted:
                    unknown_hallucination_ids.append(sample_id)
                continue
            expected = _values(target.get(field), kind)
            if expected != predicted:
                field_mismatch_ids[field].append(sample_id)
    return {
        "field_mismatch": {
            field: {"count": len(ids), "sample_ids": ids}
            for field, ids in field_mismatch_ids.items()
        },
        "unknown_field_hallucination": {
            "count": len(unknown_hallucination_ids),
            "sample_ids": unknown_hallucination_ids,
        },
        "invalid_final_json": {"count": len(invalid_json_ids), "sample_ids": invalid_json_ids},
        "operational_failure": {"count": len(failed_ids), "sample_ids": failed_ids},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--consumption", type=Path, required=True)
    parser.add_argument("--implementation-commit-sha", required=True)
    parser.add_argument("--run-source-snapshot-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"audit output already exists: {args.output}")
    report = audit_final_test(
        args.dataset,
        args.raw,
        args.metrics,
        args.gate,
        args.consumption,
        implementation_commit_sha=args.implementation_commit_sha,
        run_source_snapshot_sha256=args.run_source_snapshot_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

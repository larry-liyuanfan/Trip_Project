"""Score fixed performance cells and combine them with the locked VLM quality gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evidence_v2 import summarize_performance_matrix
from src.evaluation.relevance_evidence import canonical_json_sha256, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = [row for path in args.result for row in load_jsonl(path)]
    roles = ("old_unified_adapter", "current_system_repair_checkpoint_87")
    profiles = [row["profile_id"] for row in config["performance"]["profiles"]]
    realized_lengths = validate_realized_lengths(rows, config["performance"]["profiles"])
    for role in roles:
        for profile in profiles:
            for declaration in config["performance"]["declared_not_run"]:
                rows.append({
                    "role": role,
                    "profile_id": profile,
                    "concurrency": declaration["concurrency"],
                    "transport": declaration["transport"],
                    "status": "NOT_RUN",
                    "reason": declaration["reason"],
                    "production_sla_supported": False,
                })
    matrix = summarize_performance_matrix(rows, config["performance"]["fixed_gates"])
    comparisons: dict[str, Any] = {}
    ratio_limit = float(config["performance"]["fixed_gates"]["max_candidate_to_baseline_p95_ratio"])
    for profile in profiles:
        old_key = f"old_unified_adapter|{profile}|1|component_milvus_lite"
        current_key = f"current_system_repair_checkpoint_87|{profile}|1|component_milvus_lite"
        old = matrix["measured_cells"][old_key]
        current = matrix["measured_cells"][current_key]
        old_p95 = old["stages"]["steady"]["end_to_end_ms"]["p95"]
        current_p95 = current["stages"]["steady"]["end_to_end_ms"]["p95"]
        ratio = current_p95 / max(old_p95, 1e-12)
        checks = {
            "same_hardware_class": old["hardware"] == current["hardware"],
            "same_support": old["support"] == current["support"],
            "current_absolute_gate": current["status"] == "PASS",
            "candidate_to_baseline_p95_ratio": ratio <= ratio_limit,
        }
        comparisons[profile] = {
            "candidate_to_baseline_p95_ratio": ratio,
            "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        }
    quality = json.loads(args.quality_report.read_text(encoding="utf-8"))
    quality_pass = quality.get("fixed_gates", {}).get("status") == "PASS"
    latency_pass = all(row["status"] == "PASS" for row in comparisons.values())
    matrix["comparisons"] = comparisons
    matrix["joint_quality_latency_gate"] = {
        "quality_gate": quality_pass,
        "latency_gate": latency_pass,
        "status": "PASS" if quality_pass and latency_pass else "FAIL",
    }
    matrix["production_sla_supported"] = False
    matrix["realized_token_matrix"] = realized_lengths
    matrix["raw_measured_result_sha256"] = canonical_json_sha256(
        [row for row in rows if row.get("status") == "MEASURED"]
    )
    _write_json(args.output, matrix)
    print(json.dumps(matrix, ensure_ascii=False, indent=2, sort_keys=True))


def validate_realized_lengths(
    rows: list[dict[str, Any]], profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    """Require distinct fixed input and exact forced output supports across roles."""
    profile_map = {row["profile_id"]: row for row in profiles}
    realized: dict[str, Any] = {}
    input_counts: set[int] = set()
    output_counts: set[int] = set()
    for profile_id, profile in profile_map.items():
        support = [row for row in rows if row.get("profile_id") == profile_id]
        if not support:
            raise ValueError(f"missing measured profile: {profile_id}")
        inputs = {int(row["input_token_count"]) for row in support}
        outputs = {int(row["output_token_count"]) for row in support}
        expected_output = int(profile["output_new_tokens"])
        if len(inputs) != 1:
            raise ValueError(f"input token count differs within profile: {profile_id}")
        if outputs != {expected_output}:
            raise ValueError(f"fixed output token count mismatch: {profile_id}")
        if any(row.get("forced_output_length") is not True for row in support):
            raise ValueError(f"profile lacks forced output evidence: {profile_id}")
        input_count = next(iter(inputs))
        input_counts.add(input_count)
        output_counts.add(expected_output)
        realized[profile_id] = {
            "row_support": len(support),
            "input_token_count": input_count,
            "output_token_count": expected_output,
            "input_padding_repetitions": profile["input_padding_repetitions"],
        }
    if len(input_counts) != len(profile_map):
        raise ValueError("performance profiles did not realize distinct input token lengths")
    if len(output_counts) != len(profile_map):
        raise ValueError("performance profiles did not realize distinct output token lengths")
    return realized


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

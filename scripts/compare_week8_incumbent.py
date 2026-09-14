"""Compare development-only revisions against the explicitly named incumbent."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_week8_development_revision import SEMANTIC_FIELDS
from src.data.week8_visual_holdout import write_json_new
from src.evaluation.week8_visual_silver import select_development_candidate
from src.training.week7_data import sha256_file


def compare(comparison, incumbent_role):
    if comparison.get("test_rows_read") is not False:
        raise ValueError("only development comparisons are allowed")
    summaries = comparison["summaries"]
    formal = select_development_candidate(summaries)
    incumbent = summaries[incumbent_role]
    if formal["failures"][incumbent_role]:
        raise ValueError("incumbent does not reproduce its development eligibility")
    results = {}
    for role, after in summaries.items():
        if role in {"formal_adapter", incumbent_role}:
            continue
        if after["supports"] != incumbent["supports"] or after["reference_audit"] != incumbent["reference_audit"]:
            raise ValueError("incumbent and revision must use identical references and support")
        failures = list(formal["failures"][role])
        for field in SEMANTIC_FIELDS:
            before_value, after_value = incumbent["metrics"][field], after["metrics"][field]
            if before_value is not None and (after_value is None or after_value < before_value):
                failures.append(field + "_below_incumbent")
        before_evidence = incumbent.get("evidence_consistency")
        after_evidence = after.get("evidence_consistency")
        if (before_evidence is None) != (after_evidence is None):
            raise ValueError("incumbent and revision must use the same target-free evidence audit")
        if before_evidence is not None:
            identity = ("protocol", "records_read", "successful_observations", "target_free", "selection_use")
            if any(before_evidence.get(key) != after_evidence.get(key) for key in identity):
                raise ValueError("target-free evidence audit identity changed")
            for key in ("inconsistent_evidence_labels", "samples_with_errors"):
                if after_evidence.get(key, 0) > before_evidence.get(key, 0):
                    failures.append("target_free_" + key + "_increased")
        semantic_gain = after["metrics"]["composite"] > incumbent["metrics"]["composite"]
        mean_ratio = after["latency_ms"]["mean"] / incumbent["latency_ms"]["mean"]
        token_ratio = after["tokens"]["output_mean"] / incumbent["tokens"]["output_mean"]
        speed_gain = (mean_ratio <= 0.95 and token_ratio <= 0.90
                      and all(after["latency_ms"][k] <= incumbent["latency_ms"][k] for k in ("p50", "p95")))
        if not semantic_gain and not speed_gain:
            failures.append("no_material_quality_or_speed_gain")
        results[role] = {"eligible": not failures, "failures": failures, "semantic_gain": semantic_gain,
                         "latency_mean_ratio": mean_ratio, "output_token_ratio": token_ratio,
                         "speed_gain": speed_gain}
    eligible = [role for role, value in results.items() if value["eligible"]]
    chosen = min(eligible, key=lambda role: (-summaries[role]["metrics"]["composite"], summaries[role]["latency_ms"]["mean"])) if eligible else None
    return {"status": "IMPROVED_DEVELOPMENT_CANDIDATE" if chosen else "KEEP_INCUMBENT_CANDIDATE",
            "selected_role": chosen, "incumbent_role": incumbent_role, "candidates": results,
            "test_rows_read": False, "promotion_allowed": False, "release_changed": False,
            "human_visual_accuracy_claim": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--incumbent-role", default="observation_base")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.comparison.read_text(encoding="utf-8")), args.incumbent_role)
    result["comparison_sha256"] = sha256_file(args.comparison)
    write_json_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False))

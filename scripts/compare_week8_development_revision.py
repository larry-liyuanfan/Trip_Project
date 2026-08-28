"""Compare a genuine development revision before consuming another final identity."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.week8_visual_holdout import read_json, write_json_new
from src.evaluation.week8_visual_silver import select_development_candidate
from src.training.week7_data import sha256_file


SEMANTIC_FIELDS = ("business_category_accuracy", "business_category_including_unknown_accuracy",
                   "style_precision", "style_recall", "style_f1", "facility_precision", "facility_recall",
                   "facility_f1", "price_range_accuracy", "unknown_accuracy", "label_completeness")


def compare(previous, current):
    old_selection = select_development_candidate(previous["summaries"])
    new_selection = select_development_candidate(current["summaries"])
    if not old_selection["selected_role"] or not new_selection["selected_role"]:
        return {"status": "NO_IMPROVED_DEVELOPMENT_REVISION", "reason": "no_eligible_candidate", "new_final_allowed": False}
    before = previous["summaries"][old_selection["selected_role"]]
    after = current["summaries"][new_selection["selected_role"]]
    if before["supports"] != after["supports"] or before["reference_audit"] != after["reference_audit"]:
        raise ValueError("revision requires the same fixed development and reference")
    deltas = {key: after["metrics"][key] - value if value is not None else None
              for key, value in before["metrics"].items()}
    meaningful_semantic_gain = deltas["composite"] > 0
    semantic_nonregression = all(deltas[key] is None or deltas[key] >= 0 for key in SEMANTIC_FIELDS)
    latency_ratio = after["latency_ms"]["mean"] / before["latency_ms"]["mean"]
    token_ratio = after["tokens"]["output_mean"] / before["tokens"]["output_mean"]
    # 性能候选必须在同组语义不回退，并有 token 缩减佐证，避免把微小时延抖动当新候选。
    meaningful_speed_gain = (semantic_nonregression and latency_ratio <= 0.95 and token_ratio <= 0.90
                            and all(after["latency_ms"][key] <= before["latency_ms"][key] for key in ("p50", "p95")))
    accepted = meaningful_semantic_gain or meaningful_speed_gain
    return {"status": "IMPROVED_DEVELOPMENT_REVISION" if accepted else "NO_IMPROVED_DEVELOPMENT_REVISION",
            "new_final_allowed": accepted, "metric_deltas": deltas, "latency_mean_ratio": latency_ratio,
            "output_token_ratio": token_ratio, "semantic_gain": meaningful_semantic_gain,
            "speed_gain_without_semantic_regression": meaningful_speed_gain,
            "test_rows_read": False, "reference_changed": False, "human_visual_accuracy_claim": False}


def run(previous, current, output):
    result = compare(read_json(previous), read_json(current))
    result.update(previous_comparison_sha256=sha256_file(previous), current_comparison_sha256=sha256_file(current))
    write_json_new(output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.previous, args.current, args.output)

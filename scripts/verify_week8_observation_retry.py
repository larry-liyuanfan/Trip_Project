"""Replay the correction diagnostic without model requests or final data."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_week8_observation_retry import load_cases, load_continuation
from src.data.week8_visual_holdout import read_json, within, write_json_new
from src.training.week7_data import iter_jsonl, sha256_file
from src.inference.product_observation import canonical_config_sha256, observation_messages, observation_correction_messages, parse_observation, map_observation
from src.inference.product_observation import observation_correction_response_format


def replay_records(root, cases, records, observation, generation_root=None):
    expected = {case["case_id"]: case for case in cases}
    if len(records) != len(cases) or {row["case_id"] for row in records} != set(expected):
        raise ValueError("all correction cases must be retained exactly once")
    failures, latencies, tokens = 0, [], []
    for row in records:
        case = expected[row["case_id"]]
        image = str(generation_root or root).replace("\\", "/").rstrip("/") + "/" + case["image_path"]
        messages = observation_correction_messages(observation_messages(image, observation),
            case["previous_raw"], case["validation_error"], observation)
        if row["sample_id"] != case["sample_id"] or row["input_messages_sha256"] != canonical_config_sha256(messages):
            raise ValueError("correction model inputs differ from locked development case")
        response_format = observation_correction_response_format(observation)
        if (response_format is not None or "response_format_sha256" in row) and row.get("response_format_sha256") != canonical_config_sha256(response_format):
            raise ValueError("correction decoder constraints differ from the tested protocol")
        value = None
        if row.get("raw_output") is not None:
            try:
                value = map_observation(parse_observation(row["raw_output"], observation), observation)
            except ValueError as exc:
                if row.get("passed") or row.get("error") != str(exc):
                    raise ValueError("invalid model output was masked or changed") from exc
        if row.get("passed"):
            if value is None or value != row.get("result") or row.get("error") is not None:
                raise ValueError("passing correction differs from real model raw output")
        elif value is not None or not row.get("error"):
            raise ValueError("correction failure differs from raw replay")
        failures += int(not row["passed"])
        latency = row["elapsed_ms"]
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("invalid diagnostic latency")
        latencies.append(latency)
        tokens.append((row.get("input_tokens") or 0, row.get("output_tokens") or 0))
    return {"count": len(records), "failures": failures, "mean_ms": statistics.fmean(latencies),
            "input_tokens": sum(pair[0] for pair in tokens), "output_tokens": sum(pair[1] for pair in tokens)}


def verify(root, config_path, generation_root=None):
    config = read_json(config_path)
    cases, source_audit = load_cases(root, config)
    output = within(root, config["output_root"])
    identity, summary = read_json(output / "identity.json"), read_json(output / "summary.json")
    preserved, continuation = load_continuation(root, config, cases, source_audit, generation_root)
    if identity.get("continuation") != continuation or summary.get("continuation") != continuation:
        raise ValueError("continuation provenance differs from preserved partial evidence")
    if (identity["config_sha256"] != sha256_file(config_path) or identity["final_test_access"] is not False
            or identity["reference_targets_supplied"] is not False or identity["human_annotation_count"] != 0
            or identity["runner_sha256"] != sha256_file(root / "scripts/review_week8_observation_retry.py")
            or identity["correction_implementation_sha256"] != sha256_file(root / "src/inference/product_observation.py")
            or identity.get("decoder_implementation_sha256") != sha256_file(root / "src/inference/observation_constraints.py")
            or identity.get("backend_implementation_sha256") != sha256_file(root / "src/inference/system_runtime.py")
            or any(identity.get(key) != value for key, value in source_audit.items())
            or summary["status"] != "COMPLETED" or summary["final_test_access"] is not False
            or summary["case_manifest_sha256"] != sha256_file(output / "cases.jsonl")
            or list(iter_jsonl(output / "cases.jsonl")) != cases or set(summary["profiles"]) != set(config["profiles"])):
        raise ValueError("correction diagnostic identity or coverage changed")
    results = {}
    for name, path in config["profiles"].items():
        if identity["profile_config_hashes"][name] != sha256_file(within(root, path)):
            raise ValueError("tested correction configuration changed")
        raw_path = within(output, name + ".jsonl")
        declared = summary["profiles"][name]
        if declared["raw_sha256"] != sha256_file(raw_path):
            raise ValueError("diagnostic raw changed")
        records = list(iter_jsonl(raw_path))
        if records[:len(preserved.get(name, []))] != preserved.get(name, []):
            raise ValueError("completed diagnostic altered preserved prefix records")
        results[name] = replay_records(root, cases, records, read_json(within(root, path)), generation_root)
        if any(results[name][key] != declared[key] for key in ("count", "failures")):
            raise ValueError("diagnostic reported counts differ from raw replay")
    return {"status": "REPLAY_VERIFIED", "profiles": results, "case_count": len(cases),
            "unique_development_images": source_audit["unique_development_images"],
            "case_manifest_sha256": summary["case_manifest_sha256"], "summary_sha256": sha256_file(output / "summary.json"),
            "reference_targets_supplied": False, "final_test_access": False,
            "continuation": continuation, "execution_interruptions": int(continuation is not None),
            "product_quality_selection_requires_complete_fixed_development": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-root", help="Original execution root for exact transport-URL input hash replay across hosts.")
    args = parser.parse_args()
    result = verify(ROOT, args.config.resolve(), args.generation_root)
    write_json_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False))

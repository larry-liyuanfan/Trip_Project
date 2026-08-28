"""Audit repeated development correction cases; never select visual quality from this subset."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_week8_observation_retry import load_cases
from scripts.score_week8_reference_revision import verified_references
from scripts.verify_week8_observation_retry import replay_records
from src.data.week8_visual_holdout import read_json, write_json_new
from src.evaluation.week8_visual_silver import _labels
from src.training.week7_data import iter_jsonl, sha256_file


def semantic_counts(cases, records, references):
    expected = [case["case_id"] for case in cases]
    if [row["case_id"] for row in records] != expected or len(set(expected)) != len(expected):
        raise ValueError("correction semantic audit requires the complete ordered case list")
    groups = {}
    for case, record in zip(cases, records):
        target = references[case["sample_id"]]["target"]
        if record["sample_id"] != case["sample_id"]:
            raise ValueError("correction semantic audit sample identity changed")
        prediction = record.get("result") if record["passed"] else None
        for key in ("all_cases", case["validation_error"]):
            value = groups.setdefault(key, {"count": 0, "failures": 0, "category_errors": 0,
                "style": Counter(), "facility": Counter()})
            value["count"] += 1
            value["failures"] += int(prediction is None)
            value["category_errors"] += int(prediction is None or prediction["business_category"] != target["business_category"])
            for metric, field in (("style", "style_tags"), ("facility", "visible_facilities")):
                truth, proposed = _labels(target[field]), _labels((prediction or {}).get(field, []))
                value[metric].update({"tp": len(truth & proposed), "fp": len(proposed - truth), "fn": len(truth - proposed)})
    return groups


def run(config_path, reference_config_path, output, generation_root=None):
    config = read_json(config_path)
    cases, source_audit = load_cases(ROOT, config)
    references, audit, revision, _ = verified_references(reference_config_path)
    if revision["manifest_sha256"] != config["development_manifest_sha256"]:
        raise ValueError("diagnostic reference must use the exact fixed development manifest")
    directory = ROOT / config["output_root"]
    summary = read_json(directory / "summary.json")
    identity = read_json(directory / "identity.json")
    if (summary["status"] != "COMPLETED" or summary["final_test_access"] is not False
            or list(iter_jsonl(directory / "cases.jsonl")) != cases
            or identity["config_sha256"] != sha256_file(config_path)
            or any(identity.get(key) != value for key, value in source_audit.items())):
        raise ValueError("only complete immutable correction evidence can be audited")
    groups, raw_hashes = {}, {}
    for role, observation_path in config["profiles"].items():
        raw_path = directory / (role + ".jsonl")
        if (sha256_file(raw_path) != summary["profiles"][role]["raw_sha256"]
                or identity["profile_config_hashes"][role] != sha256_file(ROOT / observation_path)):
            raise ValueError("correction raw identity changed")
        records = list(iter_jsonl(raw_path))
        replay_records(ROOT, cases, records, read_json(ROOT / observation_path), generation_root)
        groups[role] = semantic_counts(cases, records, {row["sample_id"]: row for row in references})
        raw_hashes[role] = sha256_file(raw_path)
    result = {"scope": "repeated_historical_development_error_cases_not_independent_quality_estimate",
        "profiles": groups, "case_count": len(cases), "unique_images": source_audit["unique_development_images"],
        "source_config_sha256": sha256_file(config_path), "source_summary_sha256": sha256_file(directory / "summary.json"),
        "raw_sha256": raw_hashes, "reference_audit": audit, "test_rows_read": False,
        "human_annotation_count": 0, "label_source": "model_generated_silver", "selection_allowed": False,
        "new_model_requests": 0, "generation_runtime_revalidated": False,
        "interpretation": "Semantic raw replay supplements, and does not replace, the original source-bound execution verification."}
    write_json_new(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-revision-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-root")
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.reference_revision_config.resolve(), args.output.resolve(), args.generation_root), ensure_ascii=False))

"""Replay immutable quality evidence before declaring an automatic-silver candidate."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.week8_visual_holdout import read_json, write_json_new, within
from src.inference.business_validation import itinerary_business_errors
from src.training.week7_data import sha256_file


def validate_test_log(path):
    text = path.read_text(encoding="utf-8")
    counts = re.findall(r"^Ran (\d+) tests? in ([\d.]+)s$", text, re.M)
    if (len(counts) != 1 or int(counts[0][0]) == 0 or not re.search(r"^OK$", text, re.M)
            or re.search(r"^(FAILED|ERROR:|FAIL:)", text, re.M)):
        raise ValueError("full unittest log does not establish a successful run")
    return {"count": int(counts[0][0]), "elapsed_seconds": float(counts[0][1]), "sha256": sha256_file(path)}


def validate_retrieval(value, release_id):
    if (value.get("status") != "PASS" or value.get("production_route_executed") is not True
            or value.get("reference_query_metadata_used") is not False
            or value.get("query_change_changes_results") is not True
            or not value.get("queries") or not all(row.get("filter_correct") for row in value["queries"])):
        raise ValueError("production retrieval is incomplete")
    dialogues = value.get("dialogue_routing", [])
    if not dialogues or not all(row.get("passed") and row["response"].get("tool_calls")
                                and row["response"].get("release_id") == release_id
                                and row["response"].get("task_status") == row["expected_status"] for row in dialogues):
        raise ValueError("retrieval dialogue execution is not bound to this release")


def verify(root, config_path, retrieval_path, tests_path):
    from scripts.run_week8_visual_final import replay_final, validate_runtime_probe, validate_development_identity
    config = read_json(config_path)
    directory = within(root, config["output_root"])
    result = replay_final(root, config_path)
    if result != read_json(directory / "final_comparison.json"):
        raise ValueError("final metrics differ from immutable raw replay")
    if result["acceptance"]["status"] != "PASS":
        raise ValueError("final quality has not passed")
    lock = read_json(directory / "candidate_lock.json")
    comparison = read_json(within(root, config["development_comparison"]))
    if (sha256_file(root / config["development_comparison"]) != lock["development_comparison_sha256"]
            or validate_development_identity(root, config, comparison) != lock["development_generation_identity_sha256"]
            or validate_runtime_probe(root, config) != lock["runtime_probe_files"]):
        raise ValueError("development/runtime evidence changed after lock")
    release = read_json(within(root, config["candidate_release"]))
    probe = within(root, config["runtime_probe"])
    identity = read_json(probe / "identity.json")
    smoke = read_json(probe / "model_smoke.json")
    if identity["release_config_sha256"] != sha256_file(root / config["candidate_release"]):
        raise ValueError("real smoke must execute the exact candidate release")
    for row in [*smoke["scenarios"].values(), smoke["dialogue"]]:
        if row.get("release_id") != release["release_id"]:
            raise ValueError("smoke release identity mismatch")
    dialogue = smoke["dialogue"]
    if (not dialogue.get("tool_calls") or not dialogue.get("attempts")
            or itinerary_business_errors(dialogue["task_result"]["result"], "城市：Shanghai；2天；推荐安静行程")):
        raise ValueError("dialogue did not complete real model-backed itinerary work")
    validate_retrieval(read_json(retrieval_path), release["release_id"])
    tests = validate_test_log(tests_path)
    return {"status": "PASS", "candidate_quality_accepted": True, "release_id": release["release_id"],
            "candidate_lock_sha256": lock["lock_sha256"], "final_comparison_sha256": sha256_file(directory / "final_comparison.json"),
            "release_config_sha256": sha256_file(root / config["candidate_release"]),
            "adapter_model_sha256": release["model"]["adapter_model_sha256"],
            "retrieval_sha256": sha256_file(retrieval_path), "full_unittest": tests,
            "human_annotation_count": 0, "human_visual_accuracy_claim": False,
            "label_source": "model_generated_silver", "test_used_for_tuning": False,
            "formal_release_replaced": False, "package_verification_required": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--unittest-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(ROOT, args.config.resolve(), args.retrieval.resolve(), args.unittest_log.resolve())
    write_json_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False))

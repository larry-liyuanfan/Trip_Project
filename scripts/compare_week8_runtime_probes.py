"""Compare an itinerary-only release derivative against an immutable runtime probe."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_week8_contracts import write_new
from scripts.verify_week8_candidate_runtime import attempt_metrics
from src.training.week7_data import sha256_file


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_itinerary_only_release_change(incumbent, candidate):
    expected = copy.deepcopy(incumbent)
    expected["release_id"] = candidate["release_id"]
    expected["prompts"]["itinerary_planning"] = candidate["prompts"]["itinerary_planning"]
    if expected != candidate:
        raise ValueError("runtime candidate changes more than release id and itinerary prompt")


def probe_details(root, probe_config_path, release_path):
    config = read_json(probe_config_path)
    output = root / config["output_root"]
    identity, summary = read_json(output / "identity.json"), read_json(output / "summary.json")
    if (identity["config_sha256"] != sha256_file(probe_config_path)
            or identity["release_config_sha256"] != sha256_file(release_path)
            or identity["test_rows_read"] is not False or summary["test_rows_read"] is not False
            or summary["status"] != "PASS"):
        raise ValueError("runtime probe identity or status is invalid")
    itineraries = [read_json(output / f"itinerary_{index}.json") for index in range(len(config["itinerary_requests"]))]
    if [item["request"] for item in itineraries] != config["itinerary_requests"]:
        raise ValueError("runtime probe requests changed")
    responses = [item["response"] for item in itineraries]
    smoke = read_json(output / "model_smoke.json")
    if smoke["status"] != "PASS" or smoke["business_status"] != "PASS":
        raise ValueError("runtime model smoke did not pass")
    dialogue = {**smoke["dialogue"], "passed": smoke["dialogue"]["task_status"] == "COMPLETED"}
    return config, {
        "itinerary": attempt_metrics(responses),
        "dialogue_itinerary": attempt_metrics([dialogue]),
        "summary_sha256": sha256_file(output / "summary.json"),
        "model_smoke_sha256": sha256_file(output / "model_smoke.json"),
    }


def compare(config_path, root=ROOT):
    config = read_json(config_path)
    if config["final_test_access"] is not False or config["human_annotation_count"] != 0:
        raise ValueError("runtime comparison must not use final or human evidence")
    incumbent_release_path = root / config["incumbent_release"]
    candidate_release_path = root / config["candidate_release"]
    incumbent_release, candidate_release = read_json(incumbent_release_path), read_json(candidate_release_path)
    validate_itinerary_only_release_change(incumbent_release, candidate_release)
    incumbent_config, incumbent = probe_details(
        root, root / config["incumbent_probe_config"], incumbent_release_path)
    candidate_config, candidate = probe_details(
        root, root / config["candidate_probe_config"], candidate_release_path)
    if incumbent_config["itinerary_requests"] != candidate_config["itinerary_requests"]:
        raise ValueError("runtime probes do not use the same itinerary requests")
    direct_nonregression = (
        candidate["itinerary"]["passed"] == incumbent["itinerary"]["passed"]
        and candidate["itinerary"]["first_attempt_pass"] >= incumbent["itinerary"]["first_attempt_pass"]
        and candidate["itinerary"]["attempts_total"] <= incumbent["itinerary"]["attempts_total"]
    )
    dialogue_improved = (
        candidate["dialogue_itinerary"]["passed"] == incumbent["dialogue_itinerary"]["passed"] == 1
        and candidate["dialogue_itinerary"]["first_attempt_pass"] > incumbent["dialogue_itinerary"]["first_attempt_pass"]
        and candidate["dialogue_itinerary"]["attempts_total"] < incumbent["dialogue_itinerary"]["attempts_total"]
    )
    return {
        "protocol": "week8_itinerary_runtime_derivative_v1",
        "status": "PASS" if direct_nonregression and dialogue_improved else "NO_IMPROVEMENT",
        "selected_release": candidate_release["release_id"] if direct_nonregression and dialogue_improved else incumbent_release["release_id"],
        "incumbent": incumbent,
        "candidate": candidate,
        "direct_itinerary_nonregression": direct_nonregression,
        "dialogue_first_attempt_improved": dialogue_improved,
        "release_change_scope": ["release_id", "prompts.itinerary_planning"],
        "incumbent_release_sha256": sha256_file(incumbent_release_path),
        "candidate_release_sha256": sha256_file(candidate_release_path),
        "config_sha256": sha256_file(config_path),
        "final_test_rows_read": False,
        "human_annotation_count": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.config.resolve())
    output = ROOT / read_json(args.config.resolve())["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(output, result)
    print(json.dumps(result, ensure_ascii=False))

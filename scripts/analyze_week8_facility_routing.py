"""Evaluate target-free facility-review routes on locked development outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_week8_reference_revision import generation_details, verified_references
from src.data.week8_visual_holdout import read_json, write_json_new
from src.evaluation.product_evidence_consistency import evidence_consistency_errors
from src.evaluation.week8_visual_silver import score_paired
from src.inference.product_facility_refinement import (
    _rebuild_base_observation,
    _replay_base,
    replay_facility_refined_observation,
    should_review_facilities,
)
from src.inference.product_observation import map_observation
from src.training.week7_data import iter_jsonl, sha256_file


EMPTY_FACILITY_REVIEW_SUBJECTS = {
    "hotel_space", "dining_space", "retail_space", "industrial_space",
}


def lf_sha256(path):
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def route_facility_review(name, observation, config):
    """Route only from current image observations; never read targets or metadata."""
    if name not in {"no_review", "all_eligible", "evidence_conflict", "observable_uncertainty"}:
        raise ValueError(f"unknown facility route: {name}")
    if name == "no_review":
        return False
    if not should_review_facilities(observation, config):
        return False
    if name == "all_eligible":
        return True
    conflicts = any(
        error.startswith("facility[") for error in evidence_consistency_errors(observation)
    )
    if name == "evidence_conflict":
        return conflicts
    if name == "observable_uncertainty":
        empty_identifiable = (
            observation["subject_kind"] in EMPTY_FACILITY_REVIEW_SUBJECTS
            and not observation["facility_evidence"]
        )
        return conflicts or empty_identifiable
    raise AssertionError("validated facility route was not handled")


def primary_stage(record, config):
    """Recover the shortest strictly replayable primary-stage prefix."""
    base = {key: value for key, value in config.items() if key != "facility_refinement"}
    attempts = record.get("attempts", [])
    boundary = None
    for index in range(1, len(attempts) + 1):
        try:
            _replay_base({"attempts": attempts[:index]}, base)
        except ValueError:
            continue
        boundary = index
        break
    if boundary is None:
        raise ValueError("facility record has no replayable primary stage")
    observation = _rebuild_base_observation({"attempts": attempts[:boundary]}, base)
    return boundary, observation, map_observation(observation, base)


def transformed_record(record, result, attempts, elapsed_ms):
    """Build a validated public-output record for counterfactual scoring."""
    input_tokens = sum(item.get("input_tokens") or 0 for item in attempts)
    output_tokens = sum(item.get("output_tokens") or 0 for item in attempts)
    return {
        "sample_id": record["sample_id"],
        "passed": True,
        "elapsed_ms": elapsed_ms,
        "result": result,
        "attempts": [{
            "attempt": 1,
            "error": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": elapsed_ms,
            "raw_output": json.dumps(result, ensure_ascii=False, sort_keys=True),
        }],
    }


def analyze(generation_path, revision_path, output, role="observation_candidate_base", root=ROOT):
    references, audit, revision_config, _ = verified_references(revision_path, root)
    generation, source, identity, summary = generation_details(
        generation_path, revision_config, references, root
    )
    if role not in generation["profiles"]:
        raise ValueError("facility role is not present in the generation")
    config_path = root / generation.get("observation_profile_configs", {}).get(
        role, generation["observation_config"]
    )
    config = read_json(config_path)
    if config.get("facility_refinement") is None:
        raise ValueError("selected role has no facility refinement")
    raw_path = source / f"{role}.jsonl"
    if sha256_file(raw_path) != summary["profiles"][role]["raw_sha256"]:
        raise ValueError("facility generation raw output changed")
    records = list(iter_jsonl(raw_path))
    prepared = []
    for record in records:
        if replay_facility_refined_observation(record, config) != record.get("result"):
            raise ValueError("facility result does not replay")
        boundary, observation, primary_result = primary_stage(record, config)
        prepared.append((record, boundary, observation, primary_result))

    routes = {}
    for name in ("no_review", "evidence_conflict", "observable_uncertainty", "all_eligible"):
        counterfactual, triggered = [], 0
        for record, boundary, observation, primary_result in prepared:
            use_review = route_facility_review(name, observation, config)
            triggered += int(use_review)
            attempts = record["attempts"] if use_review else record["attempts"][:boundary]
            elapsed_ms = (
                record["elapsed_ms"] if use_review
                else sum(item.get("latency_ms") or 0 for item in attempts)
            )
            counterfactual.append(transformed_record(
                record, record["result"] if use_review else primary_result,
                attempts, elapsed_ms,
            ))
        scored = score_paired(root, references, counterfactual, reference_audit=audit)
        routes[name] = {
            "triggered": triggered,
            "trigger_rate": triggered / len(prepared),
            "metrics": scored["metrics"],
            "multilabel_counts": scored["multilabel_counts"],
            "latency_ms": scored["latency_ms"],
            "tokens": scored["tokens"],
        }

    result = {
        "protocol": "week8_target_free_facility_routing_tradeoff_v1",
        "phase": "development_counterfactual",
        "selection_use": "diagnostic_only",
        "test_rows_read": False,
        "human_annotation_count": 0,
        "label_source": "model_generated_silver",
        "route_inputs": "current_image_observation_only",
        "latency_interpretation": "recorded full elapsed when reviewed; recorded attempt latency when skipped",
        "analyzer_lf_sha256": lf_sha256(Path(__file__)),
        "generation_config_sha256": sha256_file(generation_path),
        "generation_identity_sha256": sha256_file(source / "identity.json"),
        "raw_sha256": sha256_file(raw_path),
        "reference_raw_sha256": audit["reference_raw_sha256"],
        "observation_config_sha256": identity["observation_profile_config_hashes"][role],
        "routes": routes,
        "decision": {
            "default_product_route": "no_review",
            "reason": "Only the no-review v12 product identity has locked final and packaged acceptance; development-only routes cannot replace it.",
            "facility_review_status": "DEVELOPMENT_ONLY_NOT_PROMOTED",
        },
    }
    write_json_new(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-config", required=True, type=Path)
    parser.add_argument("--revision-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--role", default="observation_candidate_base")
    args = parser.parse_args()
    value = analyze(
        args.generation_config.resolve(), args.revision_config.resolve(),
        args.output.resolve(), args.role,
    )
    print(json.dumps(value, ensure_ascii=False))

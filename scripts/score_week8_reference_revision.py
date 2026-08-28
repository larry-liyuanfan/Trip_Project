"""Re-score immutable development generation with a separately verified silver revision."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_week8_incumbent import compare
from scripts.repair_week8_visual_reference import load_revision_inputs
from src.data.week8_visual_holdout import read_json, write_json_new
from src.evaluation.visual_reference_revision import PROTOCOL, replay_revision, supports
from src.evaluation.week8_visual_silver import score_paired
from src.inference.product_style_scope import venue_style_evidence
from src.training.week7_data import iter_jsonl, sha256_file


def verified_references(config_path, root=ROOT):
    config = read_json(config_path)
    rows, original, observation, audit = load_revision_inputs(config, root)
    output = root / config["output_root"]
    identity = read_json(output / "identity.json")
    summary = read_json(output / "summary.json")
    raw_path = output / "raw_outputs.jsonl"
    if (identity["protocol"] != PROTOCOL or identity["config_sha256"] != sha256_file(config_path)
            or identity["source_raw_sha256"] != config["source_raw_sha256"]
            or identity["manifest_sha256"] != config["manifest_sha256"]
            or identity["observation_config_sha256"] != sha256_file(root / config["observation_config"])
            or identity["scope_audit_sha256"] != sha256_file(output / "scope_audit.json")
            or read_json(output / "scope_audit.json") != audit
            or summary["status"] != "COMPLETED" or summary["failures"] != 0
            or summary["raw_sha256"] != sha256_file(raw_path)
            or any(identity.get(key) is not False for key in (
                "test_rows_read", "prior_targets_supplied", "candidate_outputs_supplied", "metadata_supplied"))
            or identity["human_annotation_count"] != 0
            or identity["selected_sample_ids"] != [row["sample_id"] for row in rows]):
        raise ValueError("development reference revision identity mismatch")
    references = list(iter_jsonl(raw_path))
    if len(references) != len(original):
        raise ValueError("reference revision cannot drop fixed samples")
    for source, revised, entry in zip(original, references, audit):
        replay_revision(source, revised, entry, config, observation)
        if venue_style_evidence(revised["observation"], observation)[1]:
            raise ValueError("known nonvenue reference evidence remains unresolved")
    if summary["supports_before"] != supports(original) or summary["supports_after"] != supports(references):
        raise ValueError("reference support disclosure does not reproduce")
    reference_audit = {"protocol": PROTOCOL, "metadata_supplied": False, "candidate_outputs_supplied": False,
        "prior_targets_supplied": False, "test_rows_read": False, "model_independent": True,
        "reference_raw_sha256": sha256_file(raw_path), "source_raw_sha256": config["source_raw_sha256"],
        "scope_audit_sha256": identity["scope_audit_sha256"], "revision_identity_sha256": sha256_file(output / "identity.json"),
        "five_dimensional_identity_verified": True, "style_only_raw_replay_verified": True, "scope_errors_remaining": 0}
    return references, reference_audit, config, summary


def generation_details(generation_path, revision_config, references):
    config = read_json(generation_path)
    source = ROOT / config["output_root"]
    identity = read_json(source / "identity.json")
    generation_summary = read_json(source / "summary.json")
    if (identity["config_sha256"] != sha256_file(generation_path)
            or config["final_test_access"] is not False or identity["test_rows_read"] is not False
            or generation_summary["status"] != "COMPLETED"
            or config["development_indices"] != "all"
            or identity["development_sha256"] != revision_config["manifest_sha256"]
            or identity["selected_sample_ids"] != [row["sample_id"] for row in references]
            or identity["base_model"] == revision_config["model"]):
        raise ValueError("immutable generation and revised development reference do not match")
    return config, source, identity, generation_summary


def run(generation_path, revision_path, output):
    references, audit, revision_config, reference_summary = verified_references(revision_path)
    config, source, identity, generation_summary = generation_details(generation_path, revision_config, references)
    summaries = {}
    for role in config["profiles"]:
        raw_path = source / f"{role}.jsonl"
        if sha256_file(raw_path) != generation_summary["profiles"][role]["raw_sha256"]:
            raise ValueError("immutable development model output changed")
        observation = None
        if role.startswith("observation_"):
            path = ROOT / config.get("observation_profile_configs", {}).get(role, config["observation_config"])
            expected = identity["observation_profile_config_hashes"].get(role, identity["observation_config_sha256"])
            if sha256_file(path) != expected:
                raise ValueError("generation observation config identity changed")
            observation = read_json(path)
        summaries[role] = score_paired(ROOT, references, list(iter_jsonl(raw_path)), observation, reference_audit=audit)
    historical_formal = None
    if "formal_adapter" not in summaries:
        formal_config_path = ROOT / config["formal_baseline_generation_config"]
        _, formal_source, _, formal_summary = generation_details(formal_config_path, revision_config, references)
        formal_raw = formal_source / "formal_adapter.jsonl"
        if sha256_file(formal_raw) != formal_summary["profiles"]["formal_adapter"]["raw_sha256"]:
            raise ValueError("historical formal baseline raw changed")
        summaries["formal_adapter"] = score_paired(ROOT, references, list(iter_jsonl(formal_raw)), reference_audit=audit)
        historical_formal = {"config": config["formal_baseline_generation_config"],
                            "config_sha256": sha256_file(formal_config_path), "raw_sha256": sha256_file(formal_raw),
                            "latency_source": "historical_not_same_session_performance_comparator"}
    comparison = {"protocol": "paired_development_reference_revision_replay_v1", "summaries": summaries,
                  "generation_identity_sha256": sha256_file(source / "identity.json"),
                  "generation_config_sha256": sha256_file(generation_path),
                  "reference_revision_config_sha256": sha256_file(revision_path),
                  "reference_supports_before": reference_summary["supports_before"],
                  "reference_supports_after": reference_summary["supports_after"],
                  "test_rows_read": False, "new_model_requests": 0, "label_source": "model_generated_silver",
                  "historical_formal_baseline": historical_formal,
                  "interpretation": "All candidates and v9 use the same revised reference; not comparable with old-reference scores."}
    decision = compare(comparison, config.get("incumbent_role", "observation_base"))
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "comparison.json", comparison)
    write_json_new(output / "incumbent_comparison.json", {**decision, "comparison_sha256": sha256_file(output / "comparison.json")})
    print(json.dumps({"decision": decision, "metrics": {role: value["metrics"] for role, value in summaries.items()},
                      "supports": reference_summary["supports_after"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-config", required=True, type=Path)
    parser.add_argument("--revision-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.generation_config.resolve(), args.revision_config.resolve(), args.output.resolve())

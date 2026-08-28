"""Verify raw identities and compare paired development outputs against blind silver."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.week7_data import sha256_file
from src.evaluation.week8_visual_silver import score_paired, select_development_candidate
from scripts.review_week8_contracts import write_new


def run(config_path, output):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = ROOT / config["output_root"]
    reference = ROOT / config["reference_run"]
    reference_identity = json.loads((reference / "identity.json").read_text(encoding="utf-8"))
    identity = json.loads((source / "identity.json").read_text(encoding="utf-8"))
    run_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if (identity["config_sha256"] != sha256_file(config_path)
            or run_summary["status"] != "COMPLETED" or identity["test_rows_read"] is not False
            or reference_identity["candidate_outputs_supplied"] is not False
            or reference_identity["metadata_supplied"] is not False
            or reference_identity["test_rows_read"] is not False
            or reference_identity["manifest_sha256"] != config["reference_manifest_sha256"]
            or identity["development_sha256"] != config["reference_manifest_sha256"]
            or sha256_file(reference / "raw_outputs.jsonl") != config["reference_raw_sha256"]):
        raise ValueError("development/reference identity mismatch")
    references = [json.loads(line) for line in (reference / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    teacher_models = {row.get("response_model") for row in references}
    if len(teacher_models) != 1 or not next(iter(teacher_models)):
        raise ValueError("teacher model identity is incomplete")
    reference_audit = {"protocol": reference_identity["protocol"], "metadata_supplied": False,
                       "candidate_outputs_supplied": False, "test_rows_read": False,
                       "model_independent": next(iter(teacher_models)) != identity["base_model"],
                       "reference_raw_sha256": config["reference_raw_sha256"]}
    summaries = {}
    for role in config["profiles"]:
        raw_path = source / f"{role}.jsonl"
        if sha256_file(raw_path) != run_summary["profiles"][role]["raw_sha256"]:
            raise ValueError("candidate raw output changed")
        observation = None
        if role.startswith("observation_"):
            path = config.get("observation_profile_configs", {}).get(role, config["observation_config"])
            observation = json.loads((ROOT / path).read_text(encoding="utf-8"))
            expected = identity["observation_profile_config_hashes"].get(role, identity["observation_config_sha256"])
            if sha256_file(ROOT / path) != expected:
                raise ValueError("observation mapping identity changed")
        records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
        summaries[role] = score_paired(ROOT, references, records, observation, reference_audit=reference_audit)
    selection = select_development_candidate(summaries)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "comparison.json", {"summaries": summaries, "selection": selection,
              "reference_raw_sha256": config["reference_raw_sha256"],
              "generation_identity_sha256": sha256_file(source / "identity.json"),
              "test_rows_read": False, "label_source": "model_generated_silver"})
    print(json.dumps(selection, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week8/contract_ablation_v3.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())

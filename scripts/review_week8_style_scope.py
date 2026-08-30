"""Development-only deterministic scope replay; not a new GPU or final result."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_week8_incumbent import compare
from src.data.week8_visual_holdout import write_json_new
from src.evaluation.week8_visual_silver import replay_record, score_paired
from src.inference.product_observation import load_observation_config, canonical_config_sha256, parse_observation, map_observation, observation_messages
from src.inference.product_style_scope import venue_style_evidence
from src.training.week7_data import iter_jsonl, sha256_file


def run(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["final_test_access"] is not False or config["human_annotation_count"] != 0:
        raise ValueError("scope replay must remain development-only")
    for spec in config["sources"].values():
        path = (ROOT / spec["path"]).resolve()
        path.relative_to(ROOT.resolve())
        if sha256_file(path) != spec["sha256"]:
            raise ValueError("immutable development source changed")
    sources = {key: ROOT / value["path"] for key, value in config["sources"].items()}
    identity = json.loads(sources["generation_identity"].read_text(encoding="utf-8"))
    prior = json.loads(sources["comparison"].read_text(encoding="utf-8"))
    if identity["test_rows_read"] is not False or prior["test_rows_read"] is not False:
        raise ValueError("final outputs cannot be used for scope selection")
    original = json.loads((ROOT / config["original_observation"]).read_text(encoding="utf-8"))
    scoped = json.loads((ROOT / config["scoped_observation"]).read_text(encoding="utf-8"))
    if scoped.get("style_refinement") is not None:
        raise ValueError("additional model stages cannot be simulated by a deterministic scope replay")
    load_observation_config(ROOT / config["scoped_observation"], canonical_config_sha256(scoped))
    if observation_messages("same-image", original) != observation_messages("same-image", scoped):
        raise ValueError("raw reuse requires byte-identical generation messages")
    references = list(iter_jsonl(sources["references"]))
    records = list(iter_jsonl(sources["incumbent_raw"]))
    formal = list(iter_jsonl(sources["formal_raw"]))
    reference_audit = prior["summaries"]["observation_base"]["reference_audit"]
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    derived = []
    exclusions = []
    with (output / "derived_scope_records.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            replay_record(ROOT, record, original)
            changed = copy.deepcopy(record)
            started = time.perf_counter()
            if record["passed"]:
                observation = parse_observation(record["attempts"][-1]["raw_output"], original)
                changed["result"] = map_observation(observation, scoped)
                changed["style_scope_exclusions"] = venue_style_evidence(observation, scoped)[1]
                exclusions.extend({"sample_id": record["sample_id"], **cue} for cue in changed["style_scope_exclusions"])
            changed["scope_mapping_ms"] = (time.perf_counter() - started) * 1000
            changed["latency_source"] = "historical_primary_generation_not_new_pipeline_measurement"
            changed["derived_mapping_only"] = True
            handle.write(json.dumps(changed, ensure_ascii=False, sort_keys=True) + "\n")
            derived.append(changed)
    summaries = {
        "formal_adapter": score_paired(ROOT, references, formal, reference_audit=reference_audit),
        "observation_base": score_paired(ROOT, references, records, original, reference_audit=reference_audit),
        "observation_scope": score_paired(ROOT, references, derived, scoped, reference_audit=reference_audit),
    }
    comparison = {"summaries": summaries, "test_rows_read": False, "label_source": "model_generated_silver",
                  "performance_assessed": False, "new_model_requests": 0, "derived_mapping_only": True}
    decision = compare(comparison, "observation_base")
    result = {"status": "MAPPING_REPLAY_ONLY", "comparison": comparison, "development_selection": decision,
              "exclusions": exclusions, "requires_live_runtime_verification": True, "final_access_allowed": False,
              "sources": config["sources"], "config_sha256": sha256_file(config_path),
              "scoped_observation_sha256": sha256_file(ROOT / config["scoped_observation"]),
              "derived_raw_sha256": sha256_file(output / "derived_scope_records.jsonl"),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
              "implementation_hashes": {path: sha256_file(ROOT / path) for path in (
                  "scripts/review_week8_style_scope.py", "src/inference/product_style_scope.py",
                  "src/inference/product_observation.py", "src/evaluation/week8_visual_silver.py")}}
    write_json_new(output / "summary.json", result)
    print(json.dumps({"status": result["status"], "selection": decision, "exclusions": exclusions,
                      "before": summaries["observation_base"]["metrics"], "after": summaries["observation_scope"]["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config.resolve())

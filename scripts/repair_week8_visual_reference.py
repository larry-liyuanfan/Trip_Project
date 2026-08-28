"""Reobserve only invalid development style evidence with an independent image teacher."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_week8_visual_silver import load_inputs
from src.data.week8_visual_holdout import read_json, write_json_new
from src.evaluation.visual_reference_revision import (
    PROTOCOL, repair_record, replay_revision, supports, validate_config, validate_sources,
)
from src.inference.client import _read_api_key
from src.inference.product_observation import load_observation_config, canonical_config_sha256
from src.training.week7_data import iter_jsonl, sha256_file


def load_revision_inputs(config, root=ROOT):
    validate_config(config)
    rows, manifest = load_inputs(config, root)
    source = (root / config["source_run"]).resolve()
    source.relative_to(root.resolve())
    identity = read_json(source / "identity.json")
    teacher_path = root / config["source_teacher_config"]
    teacher = read_json(teacher_path)
    original_path = root / teacher["observation_config"]
    if (identity["manifest_sha256"] != config["manifest_sha256"]
            or sha256_file(manifest) != config["manifest_sha256"]
            or sha256_file(source / "raw_outputs.jsonl") != config["source_raw_sha256"]
            or identity["config_sha256"] != sha256_file(teacher_path)
            or identity["observation_config_sha256"] != sha256_file(original_path)
            or identity["protocol"] != "independent_image_model_observation_silver_v3"
            or identity["human_annotation_count"] != 0
            or any(identity.get(key) is not False for key in (
                "candidate_outputs_supplied", "metadata_supplied", "test_rows_read"))):
        raise ValueError("original independent development reference identity mismatch")
    references = list(iter_jsonl(source / "raw_outputs.jsonl"))
    if any(row.get("response_model") != config["model"] for row in references):
        raise ValueError("source and correction teacher identities must match")
    observation_path = root / config["observation_config"]
    observation = read_json(observation_path)
    load_observation_config(observation_path, canonical_config_sha256(observation))
    audit = validate_sources(rows, references, read_json(original_path), observation)
    return rows, references, observation, audit


def run(path, audit_only=False):
    config = read_json(path)
    rows, references, observation, audit = load_revision_inputs(config)
    if audit_only:
        print(json.dumps({"rows": len(rows), "scope_errors": [r for r in audit if r["reobserve_style"]],
                          "test_rows_read": False, "new_model_requests": 0}, ensure_ascii=False))
        return
    key = _read_api_key()
    if not key:
        raise ValueError("independent image teacher credential missing")
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "scope_audit.json", audit)
    identity = {"protocol": PROTOCOL, "config_sha256": sha256_file(path),
                "source_raw_sha256": config["source_raw_sha256"], "manifest_sha256": config["manifest_sha256"],
                "observation_config_sha256": sha256_file(ROOT / config["observation_config"]),
                "scope_audit_sha256": sha256_file(output / "scope_audit.json"),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "implementation_hashes": {name: sha256_file(ROOT / name) for name in (
                    "scripts/repair_week8_visual_reference.py", "src/evaluation/visual_reference_revision.py",
                    "src/inference/product_style_refinement.py", "src/inference/product_style_scope.py")},
                "human_annotation_count": 0, "sample_weight": 0.5, "label_source": "model_generated_silver",
                "prior_targets_supplied": False, "candidate_outputs_supplied": False, "metadata_supplied": False,
                "test_rows_read": False, "selected_sample_ids": [row["sample_id"] for row in rows]}
    write_json_new(output / "identity.json", identity)
    base_url = os.environ.get("MODEL_API_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").rstrip("/")
    revised = []
    with ThreadPoolExecutor(max_workers=config["concurrency"]) as executor, (output / "raw_outputs.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        results = executor.map(lambda args: repair_record(*args, config, observation, ROOT, base_url, key),
                               zip(rows, references, audit))
        for original, entry, result in zip(references, audit, results):
            if not result.get("error"):
                replay_revision(original, result, entry, config, observation)
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            revised.append(result)
            if entry["reobserve_style"]:
                print(json.dumps({"sample_id": result["sample_id"], "error": result.get("error"),
                                  "style_before": original["target"]["style_tags"],
                                  "style_after": result["target"]["style_tags"] if result.get("target") else None}), flush=True)
    failures = sum(bool(row.get("error")) for row in revised)
    summary = {"status": "COMPLETED" if not failures else "REFERENCE_REPAIR_FAILED", "rows": len(rows),
               "failures": failures, "style_reobserved": sum(r["reobserve_style"] for r in audit),
               "new_requests": sum(len(r["attempts"]) for r in revised if r["reference_revision"]["style_reobserved"]),
               "raw_sha256": sha256_file(output / "raw_outputs.jsonl"), "source_raw_sha256": config["source_raw_sha256"],
               "supports_before": supports(references), "supports_after": supports(revised) if not failures else None,
               "test_rows_read": False, "human_annotation_count": 0, "candidate_selection": "NOT_PERFORMED"}
    write_json_new(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    run(args.config.resolve(), args.audit_only)

"""Development-only, immutable product contract/adapter ablation and reference audit."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.product_semantics import audit_product_references, product_consistency_errors
from src.inference.system_runtime import ReleaseSettings, TransformersPeftBackend
from src.inference.transport_utils import strip_json_fence
from src.training.week7_data import iter_jsonl, sha256_file
from src.training.week8_product import (
    _run_one_product, _write_json_new, load_week8_product_config,
    summarize_product_run, validate_week8_product_lock,
)
from src.training.week8_product_two_stage import _generate_evidence, load_two_stage_config


def load_review_inputs(root: Path, path: Path):
    config = json.loads(path.read_text(encoding="utf-8"))
    expected_profiles = {
        "week8_product_review_v1": ["product_release", "evidence_legacy", "evidence_contract", "evidence_contract_base"],
        "week8_product_review_v2": ["product_release", "evidence_contract_unconstrained_base"],
    }
    if (
        config.get("schema_version") not in expected_profiles
        or config.get("final_test_access") != "forbidden"
        or any(config.get(key) is not False for key in ("human_annotation", "human_review", "human_acceptance"))
        or config.get("profiles") != expected_profiles.get(config.get("schema_version"))
    ):
        raise ValueError("invalid development-only review policy")
    product_path = (root / config["product_config"]).resolve()
    product_path.relative_to(root.resolve())
    product = load_week8_product_config(product_path)
    validation = validate_week8_product_lock(root, product_path, include_test=False)
    if validation["status"] != "PASS" or validation["lock_sha256"] != config["dataset_lock_sha256"]:
        raise ValueError("review data identity mismatch")
    lock_root = root / product["dataset"]["output_root"] / product["week8"]["dataset_version"]
    development = lock_root / "development/image_product_search.jsonl"
    rows = list(iter_jsonl(development))
    if len(rows) != config["development_count"] or any(row["split"] != "development" for row in rows):
        raise ValueError("review requires the complete fixed development set")
    for row in rows:
        image = (root / row["image_path"]).resolve()
        image.relative_to(root.resolve())
        if sha256_file(image) != row["image_sha256"]:
            raise ValueError("review image identity mismatch")
    return config, product, rows, validation, sha256_file(development)


def output_diagnostics(records):
    consistent = nonempty = first_pass = syntax = schema = 0
    for record in records:
        # 原始模型语法率单列；失败占位零分是端到端计分，不等于 JSON 语法错误。
        try:
            json.loads(strip_json_fence(record.get("evidence_raw_output", record["raw_output"])))
        except (ValueError, TypeError):
            pass
        else:
            syntax += 1
        schema += int(record.get("evidence_schema_pass", not record["failed"]))
        try:
            payload = json.loads(strip_json_fence(record["raw_output"]))
        except (ValueError, TypeError):
            payload = None
        consistent += int(not record["failed"] and not product_consistency_errors(payload))
        nonempty += int(not record["failed"] and isinstance(payload, dict) and bool(payload.get("style_tags") or payload.get("visible_facilities")))
        first_pass += int(bool(record.get("attempts")) and record["attempts"][0]["error"] is None)
    return {
        "model_json_syntax_rate": syntax / len(records),
        "model_schema_pass_rate": schema / len(records),
        "internal_consistency_rate": consistent / len(records),
        "positive_style_or_facility_output_count": nonempty,
        "first_attempt_pass_rate": first_pass / len(records),
        "not_a_visual_accuracy_measure": True,
    }


def run_review(root: Path, path: Path, output: Path, *, audit_only=False):
    if output.exists():
        raise ValueError("refusing to overwrite review output")
    config, product, rows, validation, development_sha = load_review_inputs(root, path)
    output.mkdir(parents=True, exist_ok=False)
    identity = {
        "run_id": config["run_id"], "config_sha256": sha256_file(path),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dataset_lock_sha256": validation["lock_sha256"],
        "development_sha256": development_sha,
        "input_config_hashes": {config[key]: sha256_file(root / config[key]) for key in ("product_config", "two_stage_config", "release_config")},
        "validation": validation, "test_rows_read": False,
        "human_annotation_review_acceptance": [0, 0, 0],
    }
    _write_json_new(output / "identity.json", identity)
    audit = audit_product_references(rows)
    _write_json_new(output / "reference_audit.json", audit)
    if audit_only:
        return {"status": "AUDIT_COMPLETED", "audit": audit}
    settings = ReleaseSettings.load(root=root, config_path=root / config["release_config"])
    if any(getattr(settings, key) != product["model"][key] for key in ("base_model", "base_revision", "adapter_model_sha256")):
        raise ValueError("review model identity mismatch")
    backend = TransformersPeftBackend(settings)
    cold_start = time.perf_counter()
    ok, detail = backend.ready()
    if not ok:
        raise ValueError(detail)
    cold_start = (time.perf_counter() - cold_start) * 1000
    legacy = load_two_stage_config(root / config["two_stage_config"])
    repaired = copy.deepcopy(legacy)
    repaired["two_stage"].update({
        "include_schema_in_prompt": True,
        "mapping_version": "evidence_consistent_v2",
        "max_new_tokens": config["evidence_max_new_tokens"],
    })
    summaries = {}
    for role in config["profiles"]:
        role_dir = output / role
        role_dir.mkdir()
        raw_path = role_dir / "raw_outputs.jsonl"
        records = []
        # 基座消融只临时关闭 adapter；不改权重、不合并、不导出新 adapter。
        context = backend._model.disable_adapter() if role.endswith("_base") else nullcontext()
        with context, raw_path.open("x", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(rows):
                if role == "product_release":
                    record = _run_one_product(
                        root, backend, row, run_id=f"{config['run_id']}__{role}",
                        prompt_version=settings.prompt_versions["image_product_search"],
                        max_new_tokens=config["product_max_new_tokens"],
                    )
                else:
                    active_config = copy.deepcopy(legacy if role == "evidence_legacy" else repaired)
                    if role == "evidence_contract_unconstrained_base":
                        # 解码器消融仍逐条执行完整 Schema 校验；不修补或猜填非法输出。
                        active_config["two_stage"]["constrained_decoding"] = False
                    record = _generate_evidence(
                        root, backend, row, active_config,
                        f"{config['run_id']}__{role}",
                    )
                record["adapter_enabled"] = not role.endswith("_base")
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                records.append(record)
                print(json.dumps({"profile": role, "completed": index + 1, "count": len(rows), "failed": record["failed"]}), flush=True)
        summary = summarize_product_run(root, rows, records)
        summary["output_diagnostics"] = output_diagnostics(records)
        summary["raw_sha256"] = sha256_file(raw_path)
        summary["adapter_enabled"] = not role.endswith("_base")
        _write_json_new(role_dir / "metrics.json", summary)
        summaries[role] = summary
    result = {
        "status": "COMPLETED", "identity": identity, "cold_start_ms": cold_start,
        "profiles": summaries, "release_changed": False,
        "conclusion": "Diagnostic only: mixed-metadata silver does not validate visual label accuracy.",
        "peak_gpu_allocated_bytes": backend._torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": backend._torch.cuda.max_memory_reserved(),
        "hardware": backend._torch.cuda.get_device_name(0),
    }
    _write_json_new(output / "summary.json", result)
    return {"status": "COMPLETED", "summary_sha256": sha256_file(output / "summary.json")}


def rescore_review(root: Path, path: Path, source: Path, output: Path):
    """Apply repaired failure accounting to immutable development generations."""
    if output.exists():
        raise ValueError("refusing to overwrite rescored review output")
    config, _, rows, validation, development_sha = load_review_inputs(root, path)
    identity_path = source / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        identity["config_sha256"] != sha256_file(path)
        or identity["development_sha256"] != development_sha
        or identity["dataset_lock_sha256"] != validation["lock_sha256"]
        or identity.get("test_rows_read") is not False
    ):
        raise ValueError("source review identity mismatch")
    profiles, raw_hashes = {}, {}
    for role in config["profiles"]:
        raw = source / role / "raw_outputs.jsonl"
        old_metrics = json.loads((source / role / "metrics.json").read_text(encoding="utf-8"))
        if sha256_file(raw) != old_metrics["raw_sha256"]:
            raise ValueError("source review generations changed")
        records = list(iter_jsonl(raw))
        profiles[role] = summarize_product_run(root, rows, records)
        profiles[role]["output_diagnostics"] = output_diagnostics(records)
        raw_hashes[role] = sha256_file(raw)
    result = {
        "status": "COMPLETED", "scoring_protocol": "week8_product_failure_zero_credit_v2",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_identity_sha256": sha256_file(identity_path), "source_raw_sha256": raw_hashes,
        "source_generation_commit": identity["git_commit"],
        "development_sha256": development_sha, "dataset_lock_sha256": validation["lock_sha256"],
        "test_rows_read": False, "new_model_requests": 0, "profiles": profiles,
        "release_changed": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_json_new(output / "summary.json", result)
    return {"status": "RESCORED", "summary_sha256": sha256_file(output / "summary.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/week8/product_review_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--rescore-dir", type=Path, help="completed review to rescore without new inference")
    args = parser.parse_args()
    if args.rescore_dir:
        if args.audit_only:
            parser.error("--audit-only and --rescore-dir are mutually exclusive")
        result = rescore_review(ROOT, args.config.resolve(), args.rescore_dir.resolve(), args.output_dir.resolve())
    else:
        result = run_review(ROOT, args.config.resolve(), args.output_dir.resolve(), audit_only=args.audit_only)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

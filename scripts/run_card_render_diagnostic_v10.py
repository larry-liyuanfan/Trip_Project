"""Paired rendering diagnostic on already seen v9 development, without training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_synthetic_card_layout_v10 import audit
from scripts.build_exploration_pool_v4 import _write_json, _write_jsonl
from scripts.build_semantic_robustness_pool_v9 import (
    COUNTS, SPLIT_OFFSET, PRODUCT_PROMPT, _clear_product, _unknown_product,
    _multi_product_v9, _negated_product, _conflict_product,
)
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl
from src.evaluation.semantic_robustness_v7 import score_semantic_robustness_v7

ADAPTER_HASHES = {
    "v7": "a06742ebf6344567650e3ae95e67c56116db212d07eeb5eb5c31fd6ae519b5b1",
    "v9": "6332a2baa03d88751c022cb6efdd9dbfb21e6f88f32f31f4d9cb775a0bf38f17",
}
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MANIFEST_LOCKS = {
    "original": "df69b729a674b86fa2aaa5a09781805a3a2622cfb262d6cec5884961865d569c",
    "repaired": "b51764b57ee4a226719f76f2a3a42f396bdc1304792fd927f719159646288419",
}


def build_bundles(root: Path) -> dict:
    audit(root)
    builders = {"clear": _clear_product, "unknown": _unknown_product,
                "multi": _multi_product_v9, "negated": _negated_product,
                "conflict": _conflict_product}
    locks = {}
    for rendering in ("original", "repaired"):
        rows = []
        for kind, builder in builders.items():
            for index in range(COUNTS["development"][kind]):
                _, gold, slices = builder(index, SPLIT_OFFSET["development"])
                sample_id = f"v9_dev_{kind}_{index:03d}"
                rows.append({"sample_id": sample_id, "scenario": "product", "split": "development",
                             "image_relative_path": f"{sample_id}.ppm",
                             "image_sha256": file_sha256(root / rendering / f"{sample_id}.ppm"),
                             "prompt": PRODUCT_PROMPT, "gold": gold, "slices": slices,
                             "label_provenance": "synthetic_paired_seen_v9_development_not_independent_test"})
        if canonical_json_sha256(rows) != MANIFEST_LOCKS[rendering]:
            raise ValueError("paired manifest differs from preregistered source lock")
        manifest = root / rendering / "vlm_development_manifest.jsonl"
        _write_jsonl(manifest, rows)
        lock = {"vlm": {"development": {"sample_support": len(rows),
                                         "manifest_canonical_sha256": canonical_json_sha256(rows),
                                         "manifest_file_sha256": file_sha256(manifest)}}}
        lock_path = root / rendering / "bundle_lock.json"
        _write_json(lock_path, lock)
        _write_json(root / rendering / "config.json", {"cycle_id": "v10_render_diagnostic",
                    "pool": {"committed_lock": str(lock_path.resolve())}})
        locks[rendering] = lock
    return locks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v7-adapter", type=Path, required=True)
    parser.add_argument("--v9-adapter", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()
    for role in ADAPTER_HASHES:
        path = getattr(args, f"{role}_adapter") / "adapter_model.safetensors"
        if file_sha256(path) != ADAPTER_HASHES[role]:
            raise ValueError(f"{role} adapter SHA mismatch")
    root = args.output_dir.resolve()
    locks = build_bundles(root)
    _write_json(root / "preregistered_diagnostic.json", {
        "implementation_commit": args.implementation_commit,
        "hypothesis": "complete_visible_text_reduces_truncated_style_predictions",
        "primary_factor": "rendering_only_within_each_adapter",
        "roles": ADAPTER_HASHES, "locks": locks, "support_per_cell": 84,
        "cell_count": 4, "training": False, "fresh_test_used": False,
        "independent_test": False, "promotion_allowed": False,
        "success_rule": "v9_style_f1_increases_and_other_field_f1_regressions_do_not_exceed_0.05",
        "generation": {"do_sample": False, "max_new_tokens": 256, "max_retries": 1}})
    reports = {}
    for rendering in ("original", "repaired"):
        output_rows = []
        for role in ("v7", "v9"):
            result = root / rendering / f"{role}.jsonl"
            subprocess.run([sys.executable, "scripts/run_vlm_semantic_evidence_v4.py",
                            "--config", str(root / rendering / "config.json"),
                            "--bundle-dir", str(root / rendering), "--split", "development",
                            "--role", role, "--base-revision", REVISION,
                            "--adapter-path", str(getattr(args, f"{role}_adapter")),
                            "--adapter-sha256", ADAPTER_HASHES[role], "--output", str(result)], check=True)
            output_rows.extend(load_jsonl(result))
        reports[rendering] = score_semantic_robustness_v7(output_rows,
            cycle_id="v10_render_diagnostic", primary_factor="rendering_only")
    before = reports["original"]["variants"]["v9"]["field_metrics"]
    after = reports["repaired"]["variants"]["v9"]["field_metrics"]
    delta = {field: after[field]["f1"] - before[field]["f1"] for field in before}
    summary = {"schema_version": "paired_card_render_diagnostic_v10",
               "slurm_job_id": os.getenv("SLURM_JOB_ID"),
               "hardware": {"nodes": os.getenv("SLURM_JOB_NODELIST"),
                            "cpus": os.getenv("SLURM_CPUS_PER_TASK"),
                            "memory_mib": os.getenv("SLURM_MEM_PER_NODE")},
               "implementation_commit": args.implementation_commit,
               "status": "COMPLETED", "independent_test": False, "fresh_test_used": False,
               "promotion_allowed": False, "training_run": False,
               "support_per_cell": 84, "raw_row_support": 336,
               "v9_field_f1_delta": delta,
               "diagnostic_gate": "PASS" if delta["style_tags"] > 0 and all(
                   value >= -0.05 for field, value in delta.items() if field != "style_tags") else "FAIL",
               "reports": reports,
               "artifacts": {str(p.relative_to(root)): file_sha256(p)
                             for p in root.rglob("*.jsonl")}}
    _write_json(root / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "reports"}, indent=2))


if __name__ == "__main__":
    main()

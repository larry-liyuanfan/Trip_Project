"""Offline, hash-locked rescore of retained search predictions (no model loading)."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl, score_search_results
from src.evaluation.search_scorer_v2 import build_weak_qrels

REPO = Path(__file__).resolve().parents[1]
CASES = {
    "v4_final": {
        "evidence": "experiments/search_algorithm_evidence_v4.json",
        "lock": "configs/evaluation/evidence_enhancement/exploration_pool_lock_v4.json",
        "config": "configs/evaluation/automated_evidence_v4.json",
        "split": "final", "query_dir": "v4", "result_dir": "v4",
        "source_commit": "3c6beecae5c555372ad2585eedee0fd0efe9cde7",
        "artifact_section": "search", "artifact_prefix": "final",
        "hardware": {"gpu": "NVIDIA L40S"}, "job_id": "29996439",
    },
    "v8_validation": {
        "evidence": "experiments/no_result_stress_evidence_v8.json",
        "lock": "configs/evaluation/evidence_enhancement/no_result_stress_pool_lock_v8.json",
        "config": "configs/evaluation/automated_evidence_v8_no_result.json",
        "split": "validation", "query_dir": "v8-pool", "result_dir": "v8",
        "source_commit": "2acf4018aeba3cc717edcac8d2d7836ae15fe1de",
        "artifact_section": None, "artifact_prefix": "validation",
        "hardware": {"gpu": "NVIDIA L40S", "gpu_count": 1, "cpu_count": 8, "memory_gib": 48},
        "job_id": "30005527",
    },
}


def checked_file(path: Path, expected: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"historical file SHA mismatch: {path.name}")
    return actual


def checked_metadata(archive_path: Path, expected: str) -> tuple[list, str]:
    checked_file(archive_path, expected)
    # Read one member in memory; do not extract or load vectors/model assets.
    name = "retrieval/clip_metadata_1000.jsonl"
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember(name)
        if not member.isfile() or member.size > 20_000_000:
            raise ValueError("unexpected metadata member type or size")
        payload = archive.extractfile(member).read()
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line], hashlib.sha256(payload).hexdigest()


def verify_bindings(queries: list, annotations: list, results: list, old: dict) -> None:
    for key, value in (("query_manifest_sha256", queries), ("annotation_sha256", annotations), ("result_sha256", results)):
        if old.get(key) != canonical_json_sha256(value):
            raise ValueError(f"old metrics {key} mismatch; not safely recomputable")


def compare(old: dict, new: dict) -> dict:
    comparison = {}
    names = ("recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10", "no_result_rate", "no_result_accuracy", "filter_correctness", "failure_rate")
    for method, metrics in new["methods"].items():
        previous = old["methods"][method]
        old_rows = previous["per_query"]
        eligible = {r["query_id"] for r in old_rows if r["ranking_evaluable"] and not r["failed"]}
        paired = [r for r in metrics["per_query"] if r["query_id"] in eligible]
        def paired_average(key):
            values = [r[key] for r in paired if r[key] is not None]
            return {"value": sum(values) / len(values) if values else None, "query_denominator": len(values)}
        comparison[method] = {
            "old": {"query_denominator": previous["support"], "ranking_denominator": previous["ranking_support"],
                    "metrics": {key: previous.get(key) for key in names},
                    "filter_denominator": previous["support"],
                    "no_result_rate_was_dataset_slice_ratio": True},
            "scorer_v2": {key: value for key, value in metrics.items() if key not in {"per_query"}},
            "new_on_old_ranking_query_subset": {key: paired_average(key) for key in names[:4]},
            "delta_new_minus_old": {
                key: metrics[key] - previous[key]
                if isinstance(metrics.get(key), (int, float)) and isinstance(previous.get(key), (int, float)) else None
                for key in names
            },
            "comparison_boundary": "scorer correction only; overall deltas include denominator/definition changes, not model improvement",
        }
    return comparison


def audit(case_name: str, input_root: Path, retrieval_archive: Path) -> dict:
    case = CASES[case_name]
    evidence = json.loads((REPO / case["evidence"]).read_text(encoding="utf-8"))
    lock = json.loads((REPO / case["lock"]).read_text(encoding="utf-8"))["search"][case["split"]]
    section = evidence[case["artifact_section"]] if case["artifact_section"] else evidence
    artifacts = section["artifact_sha256"]
    split = case["split"]
    paths = {
        "queries": input_root / case["query_dir"] / f"search_{split}_manifest.jsonl",
        "annotations": input_root / case["query_dir"] / f"search_{split}_annotations.jsonl",
        "predictions": input_root / case["result_dir"] / f"{split}_results.jsonl",
        "old_metrics": input_root / case["result_dir"] / f"{split}_metrics.json",
    }
    expected = {
        "queries": lock["query_manifest_file_sha256"], "annotations": lock["annotation_file_sha256"],
        "predictions": artifacts[f"{case['artifact_prefix']}_results_file"],
        "old_metrics": artifacts[f"{case['artifact_prefix']}_metrics_file"],
    }
    input_hashes = {key: checked_file(path, expected[key]) for key, path in paths.items()}
    queries, annotations, results = [load_jsonl(paths[key]) for key in ("queries", "annotations", "predictions")]
    old = json.loads(paths["old_metrics"].read_text(encoding="utf-8"))
    verify_bindings(queries, annotations, results, old)
    expected_archive = evidence["source"]["retrieval_archive_sha256"]
    metadata, metadata_sha = checked_metadata(retrieval_archive, expected_archive)
    qrels = build_weak_qrels(queries, annotations, metadata)
    new = score_search_results(queries, annotations, results, methods=tuple(old["methods"]), qrels=qrels)
    code_paths = ["src/evaluation/relevance_evidence.py", "src/evaluation/search_scorer_v2.py", "scripts/audit_search_scorer_v2.py"]
    return {
        "schema_version": "offline_scorer_v2_audit_20260914", "case": case_name,
        "status": "RECOMPUTED_FROM_HASH_VERIFIED_EXISTING_PREDICTIONS",
        "new_inference": False, "fresh_test_120_opened": False, "human_support": 0,
        "historical_source_commit": case["source_commit"],
        "historical_source_snapshot_sha256": evidence["source"].get("search_source_snapshot_sha256", evidence["source"].get("source_snapshot_sha256")),
        "historical_job_id": case["job_id"], "historical_inference_hardware": case["hardware"],
        "scorer_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "scorer_files_sha256": {path: file_sha256(REPO / path) for path in code_paths},
        "offline_hardware": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version(), "gpu_used": False},
        "input_files_sha256": input_hashes,
        "evidence_record_sha256": file_sha256(REPO / case["evidence"]),
        "pool_lock_canonical_sha256": canonical_json_sha256(json.loads((REPO / case["lock"]).read_text(encoding="utf-8"))),
        "configuration_canonical_sha256": canonical_json_sha256(json.loads((REPO / case["config"]).read_text(encoding="utf-8"))),
        "retrieval_archive_sha256": expected_archive, "metadata_member_sha256": metadata_sha,
        "query_universe_sha256": canonical_json_sha256([q["query_id"] for q in queries]),
        "query_manifest_sha256": new["query_manifest_sha256"], "annotation_sha256": new["annotation_sha256"],
        "predictions_canonical_sha256": new["result_sha256"], "qrels_sha256": new["qrels_sha256"],
        "qrels_reconstruction": "historical annotation rules applied unchanged to all 1000 locked metadata records; not human truth",
        "query_reconstruction": "original commit generator, exact committed bundle lock" if case_name == "v8_validation" else "retained original manifest, exact committed file hash",
        "denominators": {"queries": len(queries), "corpus_documents": len(metadata), "qrels_pairs": len(queries) * len(metadata)},
        "contract": new["contract"], "comparison": compare(old, new),
        "gate_disposition": "OLD_GATE_HISTORICAL_ONLY_SCORER_CHANGED_NO_REQUALIFICATION_OR_THRESHOLD_RETUNING",
        "performance_disposition": "UNCHANGED_NO_NEW_TIMING_NO_QUALITY_LATENCY_PROMOTION",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--retrieval-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("audit outputs are immutable; choose a new versioned path")
    report = audit(args.case, args.input_root, args.retrieval_archive)
    report["report_content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"case": args.case, "status": report["status"], "output_sha256": file_sha256(args.output),
                      "denominators": report["denominators"], "metrics": {name: value["scorer_v2"]["ndcg_at_10"]
                      for name, value in report["comparison"].items()}}, indent=2))


if __name__ == "__main__":
    main()

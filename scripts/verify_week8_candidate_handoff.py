"""Verify the packaged automatic-silver candidate, not a formal release promotion."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_release_bundle import verify_runtime_archive
from scripts.upload_release_oss import verify_release_dir
from src.data.week8_visual_holdout import write_json_new
from src.evaluation.week8_visual_silver import validate_locked_final, validate_incumbent_nonregression
from src.inference.product_observation import canonical_config_sha256
from scripts.validate_week8_correction_evidence import validate_nonregression


def bind_acceptance(acceptance, release):
    if (acceptance.get("status") != "PASS" or acceptance.get("candidate_quality_accepted") is not True
            or acceptance.get("human_annotation_count") != 0
            or acceptance.get("human_visual_accuracy_claim") is not False
            or acceptance.get("label_source") != "model_generated_silver"
            or acceptance.get("formal_release_replaced") is not False):
        raise ValueError("missing candidate quality evidence; never substitute a formal or human claim")
    for key, source in (("release_id", "release_id"), ("release_config_sha256", "config_sha256"),
                        ("adapter_model_sha256", "adapter_model_sha256")):
        if acceptance.get(key) != release[source]:
            raise ValueError("packaged candidate identity mismatch")


def validate_inference_coverage(summary, count, expected_roles, scores, records):
    if (type(count) is not int or count <= 0 or not {"formal_adapter", "locked_candidate"} <= expected_roles
            or set(summary["roles"]) != expected_roles or set(records) != expected_roles):
        raise ValueError("packaged paired inference roles changed")
    for role, declared in summary["roles"].items():
        rows = records[role]
        if (declared["count"] != count or len(rows) != count
                or any(type(row.get("passed")) is not bool for row in rows)):
            raise ValueError("packaged paired inference coverage changed")
        failures = sum(not row["passed"] for row in rows)
        if declared["failures"] != failures or abs(scores[role]["metrics"]["request_failure_rate"] - failures / count) > 1e-12:
            raise ValueError("packaged inference failure disclosure differs from raw")
        # 基线失败可以用于衡量修复；只有新候选必须零失败，不能要求旧模型先被修好。
        if role == "locked_candidate" and failures:
            raise ValueError("packaged candidate inference has request failures")


def verify(release_dir, acceptance_member):
    manifest = verify_release_dir(release_dir)
    runtime = verify_runtime_archive(release_dir / manifest["layers"]["runtime"]["file"])
    if runtime["release_id"] != manifest["release"]["release_id"]:
        raise ValueError("isolated runtime loaded a different release")
    member = PurePosixPath(acceptance_member)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError("unsafe evidence member")
    with tarfile.open(release_dir / manifest["layers"]["evidence"]["file"], "r:gz") as archive:
        def raw(name):
            info = archive.getmember(str(name))
            if not info.isfile():
                raise ValueError("evidence must be a regular file")
            return archive.extractfile(info).read()
        def read(name):
            return json.loads(raw(name))
        def digest(name):
            return hashlib.sha256(raw(name)).hexdigest()
        acceptance = read(member)
        bind_acceptance(acceptance, manifest["release"])
        root = member.parent
        comparison = read(root / "final_comparison.json")
        lock = read(root / "candidate_lock.json")
        if (digest(root / "final_comparison.json") != acceptance["final_comparison_sha256"]
                or lock["lock_sha256"] != acceptance["candidate_lock_sha256"]
                or lock["lock_sha256"] != canonical_config_sha256({k: v for k, v in lock.items() if k != "lock_sha256"})
                or comparison["candidate_lock_sha256"] != lock["lock_sha256"]):
            raise ValueError("packaged quality evidence changed")
        correction = lock.get("correction_evidence")
        if correction is not None:
            if correction.get("status") != "PASS" or not correction.get("packaged_artifact_sha256"):
                raise ValueError("packaged correction evidence is incomplete")
            for name, expected in correction["packaged_artifact_sha256"].items():
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "evidence" or digest(path) != expected:
                    raise ValueError("packaged correction evidence changed")
            if correction.get("semantic_evidence_member") not in correction["packaged_artifact_sha256"]:
                raise ValueError("packaged correction semantics is not hash-bound")
            semantics = read(correction["semantic_evidence_member"])
            validate_nonregression(semantics)
            if semantics["reference_audit"]["reference_raw_sha256"] != correction["reference_raw_sha256"]:
                raise ValueError("packaged correction reference identity changed")
        gate = validate_locked_final(comparison["scores"]["formal_adapter"], comparison["scores"]["locked_candidate"])
        if gate["status"] != "PASS" or gate != comparison["acceptance"]:
            raise ValueError("packaged final quality has not passed")
        expected_roles = set(lock.get("inference_roles", ["formal_adapter", "locked_candidate"]))
        if "incumbent" in expected_roles:
            incumbent_gate = validate_incumbent_nonregression(comparison["scores"]["incumbent"], comparison["scores"]["locked_candidate"])
            if (comparison.get("overall_status") != "PASS" or incumbent_gate["status"] != "PASS" or incumbent_gate != comparison.get("incumbent_acceptance")
                    or incumbent_gate != acceptance.get("incumbent_acceptance")):
                raise ValueError("packaged candidate regressed against its incumbent")
        data = read(root / "dataset_lock.json")
        if (data["lock_sha256"] != lock["data_lock_sha256"]
                or data["lock_sha256"] != canonical_config_sha256({k: v for k, v in data.items() if k != "lock_sha256"})
                or digest(root / "manifest.jsonl") != data["manifest_sha256"]
                or digest(root / "isolation.json") != data["isolation_sha256"]):
            raise ValueError("packaged final data identity changed")
        if len(data["images_sha256"]) != data["count"]:
            raise ValueError("packaged image coverage changed")
        if data.get("unlabeled_source_pool_lock_sha256"):
            from src.data.week8_unlabeled_pool import validate_pool_snapshot
            pool = root / "source_pool"
            final_rows = [json.loads(line) for line in raw(root / "manifest.jsonl").splitlines() if line.strip()]
            validate_pool_snapshot(data, final_rows, read(pool / "source_pool_lock.json"),
                lambda name: raw(pool / name).decode("utf-8"), lambda name: digest(pool / name))
        for path, expected in data["images_sha256"].items():
            if digest(root / "images" / PurePosixPath(path).name) != expected:
                raise ValueError("packaged final image SHA-256 mismatch")
        for role in ("teacher", "inference"):
            summary = read(root / role / "summary.json")
            if summary["status"] != "COMPLETED" or summary["candidate_lock_sha256"] != lock["lock_sha256"]:
                raise ValueError("packaged final attempt incomplete")
            if role == "teacher":
                if summary["count"] != data["count"] or summary["failures"] != 0:
                    raise ValueError("packaged teacher coverage or validity changed")
            else:
                records = {name: [json.loads(line) for line in raw(root / role / (name + ".jsonl")).splitlines()]
                           for name in expected_roles}
                validate_inference_coverage(summary, data["count"], expected_roles, comparison["scores"], records)
            hashes = {"raw_outputs": summary["raw_sha256"]} if role == "teacher" else {
                name: item["raw_sha256"] for name, item in summary["roles"].items()}
            for name, expected in hashes.items():
                if digest(root / role / (name + ".jsonl")) != expected:
                    raise ValueError("packaged raw output SHA-256 mismatch")
    return {"status": "PASS", "eligible_for_automatic_silver_candidate": True,
            "release_id": manifest["release"]["release_id"], "candidate_lock_sha256": lock["lock_sha256"],
            "runtime_isolated_import": runtime, "layers": manifest["layers"],
            "human_visual_accuracy_claim": False, "formal_release_replaced": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--acceptance-member", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.release_dir.resolve(), args.acceptance_member)
    write_json_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False))

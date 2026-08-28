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
from src.evaluation.week8_visual_silver import validate_locked_final
from src.inference.product_observation import canonical_config_sha256


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
        gate = validate_locked_final(comparison["scores"]["formal_adapter"], comparison["scores"]["locked_candidate"])
        if gate["status"] != "PASS" or gate != comparison["acceptance"]:
            raise ValueError("packaged final quality has not passed")
        data = read(root / "dataset_lock.json")
        if (data["lock_sha256"] != lock["data_lock_sha256"]
                or data["lock_sha256"] != canonical_config_sha256({k: v for k, v in data.items() if k != "lock_sha256"})
                or digest(root / "manifest.jsonl") != data["manifest_sha256"]
                or digest(root / "isolation.json") != data["isolation_sha256"]):
            raise ValueError("packaged final data identity changed")
        if len(data["images_sha256"]) != data["count"]:
            raise ValueError("packaged image coverage changed")
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
            elif (set(summary["roles"]) != {"formal_adapter", "locked_candidate"}
                  or any(item["count"] != data["count"] or item["failures"] != 0 for item in summary["roles"].values())):
                raise ValueError("packaged paired inference coverage changed")
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

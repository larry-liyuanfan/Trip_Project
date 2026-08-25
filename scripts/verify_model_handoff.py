#!/usr/bin/env python3
"""Verify the local model handoff package without cloud services."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.upload_release_oss import ReleaseVerificationError, verify_release_dir


SMOKE_MEMBER = "evidence/system_release_model_smoke_20260825_v6.json"
FINAL_GATE_MEMBER = "evidence/final_test_gate.json"
RETRIEVAL_BENCHMARK_MEMBER = "retrieval/milvus_benchmark_1000.json"


def verify_model_handoff(release_dir: Path) -> dict[str, Any]:
    release_dir = Path(release_dir).resolve()
    manifest = verify_release_dir(release_dir)
    release = manifest["release"]

    evidence_archive = release_dir / manifest["layers"]["evidence"]["file"]
    final_gate = _read_json_member(evidence_archive, FINAL_GATE_MEMBER)
    smoke = _read_json_member(evidence_archive, SMOKE_MEMBER)
    retrieval_archive = release_dir / manifest["layers"]["retrieval"]["file"]
    benchmark = _read_json_member(
        retrieval_archive,
        RETRIEVAL_BENCHMARK_MEMBER,
    )

    failures: list[str] = []
    if final_gate.get("status") != "PASS" or final_gate.get("release_allowed") is not True:
        failures.append("final_gate:not_passed")
    if smoke.get("status") != "PASS":
        failures.append("model_smoke:not_passed")
    for scenario in (
        "image_product_search",
        "after_sales",
        "itinerary_planning",
    ):
        result = smoke.get("scenarios", {}).get(scenario, {})
        if result.get("schema_valid") is not True:
            failures.append(f"model_smoke:{scenario}:schema_invalid")
        if result.get("release_id") != release["release_id"]:
            failures.append(f"model_smoke:{scenario}:release_mismatch")
    dialogue = smoke.get("dialogue", {})
    if dialogue.get("quality_tier") != "DIALOGUE_BETA":
        failures.append("model_smoke:dialogue:not_beta")
    if dialogue.get("release_id") != release["release_id"]:
        failures.append("model_smoke:dialogue:release_mismatch")
    if smoke.get("evidence", {}).get("adapter_model_sha256") != release[
        "adapter_model_sha256"
    ]:
        failures.append("model_smoke:adapter_mismatch")

    if benchmark.get("status") != "completed":
        failures.append("retrieval:benchmark_incomplete")
    if benchmark.get("collection") != "ota_business_image_vector":
        failures.append("retrieval:collection_mismatch")
    if benchmark.get("vector_dimension") != 512:
        failures.append("retrieval:dimension_mismatch")
    if benchmark.get("actual_vector_count_inserted") != 1000:
        failures.append("retrieval:vector_count_mismatch")

    if failures:
        raise ReleaseVerificationError("; ".join(failures))
    return {
        "status": "PASS",
        "release_id": release["release_id"],
        "adapter_model_sha256": release["adapter_model_sha256"],
        "layers": manifest["layers"],
        "model_smoke": {
            "status": smoke["status"],
            "scenarios": 3,
            "dialogue_quality_tier": dialogue["quality_tier"],
        },
        "retrieval": {
            "collection": benchmark["collection"],
            "vector_count": benchmark["actual_vector_count_inserted"],
            "vector_dimension": benchmark["vector_dimension"],
            "recall_at_k": benchmark.get("search", {}).get("recall_at_k"),
        },
    }


def _read_json_member(archive_path: Path, member: str) -> dict[str, Any]:
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            handle = archive.extractfile(member)
            if handle is None:
                raise ReleaseVerificationError(f"handoff member is missing: {member}")
            payload = json.load(handle)
    except (OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError(f"handoff member is invalid: {member}") from exc
    if not isinstance(payload, dict):
        raise ReleaseVerificationError(f"handoff member is not an object: {member}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    args = parser.parse_args()
    result = verify_model_handoff(args.release_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify the compact Week 8 final model handoff package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_release_bundle import verify_runtime_archive
from scripts.upload_release_oss import ReleaseVerificationError, verify_release_dir
from src.inference.product_observation import canonical_config_sha256


FINAL_RELEASE_ID = "trip-qwen3-vl-8b-week8-final-v1"
PRODUCT_RELEASE_ID = "trip-qwen3-vl-8b-week8-visual-silver-v12"
ITINERARY_RELEASE_ID = "trip-qwen3-vl-8b-week8-visual-silver-v13"


def verify_final_delivery(release_dir: Path) -> dict[str, Any]:
    """Verify package layers, runtime import, and Week 8 evidence lineage."""

    release_dir = Path(release_dir).resolve()
    manifest = verify_release_dir(release_dir)
    release = manifest["release"]
    runtime = verify_runtime_archive(release_dir / manifest["layers"]["runtime"]["file"])
    if release.get("release_id") != FINAL_RELEASE_ID or runtime.get("release_id") != FINAL_RELEASE_ID:
        raise ReleaseVerificationError("final release identity mismatch")

    evidence_archive = release_dir / manifest["layers"]["evidence"]["file"]
    with tarfile.open(evidence_archive, "r:gz") as archive:
        final_config = _read_json(archive, "evidence/qwen3_vl_system_final_v1.json")
        product_config = _read_json(archive, "evidence/qwen3_vl_system_week8_v12.json")
        itinerary_config = _read_json(archive, "evidence/qwen3_vl_system_week8_v13.json")
        acceptance = _read_json(archive, "evidence/promotion_acceptance.json")
        comparison_bytes = _read_bytes(archive, "evidence/final_comparison.json")
        comparison = json.loads(comparison_bytes)
        lock = _read_json(archive, "evidence/candidate_lock.json")
        itinerary = _read_json(
            archive,
            "evidence/week8_itinerary_runtime_comparison_20260829_v1.json",
        )
        test_log = _read_bytes(archive, "evidence/final_unittest.log").decode("utf-8")

    validate_release_lineage(
        final_config,
        product_config,
        itinerary_config,
        acceptance,
        itinerary,
    )
    if _sha256(comparison_bytes) != acceptance.get("final_comparison_sha256"):
        raise ReleaseVerificationError("product comparison hash mismatch")
    lock_payload = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if (
        lock.get("lock_sha256") != acceptance.get("candidate_lock_sha256")
        or lock.get("lock_sha256") != canonical_config_sha256(lock_payload)
        or comparison.get("candidate_lock_sha256") != lock.get("lock_sha256")
        or comparison.get("overall_status") != "PASS"
    ):
        raise ReleaseVerificationError("product acceptance lock mismatch")
    match = re.search(r"Ran\s+(\d+)\s+tests", test_log)
    if not match or int(match.group(1)) < 945 or not re.search(r"\r?\nOK(?:\r?\n|$)", test_log):
        raise ReleaseVerificationError("final unittest evidence is incomplete")
    if release.get("adapter_model_sha256") != final_config["model"]["adapter_model_sha256"]:
        raise ReleaseVerificationError("adapter identity mismatch")

    return {
        "status": "PASS",
        "release_id": FINAL_RELEASE_ID,
        "adapter_model_sha256": release["adapter_model_sha256"],
        "runtime_isolated_import": runtime,
        "product_evidence": {
            "release_id": PRODUCT_RELEASE_ID,
            "label_source": "model_generated_silver",
            "human_visual_accuracy_claim": False,
        },
        "itinerary_evidence": {
            "release_id": ITINERARY_RELEASE_ID,
            "direct_itinerary_nonregression": True,
            "dialogue_first_attempt_improved": True,
        },
        "tests": int(match.group(1)),
        "layers": manifest["layers"],
        "known_pending": [
            "product_price_range_supported_metric",
            "human_visual_accuracy_validation",
            "strict_dialogue_research_gate",
            "independent_business_retrieval_relevance",
        ],
    }


def validate_release_lineage(
    final_config: dict[str, Any],
    product_config: dict[str, Any],
    itinerary_config: dict[str, Any],
    acceptance: dict[str, Any],
    itinerary: dict[str, Any],
) -> None:
    """Enforce the authorized v12 product plus v13 itinerary composition."""

    if final_config.get("release_id") != FINAL_RELEASE_ID or final_config.get("status") != "final_delivery":
        raise ReleaseVerificationError("final config is not the authorized delivery identity")
    if product_config.get("release_id") != PRODUCT_RELEASE_ID:
        raise ReleaseVerificationError("product source release mismatch")
    if itinerary_config.get("release_id") != ITINERARY_RELEASE_ID:
        raise ReleaseVerificationError("itinerary source release mismatch")

    inherited = ("model", "product_pipeline", "schemas", "dialogue", "generation")
    if any(final_config.get(key) != itinerary_config.get(key) for key in inherited):
        raise ReleaseVerificationError("final runtime differs from the selected v13 candidate")
    if final_config.get("prompts") != itinerary_config.get("prompts"):
        raise ReleaseVerificationError("final prompts differ from the selected v13 candidate")
    if any(product_config.get(key) != itinerary_config.get(key) for key in inherited):
        raise ReleaseVerificationError("v13 changed more than the approved itinerary prompt")
    prompt_changes = {
        key
        for key in set(product_config.get("prompts", {})) | set(itinerary_config.get("prompts", {}))
        if product_config.get("prompts", {}).get(key) != itinerary_config.get("prompts", {}).get(key)
    }
    if prompt_changes != {"itinerary_planning"}:
        raise ReleaseVerificationError("v12 to v13 prompt scope changed")

    quality = final_config.get("quality", {})
    if (
        quality.get("formal_delivery_authorized_by_user") is not True
        or quality.get("human_visual_accuracy_claim") is not False
        or quality.get("label_source") != "model_generated_silver"
        or quality.get("product_price_range_status") != "PENDING_NO_SUPPORTED_REFERENCE"
    ):
        raise ReleaseVerificationError("final quality boundary is missing")
    if (
        acceptance.get("status") != "PASS"
        or acceptance.get("candidate_quality_accepted") is not True
        or acceptance.get("release_id") != PRODUCT_RELEASE_ID
        or acceptance.get("human_annotation_count") != 0
        or acceptance.get("human_visual_accuracy_claim") is not False
        or acceptance.get("label_source") != "model_generated_silver"
        or acceptance.get("formal_release_replaced") is not False
    ):
        raise ReleaseVerificationError("product candidate evidence is invalid")
    if (
        itinerary.get("status") != "PASS"
        or itinerary.get("selected_release") != ITINERARY_RELEASE_ID
        or itinerary.get("release_change_scope") != ["release_id", "prompts.itinerary_planning"]
        or itinerary.get("direct_itinerary_nonregression") is not True
        or itinerary.get("dialogue_first_attempt_improved") is not True
        or itinerary.get("final_test_rows_read") is not False
    ):
        raise ReleaseVerificationError("itinerary derivative evidence is invalid")


def _read_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    try:
        info = archive.getmember(name)
        handle = archive.extractfile(info)
    except (KeyError, tarfile.TarError) as exc:
        raise ReleaseVerificationError(f"handoff member is missing: {name}") from exc
    if handle is None or not info.isfile():
        raise ReleaseVerificationError(f"handoff member is invalid: {name}")
    return handle.read()


def _read_json(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(_read_bytes(archive, name))
    except json.JSONDecodeError as exc:
        raise ReleaseVerificationError(f"handoff JSON is invalid: {name}") from exc
    if not isinstance(payload, dict):
        raise ReleaseVerificationError(f"handoff JSON is not an object: {name}")
    return payload


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_final_delivery(args.release_dir),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

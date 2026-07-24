"""Deterministic stratified candidate sampling for Week 3 evaluation sets."""

import copy
import hashlib
import random
from typing import Any

from pathlib import Path

from src.evaluation.manifests import (
    ManifestValidationError,
    calculate_image_sha256,
    validate_manifest_record,
)


def stratified_sample(
    candidates: list[dict[str, Any]],
    *,
    scenario: str,
    dataset_version: str,
    seed: int,
    stratum_field: str,
    quotas: dict[str, int],
    root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select fixed-seed per-stratum candidates and leave human states pending."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ManifestValidationError("sampling seed must be an integer")
    if not isinstance(stratum_field, str) or not stratum_field.strip():
        raise ManifestValidationError("stratum_field must be a non-empty string")
    for stratum, quota in quotas.items():
        if not isinstance(stratum, str) or not stratum.strip():
            raise ManifestValidationError("sampling strata must be non-empty strings")
        if isinstance(quota, bool) or not isinstance(quota, int) or quota < 0:
            raise ManifestValidationError(f"quota for {stratum!r} must be a non-negative integer")

    grouped: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in quotas}
    unconfigured_count = 0
    source_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ManifestValidationError("sampling candidates must be JSON objects")
        source_id = candidate.get("source_id")
        if isinstance(source_id, str):
            if source_id in source_ids:
                raise ManifestValidationError(f"duplicate source_id in sampling candidates: {source_id}")
            source_ids.add(source_id)
        stratum = candidate.get(stratum_field)
        if stratum not in grouped:
            unconfigured_count += 1
            continue
        grouped[stratum].append(candidate)

    selected: list[tuple[str, dict[str, Any]]] = []
    strata_log: dict[str, dict[str, Any]] = {}
    for stratum in sorted(quotas):
        available = sorted(
            grouped[stratum],
            key=lambda item: (str(item.get("source_id", "")), str(item.get("image_sha256", ""))),
        )
        stratum_seed = int.from_bytes(
            hashlib.sha256(f"{seed}|{scenario}|{stratum}".encode("utf-8")).digest(),
            "big",
        )
        randomizer = random.Random(stratum_seed)
        randomizer.shuffle(available)
        chosen = available[: quotas[stratum]]
        selected.extend((stratum, candidate) for candidate in chosen)
        selected_source_ids = sorted(str(candidate.get("source_id", "")) for candidate in chosen)
        strata_log[stratum] = {
            "available_count": len(available),
            "requested_count": quotas[stratum],
            "selected_count": len(chosen),
            "shortfall_count": max(quotas[stratum] - len(chosen), 0),
            "selected_source_ids": selected_source_ids,
        }

    records = [
        _candidate_to_pending_record(
            candidate,
            scenario=scenario,
            dataset_version=dataset_version,
            sampling_stratum=stratum,
            root=root,
        )
        for stratum, candidate in sorted(
            selected,
            key=lambda item: (item[0], str(item[1].get("source_id", ""))),
        )
    ]
    sampling_log = {
        "scenario": scenario,
        "dataset_version": dataset_version,
        "seed": seed,
        "stratum_field": stratum_field,
        "candidate_input_count": len(candidates),
        "unconfigured_candidate_count": unconfigured_count,
        "requested_total": sum(quotas.values()),
        "selected_total": len(records),
        "strata": strata_log,
    }
    return records, sampling_log


def _candidate_to_pending_record(
    candidate: dict[str, Any],
    *,
    scenario: str,
    dataset_version: str,
    sampling_stratum: str,
    root: Path | None,
) -> dict[str, Any]:
    source_id = candidate.get("source_id")
    sample_digest = hashlib.sha256(f"{scenario}|{source_id}".encode("utf-8")).hexdigest()[:16]
    input_payload = copy.deepcopy(candidate.get("input"))
    image_sha256 = candidate.get("image_sha256")
    if root is not None:
        if not isinstance(input_payload, dict) or not isinstance(input_payload.get("images"), list):
            raise ManifestValidationError("candidate input.images must be an array")
        for index, image in enumerate(input_payload["images"]):
            if not isinstance(image, dict) or not isinstance(image.get("path"), str):
                raise ManifestValidationError(f"candidate input.images[{index}] requires a path")
            actual_hash = calculate_image_sha256(Path(root), image["path"])
            supplied_hash = image.get("sha256")
            if supplied_hash is not None and supplied_hash != actual_hash:
                raise ManifestValidationError(
                    f"candidate input.images[{index}].sha256 does not match image bytes"
                )
            image["sha256"] = actual_hash
        if input_payload["images"]:
            computed_primary_hash = input_payload["images"][0]["sha256"]
            if image_sha256 is not None and image_sha256 != computed_primary_hash:
                raise ManifestValidationError("candidate image_sha256 does not match image bytes")
            image_sha256 = computed_primary_hash
    record = {
        "sample_id": f"{scenario}-{sample_digest}",
        "scenario": scenario,
        "source_type": candidate.get("source_type"),
        "source_id": source_id,
        "source_license": candidate.get("source_license"),
        "image_sha256": image_sha256,
        "input": input_payload,
        "split": "evaluation",
        "dataset_version": dataset_version,
        "annotation_status": "pending",
        "annotator": None,
        "review_status": "pending",
        "reviewer": None,
        "file_status": candidate.get("file_status", "pending"),
        "annotation": None,
        "notes": candidate.get("notes"),
        "sampling_stratum": sampling_stratum,
    }
    if "provenance" in candidate:
        record["provenance"] = copy.deepcopy(candidate["provenance"])
    return validate_manifest_record(record, root=root)

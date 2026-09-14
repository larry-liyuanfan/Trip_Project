"""Audit identity-only evidence availability without opening protected examples.

This is a prerequisite inventory, not a leakage comparator or permission to run a model.
Missing dimensions stay missing; aggregate locks never substitute for row identities.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256


IDENTITY_FIELDS = (
    "sample_id", "source_id", "image_sha256", "query_sha256", "source_record_sha256",
    "dialogue_text_sha256", "content_sha256", "group_id", "constraint_template_id",
)
ALLOWED_FIELDS = set(IDENTITY_FIELDS) | {
    "record_id", "sha256", "split", "scenario", "label_source", "dataset_version",
    "image_path", "image_perceptual_hash", "pii_review_status", "source_uri",
    "source_version", "synthetic_recipe_version", "relative_path", "width", "height",
    "purpose", "contains_image_bytes",
}
SHA_FIELDS = {key for key in IDENTITY_FIELDS if key.endswith("sha256")} | {"sha256"}


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("registry path escapes identity-only directory")
    if path.suffix not in {".json", ".jsonl"}:
        raise ValueError("registry must be JSON or JSONL")
    return path


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_bound_rows(path: Path, spec: dict[str, Any]) -> tuple[list[dict], dict]:
    expected_file = spec.get("file_sha256")
    expected_canonical = spec.get("canonical_sha256")
    if not expected_file and not expected_canonical:
        raise ValueError("registry requires a pre-existing SHA binding")
    for digest in (expected_file, expected_canonical):
        if digest is not None and not _valid_sha(digest):
            raise ValueError("invalid expected registry SHA-256")
    blob = path.read_bytes()
    actual_file = hashlib.sha256(blob).hexdigest()
    if expected_file and actual_file != expected_file:
        raise ValueError("registry file SHA-256 mismatch")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(blob.decode("utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("registry must be a nonempty list of identity records")
    actual_canonical = canonical_json_sha256(rows)
    if expected_canonical and actual_canonical != expected_canonical:
        raise ValueError("registry canonical SHA-256 mismatch")
    if type(spec.get("expected_rows")) is not int or len(rows) != spec["expected_rows"]:
        raise ValueError("registry row support mismatch")
    for row in rows:
        if not isinstance(row, dict) or set(row) - ALLOWED_FIELDS:
            raise ValueError("registry contains non-identity fields; raw examples are forbidden")
        if any(isinstance(value, (dict, list)) for value in row.values()):
            raise ValueError("registry must contain only flat identity metadata")
        if row.get("contains_image_bytes") not in (None, False):
            raise ValueError("registry claims image bytes")
        for key in SHA_FIELDS:
            if row.get(key) not in (None, "") and not _valid_sha(row[key]):
                raise ValueError(f"invalid identity digest: {key}")
        for key in set(IDENTITY_FIELDS) - SHA_FIELDS:
            if row.get(key) not in (None, "") and not isinstance(row[key], str):
                raise ValueError(f"identity must be a string: {key}")
    return rows, {"file_sha256": actual_file, "canonical_sha256": actual_canonical}


def audit_inventory(config: dict[str, Any], root: Path) -> dict[str, Any]:
    specs = config.get("protected_identity_inventory")
    if not isinstance(specs, list) or not specs:
        raise ValueError("protected inventory cannot be empty")
    names = [spec.get("id") for spec in specs]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("registry IDs must be nonempty and unique")
    entries = []
    for spec in specs:
        report = {"id": spec["id"], "scope": spec["scope"], "status": "UNAVAILABLE"}
        relative = spec.get("file")
        if not relative:
            report["reason"] = spec.get("reason", "identity-only registry not located")
            entries.append(report)
            continue
        path = _safe_path(root, relative)
        if not path.is_file():
            report["reason"] = "registered identity-only file missing"
            entries.append(report)
            continue
        rows, hashes = _read_bound_rows(path, spec)
        aliases = spec.get("field_aliases", {})
        allowed_aliases = {"sample_id": "record_id", "image_sha256": "sha256"}
        if any(allowed_aliases.get(key) != value for key, value in aliases.items()):
            raise ValueError("invalid identity field aliases")
        support = {}
        for field in IDENTITY_FIELDS:
            key = aliases.get(field, field)
            values = [str(row[key]) for row in rows if row.get(key) not in (None, "")]
            support[field] = {"nonempty_rows": len(values), "unique_values": len(set(values)),
                              "row_denominator": len(rows)}
        required = spec.get("required_fields")
        if not required or set(required) - set(IDENTITY_FIELDS):
            raise ValueError("registry must declare recognized required identity fields")
        incomplete = [field for field in required if support[field]["nonempty_rows"] != len(rows)]
        observed_splits = dict(sorted(Counter(str(row.get("split", "not_recorded"))
                                             for row in rows).items()))
        expected_splits = spec.get("expected_splits")
        if expected_splits is not None and observed_splits != expected_splits:
            raise ValueError("registry split support mismatch")
        report.update({
            **hashes, "row_support": len(rows), "fields": support,
            "split_support": observed_splits,
            "missing_or_partial_dimensions": incomplete,
            "status": "INCOMPLETE_DIMENSIONS" if incomplete else "AVAILABLE_BOUND",
        })
        entries.append(report)
    blocked = [item["id"] for item in entries if item["status"] != "AVAILABLE_BOUND"]
    return {
        "schema_version": "protected_identity_inventory_v1",
        "status": "BLOCKED_PROTECTED_IDENTITY_COVERAGE" if blocked else "INVENTORY_READY_ONLY",
        "config_canonical_sha256": canonical_json_sha256(config),
        "registry_support": len(entries),
        "hash_verified_registry_support": sum("file_sha256" in item for item in entries),
        "available_bound_registry_support": sum(item["status"] == "AVAILABLE_BOUND" for item in entries),
        "blocking_registry_ids": blocked,
        "registries": entries,
        "overlap_check_status": "NOT_RUN_NO_NEW_DATA_LOCK",
        "overlap_count": None,
        "model_execution_authorized": False,
        "raw_protected_samples_or_predictions_opened": False,
        "interpretation": "inventory only; availability is not proof of isolation or model quality",
    }


def audit_from_paths(config_path: Path, identity_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {**audit_inventory(config, identity_root), "config_file_sha256": file_sha256(config_path)}

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
IMPORT_CANONICALIZATION = "json_utf8_sort_keys_compact_v1"
APPROVAL_SCHEMA = "identity_import_approved_sources_v1"
EXPORT_SCHEMA = "identity_only_export_manifest_v1"
COMMON_IMPORT_FIELDS = {
    "sample_id", "source_id", "source_record_sha256", "query_sha256", "content_sha256",
}
IMAGE_IDENTITY_POLICIES = {"required", "not_applicable_text_only"}


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("registry path escapes identity-only directory")
    if path.suffix not in {".json", ".jsonl"}:
        raise ValueError("registry must be JSON or JSONL")
    return path


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _load_strict_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def _load_strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line, object_pairs_hook=_unique_object))
    return rows


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} fields mismatch")
    return value


def _require_sha(value: Any, label: str) -> str:
    if not _valid_sha(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _hash_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    expected = _require_sha(expected_sha256, f"expected {label} SHA-256")
    if file_sha256(path) != expected:
        raise ValueError(f"{label} file SHA-256 mismatch")
    value = _load_strict_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _bundle_file(root: Path, relative: Any) -> Path:
    name = _require_string(relative, "identity export path")
    candidate = root / name
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved_root not in resolved.parents or candidate.is_symlink() or not candidate.is_file():
        raise ValueError("identity export path escapes bundle or is not a regular file")
    if candidate.suffix != ".jsonl":
        raise ValueError("identity exports must use JSONL")
    return candidate


def _validate_approval_registry(value: dict[str, Any], expected_scope_count: int) -> dict[str, dict]:
    _require_exact_keys(
        value,
        {"schema_version", "canonicalization_version", "approval_id", "required_scope_ids", "sources"},
        "approved source registry",
    )
    if value["schema_version"] != APPROVAL_SCHEMA:
        raise ValueError("unsupported approved source registry schema")
    if value["canonicalization_version"] != IMPORT_CANONICALIZATION:
        raise ValueError("unsupported approved canonicalization version")
    _require_string(value["approval_id"], "approval_id")
    scope_ids = value["required_scope_ids"]
    sources = value["sources"]
    if (
        not isinstance(scope_ids, list)
        or len(scope_ids) != expected_scope_count
        or any(not isinstance(item, str) or not item for item in scope_ids)
        or len(set(scope_ids)) != len(scope_ids)
    ):
        raise ValueError("approved scope coverage must be unique and complete")
    if not isinstance(sources, list) or len(sources) != expected_scope_count:
        raise ValueError("approved source support does not cover every scope")
    result: dict[str, dict] = {}
    for source in sources:
        _require_exact_keys(
            source,
            {"scope_id", "source_manifest_file_sha256", "data_lock_file_sha256",
             "required_fields", "image_identity_policy"},
            "approved source",
        )
        scope_id = _require_string(source["scope_id"], "approved scope_id")
        if scope_id in result:
            raise ValueError("duplicate approved scope_id")
        _require_sha(source["source_manifest_file_sha256"], "approved source manifest SHA-256")
        _require_sha(source["data_lock_file_sha256"], "approved data lock SHA-256")
        policy = source["image_identity_policy"]
        if not isinstance(policy, str) or policy not in IMAGE_IDENTITY_POLICIES:
            raise ValueError("unsupported image identity policy")
        fields = source["required_fields"]
        if (
            not isinstance(fields, list)
            or any(not isinstance(field, str) for field in fields)
            or len(set(fields)) != len(fields)
            or set(fields) - set(IDENTITY_FIELDS)
            or not COMMON_IMPORT_FIELDS.issubset(fields)
        ):
            raise ValueError("approved required identity fields are incomplete or unsupported")
        image_fields = {"image_sha256"} if policy == "required" else {"dialogue_text_sha256"}
        if not image_fields.issubset(fields):
            raise ValueError("approved image/text identity policy lacks its required digest")
        result[scope_id] = source
    if set(scope_ids) != set(result):
        raise ValueError("approved scope list and source identities differ")
    return result


def _validate_export_manifest(value: dict[str, Any], expected_scope_ids: set[str]) -> dict[str, dict]:
    _require_exact_keys(
        value,
        {"schema_version", "canonicalization_version", "export_id", "approval_id",
         "approved_sources_file_sha256", "scopes"},
        "export manifest",
    )
    if value["schema_version"] != EXPORT_SCHEMA:
        raise ValueError("unsupported identity export manifest schema")
    if value["canonicalization_version"] != IMPORT_CANONICALIZATION:
        raise ValueError("unsupported export canonicalization version")
    _require_string(value["export_id"], "export_id")
    _require_string(value["approval_id"], "export approval_id")
    _require_sha(value["approved_sources_file_sha256"], "export approved sources SHA-256")
    scopes = value["scopes"]
    if not isinstance(scopes, list) or len(scopes) != len(expected_scope_ids):
        raise ValueError("identity export does not cover every approved scope")
    result: dict[str, dict] = {}
    paths: set[str] = set()
    canonical_hashes: set[str] = set()
    for scope in scopes:
        _require_exact_keys(
            scope,
            {"scope_id", "path", "file_sha256", "canonical_rows_sha256", "row_count",
             "source_manifest_file_sha256", "data_lock_file_sha256"},
            "export scope",
        )
        scope_id = _require_string(scope["scope_id"], "export scope_id")
        path = _require_string(scope["path"], "export path")
        if scope_id in result or path in paths:
            raise ValueError("duplicate export scope_id or path")
        for key in ("file_sha256", "canonical_rows_sha256", "source_manifest_file_sha256",
                    "data_lock_file_sha256"):
            _require_sha(scope[key], f"export {key}")
        if type(scope["row_count"]) is not int or scope["row_count"] <= 0:
            raise ValueError("export row_count must be a positive integer")
        if scope["canonical_rows_sha256"] in canonical_hashes:
            raise ValueError("multiple scopes claim identical canonical row exports")
        result[scope_id] = scope
        paths.add(path)
        canonical_hashes.add(scope["canonical_rows_sha256"])
    if set(result) != expected_scope_ids:
        raise ValueError("identity export scope coverage differs from approval")
    return result


def _validate_import_rows(
    rows: Any, scope: dict[str, Any], approved: dict[str, Any], global_samples: dict[str, str]
) -> dict[str, int]:
    if not isinstance(rows, list) or len(rows) != scope["row_count"] or not rows:
        raise ValueError("identity export row support mismatch")
    required = set(approved["required_fields"])
    seen_samples: set[str] = set()
    exact_cross_scope_duplicates = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) - ALLOWED_FIELDS:
            raise ValueError("identity export contains non-identity fields")
        if any(isinstance(item, (dict, list)) for item in row.values()):
            raise ValueError("identity export must contain flat metadata only")
        if any(row.get(field) in (None, "") for field in required):
            raise ValueError("identity export is missing an approved required field")
        if any(not isinstance(row[field], str) for field in required):
            raise ValueError("required identity values must be strings")
        for field in required & SHA_FIELDS:
            _require_sha(row[field], f"identity {field}")
        for field in required - SHA_FIELDS:
            _require_string(row[field], f"identity {field}")
        for field in SHA_FIELDS:
            if row.get(field) not in (None, ""):
                _require_sha(row[field], f"optional identity {field}")
        for field in (set(IDENTITY_FIELDS) - SHA_FIELDS) | {"record_id"}:
            if row.get(field) not in (None, ""):
                _require_string(row[field], f"optional identity {field}")
        if row.get("contains_image_bytes") not in (None, False):
            raise ValueError("identity export claims image bytes")
        sample_id = row["sample_id"]
        if sample_id in seen_samples:
            raise ValueError("duplicate sample_id within export scope")
        seen_samples.add(sample_id)
        if approved["image_identity_policy"] == "not_applicable_text_only" and row.get("image_sha256"):
            raise ValueError("text-only scope unexpectedly contains image identity")
        identity = canonical_json_sha256({
            field: row.get(field) if row.get(field) not in (None, "") else None
            for field in IDENTITY_FIELDS
        })
        previous = global_samples.get(sample_id)
        if previous is not None and previous != identity:
            raise ValueError("conflicting duplicate sample_id across export scopes")
        if previous == identity:
            exact_cross_scope_duplicates += 1
        global_samples[sample_id] = identity
    return {"row_support": len(rows), "exact_cross_scope_duplicate_sample_ids": exact_cross_scope_duplicates}


def validate_identity_only_import(
    *,
    bundle_root: Path,
    export_manifest_sha256: str,
    approved_sources_path: Path,
    approved_sources_sha256: str,
    expected_scope_count: int = 12,
) -> dict[str, Any]:
    """Validate a transport-bound export against an independently hash-bound approval registry."""
    if type(expected_scope_count) is not int or expected_scope_count <= 0:
        raise ValueError("expected_scope_count must be a positive integer")
    root = bundle_root.resolve()
    if not root.is_dir():
        raise ValueError("identity export bundle root is missing")
    approval = _hash_bound_json(
        approved_sources_path, approved_sources_sha256, "approved source registry"
    )
    approved = _validate_approval_registry(approval, expected_scope_count)
    manifest_path = root / "export_manifest.json"
    manifest = _hash_bound_json(manifest_path, export_manifest_sha256, "export manifest")
    if (
        manifest.get("approval_id") != approval["approval_id"]
        or manifest.get("approved_sources_file_sha256") != approved_sources_sha256
    ):
        raise ValueError("export manifest is not bound to the approved source registry")
    scopes = _validate_export_manifest(manifest, set(approved))
    global_samples: dict[str, str] = {}
    scope_reports = []
    for scope_id in approval["required_scope_ids"]:
        scope = scopes[scope_id]
        source = approved[scope_id]
        if (
            scope["source_manifest_file_sha256"] != source["source_manifest_file_sha256"]
            or scope["data_lock_file_sha256"] != source["data_lock_file_sha256"]
        ):
            raise ValueError("export source manifest or data lock identity is not approved")
        path = _bundle_file(root, scope["path"])
        if file_sha256(path) != scope["file_sha256"]:
            raise ValueError("identity export file SHA-256 mismatch")
        rows = _load_strict_jsonl(path)
        if canonical_json_sha256(rows) != scope["canonical_rows_sha256"]:
            raise ValueError("identity export canonical rows SHA-256 mismatch")
        counts = _validate_import_rows(rows, scope, source, global_samples)
        scope_reports.append({
            "scope_id": scope_id,
            "status": "STRUCTURE_AND_APPROVED_SOURCE_BINDING_VALID",
            "file_sha256": scope["file_sha256"],
            "canonical_rows_sha256": scope["canonical_rows_sha256"],
            "source_manifest_file_sha256": scope["source_manifest_file_sha256"],
            "data_lock_file_sha256": scope["data_lock_file_sha256"],
            **counts,
        })
    return {
        "schema_version": "identity_only_import_validation_v1",
        "status": "VALIDATED_IMPORT_NOT_TRAINING_AUTHORIZATION",
        "canonicalization_version": IMPORT_CANONICALIZATION,
        "approval_id": approval["approval_id"],
        "approved_sources_file_sha256": approved_sources_sha256,
        "export_manifest_file_sha256": export_manifest_sha256,
        "scope_support": len(scope_reports),
        "scope_denominator": expected_scope_count,
        "row_support": sum(item["row_support"] for item in scope_reports),
        "scope_reports": scope_reports,
        "format_validation": "PASS",
        "actual_file_hash_validation": "PASS",
        "approved_source_manifest_and_data_lock_binding": "PASS",
        "coverage_validation": "PASS",
        "custodian_attestation_verified": False,
        "human_annotation_support": None,
        "leakage_check_status": "NOT_RUN",
        "overlap_count": None,
        "model_execution_authorized": False,
        "coordinator_release_required": True,
        "interpretation": (
            "Structural import validation only; digest derivation from original protected content, "
            "custodian authority, annotation provenance, leakage, and training eligibility remain unverified."
        ),
    }


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

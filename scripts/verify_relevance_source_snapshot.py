"""Fail closed unless a relevance-evidence source snapshot matches byte for byte."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_snapshot(
    project_root: Path,
    manifest_path: Path,
    expected_snapshot_sha256: str,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "relevance_source_snapshot_v1":
        raise ValueError("unsupported relevance source snapshot schema")
    actual_snapshot_sha256 = canonical_sha256(manifest)
    if actual_snapshot_sha256 != expected_snapshot_sha256:
        raise ValueError("source snapshot manifest SHA-256 mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("source snapshot must bind at least one file")
    observed: set[str] = set()
    for item in files:
        relative = item.get("path") if isinstance(item, dict) else None
        expected = item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(relative, str) or relative in observed:
            raise ValueError("source snapshot paths must be unique strings")
        observed.add(relative)
        path = (project_root / relative).resolve()
        if project_root not in path.parents:
            raise ValueError(f"source snapshot path escapes project root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"source snapshot file is missing: {relative}")
        if file_sha256(path) != expected:
            raise ValueError(f"source snapshot file SHA-256 mismatch: {relative}")
    return {
        "status": "PASS",
        "implementation_commit_sha": manifest.get("implementation_commit_sha"),
        "git_base_sha": manifest.get("git_base_sha"),
        "file_support": len(files),
        "run_source_snapshot_sha256": actual_snapshot_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-snapshot-sha256", required=True)
    args = parser.parse_args()
    report = validate_snapshot(
        args.project_root,
        args.manifest,
        args.expected_snapshot_sha256,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

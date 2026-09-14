"""Create an exclusive per-file SHA manifest for an immutable source snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-base-sha", required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()
    manifest = build_snapshot_manifest(
        project_root=args.project_root,
        output=args.output,
        git_base_sha=args.git_base_sha,
        implementation_commit=args.implementation_commit,
    )
    print(json.dumps({
        "status": "PASS",
        "file_support": len(manifest["files"]),
        "canonical_sha256": canonical_json_sha256(manifest),
        "manifest_file_sha256": file_sha256(args.output),
    }, indent=2, sort_keys=True))


def build_snapshot_manifest(
    *,
    project_root: Path,
    output: Path,
    git_base_sha: str,
    implementation_commit: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"snapshot project root is missing: {root}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite source manifest: {output}")
    base = _full_git_sha(git_base_sha, "git base")
    implementation = _full_git_sha(implementation_commit, "implementation commit")
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    if not files:
        raise ValueError("source snapshot must contain files")
    manifest = {
        "schema_version": "relevance_source_snapshot_v1",
        "git_base_sha": base,
        "implementation_commit_sha": implementation,
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def _full_git_sha(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 40 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{label} must be a full 40-character Git SHA")
    return normalized


if __name__ == "__main__":
    main()

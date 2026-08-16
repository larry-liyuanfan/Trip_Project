"""Build and audit immutable image manifests for Week 6 training shards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Iterator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_image_paths(jsonl_paths: list[Path]) -> Iterator[str]:
    for jsonl_path in jsonl_paths:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for message in row.get("messages", []):
                    content = message.get("content")
                    if not isinstance(content, list):
                        continue
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            image_path = item.get("path")
                            if not isinstance(image_path, str) or not image_path:
                                raise ValueError(
                                    f"invalid image path in {jsonl_path}:{line_number}"
                                )
                            yield image_path


def resolve_scoped(project_root: Path, relative_path: str) -> Path:
    candidate = (project_root / relative_path).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"image escapes project root: {relative_path}") from exc
    return candidate


def build_manifest(
    *, inputs: list[Path], project_root: Path, output: Path, summary: Path
) -> dict:
    unique_paths = sorted(set(iter_image_paths(inputs)))
    missing: list[str] = []
    records: list[dict] = []
    for relative_path in unique_paths:
        resolved = resolve_scoped(project_root, relative_path)
        if not resolved.is_file():
            missing.append(relative_path)
            continue
        records.append(
            {
                "path": relative_path,
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    if missing:
        raise ValueError(f"local image manifest has {len(missing)} missing files")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    payload = {
        "status": "ok",
        "records": len(records),
        "manifest_sha256": sha256_file(output),
        "inputs": [str(path) for path in inputs],
    }
    summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def audit_shard(
    *, manifest: Path, project_root: Path, shard_index: int, shard_count: int, output: Path
) -> dict:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard coordinates")
    checked = 0
    failures: list[dict] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for record_index, line in enumerate(handle):
            if record_index % shard_count != shard_index:
                continue
            record = json.loads(line)
            checked += 1
            resolved = resolve_scoped(project_root, record["path"])
            if not resolved.is_file():
                failures.append({"path": record["path"], "reason": "missing"})
                continue
            if resolved.stat().st_size != int(record["size_bytes"]):
                failures.append({"path": record["path"], "reason": "size_mismatch"})
                continue
            if sha256_file(resolved) != record["sha256"]:
                failures.append({"path": record["path"], "reason": "sha256_mismatch"})
    payload = {
        "status": "ok" if not failures else "failed",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "checked": checked,
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def merge_audits(*, input_dir: Path, shard_count: int, output: Path) -> dict:
    shard_payloads = []
    for path in sorted(input_dir.glob("audit-*.json")):
        shard_payloads.append(json.loads(path.read_text(encoding="utf-8")))
    indices = {int(payload["shard_index"]) for payload in shard_payloads}
    expected = set(range(shard_count))
    failures = [
        failure
        for payload in shard_payloads
        for failure in payload.get("failures", [])
    ]
    reasons: list[str] = []
    if indices != expected:
        reasons.append("missing_audit_shards")
    if failures:
        reasons.append("image_failures")
    payload = {
        "status": "ok" if not reasons else "failed",
        "shard_count": shard_count,
        "completed_shards": sorted(indices),
        "checked": sum(int(item.get("checked", 0)) for item in shard_payloads),
        "failures": failures,
        "reasons": reasons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def pack_failures(
    *, failure_list: Path, project_root: Path, output: Path, summary: Path
) -> dict:
    with gzip.open(failure_list, "rt", encoding="utf-8") as handle:
        paths = sorted({line.strip() for line in handle if line.strip()})
    if not paths:
        raise ValueError("failure list is empty")
    total_bytes = 0
    resolved_paths: list[tuple[str, Path]] = []
    for relative_path in paths:
        resolved = resolve_scoped(project_root, relative_path)
        if not resolved.is_file():
            raise ValueError(f"local recovery image is missing: {relative_path}")
        total_bytes += resolved.stat().st_size
        resolved_paths.append((relative_path, resolved))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for relative_path, resolved in resolved_paths:
            archive.add(resolved, arcname=relative_path, recursive=False)
    payload = {
        "status": "ok",
        "records": len(resolved_paths),
        "source_bytes": total_bytes,
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
    }
    summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def split_file(*, source: Path, output_dir: Path, part_size_bytes: int, summary: Path) -> dict:
    if part_size_bytes <= 0:
        raise ValueError("part size must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    parts: list[dict] = []
    with source.open("rb") as handle:
        index = 0
        while block := handle.read(part_size_bytes):
            part = output_dir / f"part-{index:04d}"
            part.write_bytes(block)
            parts.append(
                {
                    "name": part.name,
                    "size_bytes": len(block),
                    "sha256": sha256_file(part),
                }
            )
            index += 1
    if not parts:
        raise ValueError("source file is empty")
    payload = {
        "status": "ok",
        "source": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "part_size_bytes": part_size_bytes,
        "parts": parts,
    }
    summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--input", type=Path, action="append", required=True)
    build.add_argument("--project-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--summary", type=Path, required=True)
    audit = sub.add_parser("audit-shard")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--project-root", type=Path, required=True)
    audit.add_argument("--shard-index", type=int, required=True)
    audit.add_argument("--shard-count", type=int, required=True)
    audit.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge-audits")
    merge.add_argument("--input-dir", type=Path, required=True)
    merge.add_argument("--shard-count", type=int, required=True)
    merge.add_argument("--output", type=Path, required=True)
    pack = sub.add_parser("pack-failures")
    pack.add_argument("--failure-list", type=Path, required=True)
    pack.add_argument("--project-root", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--summary", type=Path, required=True)
    split = sub.add_parser("split-file")
    split.add_argument("--source", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--part-size-bytes", type=int, required=True)
    split.add_argument("--summary", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "build":
        payload = build_manifest(
            inputs=args.input,
            project_root=args.project_root,
            output=args.output,
            summary=args.summary,
        )
    elif args.command == "audit-shard":
        payload = audit_shard(
            manifest=args.manifest,
            project_root=args.project_root,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            output=args.output,
        )
    elif args.command == "merge-audits":
        payload = merge_audits(
            input_dir=args.input_dir,
            shard_count=args.shard_count,
            output=args.output,
        )
    elif args.command == "pack-failures":
        payload = pack_failures(
            failure_list=args.failure_list,
            project_root=args.project_root,
            output=args.output,
            summary=args.summary,
        )
    else:
        payload = split_file(
            source=args.source,
            output_dir=args.output_dir,
            part_size_bytes=args.part_size_bytes,
            summary=args.summary,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

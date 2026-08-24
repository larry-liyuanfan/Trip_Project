"""Build layered private-OSS release archives and a checksum manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_CONFIG = ROOT / "configs/releases/qwen3_vl_system_v1.json"
RUNTIME_PATHS = [
    "src/api",
    "src/inference",
    "src/retrieval",
    "src/evaluation/schema_validation.py",
    "src/evaluation/prompting.py",
    "configs/evaluation/prompts",
    "configs/evaluation/schemas",
    "configs/releases",
    "docker/system",
    "scripts/tripctl.py",
    "requirements-api.txt",
    "requirements-training.txt",
    "requirements-milvus.txt",
]


def build_bundle(
    output_dir: Path,
    *,
    adapter_dir: Path,
    retrieval_dir: Path,
    evidence_paths: list[Path],
    release_config: Path = DEFAULT_RELEASE_CONFIG,
) -> dict:
    release_config = Path(release_config).resolve()
    adapter_dir = Path(adapter_dir).resolve()
    release = json.loads(release_config.read_text(encoding="utf-8"))
    adapter_model = adapter_dir / "adapter_model.safetensors"
    expected_adapter_sha = release.get("model", {}).get("adapter_model_sha256")
    if not adapter_model.is_file() or _sha256(adapter_model) != expected_adapter_sha:
        raise ValueError("adapter does not match the release config SHA-256")
    if output_dir.exists():
        raise FileExistsError(f"release output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    layers = {
        "runtime": _archive(
            output_dir / "runtime.tar.gz",
            [
                *((ROOT / path, Path(path)) for path in RUNTIME_PATHS),
                (release_config, Path("release/release_config.json")),
            ],
        ),
        "adapter": _archive(
            output_dir / "adapter.tar.gz",
            [(adapter_dir, Path("adapter"))],
        ),
        "retrieval": _archive(
            output_dir / "retrieval.tar.gz",
            [(retrieval_dir, Path("retrieval"))],
        ),
        "evidence": _archive(
            output_dir / "evidence.tar.gz",
            [(path, Path("evidence") / path.name) for path in evidence_paths],
        ),
    }
    manifest = {
        "schema_version": "private_oss_release_v1",
        "visibility": "private",
        "release": {
            "release_id": release.get("release_id"),
            "config_member": "release/release_config.json",
            "config_sha256": _sha256(release_config),
            "adapter_model_sha256": expected_adapter_sha,
        },
        "layers": layers,
    }
    manifest_path = output_dir / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _archive(output: Path, sources: list[tuple[Path, Path]]) -> dict:
    for source, _ in sources:
        if not source.exists():
            raise FileNotFoundError(source)
    members = 0
    with output.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for source, arcname in sources:
                    members += _add_path(archive, source, arcname)
    return {
        "file": output.name,
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
        "member_count": members,
    }


def _add_path(archive: tarfile.TarFile, source: Path, arcname: Path) -> int:
    files = [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.is_file())
    count = 0
    for path in files:
        if "__pycache__" in path.parts or path.suffix == ".pyc" or path.name == ".env":
            continue
        relative = arcname if source.is_file() else arcname / path.relative_to(source)
        info = archive.gettarinfo(str(path), arcname=relative.as_posix())
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        with path.open("rb") as handle:
            archive.addfile(info, handle)
        count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--retrieval-dir", required=True, type=Path)
    parser.add_argument("--release-config", default=DEFAULT_RELEASE_CONFIG, type=Path)
    parser.add_argument("--evidence", action="append", default=[], type=Path)
    args = parser.parse_args()
    manifest = build_bundle(
        args.output_dir,
        adapter_dir=args.adapter_dir,
        retrieval_dir=args.retrieval_dir,
        evidence_paths=args.evidence,
        release_config=args.release_config,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

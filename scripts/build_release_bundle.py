"""Build layered private-OSS release archives and a checksum manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_CONFIG = ROOT / "configs/releases/qwen3_vl_system_v1.json"
RUNTIME_PATHS = [
    "src/__init__.py",
    "src/api",
    "src/inference",
    "src/retrieval",
    "src/planning",
    "src/data/__init__.py",
    "src/data/yelp_paths.py",
    "src/data/product_labels.py",
    "src/evaluation/__init__.py",
    "src/evaluation/scenarios.py",
    "src/evaluation/schema_validation.py",
    "src/evaluation/prompting.py",
    "src/evaluation/product_semantics.py",
    "configs/evaluation/prompts",
    "configs/evaluation/schemas",
    "configs/releases",
    "configs/retrieval",
    "configs/week8/product_observation_v1.json",
    "configs/week8/product_observation_v2.json",
    "configs/week8/product_observation_v3.json",
    "docker/system",
    "scripts/tripctl.py",
    "scripts/load_system_retrieval.py",
    "requirements-api.txt",
    "requirements-training.txt",
    "requirements-milvus.txt",
]


def runtime_paths(release: dict) -> list[str]:
    """Include the actual selected observation config, not just historical versions."""
    paths = list(RUNTIME_PATHS)
    selected = release.get("product_pipeline", {}).get("config")
    if selected is not None:
        if not isinstance(selected, str):
            raise ValueError("runtime observation config must be a repository JSON path")
        relative = PurePosixPath(selected.replace("\\", "/"))
        if (relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("configs",)
                or relative.suffix != ".json"):
            raise ValueError("runtime observation config must stay inside repository configs")
        (ROOT / relative).resolve().relative_to(ROOT.resolve())
        # 目录已包含目标时不重复归档；禁止通过目标配置带入密钥等非配置文件。
        if not any(relative == PurePosixPath(path) or PurePosixPath(path) in relative.parents for path in paths):
            paths.append(relative.as_posix())
    return paths


def verify_runtime_archive(archive_path: Path) -> dict:
    """隔离导入打包后服务，不能借用源工作树掩盖缺失的运行依赖。"""
    with tempfile.TemporaryDirectory(prefix="trip-runtime-verify-") as temporary:
        root = Path(temporary).resolve()
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                destination = (root / member.name).resolve()
                destination.relative_to(root)
                if not member.isfile():
                    raise ValueError("runtime archive may contain only regular files")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
        code = ("import json,sys; from pathlib import Path; root=Path.cwd(); sys.path.insert(0,str(root)); "
                "from src.api.app import app; from src.inference.system_runtime import ReleaseSettings; "
                "from src.retrieval.visual_search import VisualSearchService; "
                "settings=ReleaseSettings.load(root,root/'release/release_config.json'); "
                "paths=sorted(app.openapi()['paths']); "
                "required={'/health','/ready','/v1/tasks/image-product-search','/v1/tasks/after-sales',"
                "'/v1/tasks/itinerary-planning','/v1/dialogue','/v1/visual-search'}; "
                "missing=required-set(paths)\n"
                "if missing: raise ValueError('required business routes missing: '+','.join(sorted(missing)))\n"
                "print(json.dumps({'status':'PASS','release_id':settings.release_id,"
                "'route_count':len(app.routes),'registered_paths':paths,'required_business_paths_present':True,"
                "'observation_loaded':settings.product_observation is not None}))")
        result = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code], cwd=root,
                                env=dict(os.environ, APP_ENV="production", PYTHONIOENCODING="utf-8"),
                                capture_output=True, text=True, encoding="utf-8", timeout=60)
        if result.returncode:
            raise ValueError("isolated runtime import failed: " + result.stderr[-2000:])
        return json.loads(result.stdout)


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
                *((ROOT / path, Path(path)) for path in runtime_paths(release)),
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

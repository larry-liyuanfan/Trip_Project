"""Verify local four-layer handoff packages without cloud dependencies."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import BinaryIO


SCHEMA_VERSION = "local_release_v1"
LAYER_NAMES = {"runtime", "adapter", "retrieval", "evidence"}


class ReleaseVerificationError(ValueError):
    """Raised when a local release differs from its manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release_dir(release_dir: Path) -> dict:
    release_dir = Path(release_dir)
    manifest_path = release_dir / "release_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError("release_manifest.json is missing or invalid") from exc
    identity = (
        (manifest.get("schema_version") == SCHEMA_VERSION and manifest.get("distribution") == "local_handoff")
        or (
            manifest.get("schema_version") == "private_oss_release_v1"
            and manifest.get("visibility") == "private"
        )
    )
    if not identity or set(manifest.get("layers", {})) != LAYER_NAMES:
        raise ReleaseVerificationError("release manifest identity is invalid")
    release = manifest.get("release", {})
    if (
        not release.get("release_id")
        or not release.get("config_member")
        or len(str(release.get("config_sha256", ""))) != 64
        or len(str(release.get("adapter_model_sha256", ""))) != 64
    ):
        raise ReleaseVerificationError("release model identity is invalid")
    for name, expected in manifest["layers"].items():
        path = release_dir / str(expected.get("file", ""))
        if not path.is_file():
            raise ReleaseVerificationError(f"release layer is missing: {name}")
        if path.stat().st_size != expected.get("size_bytes"):
            raise ReleaseVerificationError(f"release layer size mismatch: {name}")
        if sha256_file(path) != expected.get("sha256"):
            raise ReleaseVerificationError(f"release layer SHA-256 mismatch: {name}")
    _verify_embedded_release(release_dir, manifest)
    return manifest


def _verify_embedded_release(release_dir: Path, manifest: dict) -> None:
    release = manifest["release"]
    try:
        with tarfile.open(release_dir / manifest["layers"]["runtime"]["file"], "r:gz") as archive:
            config_handle = archive.extractfile(release["config_member"])
            if config_handle is None:
                raise ReleaseVerificationError("embedded release config is missing")
            config_bytes = config_handle.read()
        config = json.loads(config_bytes.decode("utf-8"))
        if hashlib.sha256(config_bytes).hexdigest() != release["config_sha256"]:
            raise ReleaseVerificationError("embedded release config SHA-256 mismatch")
        if (
            config.get("release_id") != release["release_id"]
            or config.get("model", {}).get("adapter_model_sha256")
            != release["adapter_model_sha256"]
        ):
            raise ReleaseVerificationError("embedded release identity mismatch")
        with tarfile.open(release_dir / manifest["layers"]["adapter"]["file"], "r:gz") as archive:
            adapter_handle = archive.extractfile("adapter/adapter_model.safetensors")
            if adapter_handle is None:
                raise ReleaseVerificationError("packaged adapter model is missing")
            adapter_sha = _sha256_handle(adapter_handle)
        if adapter_sha != release["adapter_model_sha256"]:
            raise ReleaseVerificationError("packaged adapter SHA-256 mismatch")
    except (OSError, KeyError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError("release archive identity is invalid") from exc


def _sha256_handle(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()

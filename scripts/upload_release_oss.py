"""Upload and download-verify a release directory at a private OSS prefix."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


class ReleaseVerificationError(ValueError):
    """Raised when a local or downloaded release differs from its manifest."""


def verify_release_dir(release_dir: Path) -> dict:
    manifest_path = Path(release_dir) / "release_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError("release_manifest.json is missing or invalid") from exc
    if (
        manifest.get("schema_version") != "private_oss_release_v1"
        or manifest.get("visibility") != "private"
        or set(manifest.get("layers", {}))
        != {"runtime", "adapter", "retrieval", "evidence"}
    ):
        raise ReleaseVerificationError("release manifest identity is invalid")
    for name, expected in manifest["layers"].items():
        path = Path(release_dir) / str(expected.get("file", ""))
        if not path.is_file():
            raise ReleaseVerificationError(f"release layer is missing: {name}")
        if path.stat().st_size != expected.get("size_bytes"):
            raise ReleaseVerificationError(f"release layer size mismatch: {name}")
        if _sha256(path) != expected.get("sha256"):
            raise ReleaseVerificationError(f"release layer SHA-256 mismatch: {name}")
    return manifest


def upload_and_verify(release_dir: Path, destination: str) -> dict:
    release_dir = Path(release_dir).resolve()
    manifest = verify_release_dir(release_dir)
    prefix = destination.rstrip("/")
    files = [
        *(release_dir / layer["file"] for layer in manifest["layers"].values()),
        release_dir / "release_manifest.json",
    ]
    for path in files:
        subprocess.run(
            [
                "ossutil",
                "cp",
                "--force",
                "--acl",
                "private",
                str(path),
                f"{prefix}/{path.name}",
            ],
            check=True,
        )
    with tempfile.TemporaryDirectory(prefix="trip-release-verify-") as tmpdir:
        downloaded = Path(tmpdir)
        for path in files:
            subprocess.run(
                [
                    "ossutil",
                    "cp",
                    "--force",
                    f"{prefix}/{path.name}",
                    str(downloaded / path.name),
                ],
                check=True,
            )
        downloaded_manifest = verify_release_dir(downloaded)
    return {
        "status": "verified",
        "destination": prefix,
        "layers": downloaded_manifest["layers"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("destination", help="private oss://bucket/prefix")
    args = parser.parse_args()
    if not args.destination.startswith("oss://"):
        raise SystemExit("destination must be an oss:// URI")
    result = upload_and_verify(args.release_dir, args.destination)
    print(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()

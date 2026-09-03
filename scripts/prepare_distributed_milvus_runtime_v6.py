#!/usr/bin/env python3
"""Extract and configure one job-local Milvus 2.6 distributed runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


DEFAULT_MINIO_CREDENTIAL = "minio" + "admin"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-node", required=True)
    parser.add_argument("--port-base", type=int, required=True)
    parser.add_argument("--expected-rpm-sha256", required=True)
    args = parser.parse_args()
    access_key = os.environ.get("TRIP_MINIO_ACCESS_KEY", "")
    secret_key = os.environ.get("TRIP_MINIO_SECRET_KEY", "")
    prepare_runtime(
        rpm=args.rpm,
        output_dir=args.output_dir,
        control_node=args.control_node,
        port_base=args.port_base,
        expected_rpm_sha256=args.expected_rpm_sha256,
        access_key=access_key,
        secret_key=secret_key,
    )


def prepare_runtime(
    *,
    rpm: Path,
    output_dir: Path,
    control_node: str,
    port_base: int,
    expected_rpm_sha256: str,
    access_key: str,
    secret_key: str,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"distributed Milvus runtime already exists: {output_dir}")
    if hashlib.sha256(rpm.read_bytes()).hexdigest() != expected_rpm_sha256:
        raise ValueError("Milvus RPM SHA-256 differs from the fixed configuration")
    if not control_node or any(character.isspace() for character in control_node):
        raise ValueError("control node must be a non-empty hostname")
    if not 20000 <= port_base <= 50000 or port_base + 20 > 65535:
        raise ValueError("port base is outside the permitted job-local range")
    if len(access_key) < 16 or len(secret_key) < 32:
        raise ValueError("job-local MinIO credentials do not meet the minimum length")
    if access_key.lower() == DEFAULT_MINIO_CREDENTIAL or secret_key.lower() == DEFAULT_MINIO_CREDENTIAL:
        raise ValueError("default MinIO credentials are forbidden")

    output_dir.mkdir(parents=True)
    rpm_process = subprocess.Popen(["rpm2cpio", str(rpm)], stdout=subprocess.PIPE)
    if rpm_process.stdout is None:
        raise RuntimeError("rpm2cpio did not expose stdout")
    unpack = subprocess.run(
        ["cpio", "-idm", "--quiet"],
        cwd=output_dir,
        stdin=rpm_process.stdout,
        check=False,
    )
    rpm_process.stdout.close()
    rpm_code = rpm_process.wait()
    if rpm_code or unpack.returncode:
        raise RuntimeError(f"Milvus RPM extraction failed: rpm2cpio={rpm_code}, cpio={unpack.returncode}")

    config_path = output_dir / "etc" / "milvus" / "configs" / "milvus.yaml"
    text = config_path.read_text(encoding="utf-8")
    text = configure_milvus_text(
        text,
        control_node=control_node,
        port_base=port_base,
        access_key=access_key,
        secret_key=secret_key,
        output_dir=output_dir,
    )
    config_path.write_text(text, encoding="utf-8")
    config_path.chmod(0o600)
    identity = {
        "schema_version": "distributed_milvus_runtime_v6",
        "control_node": control_node,
        "port_base": port_base,
        "rpm_sha256": expected_rpm_sha256,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "credentials_written_only_to_job_local_config": True,
    }
    print(json.dumps(identity, indent=2, sort_keys=True))
    return identity


def configure_milvus_text(
    text: str,
    *,
    control_node: str,
    port_base: int,
    access_key: str,
    secret_key: str,
    output_dir: Path,
) -> str:
    local_data = output_dir / "data"
    replacements = {
        "localhost:2379": f"{control_node}:{port_base}",
        "    embed: true # Whether to enable embedded Etcd (an in-process EtcdServer).": (
            "    embed: false # Distributed v6 uses the external job-local etcd service."
        ),
        "  address: localhost:9000": f"  address: {control_node}:{port_base + 1}",
        "  port: 9000 # Port of MinIO or S3 service.": (
            f"  port: {port_base + 1} # Port of MinIO or S3 service."
        ),
        f"  accessKeyID: {DEFAULT_MINIO_CREDENTIAL}": f"  accessKeyID: {access_key}",
        f"  secretAccessKey: {DEFAULT_MINIO_CREDENTIAL}": f"  secretAccessKey: {secret_key}",
        "  type: default\n": "  type: woodpecker\n",
        "  storageType: local # please adjust in embedded Milvus: local, available values are [local, remote, opendal], value minio is deprecated, use remote instead": (
            "  storageType: remote # distributed v6 job-local MinIO"
        ),
        "/var/lib/milvus/data/": f"{local_data}/",
        "/var/lib/milvus/rdb_data": str(output_dir / "rdb_data"),
        "/tmp/milvus_access": str(output_dir / "access"),
        "  port: 22125 # TCP port of rootCoord": f"  port: {port_base + 3} # TCP port of rootCoord",
        "  port: 19530 # TCP port of proxy": f"  port: {port_base + 4} # TCP port of proxy",
        "  internalPort: 19529": f"  internalPort: {port_base + 5}",
        "  port: 19531 # TCP port of queryCoord": f"  port: {port_base + 6} # TCP port of queryCoord",
        "  port: 21123 # TCP port of queryNode": f"  port: {port_base + 7} # TCP port of queryNode",
        "  port: 13333 # TCP port of dataCoord": f"  port: {port_base + 8} # TCP port of dataCoord",
        "  port: 21124 # TCP port of dataNode": f"  port: {port_base + 9} # TCP port of dataNode",
        "  port: 22222 # TCP port of streamingNode": f"  port: {port_base + 10} # TCP port of streamingNode",
        "  minSegmentSizeToEnableIndex: 1024": "  minSegmentSizeToEnableIndex: 0",
        "    enabled: true # Whether to enable the http server": "    enabled: false # Whether to enable the http server",
    }
    for old, new in replacements.items():
        count = text.count(old)
        if count != 1:
            raise ValueError(f"expected one Milvus config token, found {count}: {old!r}")
        text = text.replace(old, new, 1)
    return text


if __name__ == "__main__":
    main()

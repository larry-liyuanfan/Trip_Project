"""Single operator CLI for validating, serving, and smoking the packaged system."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "configs/releases/qwen3_vl_system_v1.json"


def doctor() -> dict[str, Any]:
    """Inspect required tools and local artifacts without starting services."""
    adapter = Path(os.getenv("TRIP_ADAPTER_DIR", "")) if os.getenv("TRIP_ADAPTER_DIR") else None
    checks = {
        "python": {"ok": sys.version_info >= (3, 10), "detail": sys.version.split()[0]},
        "docker_cli": {"ok": shutil.which("docker") is not None},
        "nvidia_smi": {"ok": shutil.which("nvidia-smi") is not None},
        "release_config": {"ok": DEFAULT_RELEASE.is_file(), "detail": str(DEFAULT_RELEASE)},
        "adapter_dir": {
            "ok": bool(adapter and (adapter / "adapter_model.safetensors").is_file()),
            "detail": str(adapter) if adapter else "TRIP_ADAPTER_DIR is unset",
        },
    }
    return {"status": "ok" if all(item["ok"] for item in checks.values()) else "not_ready", "checks": checks}


def validate() -> dict[str, Any]:
    """Validate tracked release and Compose configuration without model loading."""
    errors = []
    try:
        payload = json.loads(DEFAULT_RELEASE.read_text(encoding="utf-8"))
        if payload.get("model", {}).get("base_model") != "Qwen/Qwen3-VL-8B-Instruct":
            errors.append("release base_model is not Qwen3-VL-8B-Instruct")
        if payload.get("quality", {}).get("dialogue") != "DIALOGUE_BETA":
            errors.append("release dialogue tier is not DIALOGUE_BETA")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"release config invalid: {exc}")
    compose = ROOT / "docker/system/docker-compose.yml"
    if not compose.is_file():
        errors.append("system Compose is missing")
    return {"status": "ok" if not errors else "failed", "errors": errors}


def serve() -> int:
    """Replace the current process with the documented API server."""
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.app:app",
            "--host",
            os.getenv("API_BIND_ADDRESS", "127.0.0.1"),
            "--port",
            os.getenv("API_PORT", "8000"),
        ],
        cwd=ROOT,
    )


def smoke(base_url: str) -> dict[str, Any]:
    """Check both liveness and strict readiness of a running release."""
    results = {}
    for endpoint in ("health", "ready"):
        try:
            response = requests.get(f"{base_url.rstrip('/')}/{endpoint}", timeout=15)
            results[endpoint] = {
                "ok": response.ok,
                "status_code": response.status_code,
                "body": response.json(),
            }
        except Exception as exc:
            results[endpoint] = {"ok": False, "error": str(exc)}
    return {
        "status": "ok" if all(item["ok"] for item in results.values()) else "failed",
        "checks": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    subparsers.add_parser("validate")
    subparsers.add_parser("serve")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.command == "serve":
        raise SystemExit(serve())
    result = smoke(args.base_url) if args.command == "smoke" else globals()[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()

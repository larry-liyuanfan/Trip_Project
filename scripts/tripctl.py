"""Single operator CLI for validating, serving, and smoking the packaged system."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference.release_config import resolve_release_config
from src.inference.system_runtime import ReleaseSettings, RuntimeConfigurationError
DEFAULT_RELEASE = ROOT / "configs/releases/qwen3_vl_system_v1.json"
DEFAULT_SMOKE_IMAGE = ROOT / "data/samples/images/cafe_001.jpg"


def doctor() -> dict[str, Any]:
    """Inspect required tools and local artifacts without starting services."""
    adapter = Path(os.getenv("TRIP_ADAPTER_DIR", "")) if os.getenv("TRIP_ADAPTER_DIR") else None
    try:
        selected = resolve_release_config(ROOT)
        release_check = {"ok": True, "detail": str(selected)}
    except ValueError as exc:
        release_check = {"ok": False, "detail": str(exc)}
    checks = {
        "python": {"ok": sys.version_info >= (3, 10), "detail": sys.version.split()[0]},
        "docker_cli": {"ok": shutil.which("docker") is not None},
        "nvidia_smi": {"ok": shutil.which("nvidia-smi") is not None},
        "release_config": release_check,
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
        selected = resolve_release_config(ROOT)
        ReleaseSettings.load(root=ROOT, config_path=selected)
        payload = json.loads(selected.read_text(encoding="utf-8"))
        if payload.get("model", {}).get("base_model") != "Qwen/Qwen3-VL-8B-Instruct":
            errors.append("release base_model is not Qwen3-VL-8B-Instruct")
        if payload.get("quality", {}).get("dialogue") != "DIALOGUE_BETA":
            errors.append("release dialogue tier is not DIALOGUE_BETA")
    except (OSError, ValueError, RuntimeConfigurationError) as exc:
        errors.append(f"release config invalid: {exc}")
    compose = ROOT / "docker/system/docker-compose.yml"
    if not compose.is_file():
        errors.append("system Compose is missing")
    return {"status": "ok" if not errors else "failed", "errors": errors,
            "release_config_sha256": hashlib.sha256(selected.read_bytes()).hexdigest() if "selected" in locals() and selected.is_file() else None,
            "release_config": str(selected) if "selected" in locals() else None}


def compose(arguments: list[str]) -> int:
    """将同一配置绝对路径交给 Compose，消除相对路径基准差异。"""
    selected = resolve_release_config(ROOT)
    ReleaseSettings.load(root=ROOT, config_path=selected)
    environment = dict(os.environ, TRIP_RELEASE_CONFIG=str(selected))
    return subprocess.call(["docker", "compose", "-f", str(ROOT / "docker/system/docker-compose.yml"), *arguments],
                           cwd=ROOT, env=environment)


def serve() -> int:
    """Replace the current process with the documented API server."""
    selected = resolve_release_config(ROOT)
    ReleaseSettings.load(root=ROOT, config_path=selected)
    os.environ["TRIP_RELEASE_CONFIG"] = str(selected)
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


def smoke(base_url: str, image_path: Path = DEFAULT_SMOKE_IMAGE) -> dict[str, Any]:
    """Exercise readiness, four model scenarios, and production retrieval."""
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        return {
            "status": "failed",
            "checks": {"smoke_image": {"ok": False, "detail": str(image_path)}},
        }
    base_url = base_url.rstrip("/")
    results = {}
    for endpoint in ("health", "ready"):
        try:
            response = requests.get(f"{base_url}/{endpoint}", timeout=30)
            results[endpoint] = {
                "ok": response.ok,
                "status_code": response.status_code,
                "body": response.json(),
            }
        except Exception as exc:
            results[endpoint] = {"ok": False, "error": str(exc)}
    task_specs = {
        "image_product_search": (
            "/v1/tasks/image-product-search",
            None,
        ),
        "after_sales": (
            "/v1/tasks/after-sales",
            None,
        ),
        "itinerary_planning": (
            "/v1/tasks/itinerary-planning",
            "规划上海两日行程，预算适中，偏好安静的文化体验。",
        ),
    }
    for scenario, (endpoint, text_context) in task_specs.items():
        _post_smoke(
            results,
            scenario,
            f"{base_url}{endpoint}",
            {"image_urls": [str(image_path)], "text_context": text_context},
            lambda body, expected=scenario: (
                body.get("scenario") == expected and body.get("schema_valid") is True
                and (expected != "itinerary_planning" or body.get("business_valid") is True)
            ),
        )
    _post_smoke(
        results,
        "dialogue",
        f"{base_url}/v1/dialogue",
        {
            "messages": [{"role": "user", "content": "参考这张图，推荐安静的两日行程。"}],
            "image_urls": [str(image_path)],
            "state": {"city": "Shanghai", "days": 2},
        },
        lambda body: body.get("quality_tier") == "DIALOGUE_BETA" and body.get("task_status") == "COMPLETED",
    )
    _post_smoke(
        results,
        "visual_search",
        f"{base_url}/v1/visual-search",
        {
            "image_urls": [str(image_path)],
            "query_text": "安静的咖啡馆",
            "top_k": 3,
            "retrieval_mode": "embedding",
        },
        lambda body: (
            body.get("retrieval_mode") == "clip_milvus_hnsw_cosine"
            and isinstance(body.get("results"), list)
        ),
    )
    return {
        "status": "ok" if all(item["ok"] for item in results.values()) else "failed",
        "checks": results,
    }


def _post_smoke(
    results: dict[str, Any],
    name: str,
    url: str,
    payload: dict[str, Any],
    validator,
) -> None:
    try:
        response = requests.post(url, json=payload, timeout=180)
        body = response.json()
        results[name] = {
            "ok": bool(response.ok and validator(body)),
            "status_code": response.status_code,
            "body": body,
        }
    except Exception as exc:
        results[name] = {"ok": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-config", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    subparsers.add_parser("validate")
    subparsers.add_parser("serve")
    compose_parser = subparsers.add_parser("compose")
    compose_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    smoke_parser.add_argument("--image-path", type=Path, default=DEFAULT_SMOKE_IMAGE)
    args = parser.parse_args()
    if args.release_config is not None:
        os.environ["TRIP_RELEASE_CONFIG"] = str(args.release_config)
    if args.command == "serve":
        raise SystemExit(serve())
    if args.command == "compose":
        raise SystemExit(compose(args.arguments))
    result = (
        smoke(args.base_url, args.image_path)
        if args.command == "smoke"
        else globals()[args.command]()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()

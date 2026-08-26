#!/usr/bin/env python3
"""Run immutable Week 8 dialogue-routing and product-latency benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.week8_runtime_optimization import (
    load_runtime_benchmark_config,
    run_dialogue_first_turn_comparison,
    run_product_latency_benchmark,
)
from src.inference.system_runtime import ReleaseSettings, TransformersPeftBackend


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(backend: TransformersPeftBackend) -> dict[str, Any]:
    torch_module = backend._torch
    cuda_available = bool(torch_module and torch_module.cuda.is_available())
    return {
        "cuda_available": cuda_available,
        "cuda_device": (
            torch_module.cuda.get_device_name(0) if cuda_available else None
        ),
        "torch_version": getattr(torch_module, "__version__", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument(
        "--release-config",
        default=ROOT / "configs/releases/qwen3_vl_system_week8_v1.json",
        type=Path,
    )
    parser.add_argument(
        "--benchmark-config",
        default=ROOT / "configs/week8/runtime_optimization_v1.json",
        type=Path,
    )
    parser.add_argument(
        "--product-image",
        type=Path,
        help="optional untracked fixed image override for a real-size benchmark",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite benchmark evidence: {output}")
    os.environ["TRIP_ADAPTER_DIR"] = str(args.adapter_dir.resolve())
    settings = ReleaseSettings.load(
        root=ROOT,
        config_path=args.release_config.resolve(),
    )
    config = load_runtime_benchmark_config(ROOT, args.benchmark_config.resolve())
    sections = config.get(
        "benchmark_sections",
        ["dialogue", "product_latency"],
    )
    fixed_image = (
        args.product_image.resolve()
        if args.product_image is not None
        else (ROOT / config["product_latency"]["image"]).resolve()
    )
    if not fixed_image.is_file():
        raise SystemExit(f"fixed product image is missing: {fixed_image}")
    if args.product_image is not None:
        config["product_latency"]["image"] = str(fixed_image)
    backend = TransformersPeftBackend(settings)
    started = time.perf_counter()
    ready, reason = backend.ready()
    cold_start_ms = (time.perf_counter() - started) * 1000
    if not ready:
        raise SystemExit(f"runtime backend is not ready: {reason}")

    result: dict[str, Any] = {
        "schema_version": "week8_runtime_optimization_evidence_v1",
        "status": "COMPLETED",
        "benchmark_sections": sections,
        "cold_start_ms": cold_start_ms,
        "environment": _environment(backend),
        "identity": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip(),
            "release_id": settings.release_id,
            "base_model": settings.base_model,
            "base_revision": settings.base_revision,
            "adapter_name": settings.adapter_name,
            "adapter_model_sha256": _sha256(
                args.adapter_dir / "adapter_model.safetensors"
            ),
            "release_config_sha256": _sha256(args.release_config),
            "benchmark_config_sha256": _sha256(args.benchmark_config),
            "fixed_image_sha256": _sha256(
                fixed_image
            ),
            "fixed_image_override": args.product_image is not None,
        },
    }
    if "dialogue" in sections:
        result["dialogue"] = run_dialogue_first_turn_comparison(
            settings,
            backend,
            config,
        )
    if "product_latency" in sections:
        result["product_latency"] = run_product_latency_benchmark(
            settings,
            backend,
            config,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

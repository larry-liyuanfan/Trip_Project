#!/usr/bin/env python3
"""Run production-model smoke; the default image is only a transport fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.schemas import DialogueRequest, TaskRequest
from src.evaluation.schema_validation import validate_output
from src.inference.transport_utils import strip_json_fence
from src.inference.system_runtime import (
    ModelGenerationError,
    ReleaseSettings,
    ScenarioService,
    TransformersPeftBackend,
)


def run_model_smoke(service: ScenarioService, image_path: Path) -> dict[str, Any]:
    """Exercise the same service object used by the production API."""

    image = str(Path(image_path).resolve())
    tasks = {
        "image_product_search": None,
        "after_sales": None,
        "itinerary_planning": "规划上海两日行程，预算适中，偏好安静的文化体验。",
    }
    results = {}
    for scenario, text_context in tasks.items():
        try:
            results[scenario] = service.run_task(scenario, TaskRequest(image_urls=[image], text_context=text_context)).model_dump()
        except ModelGenerationError as exc:
            schema_valid = False
            if exc.attempts:
                try:
                    payload = json.loads(strip_json_fence(exc.attempts[-1].raw_output))
                    validate_output(service.settings.root, scenario, payload, service.settings.schema_versions[scenario])
                    schema_valid = True
                except (ValueError, TypeError):
                    pass
            results[scenario] = {"schema_valid": schema_valid, "business_valid": False,
                                 "error": str(exc), "attempts": [item.model_dump() for item in exc.attempts]}
    try:
        dialogue = service.run_dialogue(DialogueRequest(
            messages=[
                {
                    "role": "user",
                    "content": "参考这张图，推荐安静的两日行程。",
                }
            ],
            image_urls=[image],
            state={"city": "Shanghai", "days": 2},
        )).model_dump()
    except ModelGenerationError as exc:
        dialogue = {"task_status": "NOT_COMPLETED", "error": str(exc),
                    "attempts": [item.model_dump() for item in exc.attempts]}
    technical = all(item.get("schema_valid") is True for item in results.values())
    technical = technical and dialogue.get("quality_tier") == "DIALOGUE_BETA"
    passed = technical and results["itinerary_planning"].get("business_valid") is True and dialogue.get("task_status") == "COMPLETED"
    return {
        "status": "PASS" if passed else "FAIL",
        "technical_status": "PASS" if technical else "FAIL",
        "business_status": "PASS" if passed else "FAIL",
        "product_visual_accuracy": "NOT_ASSESSED_BY_SMOKE",
        "scenarios": results,
        "dialogue": dialogue,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument(
        "--release-config",
        default=ROOT / "configs/releases/qwen3_vl_system_v1.json",
        type=Path,
    )
    parser.add_argument(
        "--image",
        default=ROOT / "data/samples/images/cafe_001.jpg",
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-image-sha256", help="optional immutable real-photo identity")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite model smoke evidence: {output}")
    if args.expected_image_sha256 and _sha256(args.image) != args.expected_image_sha256:
        raise SystemExit("smoke image SHA-256 mismatch")
    from PIL import Image
    with Image.open(args.image) as image_source:
        image_dimensions = list(image_source.size)
    os.environ["TRIP_ADAPTER_DIR"] = str(args.adapter_dir.resolve())
    settings = ReleaseSettings.load(
        root=ROOT,
        config_path=args.release_config.resolve(),
    )
    service = ScenarioService(settings, TransformersPeftBackend(settings))
    try:
        result = run_model_smoke(service, args.image)
    except ModelGenerationError as exc:
        result = {
            "status": "FAIL",
            "error": str(exc),
            "failed_attempts": [item.model_dump() for item in exc.attempts],
        }
    result["evidence"] = {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip(),
        "release_config": str(args.release_config.resolve()),
        "release_config_sha256": _sha256(args.release_config),
        "adapter_model_sha256": _sha256(
            args.adapter_dir / "adapter_model.safetensors"
        ),
        "sample_image_sha256": _sha256(args.image),
        "sample_image_dimensions": image_dimensions,
        "sample_image_identity_pinned": bool(args.expected_image_sha256),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

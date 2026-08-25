#!/usr/bin/env python3
"""Run four real production-model scenarios without consuming evaluation data."""

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
    results = {
        scenario: service.run_task(
            scenario,
            TaskRequest(image_urls=[image], text_context=text_context),
        ).model_dump()
        for scenario, text_context in tasks.items()
    }
    dialogue = service.run_dialogue(
        DialogueRequest(
            messages=[
                {
                    "role": "user",
                    "content": "参考这张图，推荐安静的两日行程。",
                }
            ],
            image_urls=[image],
            state={"city": "Shanghai", "days": 2},
        )
    ).model_dump()
    passed = all(item.get("schema_valid") is True for item in results.values())
    passed = passed and dialogue.get("quality_tier") == "DIALOGUE_BETA"
    return {
        "status": "PASS" if passed else "FAIL",
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
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite model smoke evidence: {output}")
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

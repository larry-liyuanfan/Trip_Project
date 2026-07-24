"""Capture immutable, local evidence for a Week 3 /v1/models readiness probe."""

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.config import load_evaluation_config
from src.evaluation.readiness import parse_models_payload


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week3.yaml")
    parser.add_argument("--evidence-id", required=True)
    args = parser.parse_args()
    config = load_evaluation_config(args.config)
    base_url = config["runtime"]["live_base_url"].rstrip("/")
    endpoint = base_url + "/v1/models"
    timeout = config["runtime"]["timeout_seconds"]
    with urllib.request.urlopen(endpoint, timeout=timeout) as response:
        status_code = response.status
        payload = json.loads(response.read().decode("utf-8"), parse_constant=_reject_constant)
    if status_code != 200:
        raise RuntimeError(f"readiness probe returned HTTP {status_code}")
    model_ids = parse_models_payload(payload)
    root = Path.cwd()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    output_dir = root / config["paths"]["readiness_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.evidence_id}.json"
    record = {
        "evidence_id": args.evidence_id,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "command": (
            "python scripts/capture_week3_readiness.py "
            f"--config {args.config} --evidence-id {args.evidence_id}"
        ),
        "status_code": status_code,
        "model_ids": model_ids,
        "git_head": git_head,
        "git_worktree_dirty": bool(git_status.strip()),
        "scope": "readiness_only_no_week3_image_request",
    }
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "captured", "path": str(output), **record}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

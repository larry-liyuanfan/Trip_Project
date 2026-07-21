"""Safely refresh Week 3 after-sales business-synthetic evidence cards."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.config import load_evaluation_config
from src.evaluation.synthetic_evidence_refresh import (
    execute_synthetic_evidence_refresh,
    plan_synthetic_evidence_refresh,
)


def run_refresh(
    *,
    config: dict[str, Any],
    root: Path,
    run_id: str,
    check_only: bool,
) -> dict[str, Any]:
    """Plan one refresh and optionally apply its transactional replacements."""
    plan = plan_synthetic_evidence_refresh(
        root=root,
        config=config,
        run_id=run_id,
    )
    if check_only:
        return {
            "status": "ready",
            "run_id": plan.run_id,
            "recipe_version": plan.recipe_version,
            "target_count": len(plan.images),
        }
    return execute_synthetic_evidence_refresh(
        plan,
        root=root,
        config=config,
    )


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week3.yaml")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = load_evaluation_config(args.config)
    result = run_refresh(
        config=config,
        root=Path.cwd(),
        run_id=args.run_id or _default_run_id(),
        check_only=args.check,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

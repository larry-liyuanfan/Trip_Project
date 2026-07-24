"""Replace pending Week 3 v2 after-sales candidates with reviewed photo assets."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.config import load_evaluation_config
from src.evaluation.v2_photorealistic import (
    apply_after_sales_photo_replacement,
    plan_after_sales_photo_replacement,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week3_v2.yaml")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    config = load_evaluation_config(args.config)
    plan = plan_after_sales_photo_replacement(
        root=root,
        config=config,
        image_dir=args.image_dir,
    )
    if args.apply:
        result = apply_after_sales_photo_replacement(
            plan,
            root=root,
            config=config,
            run_id=args.run_id,
        )
    else:
        result = {
            "status": "planned",
            "replacement_count": len(plan.mappings),
            "reserve_paths": list(plan.reserve_paths),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

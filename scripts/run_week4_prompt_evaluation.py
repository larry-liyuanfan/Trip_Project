"""Run a fixed Week 4 prompt pilot or the per-scenario full winners."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.week4_runner import run_week4_prompt_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week4.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--variant", choices=("standardized_v2", "fewshot_4_v2", "fewshot_7_v2"))
    parser.add_argument("--product-variant")
    parser.add_argument("--after-sales-variant")
    parser.add_argument("--itinerary-variant")
    args = parser.parse_args()
    if args.stage == "pilot":
        if args.variant is None:
            parser.error("--variant is required for pilot")
        variants = {
            "image_product_search": args.variant,
            "after_sales": args.variant,
            "itinerary_planning": args.variant,
        }
    else:
        values = (
            args.product_variant,
            args.after_sales_variant,
            args.itinerary_variant,
        )
        if any(value is None for value in values):
            parser.error("all three per-scenario variants are required for full")
        variants = {
            "image_product_search": args.product_variant,
            "after_sales": args.after_sales_variant,
            "itinerary_planning": args.itinerary_variant,
        }
    summary = run_week4_prompt_evaluation(
        root=Path.cwd(),
        config_path=Path(args.config),
        run_id=args.run_id,
        stage=args.stage,
        variants_by_scenario=variants,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

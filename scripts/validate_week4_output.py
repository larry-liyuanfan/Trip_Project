"""Apply the bounded Week 4 JSON/Schema fallback to one raw output."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.format_fallback import parse_with_schema_fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        required=True,
        choices=("image_product_search", "after_sales", "itinerary_planning"),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-output")
    source.add_argument("--raw-output-file")
    args = parser.parse_args()
    raw_output = args.raw_output
    if args.raw_output_file:
        raw_output = Path(args.raw_output_file).read_text(encoding="utf-8")
    result = parse_with_schema_fallback(Path.cwd(), args.scenario, raw_output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

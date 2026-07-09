import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.image_validation import validate_photo_images
from src.data.jsonl_utils import iter_jsonl, limit_records, write_json, write_table
from src.data.parse_business import parse_business_records
from src.data.parse_photos import parse_photo_records
from src.data.parse_reviews import parse_review_records
from src.data.yelp_paths import create_output_directories, load_config, resolve_pipeline_paths


def run_parse(config: dict[str, Any]) -> dict[str, Any]:
    create_output_directories(config)
    paths = resolve_pipeline_paths(config)
    output_format = config.get("output", {}).get("format", "parquet")
    limits = config.get("processing_limits", {})
    interim = paths["interim_dir"]

    business_records, business_json_summary = iter_jsonl(paths["business_json"])
    businesses = parse_business_records(limit_records(business_records, _optional_int(limits.get("max_businesses"))))

    review_records, review_json_summary = iter_jsonl(paths["review_json"])
    reviews, review_stats, review_filter_summary = parse_review_records(
        limit_records(review_records, _optional_int(limits.get("max_reviews"))),
        min_text_length=int(config.get("review_filters", {}).get("min_text_length", 20)),
        reject_symbol_only=bool(config.get("review_filters", {}).get("reject_symbol_only", True)),
    )

    photo_records, photo_json_summary = iter_jsonl(paths["photo_json"])
    photos = parse_photo_records(limit_records(photo_records, _optional_int(limits.get("max_photos"))), image_root=paths["image_root"])
    image_index, image_summary = validate_photo_images(photos)

    output_summaries = [
        write_table(interim / f"business.{_extension(output_format)}", businesses, output_format),
        write_table(interim / f"reviews.{_extension(output_format)}", reviews, output_format),
        write_table(interim / f"photos.{_extension(output_format)}", photos, output_format),
        write_table(interim / f"photo_image_index.{_extension(output_format)}", image_index, output_format),
        write_table(interim / f"review_business_stats.{_extension(output_format)}", review_stats, output_format),
    ]
    summary = {
        "business_count": len(businesses),
        "review_count": len(reviews),
        "photo_count": len(photos),
        "valid_image_count": image_summary["valid_images"],
        "jsonl": {
            "business": business_json_summary,
            "review": review_json_summary,
            "photo": photo_json_summary,
        },
        "review_filters": review_filter_summary,
        "image_validation": image_summary,
        "outputs": output_summaries,
    }
    write_json(interim / "validation_summary.json", summary)
    write_json(paths["validation_dir"] / "validation_summary.json", summary)
    return summary


def _optional_int(value: Any) -> int | None:
    if value in {None, "null", "None", ""}:
        return None
    return int(value)


def _extension(output_format: str) -> str:
    return "csv" if output_format.lower() == "csv" else "parquet"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse Yelp JSONL files into interim tabular outputs.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_processing.yaml"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_parse(load_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

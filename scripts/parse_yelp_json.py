"""Stream Yelp JSONL sources into validated interim tables and summaries."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.image_validation import iter_validated_photo_images
from src.data.jsonl_utils import TableStreamWriter, iter_jsonl, limit_records, write_json, write_table
from src.data.parse_business import stream_business_records
from src.data.parse_photos import stream_photo_records
from src.data.parse_reviews import stream_review_records
from src.data.yelp_paths import create_output_directories, load_config, resolve_pipeline_paths


def run_parse(config: dict[str, Any]) -> dict[str, Any]:
    """Parse all configured source records with bounded memory and image workers."""
    create_output_directories(config)
    paths = resolve_pipeline_paths(config)
    output_format = config.get("output", {}).get("format", "parquet")
    limits = config.get("processing_limits", {})
    interim = paths["interim_dir"]
    chunk_size = int(config.get("output", {}).get("chunk_size", 50000))
    image_validation_config = config.get("image_validation", {})
    image_workers = int(image_validation_config.get("workers", 8))
    image_batch_size = int(image_validation_config.get("batch_size", 512))

    business_writer = TableStreamWriter(
        interim / f"business.{_extension(output_format)}",
        output_format=output_format,
        chunk_size=chunk_size,
    )
    business_records, business_json_summary = iter_jsonl(paths["business_json"])
    business_parse_summary = stream_business_records(
        limit_records(business_records, _optional_int(limits.get("max_businesses"))),
        row_sink=business_writer.write,
    )
    business_output_summary = business_writer.close()

    review_writer = TableStreamWriter(
        interim / f"reviews.{_extension(output_format)}",
        output_format=output_format,
        fieldnames=["review_id", "business_id", "user_id", "stars", "useful", "funny", "cool", "text", "date"],
        chunk_size=chunk_size,
    )
    review_records, review_json_summary = iter_jsonl(paths["review_json"])
    review_stats, review_filter_summary = stream_review_records(
        limit_records(review_records, _optional_int(limits.get("max_reviews"))),
        row_sink=review_writer.write,
        min_text_length=int(config.get("review_filters", {}).get("min_text_length", 20)),
        reject_symbol_only=bool(config.get("review_filters", {}).get("reject_symbol_only", True)),
    )
    review_output_summary = review_writer.close()

    photo_writer = TableStreamWriter(
        interim / f"photos.{_extension(output_format)}",
        output_format=output_format,
        fieldnames=["photo_id", "business_id", "caption", "label", "image_path"],
        chunk_size=chunk_size,
    )
    image_index_writer = TableStreamWriter(
        interim / f"photo_image_index.{_extension(output_format)}",
        output_format=output_format,
        fieldnames=["photo_id", "business_id", "image_path", "image_valid", "image_width", "image_height", "validation_error"],
        chunk_size=chunk_size,
        parquet_schema=_image_index_parquet_schema(output_format),
    )
    image_summary = {"total_images": 0, "valid_images": 0, "missing_images": 0, "corrupted_images": 0}

    photo_validation_batch: list[dict[str, Any]] = []

    def flush_photo_validation_batch() -> None:
        """Validate and persist the current bounded photo batch, then release it."""
        if not photo_validation_batch:
            return
        for image_index, image_status in iter_validated_photo_images(
            photo_validation_batch,
            workers=image_workers,
        ):
            image_index_writer.write(image_index)
            image_summary["total_images"] += 1
            if image_status == "valid":
                image_summary["valid_images"] += 1
            elif image_status == "missing":
                image_summary["missing_images"] += 1
            else:
                image_summary["corrupted_images"] += 1
        photo_validation_batch.clear()

    def emit_photo_and_validation(photo: dict[str, Any]) -> None:
        """Write photo metadata and queue the same row for local image validation."""
        photo_writer.write(photo)
        photo_validation_batch.append(photo)
        if len(photo_validation_batch) >= image_batch_size:
            flush_photo_validation_batch()

    photo_records, photo_json_summary = iter_jsonl(paths["photo_json"])
    photo_parse_summary = stream_photo_records(
        limit_records(photo_records, _optional_int(limits.get("max_photos"))),
        image_root=paths["image_root"],
        row_sink=emit_photo_and_validation,
    )
    flush_photo_validation_batch()
    photo_output_summary = photo_writer.close()
    image_index_output_summary = image_index_writer.close()

    output_summaries = [
        business_output_summary,
        review_output_summary,
        photo_output_summary,
        image_index_output_summary,
        write_table(interim / f"review_business_stats.{_extension(output_format)}", review_stats, output_format),
    ]
    summary = {
        "business_count": business_parse_summary["parsed_businesses"],
        "review_count": review_filter_summary["valid_reviews"],
        "photo_count": photo_parse_summary["parsed_photos"],
        "valid_image_count": image_summary["valid_images"],
        "businesses_with_valid_reviews": len(review_stats),
        "jsonl": {
            "business": business_json_summary,
            "review": review_json_summary,
            "photo": photo_json_summary,
        },
        "review_filters": review_filter_summary,
        "business_parsing": business_parse_summary,
        "photo_parsing": photo_parse_summary,
        "image_validation": image_summary,
        "outputs": output_summaries,
    }
    write_json(interim / "validation_summary.json", summary)
    write_json(paths["validation_dir"] / "validation_summary.json", summary)
    return summary


def _optional_int(value: Any) -> int | None:
    """Normalize nullable YAML limits to either an integer or no limit."""
    if value in {None, "null", "None", ""}:
        return None
    return int(value)


def _extension(output_format: str) -> str:
    """Return the configured table filename extension."""
    return "csv" if output_format.lower() == "csv" else "parquet"


def _image_index_parquet_schema(output_format: str) -> Any | None:
    """Define stable image-index types when the PyArrow engine is available."""
    if output_format.lower() != "parquet":
        return None
    try:
        import pyarrow as pa
    except ImportError:
        return None
    return pa.schema(
        [
            ("photo_id", pa.string()),
            ("business_id", pa.string()),
            ("image_path", pa.string()),
            ("image_valid", pa.bool_()),
            ("image_width", pa.int64()),
            ("image_height", pa.int64()),
            ("validation_error", pa.string()),
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for JSONL parsing."""
    parser = argparse.ArgumentParser(description="Parse Yelp JSONL files into interim tabular outputs.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_processing.yaml"))
    return parser


def main() -> None:
    """Run the parsing stage and print its validation summary."""
    args = build_arg_parser().parse_args()
    summary = run_parse(load_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

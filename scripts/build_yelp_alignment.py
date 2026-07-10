import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.alignment import build_medium_alignment, build_strong_alignment, build_weak_alignment
from src.data.jsonl_utils import read_table, write_json, write_table
from src.data.statistics import build_dataset_statistics
from src.data.yelp_paths import create_output_directories, load_config, resolve_pipeline_paths


def run_alignment(config: dict[str, Any]) -> dict[str, Any]:
    create_output_directories(config)
    paths = resolve_pipeline_paths(config)
    output_format = config.get("output", {}).get("format", "parquet")
    extension = "csv" if output_format.lower() == "csv" else "parquet"
    interim = paths["interim_dir"]
    processed = paths["processed_dir"]

    businesses = read_table(interim / f"business.{extension}")
    photos = read_table(interim / f"photos.{extension}")
    image_index = read_table(interim / f"photo_image_index.{extension}")

    weak_config = config.get("weak_alignment", {})
    max_reviews_per_business = int(weak_config.get("max_reviews_per_business", 5))
    strong = build_strong_alignment(photos, image_index)
    medium = build_medium_alignment(photos, image_index, businesses)
    # Weak alignment only needs a bounded number of reviews per business; reading
    # the full 6.9M-review table here would defeat the streaming parse design.
    reviews = read_bounded_reviews_for_weak_alignment(
        interim / f"reviews.{extension}",
        max_reviews_per_business=max_reviews_per_business,
        target_business_ids={
            str(row["business_id"])
            for row in image_index
            if row.get("business_id") and row.get("image_valid") in {True, "True", "true", 1, "1"}
        },
    )
    weak = build_weak_alignment(
        photos,
        image_index,
        reviews,
        max_reviews_per_business=max_reviews_per_business,
        max_images_per_business=int(weak_config.get("max_images_per_business", 5)),
    )
    stats = build_dataset_statistics(businesses, reviews, photos, image_index, strong, medium, weak)
    validation_summary = _read_json(interim / "validation_summary.json")
    if validation_summary:
        stats["review_count"] = validation_summary.get("review_count", stats["review_count"])
        stats["business_count"] = validation_summary.get("business_count", stats["business_count"])
        stats["photo_metadata_count"] = validation_summary.get("photo_count", stats["photo_metadata_count"])
        stats["valid_image_count"] = validation_summary.get("valid_image_count", stats["valid_image_count"])
        stats["businesses_with_reviews"] = validation_summary.get(
            "businesses_with_valid_reviews",
            stats["businesses_with_reviews"],
        )
    output_summaries = [
        write_table(processed / f"strong_image_caption_pairs.{extension}", strong, output_format),
        write_table(processed / f"image_business_attribute_pairs.{extension}", medium, output_format),
        write_table(processed / f"business_level_weak_pairs.{extension}", weak, output_format),
    ]
    write_json(processed / "dataset_statistics.json", stats)
    summary = {
        "strong_pairs": len(strong),
        "medium_pairs": len(medium),
        "weak_pairs": len(weak),
        "statistics": stats,
        "outputs": output_summaries,
    }
    return summary


def read_bounded_reviews_for_weak_alignment(
    review_path: Path,
    max_reviews_per_business: int,
    target_business_ids: set[str],
) -> list[dict[str, Any]]:
    if review_path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            selected: list[dict[str, Any]] = []
            counts: dict[str, int] = {}
            columns = ["review_id", "business_id", "text"]
            parquet_file = pq.ParquetFile(review_path)
            for batch in parquet_file.iter_batches(batch_size=50000, columns=columns):
                for row in batch.to_pylist():
                    business_id = row.get("business_id")
                    if not business_id or str(business_id) not in target_business_ids:
                        continue
                    current = counts.get(str(business_id), 0)
                    if current >= max_reviews_per_business:
                        continue
                    counts[str(business_id)] = current + 1
                    selected.append(row)
            return selected
        except Exception:
            pass

    selected = []
    counts: dict[str, int] = {}
    for row in read_table(review_path):
        business_id = row.get("business_id")
        if not business_id or str(business_id) not in target_business_ids:
            continue
        current = counts.get(str(business_id), 0)
        if current >= max_reviews_per_business:
            continue
        counts[str(business_id)] = current + 1
        selected.append(row)
    return selected


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Yelp multimodal alignment datasets.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_processing.yaml"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_alignment(load_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

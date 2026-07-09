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
    reviews = read_table(interim / f"reviews.{extension}")
    photos = read_table(interim / f"photos.{extension}")
    image_index = read_table(interim / f"photo_image_index.{extension}")

    weak_config = config.get("weak_alignment", {})
    strong = build_strong_alignment(photos, image_index)
    medium = build_medium_alignment(photos, image_index, businesses)
    weak = build_weak_alignment(
        photos,
        image_index,
        reviews,
        max_reviews_per_business=int(weak_config.get("max_reviews_per_business", 5)),
        max_images_per_business=int(weak_config.get("max_images_per_business", 5)),
    )
    stats = build_dataset_statistics(businesses, reviews, photos, image_index, strong, medium, weak)
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

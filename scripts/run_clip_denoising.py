"""Run row-level CLIP filtering for bounded Yelp weak-alignment groups."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clip_denoising import DENOISED_PAIR_FIELDS, stream_clip_denoising
from src.data.jsonl_utils import TableStreamWriter, read_table, write_json
from src.data.yelp_paths import create_output_directories, load_config, resolve_pipeline_paths


def run_from_config(config_path: Path) -> dict[str, object]:
    """Load a YAML configuration and execute the CLIP task."""
    config = load_config(config_path)
    return run_with_config(config)


def run_with_config(config: dict[str, object]) -> dict[str, object]:
    """Run CLIP denoising and stream the retained rows into a durable table."""
    create_output_directories(config)
    paths = resolve_pipeline_paths(config)
    output_format = config.get("output", {}).get("format", "parquet")
    extension = "csv" if output_format.lower() == "csv" else "parquet"
    weak_pairs = read_table(paths["processed_dir"] / f"business_level_weak_pairs.{extension}")
    clip_config = config.get("clip_denoising", {})
    output_filename = str(clip_config.get("output_filename", "weak_pairs_denoised"))
    writer = TableStreamWriter(
        paths["processed_dir"] / f"{output_filename}.{extension}",
        output_format=output_format,
        fieldnames=DENOISED_PAIR_FIELDS,
        chunk_size=int(config.get("output", {}).get("chunk_size", 50000)),
    )
    summary = stream_clip_denoising(weak_pairs, clip_config, writer.write)
    summary["output"] = writer.close()
    write_json(paths["processed_dir"] / "clip_denoising_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the one-off CLIP task."""
    parser = argparse.ArgumentParser(description="Optionally denoise weak Yelp image-text pairs with CLIP.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_processing.yaml"))
    return parser


def main() -> None:
    """Run CLIP denoising and print its measured summary."""
    args = build_arg_parser().parse_args()
    print(json.dumps(run_from_config(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

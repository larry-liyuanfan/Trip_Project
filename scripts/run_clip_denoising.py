import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clip_denoising import run_clip_denoising
from src.data.jsonl_utils import read_table, write_json, write_table
from src.data.yelp_paths import create_output_directories, load_config, resolve_pipeline_paths


def run_from_config(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    create_output_directories(config)
    paths = resolve_pipeline_paths(config)
    output_format = config.get("output", {}).get("format", "parquet")
    extension = "csv" if output_format.lower() == "csv" else "parquet"
    weak_pairs = read_table(paths["processed_dir"] / f"business_level_weak_pairs.{extension}")
    summary, rows = run_clip_denoising(weak_pairs, config.get("clip_denoising", {}))
    if rows:
        write_table(paths["processed_dir"] / f"weak_pairs_denoised.{extension}", rows, output_format)
    write_json(paths["processed_dir"] / "clip_denoising_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optionally denoise weak Yelp image-text pairs with CLIP.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_processing.yaml"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(run_from_config(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

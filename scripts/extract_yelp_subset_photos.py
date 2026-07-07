import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.yelp_archives import extract_yelp_photo_files, read_photo_ids_from_multimodal_items


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract only Yelp photos referenced by a processed multimodal subset.")
    parser.add_argument("--photos-zip", type=Path, default=Path("data/Yelp-Photos.zip"))
    parser.add_argument(
        "--multimodal-items",
        type=Path,
        default=Path("data/yelp/processed/ota_subset_v1/multimodal_items.jsonl"),
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/yelp/raw"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    photo_ids = read_photo_ids_from_multimodal_items(args.multimodal_items)
    manifest = extract_yelp_photo_files(
        photos_zip_path=args.photos_zip,
        raw_dir=args.raw_dir,
        photo_ids=photo_ids,
    )
    summary = {
        "source": manifest["source"],
        "raw_dir": manifest["raw_dir"],
        "photos_zip_path": manifest["photos_zip_path"],
        "requested_photo_count": manifest["requested_photo_count"],
        "extracted_photo_count": manifest["extracted_photo_count"],
        "manifest_path": str(args.raw_dir / "extract_photo_manifest.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

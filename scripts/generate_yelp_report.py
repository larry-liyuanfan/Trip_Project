import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.yelp_paths import create_output_directories, load_config, resolve_pipeline_paths


def render_report(config: dict[str, Any], stats: dict[str, Any], validation: dict[str, Any] | None = None, clip: dict[str, Any] | None = None) -> str:
    validation = validation or {}
    clip = clip or {}
    storage_note = _storage_note(validation)

    def value(key: str) -> Any:
        result = stats.get(key)
        return "TODO" if result is None else result

    def nested(section: dict[str, Any], key: str) -> Any:
        result = section.get(key)
        return "TODO" if result is None else result

    lines = [
        "# Week 2 Yelp Multimodal Data Processing Report",
        "",
        "## 1. Week 2 objective",
        "Build a reproducible Yelp multimodal dataset pipeline that parses raw JSONL files, validates local images, creates strong/medium/weak image-text alignments, and produces a report-ready dataset summary.",
        "",
        "## 2. Raw dataset files used",
        f"- Business JSONL: `{config.get('paths', {}).get('business_json', 'TODO')}`",
        f"- Review JSONL: `{config.get('paths', {}).get('review_json', 'TODO')}`",
        f"- Photo metadata JSONL: `{config.get('paths', {}).get('photo_json', 'TODO')}`",
        f"- Local image root: `{config.get('paths', {}).get('image_root', 'TODO')}`",
        f"- Review cap for this smoke run: {config.get('processing_limits', {}).get('max_reviews', 'TODO')}",
        "- Download and archive extraction status: completed before Week 2; this pipeline consumes the normalized files under `data/yelp/raw/`.",
        "",
        "## 3. Parsing pipeline overview",
        "- `scripts/parse_yelp_json.py` reads business, review, and photo JSONL inputs line by line, then writes interim tables for the configured smoke-run scope.",
        "- Review parsing rejects empty, symbol-only, and too-short text according to `configs/data_processing.yaml`.",
        "- Image validation checks local file existence and readability with Pillow, then records width and height.",
        "- `scripts/build_yelp_alignment.py` creates processed alignment datasets and statistics.",
        "",
        "## 4. Extracted business/review/photo fields",
        "- Business fields: `business_id`, name, location, coordinates, stars, review count, categories, attributes, hours, and selected flattened attributes.",
        "- Review fields: `review_id`, `business_id`, user id, stars, useful/funny/cool counts, cleaned text, and date.",
        "- Photo fields: `photo_id`, `business_id`, caption, label, and local `image_path`.",
        "- Image index fields: `photo_id`, `business_id`, `image_path`, validity flag, width, height, and validation error.",
        "",
        "## 5. Local image validation result",
        f"- Photo metadata entries parsed: {value('photo_metadata_count')}",
        f"- Valid local images: {nested(validation.get('image_validation', {}), 'valid_images')}",
        f"- Missing local images: {nested(validation.get('image_validation', {}), 'missing_images')}",
        f"- Corrupted local images: {nested(validation.get('image_validation', {}), 'corrupted_images')}",
        "",
        "## 6. Multimodal alignment strategy",
        f"- Strong alignment: {value('strong_pairs')} valid image-caption-label pairs keyed by `photo_id`.",
        f"- Medium alignment: {value('medium_pairs')} valid image-business metadata pairs with generated business descriptions.",
        f"- Weak alignment: {value('weak_pairs')} business-level groups containing bounded image lists and selected review texts.",
        f"- CLIP denoising: {clip.get('status', 'TODO')} ({clip.get('reason', 'TODO')}).",
        "",
        "## 7. Output statistics",
        f"- Businesses parsed: {value('business_count')}",
        f"- Reviews parsed: {value('review_count')}",
        f"- Photo metadata entries parsed: {value('photo_metadata_count')}",
        f"- Valid local images: {value('valid_image_count')}",
        f"- Strong pairs: {value('strong_pairs')}",
        f"- Medium pairs: {value('medium_pairs')}",
        f"- Weak groups: {value('weak_pairs')}",
        f"- Businesses with capped reviews: {value('businesses_with_reviews')}",
        f"- Businesses with valid images: {value('businesses_with_valid_images')}",
        f"- Valid image ratio: {value('valid_image_ratio')}",
        f"- Photo label distribution: {stats.get('photo_label_distribution', 'TODO')}",
        f"- Caption length statistics: {stats.get('caption_length_stats', 'TODO')}",
        f"- Denoising before/after weak pairs: {clip.get('input_pairs', 'TODO')} -> {clip.get('retained_pairs', 'TODO')}",
        f"- Top categories: {stats.get('top_categories', 'TODO')}",
        "",
        "## 8. Data quality issues and limitations",
        f"- Storage behavior: {storage_note}",
        f"- CLIP denoising status: {clip.get('status', 'TODO')} ({clip.get('reason', 'TODO')})",
        "- The local image set is partial: most photo metadata rows point to files that are not currently extracted locally.",
        "- The default config caps reviews at 10,000 for fast Windows smoke validation. A full raw Yelp review pass should first add chunked table writing or another bounded-memory output path.",
        "- Full live model serving is intentionally out of scope for Week 2 and should use Docker or WSL2 later.",
        "",
        "## 9. Reproducible commands",
        "```bash",
        "pip install -r requirements-data.txt",
        "python scripts/parse_yelp_json.py --config configs/data_processing.yaml",
        "python scripts/build_yelp_alignment.py --config configs/data_processing.yaml",
        "python scripts/run_clip_denoising.py --config configs/data_processing.yaml",
        "python scripts/generate_yelp_report.py --config configs/data_processing.yaml",
        "python scripts/validate_week2_pipeline.py --config configs/data_processing.yaml",
        "python -m unittest discover -s tests -v",
        "```",
        "",
        "## 10. Follow-up TODOs",
        "- Add chunked table writes before uncapping full review parsing.",
        "- Replace the current CLIP skip/status interface with real semantic scoring when GPU or suitable CPU runtime is available.",
        "- Keep GPU-heavy dependency paths in Docker or WSL2.",
    ]
    return "\n".join(lines) + "\n"


def run_report(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    create_output_directories(config)
    paths = resolve_pipeline_paths(config)
    stats = _read_json(paths["processed_dir"] / "dataset_statistics.json")
    validation = _read_json(paths["interim_dir"] / "validation_summary.json")
    clip = _read_json(paths["processed_dir"] / "clip_denoising_summary.json")
    report = render_report(config, stats, validation, clip)
    paths["report_path"].write_text(report, encoding="utf-8")
    return {"report_path": str(paths["report_path"]), "sections": 10}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _storage_note(validation: dict[str, Any]) -> str:
    outputs = validation.get("outputs") or []
    actual_formats = {record.get("actual_format") for record in outputs if isinstance(record, dict)}
    if "csv_fallback" in actual_formats:
        return "CSV fallback was used at the configured .parquet paths because no Parquet engine was available for that run."
    if "parquet" in actual_formats:
        return "Real Parquet files were written with the available pandas Parquet engine."
    return "TODO"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the Week 2 Yelp data processing report.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_processing.yaml"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(run_report(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

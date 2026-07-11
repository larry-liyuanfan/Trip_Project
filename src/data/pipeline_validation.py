"""Validate Week 2 output presence, schemas, counts, paths, and storage format."""

import importlib.util
import re
import csv
import json
from pathlib import Path
from typing import Any

from src.data.clip_denoising import DENOISED_PAIR_FIELDS
from src.data.jsonl_utils import read_table
from src.data.yelp_paths import resolve_pipeline_paths


EXPECTED_FILES = {
    "business": ("interim_dir", "business", {"business_id", "name", "categories", "stars"}),
    "reviews": ("interim_dir", "reviews", {"review_id", "business_id", "text", "stars"}),
    "photos": ("interim_dir", "photos", {"photo_id", "business_id", "caption", "label", "image_path"}),
    "photo_image_index": ("interim_dir", "photo_image_index", {"photo_id", "business_id", "image_path", "image_valid"}),
    "review_business_stats": ("interim_dir", "review_business_stats", {"business_id", "valid_review_count"}),
    "strong_pairs": ("processed_dir", "strong_image_caption_pairs", {"business_id", "photo_id", "image_path", "caption", "label"}),
    "medium_pairs": ("processed_dir", "image_business_attribute_pairs", {"business_id", "photo_id", "image_path", "business_description", "attribute_dimension_labels"}),
    "weak_pairs": ("processed_dir", "business_level_weak_pairs", {"business_id", "photo_ids", "image_paths", "review_texts"}),
}


def validate_week2_outputs(config: dict[str, Any]) -> dict[str, Any]:
    """Run the complete output contract and return errors without hiding gaps."""
    paths = resolve_pipeline_paths(config)
    output_format = config.get("output", {}).get("format", "parquet")
    extension = "csv" if output_format == "csv" else "parquet"
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    columns: dict[str, list[str]] = {}
    table_paths: dict[str, Path] = {}

    for logical_name, (directory_key, filename, required_columns) in EXPECTED_FILES.items():
        path = paths[directory_key] / f"{filename}.{extension}"
        table_paths[logical_name] = path
        if not path.exists():
            errors.append(f"Missing expected output file: {path}")
            counts[logical_name] = 0
            columns[logical_name] = []
            continue
        counts[logical_name], observed_columns = inspect_table(path)
        columns[logical_name] = sorted(observed_columns)
        missing_columns = required_columns - observed_columns
        if missing_columns:
            errors.append(f"{logical_name} is missing columns: {sorted(missing_columns)}")

    _validate_alignment_image_paths(table_paths, errors)
    clip_errors = validate_clip_denoising_output(
        paths["processed_dir"],
        output_format,
        config.get("clip_denoising", {}),
        set(DENOISED_PAIR_FIELDS),
    )
    errors.extend(clip_errors)
    _validate_report_counts(paths["report_path"], counts, errors, warnings)
    _validate_storage_format(table_paths, output_format, paths["report_path"], errors, warnings)

    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
        "columns": columns,
        "files": {key: str(path) for key, path in table_paths.items()},
    }


def inspect_table(path: Path) -> tuple[int, set[str]]:
    """Read Parquet metadata first so full review tables are not materialized."""
    if path.suffix == ".parquet" and importlib.util.find_spec("pyarrow") is not None:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        # Arrow schema names preserve top-level list/struct fields, while the
        # physical Parquet schema exposes nested children as generic `element`.
        return parquet_file.metadata.num_rows, set(parquet_file.schema_arrow.names)
    rows = read_table(path)
    return len(rows), set(rows[0].keys()) if rows else set()


def validate_clip_denoising_output(
    processed_dir: Path,
    output_format: str,
    clip_config: dict[str, Any],
    required_columns: set[str],
) -> list[str]:
    """Verify the one-off CLIP task wrote a complete table and matching summary."""
    summary_path = processed_dir / "clip_denoising_summary.json"
    if not clip_config.get("enabled", False) and not summary_path.exists():
        return []

    errors: list[str] = []
    extension = "csv" if output_format.lower() == "csv" else "parquet"
    output_name = str(clip_config.get("output_filename", "weak_pairs_denoised"))
    output_path = processed_dir / f"{output_name}.{extension}"
    if not summary_path.exists():
        return [f"Missing CLIP denoising summary: {summary_path}"]
    if not output_path.exists():
        return [f"Missing CLIP denoised output: {output_path}"]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_table(output_path)
    columns = _table_columns(output_path)
    missing_columns = required_columns - columns
    if missing_columns:
        errors.append(f"CLIP denoised output is missing columns: {sorted(missing_columns)}")
    retained_pairs = int(summary.get("retained_pairs", 0))
    if retained_pairs != len(rows):
        errors.append(f"CLIP summary retained_pairs={retained_pairs} does not match output rows={len(rows)}")
    for index, row in enumerate(rows):
        image_path = row.get("image_path")
        if not image_path or not Path(str(image_path)).exists():
            errors.append(f"CLIP denoised row {index} has missing image_path: {image_path}")
            break
    return errors


def _table_columns(path: Path) -> set[str]:
    """Return logical top-level columns without materializing Parquet rows."""
    return inspect_table(path)[1]


def _validate_alignment_image_paths(table_paths: dict[str, Path], errors: list[str]) -> None:
    """Ensure every strong, medium, and weak output references local files."""
    for logical_name in ["strong_pairs", "medium_pairs"]:
        for index, row in enumerate(read_table(table_paths[logical_name])):
            image_path = row.get("image_path")
            if not image_path or not Path(str(image_path)).exists():
                errors.append(f"{logical_name}[{index}] image_path does not exist: {image_path}")
                break
    for index, row in enumerate(read_table(table_paths["weak_pairs"])):
        image_paths = row.get("image_paths") or []
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        missing = [path for path in image_paths if not Path(str(path)).exists()]
        if missing:
            errors.append(f"weak_pairs[{index}] has missing image paths: {missing[:3]}")
            break


def _validate_report_counts(report_path: Path, counts: dict[str, int], errors: list[str], warnings: list[str]) -> None:
    """Check that the mentor report states each measured output count."""
    if not report_path.exists():
        errors.append(f"Missing report file: {report_path}")
        return
    text = report_path.read_text(encoding="utf-8")
    expected = {
        "business": ("business_count", counts.get("business", 0)),
        "review": ("review_count", counts.get("reviews", 0)),
        "photo": ("photo_metadata_count", counts.get("photos", 0)),
        "strong": ("strong_pairs", counts.get("strong_pairs", 0)),
        "medium": ("medium_pairs", counts.get("medium_pairs", 0)),
        "weak": ("weak_pairs", counts.get("weak_pairs", 0)),
    }
    for label, (_, count) in expected.items():
        if str(count) not in text:
            errors.append(f"Report does not mention actual {label} count: {count}")
    if "CSV fallback" not in text and "Parquet" not in text:
        warnings.append("Report does not clearly mention Parquet or CSV fallback behavior.")


def _validate_storage_format(
    table_paths: dict[str, Path],
    output_format: str,
    report_path: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Confirm configured Parquet outputs are real Parquet or documented fallback."""
    if output_format != "parquet":
        return
    parquet_engine_available = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None
    parquet_files = [path for path in table_paths.values() if path.exists()]
    real_parquet = all(_is_parquet_file(path) for path in parquet_files)
    if parquet_engine_available and not real_parquet:
        errors.append("A Parquet engine is available, but at least one configured .parquet output is not a real Parquet file.")
    if not real_parquet:
        report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        if "CSV fallback" not in report_text:
            errors.append("CSV fallback is used, but the report does not clearly state it.")
        warnings.append("Configured .parquet outputs are CSV fallback files because no Parquet engine was used.")


def _is_parquet_file(path: Path) -> bool:
    """Identify Parquet files from their required leading magic bytes."""
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"PAR1"
    except OSError:
        return False

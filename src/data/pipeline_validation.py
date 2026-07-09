import importlib.util
import re
from pathlib import Path
from typing import Any

from src.data.jsonl_utils import read_table
from src.data.yelp_paths import resolve_pipeline_paths


EXPECTED_FILES = {
    "business": ("interim_dir", "business", {"business_id", "name", "categories", "stars"}),
    "reviews": ("interim_dir", "reviews", {"review_id", "business_id", "text", "stars"}),
    "photos": ("interim_dir", "photos", {"photo_id", "business_id", "caption", "label", "image_path"}),
    "photo_image_index": ("interim_dir", "photo_image_index", {"photo_id", "business_id", "image_path", "image_valid"}),
    "review_business_stats": ("interim_dir", "review_business_stats", {"business_id", "valid_review_count"}),
    "strong_pairs": ("processed_dir", "strong_image_caption_pairs", {"business_id", "photo_id", "image_path", "caption", "label"}),
    "medium_pairs": ("processed_dir", "image_business_attribute_pairs", {"business_id", "photo_id", "image_path", "business_description"}),
    "weak_pairs": ("processed_dir", "business_level_weak_pairs", {"business_id", "photo_ids", "image_paths", "review_texts"}),
}


def validate_week2_outputs(config: dict[str, Any]) -> dict[str, Any]:
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
        rows = read_table(path)
        counts[logical_name] = len(rows)
        observed_columns = set(rows[0].keys()) if rows else set()
        columns[logical_name] = sorted(observed_columns)
        missing_columns = required_columns - observed_columns
        if missing_columns:
            errors.append(f"{logical_name} is missing columns: {sorted(missing_columns)}")

    _validate_alignment_image_paths(table_paths, errors)
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


def _validate_alignment_image_paths(table_paths: dict[str, Path], errors: list[str]) -> None:
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
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"PAR1"
    except OSError:
        return False

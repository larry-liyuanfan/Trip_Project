"""Load shared Yelp pipeline configuration and resolve repository paths."""

from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "business_json": "data/yelp/raw/yelp_academic_dataset_business.json",
        "review_json": "data/yelp/raw/yelp_academic_dataset_review.json",
        "photo_json": "data/yelp/raw/photos.json",
        "image_root": "data/yelp/raw/photos",
        "interim_dir": "data/yelp/interim",
        "processed_dir": "data/yelp/processed",
        "logs_dir": "data/yelp/logs",
        "validation_dir": "data/yelp/validation",
        "report_path": "reports/yelp_multimodal_data_processing_report_part1.md",
    },
    "output": {"format": "parquet"},
    "review_filters": {"min_text_length": 20, "reject_symbol_only": True},
    "weak_alignment": {"max_reviews_per_business": 5, "max_images_per_business": 5},
    "clip_denoising": {"enabled": False, "threshold": 0.25},
}


def load_config(path: Path | str) -> dict[str, Any]:
    """Merge an optional YAML file over safe pipeline defaults."""
    config_path = Path(path)
    config = deep_merge(DEFAULT_CONFIG, {})
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        loaded = parse_simple_yaml(text)
        config = deep_merge(config, loaded)
    return config


def resolve_pipeline_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Convert configured path strings to `Path` objects without absolutizing."""
    paths = config.get("paths", {})
    return {key: Path(value) for key, value in paths.items()}


def create_output_directories(config: dict[str, Any]) -> None:
    """Create generated-data and report parent directories from configuration."""
    paths = resolve_pipeline_paths(config)
    for key in ["interim_dir", "processed_dir", "logs_dir", "validation_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    paths["report_path"].parent.mkdir(parents=True, exist_ok=True)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge nested mappings while leaving caller inputs unchanged."""
    result = {key: value.copy() if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Use PyYAML when available and a minimal mapping parser otherwise."""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        return _parse_indented_mapping(text)


def _parse_indented_mapping(text: str) -> dict[str, Any]:
    """Parse the mapping-only YAML subset used by project configuration."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, sep, raw_value = raw_line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = raw_value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    """Decode booleans, nulls, integers, floats, and quoted strings."""
    stripped = value.strip("'\"")
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped

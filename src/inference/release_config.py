"""Shared explicit release-file resolution for CLI and service."""
import os
from pathlib import Path

DEFAULT_RELEASE_CONFIG = "configs/releases/qwen3_vl_system_final_v1.json"


def resolve_release_config(root: Path, config_path=None) -> Path:
    value = config_path if config_path is not None else os.getenv("TRIP_RELEASE_CONFIG", DEFAULT_RELEASE_CONFIG)
    if not str(value).strip():
        raise ValueError("release config path is empty")
    selected = Path(value)
    selected = (selected if selected.is_absolute() else Path(root) / selected).resolve()
    if not selected.is_file():
        raise ValueError(f"release config does not exist: {selected}")
    return selected

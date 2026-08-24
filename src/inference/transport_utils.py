"""Dependency-light helpers shared by API and offline evaluation transports."""

import base64
import mimetypes
import os
from pathlib import Path
from urllib.parse import unquote, urlparse


def normalize_image_url(image_url: str) -> str:
    """Convert an existing local file URL to a data URL for model transport."""
    if not image_url.startswith("file://"):
        return image_url

    parsed = urlparse(image_url)
    raw_path = _file_url_to_path_text(parsed.netloc, unquote(parsed.path))
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else _relative_asset_candidates(path)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return image_url

    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _relative_asset_candidates(path: Path) -> list[Path]:
    if ".." in path.parts:
        return []
    candidates = [Path.cwd() / path]
    for value in os.getenv("TRIP_ASSET_ROOTS", "").split(os.pathsep):
        if not value.strip():
            continue
        root = Path(value).expanduser().resolve()
        candidate = (root / path).resolve()
        if candidate == root or root in candidate.parents:
            candidates.append(candidate)
    return candidates


def _file_url_to_path_text(netloc: str, path: str) -> str:
    """Normalize POSIX, relative, UNC-like, and Windows-drive file URL forms."""
    if netloc in ("", "localhost"):
        if len(path) >= 3 and path[0] == "/" and path[2] == ":":
            return path[1:]
        return path.lstrip("/") if not Path(path).is_absolute() else path
    if len(netloc) == 2 and netloc[1] == ":":
        return f"{netloc}{path}"
    return f"{netloc}{path}"


def strip_json_fence(content: str) -> str:
    """Remove an optional Markdown JSON fence around model output."""
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

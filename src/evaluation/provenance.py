"""Hash evaluation inputs and contracts so persisted runs remain reproducible."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.data.yelp_paths import parse_simple_yaml


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceValidationError(ValueError):
    """Raised when a persisted evaluation artifact no longer matches a run."""


def canonical_sha256(value: Any) -> str:
    """Return a stable SHA-256 for a JSON-compatible value."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact_hashes(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    """Hash files using repository-relative POSIX paths as stable keys."""
    project_root = Path(root).resolve()
    hashes: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = project_root / path
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(project_root).as_posix()
        except ValueError as exc:
            raise ProvenanceValidationError(
                f"artifact path escapes project root: {path}"
            ) from exc
        if not resolved.is_file():
            raise ProvenanceValidationError(f"artifact file is missing: {relative}")
        hashes[relative] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def verify_artifact_hashes(root: Path, hashes: dict[str, str]) -> None:
    """Reject missing, malformed, or changed files recorded by a run."""
    if not isinstance(hashes, dict) or not hashes:
        raise ProvenanceValidationError("artifact_hashes must be a non-empty object")
    project_root = Path(root).resolve()
    for relative, expected in hashes.items():
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(expected, str)
            or SHA256_PATTERN.fullmatch(expected) is None
        ):
            raise ProvenanceValidationError("artifact_hashes contains an invalid entry")
        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise ProvenanceValidationError(
                f"artifact path escapes project root: {relative}"
            ) from exc
        if not path.is_file():
            raise ProvenanceValidationError(f"artifact file is missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ProvenanceValidationError(
                f"artifact hash mismatch for {relative}: expected {expected}, got {actual}"
            )


def build_run_artifact_hashes(
    root: Path,
    config: dict[str, Any],
    prompt_version: str,
) -> dict[str, str]:
    """Hash manifests, exclusion registry, active prompts, and all Schemas."""
    project_root = Path(root)
    scenarios = config.get("scenarios")
    paths_config = config.get("paths")
    if not isinstance(scenarios, dict) or not isinstance(paths_config, dict):
        raise ProvenanceValidationError("evaluation config is missing paths or scenarios")

    paths: list[Path] = [project_root / paths_config["exclusion_manifest"]]
    for scenario, scenario_config in scenarios.items():
        paths.append(project_root / scenario_config["manifest_path"])
        paths.append(
            project_root
            / "configs"
            / "evaluation"
            / "schemas"
            / f"{scenario}_v1.schema.json"
        )

    prompt_root = project_root / "configs" / "evaluation" / "prompts"
    if prompt_version == "baseline_minimal_v1":
        paths.extend(
            prompt_root / prompt_version / f"{scenario}.txt"
            for scenario in scenarios
        )
    elif prompt_version.startswith("standardized_v"):
        paths.append(prompt_root / prompt_version / "common.yaml")
        for scenario in scenarios:
            prompt_path = prompt_root / prompt_version / f"{scenario}.yaml"
            paths.append(prompt_path)
            prompt_spec = parse_simple_yaml(prompt_path.read_text(encoding="utf-8"))
            schema_name = prompt_spec.get("schema_name") if isinstance(prompt_spec, dict) else None
            if (
                not isinstance(schema_name, str)
                or Path(schema_name).name != schema_name
                or not schema_name.startswith(f"{scenario}_")
                or not schema_name.endswith(".schema.json")
            ):
                raise ProvenanceValidationError(
                    f"invalid schema_name in prompt asset: {prompt_path}"
                )
            paths.append(
                project_root / "configs" / "evaluation" / "schemas" / schema_name
            )
    else:
        raise ProvenanceValidationError(f"unsupported prompt version: {prompt_version}")
    return build_artifact_hashes(project_root, paths)

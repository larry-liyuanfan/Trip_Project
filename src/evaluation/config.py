"""Configuration loading and validation for the Week 3 evaluation framework."""

from pathlib import Path
from typing import Any

from src.data.yelp_paths import parse_simple_yaml
from src.evaluation.manifests import ManifestValidationError, SCENARIOS


def load_evaluation_config(path: Path | str) -> dict[str, Any]:
    """Load a Week 3 YAML config and validate targets, paths, seeds, and quotas."""
    config_path = Path(path)
    payload = parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestValidationError("evaluation config must be a YAML mapping")

    dataset_version = payload.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise ManifestValidationError("dataset_version must be a non-empty string")

    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ManifestValidationError("paths must be a mapping")
    for name in (
        "images_dir",
        "exclusion_manifest",
        "sampling_logs_dir",
        "runs_dir",
    ):
        _require_relative_path(paths.get(name), f"paths.{name}")
    if "scores_dir" in paths:
        _require_relative_path(paths.get("scores_dir"), "paths.scores_dir")
    for name in ("codings_dir", "comparisons_dir", "generated_reports_dir", "readiness_dir"):
        if name in paths:
            _require_relative_path(paths.get(name), f"paths.{name}")

    candidate_sources = payload.get("candidate_sources")
    if candidate_sources is not None:
        if not isinstance(candidate_sources, dict):
            raise ManifestValidationError("candidate_sources must be a mapping")
        for name in ("yelp_business_path", "yelp_photos_path", "yelp_weak_pairs_path"):
            _require_relative_path(candidate_sources.get(name), f"candidate_sources.{name}")
        for name in ("yelp_source_license", "synthetic_recipe_version"):
            value = candidate_sources.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ManifestValidationError(
                    f"candidate_sources.{name} must be a non-empty string"
                )

    metrics = payload.get("metrics")
    if metrics is not None:
        if not isinstance(metrics, dict):
            raise ManifestValidationError("metrics must be a mapping")
        _require_relative_path(metrics.get("aliases_path"), "metrics.aliases_path")

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise ManifestValidationError("runtime must be a mapping")
    for name in ("model_config_path", "inference_config_path"):
        _require_relative_path(runtime.get(name), f"runtime.{name}")
    live_base_url = runtime.get("live_base_url")
    if not isinstance(live_base_url, str) or not live_base_url.strip():
        raise ManifestValidationError("runtime.live_base_url must be a non-empty string")
    if not live_base_url.startswith(("http://", "https://")):
        raise ManifestValidationError("runtime.live_base_url must be an HTTP(S) URL")
    timeout_seconds = runtime.get("timeout_seconds")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise ManifestValidationError("runtime.timeout_seconds must be a positive integer")

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, dict) or set(scenarios) != SCENARIOS:
        raise ManifestValidationError(f"scenarios must contain exactly {sorted(SCENARIOS)}")
    for scenario, settings in scenarios.items():
        if not isinstance(settings, dict):
            raise ManifestValidationError(f"scenario {scenario} settings must be a mapping")
        _require_relative_path(settings.get("manifest_path"), f"{scenario}.manifest_path")
        target_count = settings.get("target_count")
        if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count <= 0:
            raise ManifestValidationError(f"{scenario}.target_count must be a positive integer")
        sampling = settings.get("sampling")
        if not isinstance(sampling, dict):
            raise ManifestValidationError(f"{scenario}.sampling must be a mapping")
        seed = sampling.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ManifestValidationError(f"{scenario}.sampling.seed must be an integer")
        stratum_field = sampling.get("stratum_field")
        if not isinstance(stratum_field, str) or not stratum_field.strip():
            raise ManifestValidationError(
                f"{scenario}.sampling.stratum_field must be a non-empty string"
            )
        quotas = sampling.get("quotas")
        if not isinstance(quotas, dict) or not quotas:
            raise ManifestValidationError(f"{scenario}.sampling.quotas must be a non-empty mapping")
        for stratum, quota in quotas.items():
            if not isinstance(stratum, str) or not stratum.strip():
                raise ManifestValidationError(f"{scenario} quota strata must be non-empty strings")
            if isinstance(quota, bool) or not isinstance(quota, int) or quota < 0:
                raise ManifestValidationError(
                    f"{scenario} quota for {stratum!r} must be a non-negative integer"
                )
        if sum(quotas.values()) != target_count:
            raise ManifestValidationError(
                f"{scenario} sampling quotas must sum to target_count {target_count}"
            )
        required_source_types = settings.get("required_source_types")
        if required_source_types is not None:
            if (
                not isinstance(required_source_types, list)
                or len(required_source_types) < 2
                or any(
                    not isinstance(source_type, str) or not source_type.strip()
                    for source_type in required_source_types
                )
                or len(set(required_source_types)) != len(required_source_types)
            ):
                raise ManifestValidationError(
                    f"{scenario}.required_source_types must contain at least two "
                    "unique non-empty source types"
                )
    return payload


def _require_relative_path(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field} must be a non-empty repository-relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ManifestValidationError(f"{field} must be repository-relative")

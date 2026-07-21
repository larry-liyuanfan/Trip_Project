"""Mode-aware Week 3 multimodal evaluation runner."""

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from src.data.yelp_paths import parse_simple_yaml
from src.evaluation.config import load_evaluation_config
from src.evaluation.manifests import (
    is_release_eligible,
    load_configured_manifests,
    validate_release_provenance,
    validate_exclusion_manifest,
)
from src.evaluation.prompting import render_baseline_request, render_standard_prompt
from src.evaluation.provenance import (
    build_run_artifact_hashes,
    canonical_sha256,
)
from src.evaluation.results import ImmutableRunWriter, parse_and_validate_output
from src.inference.client import normalize_image_url


ModelTransport = Callable[[str, dict[str, Any], int], str]


class EvaluationRunError(ValueError):
    """Raised when evaluation run arguments or runtime settings are invalid."""


def load_runtime_settings(
    root: Path,
    evaluation_config: dict[str, Any],
    *,
    live_base_url: str | None = None,
) -> dict[str, Any]:
    """Load model and generation settings referenced by evaluation config."""
    project_root = Path(root)
    runtime_config = evaluation_config["runtime"]
    model = _load_yaml_mapping(
        project_root / runtime_config["model_config_path"],
        "model config",
    )
    inference = _load_yaml_mapping(
        project_root / runtime_config["inference_config_path"],
        "inference config",
    )
    for field in ("model_name", "served_model_name", "backend"):
        value = model.get(field)
        if not isinstance(value, str) or not value.strip():
            raise EvaluationRunError(f"model config {field} must be non-empty text")
    generation = {
        "temperature": inference.get("temperature"),
        "top_p": inference.get("top_p"),
        "max_tokens": inference.get("max_tokens"),
    }
    if isinstance(generation["temperature"], bool) or not isinstance(
        generation["temperature"], (int, float)
    ):
        raise EvaluationRunError("inference temperature must be numeric")
    if isinstance(generation["top_p"], bool) or not isinstance(
        generation["top_p"], (int, float)
    ):
        raise EvaluationRunError("inference top_p must be numeric")
    if (
        isinstance(generation["max_tokens"], bool)
        or not isinstance(generation["max_tokens"], int)
        or generation["max_tokens"] <= 0
    ):
        raise EvaluationRunError("inference max_tokens must be a positive integer")
    return {
        "model_name": model["model_name"],
        "served_model_name": model["served_model_name"],
        "model_config": model,
        "generation": generation,
        "live_base_url": live_base_url or runtime_config["live_base_url"],
        "timeout_seconds": runtime_config["timeout_seconds"],
    }


def run_configured_evaluation(
    *,
    root: Path,
    config_path: Path,
    run_id: str,
    mode: str,
    prompt_version: str,
    run_scope: str = "framework",
    mock_outputs: dict[str, str] | None = None,
    live_base_url: str | None = None,
    transport: ModelTransport | None = None,
) -> dict[str, Any]:
    """Load configured manifests and execute one immutable evaluation run."""
    project_root = Path(root)
    resolved_config = Path(config_path)
    if not resolved_config.is_absolute():
        resolved_config = project_root / resolved_config
    config = load_evaluation_config(resolved_config)
    configured_records = load_configured_manifests(config, root=project_root)
    records = [
        record
        for scenario in config["scenarios"]
        for record in configured_records[scenario]
    ]
    validate_exclusion_manifest(
        records,
        project_root / config["paths"]["exclusion_manifest"],
    )
    if mode == "live":
        if run_scope not in {"pilot", "full"}:
            raise EvaluationRunError("live mode requires run_scope pilot or full")
        selected_records = (
            select_pilot_records(config, configured_records)
            if run_scope == "pilot"
            else select_inference_records(records)
        )
        if not selected_records:
            raise EvaluationRunError("live mode requires at least one eligible sample")
        validate_release_provenance(selected_records)
        if run_scope == "full":
            validate_full_run_readiness(config, configured_records)
        records = selected_records
    elif run_scope != "framework":
        raise EvaluationRunError("mock and dry-run modes require run_scope framework")
    artifact_hashes = build_run_artifact_hashes(
        project_root,
        config,
        prompt_version,
    )
    runtime = load_runtime_settings(
        project_root,
        config,
        live_base_url=live_base_url,
    )
    return run_records(
        root=project_root,
        records=records,
        runs_dir=project_root / config["paths"]["runs_dir"],
        run_id=run_id,
        mode=mode,
        prompt_version=prompt_version,
        run_scope=run_scope,
        runtime=runtime,
        dataset_version=config["dataset_version"],
        artifact_hashes=artifact_hashes,
        mock_outputs=mock_outputs,
        transport=transport,
    )


def select_inference_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select records that pass the approved single-annotator release gate."""
    return [record for record in records if is_release_eligible(record)]


def select_pilot_records(
    config: dict[str, Any],
    configured_records: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Select the first eligible sample from every approved scenario stratum."""
    selected: list[dict[str, Any]] = []
    for scenario, settings in config["scenarios"].items():
        eligible = select_inference_records(configured_records.get(scenario, []))
        by_stratum: dict[str, dict[str, Any]] = {}
        for record in eligible:
            stratum = record.get("sampling_stratum")
            if isinstance(stratum, str) and stratum not in by_stratum:
                by_stratum[stratum] = record
        for stratum in settings["sampling"]["quotas"]:
            record = by_stratum.get(stratum)
            if record is None:
                raise EvaluationRunError(
                    f"{scenario} pilot has no eligible sample for stratum {stratum}"
                )
            selected.append(record)
    return selected


def validate_full_run_readiness(
    config: dict[str, Any],
    configured_records: dict[str, list[dict[str, Any]]],
) -> None:
    """Require target size, candidate strata, and configured source presence."""
    for scenario, settings in config["scenarios"].items():
        eligible = select_inference_records(configured_records.get(scenario, []))
        target = settings["target_count"]
        if len(eligible) != target:
            raise EvaluationRunError(
                f"{scenario} validated_count must equal target_count {target}; "
                f"got {len(eligible)}"
            )
        expected_quotas = settings["sampling"]["quotas"]
        actual_counts = {stratum: 0 for stratum in expected_quotas}
        for record in eligible:
            stratum = record.get("sampling_stratum")
            if stratum not in actual_counts:
                raise EvaluationRunError(
                    f"{scenario} has unapproved sampling_stratum: {stratum}"
                )
            actual_counts[stratum] += 1
        if actual_counts != expected_quotas:
            raise EvaluationRunError(
                f"{scenario} validated stratum counts do not match approved quotas: "
                f"expected {expected_quotas}, got {actual_counts}"
            )
        required_source_types = settings.get("required_source_types")
        if not isinstance(required_source_types, list):
            continue
        actual_source_types = {
            record.get("source_type")
            for record in eligible
            if isinstance(record.get("source_type"), str)
        }
        missing = sorted(set(required_source_types) - actual_source_types)
        if missing:
            raise EvaluationRunError(
                f"{scenario} source mix is incomplete: required {required_source_types}, "
                f"present {sorted(actual_source_types)}, missing {missing}"
            )


def run_records(
    *,
    root: Path,
    records: list[dict[str, Any]],
    runs_dir: Path,
    run_id: str,
    mode: str,
    prompt_version: str,
    runtime: dict[str, Any],
    run_scope: str = "framework",
    dataset_version: str | None = None,
    artifact_hashes: dict[str, str] | None = None,
    mock_outputs: dict[str, str] | None = None,
    transport: ModelTransport | None = None,
) -> dict[str, Any]:
    """Render and persist one immutable run over eligible manifest records."""
    if mode not in {"mock", "dry-run", "live"}:
        raise EvaluationRunError("mode must be mock, dry-run, or live")
    if prompt_version not in {"baseline_minimal_v1", "standardized_v1"}:
        raise EvaluationRunError("unsupported prompt_version")
    _validate_runtime(runtime)

    project_root = Path(root)
    selected = select_inference_records(records)
    resolved_dataset_version = dataset_version or _infer_dataset_version(records)
    record_model_config = {
        "model": copy.deepcopy(runtime["model_config"]),
        "served_model_name": runtime["served_model_name"],
        "generation": copy.deepcopy(runtime["generation"]),
        "live_base_url": runtime["live_base_url"],
        "timeout_seconds": runtime["timeout_seconds"],
    }
    metadata = {
        "run_id": run_id,
        "mode": mode,
        "prompt_version": prompt_version,
        "run_scope": run_scope,
        "model_name": runtime["model_name"],
        "model_config": record_model_config,
        "dataset_version": resolved_dataset_version,
        "artifact_hashes": copy.deepcopy(artifact_hashes or {"direct_run": "0" * 64}),
        "selected_sample_ids_sha256": canonical_sha256(
            [record["sample_id"] for record in selected]
        ),
        "selected_count": len(selected),
    }
    active_transport = transport or _request_chat_completion

    with ImmutableRunWriter(Path(runs_dir), run_id, metadata) as writer:
        for record in selected:
            scenario = record["scenario"]
            input_metadata = copy.deepcopy(record["input"])
            rendered = _render_request(
                project_root,
                scenario,
                input_metadata,
                prompt_version,
            )
            request_sha256 = canonical_sha256(rendered)
            started = time.perf_counter()
            raw_output: str | None = None
            parsed_output: Any = None
            json_valid = False
            schema_valid = False
            error: str | None = None

            if mode == "dry-run":
                error = "dry_run"
            elif mode == "mock" and (
                mock_outputs is None or record["sample_id"] not in mock_outputs
            ):
                error = (
                    "mock_fixture_missing: no raw output for sample_id "
                    f"{record['sample_id']}"
                )
            else:
                try:
                    if mode == "mock":
                        assert mock_outputs is not None
                        raw_output = mock_outputs[record["sample_id"]]
                    else:
                        payload = _build_chat_payload(project_root, rendered, runtime)
                        endpoint = (
                            runtime["live_base_url"].rstrip("/")
                            + "/v1/chat/completions"
                        )
                        raw_output = active_transport(
                            endpoint,
                            payload,
                            runtime["timeout_seconds"],
                        )
                    if not isinstance(raw_output, str):
                        raise EvaluationRunError("model transport must return text")
                    parsed = parse_and_validate_output(
                        project_root,
                        scenario,
                        raw_output,
                    )
                    parsed_output = parsed["parsed_output"]
                    json_valid = parsed["json_valid"]
                    schema_valid = parsed["schema_valid"]
                    error = parsed["error"]
                except Exception as exc:
                    raw_output = None
                    parsed_output = None
                    json_valid = False
                    schema_valid = False
                    error = f"model_request_error: {type(exc).__name__}: {exc}"

            latency_ms = (time.perf_counter() - started) * 1000
            writer.write(
                {
                    "run_id": run_id,
                    "sample_id": record["sample_id"],
                    "scenario": scenario,
                    "mode": mode,
                    "model_name": runtime["model_name"],
                    "model_config": record_model_config,
                    "prompt_version": prompt_version,
                    "request_sha256": request_sha256,
                    "input_metadata": input_metadata,
                    "raw_output": raw_output,
                    "parsed_output": parsed_output,
                    "json_valid": json_valid,
                    "schema_valid": schema_valid,
                    "latency_ms": latency_ms,
                    "error": error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    return json.loads(
        (Path(runs_dir) / run_id / "metadata.json").read_text(encoding="utf-8")
    )


def _render_request(
    root: Path,
    scenario: str,
    input_metadata: dict[str, Any],
    prompt_version: str,
) -> dict[str, Any]:
    if prompt_version == "baseline_minimal_v1":
        return render_baseline_request(root, scenario, input_metadata)
    return render_standard_prompt(root, scenario, input_metadata)


def _infer_dataset_version(records: list[dict[str, Any]]) -> str:
    versions = {
        record.get("dataset_version")
        for record in records
        if isinstance(record.get("dataset_version"), str)
        and record["dataset_version"].strip()
    }
    if len(versions) == 1:
        return versions.pop()
    if not records:
        return "unspecified_direct_run"
    raise EvaluationRunError("records must use one non-empty dataset_version")


def _build_chat_payload(
    root: Path,
    rendered: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    messages = copy.deepcopy(rendered["messages"])
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            url = part["image_url"]["url"]
            part["image_url"]["url"] = normalize_image_url(
                _resolve_local_image_url(root, url)
            )
    return {
        "model": runtime["served_model_name"],
        "messages": messages,
        **copy.deepcopy(runtime["generation"]),
    }


def _resolve_local_image_url(root: Path, url: str) -> str:
    if not url.startswith("file://"):
        return url
    raw_path = url[len("file://") :]
    path = Path(raw_path)
    if path.is_absolute():
        return path.as_uri()
    return (Path(root) / path).resolve().as_uri()


def _request_chat_completion(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> str:
    response = requests.post(url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise EvaluationRunError("chat completion content must be text")
    return content


def _validate_runtime(runtime: dict[str, Any]) -> None:
    if not isinstance(runtime, dict):
        raise EvaluationRunError("runtime must be a mapping")
    for field in ("model_name", "served_model_name", "live_base_url"):
        value = runtime.get(field)
        if not isinstance(value, str) or not value.strip():
            raise EvaluationRunError(f"runtime.{field} must be non-empty text")
    if not isinstance(runtime.get("model_config"), dict):
        raise EvaluationRunError("runtime.model_config must be a mapping")
    if not isinstance(runtime.get("generation"), dict):
        raise EvaluationRunError("runtime.generation must be a mapping")
    timeout = runtime.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise EvaluationRunError("runtime.timeout_seconds must be positive")


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationRunError(f"{label} does not exist: {path}")
    payload = parse_simple_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvaluationRunError(f"{label} must be a mapping")
    return payload

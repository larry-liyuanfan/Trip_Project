"""Raw output parsing and immutable Week 3 evaluation result storage."""

import json
import math
import re
from pathlib import Path
from typing import Any, TextIO

from src.evaluation.scenarios import SCENARIOS
from src.evaluation.provenance import SHA256_PATTERN
from src.evaluation.schema_validation import SchemaValidationError, validate_output
from src.inference.client import strip_json_fence


RESULT_FIELDS = {
    "run_id",
    "sample_id",
    "scenario",
    "mode",
    "model_name",
    "model_config",
    "prompt_version",
    "request_sha256",
    "input_metadata",
    "raw_output",
    "parsed_output",
    "json_valid",
    "schema_valid",
    "latency_ms",
    "error",
    "timestamp",
}
RUN_METADATA_FIELDS = {
    "run_id",
    "mode",
    "prompt_version",
    "model_name",
    "model_config",
    "dataset_version",
    "artifact_hashes",
    "selected_sample_ids_sha256",
    "selected_count",
    "status",
    "record_count",
    "error",
}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ResultValidationError(ValueError):
    """Raised when a run ID or persisted result violates its contract."""


class RunAlreadyExistsError(FileExistsError):
    """Raised when an immutable run directory already exists."""


def parse_and_validate_output(
    root: Path,
    scenario: str,
    raw_output: str,
) -> dict[str, Any]:
    """Parse raw model text and report JSON and Schema validity separately."""
    try:
        parsed = json.loads(
            strip_json_fence(raw_output),
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "parsed_output": None,
            "json_valid": False,
            "schema_valid": False,
            "error": f"json_parse_error: {exc}",
        }

    try:
        validate_output(root, scenario, parsed)
    except SchemaValidationError as exc:
        return {
            "parsed_output": parsed,
            "json_valid": True,
            "schema_valid": False,
            "error": f"schema_validation_error: {exc}",
        }
    return {
        "parsed_output": parsed,
        "json_valid": True,
        "schema_valid": True,
        "error": None,
    }


def validate_result_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate the traceability fields required on every persisted result."""
    if not isinstance(record, dict):
        raise ResultValidationError("result record must be a JSON object")
    missing = sorted(RESULT_FIELDS - record.keys())
    if missing:
        raise ResultValidationError(f"result record missing required fields: {missing}")

    for field in ("run_id", "sample_id", "model_name", "prompt_version", "timestamp"):
        _require_text(record[field], field)
    if SHA256_PATTERN.fullmatch(record["request_sha256"]) is None:
        raise ResultValidationError("request_sha256 must be a lowercase SHA-256")
    _validate_run_id(record["run_id"])
    if record["scenario"] not in SCENARIOS:
        raise ResultValidationError("result scenario is unsupported")
    if record["mode"] not in {"mock", "dry-run", "live"}:
        raise ResultValidationError("result mode must be mock, dry-run, or live")
    if not isinstance(record["model_config"], dict):
        raise ResultValidationError("model_config must be an object")
    if not isinstance(record["input_metadata"], dict):
        raise ResultValidationError("input_metadata must be an object")
    if record["raw_output"] is not None and not isinstance(record["raw_output"], str):
        raise ResultValidationError("raw_output must be a string or null")
    for field in ("json_valid", "schema_valid"):
        if not isinstance(record[field], bool):
            raise ResultValidationError(f"{field} must be boolean")
    latency = record["latency_ms"]
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
        raise ResultValidationError("latency_ms must be a non-negative number")
    error = record["error"]
    if error is not None and (not isinstance(error, str) or not error.strip()):
        raise ResultValidationError("error must be null or non-empty text")
    if record["schema_valid"] and not record["json_valid"]:
        raise ResultValidationError("schema_valid requires json_valid")
    if record["schema_valid"] and error is not None:
        raise ResultValidationError("schema-valid results cannot contain an error")
    _reject_non_finite_numbers(record)
    return dict(record)


def load_run_metadata(path: Path) -> dict[str, Any]:
    """Strictly load and validate one immutable run metadata document."""
    metadata_path = Path(path)
    try:
        payload = json.loads(
            metadata_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise ResultValidationError(f"cannot load metadata.json: {exc}") from exc
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ResultValidationError(f"invalid metadata.json: {exc}") from exc
    return validate_run_metadata(payload)


def validate_run_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate the traceability, completion, and count fields used by scoring."""
    if not isinstance(metadata, dict):
        raise ResultValidationError("metadata.json must contain a JSON object")
    missing = sorted(RUN_METADATA_FIELDS - metadata.keys())
    if missing:
        raise ResultValidationError(
            f"metadata.json missing required fields: {missing}"
        )

    _require_text(metadata["run_id"], "metadata.run_id")
    _validate_run_id(metadata["run_id"])
    if metadata["mode"] not in {"mock", "dry-run", "live"}:
        raise ResultValidationError("metadata.mode must be mock, dry-run, or live")
    for field in ("prompt_version", "model_name"):
        _require_text(metadata[field], f"metadata.{field}")
    _require_text(metadata["dataset_version"], "metadata.dataset_version")
    if SHA256_PATTERN.fullmatch(metadata["selected_sample_ids_sha256"]) is None:
        raise ResultValidationError(
            "metadata.selected_sample_ids_sha256 must be a lowercase SHA-256"
        )
    artifact_hashes = metadata["artifact_hashes"]
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ResultValidationError("metadata.artifact_hashes must be a non-empty object")
    for path, digest in artifact_hashes.items():
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ResultValidationError("metadata.artifact_hashes has an invalid entry")
    if not isinstance(metadata["model_config"], dict):
        raise ResultValidationError("metadata.model_config must be an object")
    for field in ("selected_count", "record_count"):
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ResultValidationError(
                f"metadata.{field} must be a non-negative integer"
            )
    if metadata["status"] not in {"completed", "failed"}:
        raise ResultValidationError("metadata.status must be completed or failed")
    error = metadata["error"]
    if metadata["status"] == "completed" and error is not None:
        raise ResultValidationError("completed metadata.error must be null")
    if metadata["status"] == "failed" and (
        not isinstance(error, str) or not error.strip()
    ):
        raise ResultValidationError("failed metadata.error must be non-empty text")
    _reject_non_finite_numbers(metadata)
    return dict(metadata)


class ImmutableRunWriter:
    """Incrementally write one run into a newly-created immutable directory."""

    def __init__(
        self,
        runs_dir: Path,
        run_id: str,
        metadata: dict[str, Any],
    ) -> None:
        _validate_run_id(run_id)
        if not isinstance(metadata, dict) or metadata.get("run_id") != run_id:
            raise ResultValidationError("metadata.run_id must match run_id")
        self.run_id = run_id
        self.metadata = dict(metadata)
        self.record_count = 0
        self._closed = False

        runs_root = Path(runs_dir)
        runs_root.mkdir(parents=True, exist_ok=True)
        self.run_dir = runs_root / run_id
        try:
            self.run_dir.mkdir()
        except FileExistsError as exc:
            raise RunAlreadyExistsError(f"run_id already exists: {run_id}") from exc
        self._handle: TextIO = (self.run_dir / "results.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        )

    def write(self, record: dict[str, Any]) -> None:
        """Validate, append, and flush one result row."""
        if self._closed:
            raise ResultValidationError("run writer is already closed")
        validated = validate_result_record(record)
        if validated["run_id"] != self.run_id:
            raise ResultValidationError("result run_id does not match writer run_id")
        try:
            serialized = json.dumps(
                validated,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        except ValueError as exc:
            raise ResultValidationError(f"result contains non-finite number: {exc}") from exc
        self._handle.write(serialized + "\n")
        self._handle.flush()
        self.record_count += 1

    def close(self, *, status: str = "completed", error: str | None = None) -> None:
        """Close the JSONL and write final metadata exactly once."""
        if self._closed:
            return
        self._handle.close()
        summary = {
            **self.metadata,
            "status": status,
            "record_count": self.record_count,
            "error": error,
        }
        with (self.run_dir / "metadata.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            try:
                json.dump(
                    summary,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            except ValueError as exc:
                raise ResultValidationError(
                    f"run metadata contains non-finite number: {exc}"
                ) from exc
            handle.write("\n")
        self._closed = True

    def __enter__(self) -> "ImmutableRunWriter":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        if exc is None:
            self.close()
        else:
            self.close(status="failed", error=f"{type(exc).__name__}: {exc}")
        return False


def _validate_run_id(run_id: Any) -> None:
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ResultValidationError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )


def _require_text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ResultValidationError(f"{field} must be non-empty text")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _reject_non_finite_numbers(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ResultValidationError(f"{path} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_non_finite_numbers(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_non_finite_numbers(child, path=f"{path}[{index}]")

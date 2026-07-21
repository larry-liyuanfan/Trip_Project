"""Run the Week 3 evaluation framework in mock, dry-run, or live mode."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.runner import EvaluationRunError, run_configured_evaluation


def load_mock_outputs(path: Path) -> dict[str, str]:
    """Load explicit sample-to-raw-output mappings from UTF-8 JSONL."""
    outputs: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as exc:
                raise EvaluationRunError(
                    f"invalid mock JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise EvaluationRunError(
                    f"mock response line {line_number} must be an object"
                )
            sample_id = row.get("sample_id")
            raw_output = row.get("raw_output")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise EvaluationRunError(
                    f"mock response line {line_number} requires sample_id"
                )
            if not isinstance(raw_output, str):
                raise EvaluationRunError(
                    f"mock response line {line_number} requires raw_output text"
                )
            if sample_id in outputs:
                raise EvaluationRunError(f"duplicate mock sample_id: {sample_id}")
            outputs[sample_id] = raw_output
    return outputs


def _reject_json_constant(value: str) -> None:
    raise EvaluationRunError(
        f"non-finite JSON constant is not allowed in mock responses: {value}"
    )


def run_cli(
    argv: list[str] | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Parse CLI arguments, execute the configured run, and print its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week3.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("mock", "dry-run", "live"), required=True)
    parser.add_argument(
        "--prompt-version",
        choices=("baseline_minimal_v1", "standardized_v1"),
        required=True,
    )
    parser.add_argument(
        "--run-scope",
        choices=("framework", "pilot", "full"),
        default="framework",
    )
    parser.add_argument("--mock-responses")
    parser.add_argument("--base-url")
    args = parser.parse_args(argv)

    project_root = Path(root) if root is not None else Path.cwd()
    mock_outputs = None
    if args.mode == "mock":
        if not args.mock_responses:
            raise EvaluationRunError("mock mode requires --mock-responses")
        mock_path = Path(args.mock_responses)
        if not mock_path.is_absolute():
            mock_path = project_root / mock_path
        mock_outputs = load_mock_outputs(mock_path)
    elif args.mock_responses:
        raise EvaluationRunError("--mock-responses is only valid in mock mode")

    summary = run_configured_evaluation(
        root=project_root,
        config_path=Path(args.config),
        run_id=args.run_id,
        mode=args.mode,
        prompt_version=args.prompt_version,
        run_scope=args.run_scope,
        mock_outputs=mock_outputs,
        live_base_url=args.base_url,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()

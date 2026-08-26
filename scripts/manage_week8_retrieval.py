#!/usr/bin/env python3
"""Build and evaluate the Week 8 isolated retrieval relevance protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.week8_relevance import (
    Week8RetrievalError,
    build_data_lock,
    claim_final_test,
    complete_final_test,
    evaluate_partition,
    load_config,
    load_retrieval_source,
    select_development_method,
    sha256_file,
    validate_data_lock,
    validate_development_selection,
    write_evaluation,
)


DEFAULT_CONFIG = ROOT / "configs/week8/retrieval_relevance_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-lock")
    _add_common_source_arguments(build)
    build.add_argument("--source-project-root", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)

    validate = subparsers.add_parser("validate-lock")
    validate.add_argument("--lock-dir", required=True, type=Path)

    development = subparsers.add_parser("evaluate-development")
    _add_common_source_arguments(development)
    development.add_argument("--lock-dir", required=True, type=Path)
    development.add_argument("--output-dir", required=True, type=Path)

    final_test = subparsers.add_parser("evaluate-final-test")
    _add_common_source_arguments(final_test)
    final_test.add_argument("--lock-dir", required=True, type=Path)
    final_test.add_argument("--selection", required=True, type=Path)
    final_test.add_argument("--output-dir", required=True, type=Path)
    final_test.add_argument("--consumption-marker", required=True, type=Path)
    return parser


def _add_common_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--retrieval-archive", type=Path)
    source.add_argument("--retrieval-dir", type=Path)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-lock":
        manifest, rows = validate_data_lock(args.lock_dir)
        _print(
            {
                "status": "PASS",
                "dataset_version": manifest["dataset_version"],
                "counts": {name: len(partition_rows) for name, partition_rows in rows.items()},
                "five_dimension_isolation": manifest["five_dimension_isolation"],
            }
        )
        return

    config = load_config(args.config)
    vectors, metadata, source_hashes = load_retrieval_source(
        config,
        archive_path=args.retrieval_archive,
        retrieval_dir=args.retrieval_dir,
    )
    if args.command == "build-lock":
        _print(
            build_data_lock(
                config,
                vectors,
                metadata,
                source_hashes,
                source_project_root=args.source_project_root,
                output_dir=args.output_dir,
            )
        )
        return

    manifest, rows_by_partition = validate_data_lock(args.lock_dir)
    if manifest.get("source_hashes") != source_hashes:
        raise Week8RetrievalError("runtime retrieval source does not match the locked source")

    if args.command == "evaluate-development":
        metrics, results, references = evaluate_partition(
            config,
            vectors,
            metadata,
            rows_by_partition,
            "development_query",
        )
        selection = select_development_method(config, metrics)
        hashes = write_evaluation(
            args.output_dir,
            partition="development_query",
            metrics=metrics,
            results=results,
            references=references,
            selection=selection,
            data_lock_sha256=sha256_file(args.lock_dir / "dataset_lock.json"),
            source_hashes=source_hashes,
        )
        _print({"status": "COMPLETED", "selection": selection, "hashes": hashes})
        return

    selection = validate_development_selection(
        config,
        args.selection,
        lock_dir=args.lock_dir,
        source_hashes=source_hashes,
    )
    selected_method = selection.get("selected_method")
    claim_final_test(
        args.consumption_marker,
        selection,
        selection_sha256=sha256_file(args.selection),
    )
    methods = ["clip"] if selected_method == "clip" else ["clip", selected_method]
    metrics, results, references = evaluate_partition(
        config,
        vectors,
        metadata,
        rows_by_partition,
        "final_test_query",
        methods=methods,
    )
    hashes = write_evaluation(
        args.output_dir,
        partition="final_test_query",
        metrics=metrics,
        results=results,
        references=references,
    )
    completed_marker = complete_final_test(args.consumption_marker, hashes)
    _print(
        {
            "status": "COMPLETED",
            "selected_method": selected_method,
            "metrics": metrics,
            "hashes": hashes,
            "final_test_marker": completed_marker,
        }
    )


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

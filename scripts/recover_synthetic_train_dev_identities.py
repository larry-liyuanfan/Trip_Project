"""Recover hash-bound identity-only rows for deterministic train/dev pools.

The recovery is deliberately narrower than the original pool builders: it invokes
only the training and development split builders, verifies their exact committed
manifest locks, and exports flat identity metadata.  It never builds or opens a
final/test split and it is not a training authorization.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_context_focus_pool_v5 import _build_split as build_v5_split
from scripts.build_exploration_pool_v4 import (
    _build_search_split as build_v4_search_split,
    _build_vlm_split as build_v4_vlm_split,
    _write_jsonl as write_generator_jsonl,
)
from scripts.build_semantic_robustness_pool_v7 import _build_split as build_v7_split
from scripts.build_semantic_robustness_pool_v9 import _build_split as build_v9_split
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256


ALLOWED_SPLITS = ("training", "development")
RECOVERY_SCHEMA = "synthetic_train_dev_identity_recovery_v1"
QUERY_CANONICALIZATION = "canonical_json_utf8_sort_keys_compact_v1"
VLM_QUERY_SCHEMA = "trip_vlm_query_identity_v1"
OVERLAP_FIELDS = (
    "sample_id",
    "source_id",
    "image_sha256",
    "query_sha256",
    "source_record_sha256",
    "dialogue_text_sha256",
    "content_sha256",
)
SOURCE_FILES = {
    "v4": "scripts/build_exploration_pool_v4.py",
    "v5": "scripts/build_context_focus_pool_v5.py",
    "v7": "scripts/build_semantic_robustness_pool_v7.py",
    "v9": "scripts/build_semantic_robustness_pool_v9.py",
}
LOCK_FILES = {
    "v4": "configs/evaluation/evidence_enhancement/exploration_pool_lock_v4.json",
    "v5": "configs/evaluation/evidence_enhancement/context_focus_pool_lock_v5.json",
    "v7": "configs/evaluation/evidence_enhancement/semantic_robustness_pool_lock_v7.json",
    "v9": "configs/evaluation/evidence_enhancement/semantic_robustness_pool_lock_v9.json",
}


def _require_allowed_split(split: str) -> None:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"recovery forbids non-train/dev split: {split}")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"missing or invalid {label}")
    return value


def _search_identity(row: dict[str, Any], dataset_version: str) -> dict[str, Any]:
    split = str(row.get("split", ""))
    _require_allowed_split(split)
    image = row.get("image")
    source = row.get("source")
    if not isinstance(image, dict) or not isinstance(source, dict):
        raise ValueError("search row lacks generator-bound image or source identity")
    image_sha256 = _sha(image.get("sha256"), "search image_sha256")
    return {
        "sample_id": str(row["query_id"]),
        "source_id": str(source["source_id"]),
        "image_sha256": image_sha256,
        "query_sha256": _sha(row.get("query_sha256"), "search query_sha256"),
        "source_record_sha256": _sha(
            row.get("source_record_sha256"), "search source_record_sha256"
        ),
        "content_sha256": image_sha256,
        "split": split,
        "scenario": "search",
        "dataset_version": dataset_version,
        "contains_image_bytes": False,
    }


def _vlm_identity(row: dict[str, Any], dataset_version: str) -> dict[str, Any]:
    split = str(row.get("split", ""))
    _require_allowed_split(split)
    scenario = str(row.get("scenario", ""))
    if scenario == "product":
        image_sha256 = _sha(row.get("image_sha256"), "product image_sha256")
        content_sha256 = image_sha256
        modality_identity = {"image_sha256": image_sha256}
    elif scenario == "dialogue":
        dialogue = row.get("dialogue")
        if not isinstance(dialogue, str) or not dialogue:
            raise ValueError("dialogue row lacks text needed for identity recovery")
        derived = canonical_json_sha256({"dialogue": dialogue})
        recorded = row.get("dialogue_text_sha256")
        if recorded is not None and _sha(recorded, "dialogue_text_sha256") != derived:
            raise ValueError("recorded dialogue_text_sha256 differs from generated dialogue")
        content_sha256 = derived
        modality_identity = {"dialogue_text_sha256": derived}
    else:
        raise ValueError(f"unsupported VLM scenario: {scenario}")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("VLM row lacks its actual prompt")
    query_sha256 = canonical_json_sha256(
        {
            "schema_version": VLM_QUERY_SCHEMA,
            "scenario": scenario,
            "prompt": prompt,
            "content_sha256": content_sha256,
        }
    )
    return {
        "sample_id": str(row["sample_id"]),
        "source_id": str(row["source_id"]),
        **modality_identity,
        "query_sha256": query_sha256,
        "source_record_sha256": _sha(
            row.get("source_record_sha256"), "VLM source_record_sha256"
        ),
        "content_sha256": content_sha256,
        "split": split,
        "scenario": scenario,
        "dataset_version": dataset_version,
        "contains_image_bytes": False,
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _cross_scope_overlap_check(
    scopes: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    owners: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in OVERLAP_FIELDS
    }
    for scope in scopes:
        scope_id = scope["scope_id"]
        path = output_dir / scope["identity_path"]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for field in OVERLAP_FIELDS:
                value = row.get(field)
                if value:
                    owners[field][str(value)].add(scope_id)
    counts = {
        field: sum(len(scope_ids) > 1 for scope_ids in values.values())
        for field, values in owners.items()
    }
    if any(counts.values()):
        raise ValueError(f"cross-scope identity collision: {counts}")
    return {
        "status": "PASS",
        "scope_denominator": len(scopes),
        "collision_value_counts": counts,
    }


def _verify_locked_rows(
    rows: list[dict[str, Any]], temp_manifest: Path, expected: dict[str, Any]
) -> None:
    write_generator_jsonl(temp_manifest, rows)
    if len(rows) != expected["sample_support"]:
        raise ValueError("regenerated row support differs from committed lock")
    if canonical_json_sha256(rows) != expected["manifest_canonical_sha256"]:
        raise ValueError("regenerated canonical manifest differs from committed lock")
    if file_sha256(temp_manifest) != expected["manifest_file_sha256"]:
        raise ValueError("regenerated manifest file differs from committed lock")


def _write_scope(
    *,
    output_dir: Path,
    scope_id: str,
    rows: list[dict[str, Any]],
    cycle: str,
    split: str,
    kind: str,
    source_manifest_lock: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("recovered identity scope cannot be empty")
    path = output_dir / f"{scope_id}.jsonl"
    _write_jsonl(path, rows)
    generator_path = repo_root / SOURCE_FILES[cycle]
    lock_path = repo_root / LOCK_FILES[cycle]
    return {
        "scope_id": scope_id,
        "cycle": cycle,
        "split": split,
        "kind": kind,
        "status": "RECOVERED_LOCK_VERIFIED_TRAIN_DEV_ONLY",
        "row_count": len(rows),
        "identity_path": path.name,
        "identity_file_sha256": file_sha256(path),
        "identity_canonical_rows_sha256": canonical_json_sha256(rows),
        "source_manifest_file_sha256": source_manifest_lock["manifest_file_sha256"],
        "source_manifest_canonical_sha256": source_manifest_lock[
            "manifest_canonical_sha256"
        ],
        "generator_path": SOURCE_FILES[cycle],
        "generator_file_sha256": file_sha256(generator_path),
        "committed_lock_path": LOCK_FILES[cycle],
        "committed_lock_file_sha256": file_sha256(lock_path),
    }


def _recover_v4(
    output_dir: Path, temp_root: Path, repo_root: Path, lock: dict[str, Any]
) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        split_root = temp_root / "v4" / split
        split_root.mkdir(parents=True)
        registry: list[dict[str, Any]] = []
        queries, _annotations = build_v4_search_split(split_root, split, registry)
        query_manifest = split_root / f"search_{split}_manifest.jsonl"
        write_generator_jsonl(query_manifest, queries)
        expected_search = lock["search"][split]
        if len(queries) != expected_search["query_support"]:
            raise ValueError("regenerated v4 search support differs from committed lock")
        if canonical_json_sha256(queries) != expected_search["query_manifest_canonical_sha256"]:
            raise ValueError("regenerated v4 search canonical manifest differs from committed lock")
        if file_sha256(query_manifest) != expected_search["query_manifest_file_sha256"]:
            raise ValueError("regenerated v4 search file differs from committed lock")
        search_rows = [_search_identity(row, "trip_exploration_pool_v4") for row in queries]
        search_lock = {
            "manifest_file_sha256": expected_search["query_manifest_file_sha256"],
            "manifest_canonical_sha256": expected_search["query_manifest_canonical_sha256"],
        }
        scopes.append(
            _write_scope(
                output_dir=output_dir,
                scope_id=f"v4_search_{split}",
                rows=search_rows,
                cycle="v4",
                split=split,
                kind="search",
                source_manifest_lock=search_lock,
                repo_root=repo_root,
            )
        )

        vlm_rows = build_v4_vlm_split(split_root, split, registry)
        _verify_locked_rows(
            vlm_rows,
            split_root / f"vlm_{split}_manifest.jsonl",
            lock["vlm"][split],
        )
        scopes.append(
            _write_scope(
                output_dir=output_dir,
                scope_id=f"v4_vlm_{split}",
                rows=[_vlm_identity(row, "trip_exploration_pool_v4") for row in vlm_rows],
                cycle="v4",
                split=split,
                kind="vlm",
                source_manifest_lock=lock["vlm"][split],
                repo_root=repo_root,
            )
        )
    return scopes


def _recover_vlm_cycle(
    *,
    output_dir: Path,
    temp_root: Path,
    repo_root: Path,
    cycle: str,
    dataset_version: str,
    builder: Callable[[Path, str, list[dict[str, Any]]], list[dict[str, Any]]],
    lock: dict[str, Any],
) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        split_root = temp_root / cycle / split
        split_root.mkdir(parents=True)
        registry: list[dict[str, Any]] = []
        rows = builder(split_root, split, registry)
        _verify_locked_rows(
            rows,
            split_root / f"vlm_{split}_manifest.jsonl",
            lock["vlm"][split],
        )
        scopes.append(
            _write_scope(
                output_dir=output_dir,
                scope_id=f"{cycle}_vlm_{split}",
                rows=[_vlm_identity(row, dataset_version) for row in rows],
                cycle=cycle,
                split=split,
                kind="vlm",
                source_manifest_lock=lock["vlm"][split],
                repo_root=repo_root,
            )
        )
    return scopes


def recover(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    locks = {
        cycle: json.loads((repo_root / relative).read_text(encoding="utf-8"))
        for cycle, relative in LOCK_FILES.items()
    }
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        with tempfile.TemporaryDirectory(prefix="trip-identity-recovery-") as temp:
            temp_root = Path(temp)
            scopes = _recover_v4(output_dir, temp_root, repo_root, locks["v4"])
            for cycle, version, builder in (
                ("v5", "trip_context_focus_pool_v5", build_v5_split),
                ("v7", "trip_semantic_robustness_pool_v7", build_v7_split),
                ("v9", "trip_semantic_robustness_pool_v9", build_v9_split),
            ):
                scopes.extend(
                    _recover_vlm_cycle(
                        output_dir=output_dir,
                        temp_root=temp_root,
                        repo_root=repo_root,
                        cycle=cycle,
                        dataset_version=version,
                        builder=builder,
                        lock=locks[cycle],
                    )
                )
        manifest = {
            "schema_version": RECOVERY_SCHEMA,
            "status": "PARTIAL_RECOVERY_NOT_TRAINING_AUTHORIZATION",
            "git_sha": git_sha,
            "canonicalization": {
                "json": QUERY_CANONICALIZATION,
                "vlm_query_schema": VLM_QUERY_SCHEMA,
                "product_content_sha256": "raw_image_bytes_sha256",
                "dialogue_content_sha256": "canonical_json_sha256_of_dialogue_field",
                "source_record_sha256": "copied_from_exact_lock_verified_generator_row",
            },
            "allowed_splits": list(ALLOWED_SPLITS),
            "forbidden_splits_generated_or_opened": [],
            "temporary_generated_assets_retained": False,
            "human_annotation_support": 0,
            "scope_support": len(scopes),
            "row_support": sum(scope["row_count"] for scope in scopes),
            "cross_scope_identity_check": _cross_scope_overlap_check(scopes, output_dir),
            "scopes": scopes,
            "training_authorized": False,
            "coordinator_approval_required": True,
        }
        _write_json(output_dir / "recovery_manifest.json", manifest)
        return manifest
    except Exception:
        for path in output_dir.glob("*"):
            path.unlink()
        output_dir.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = recover(args.output_dir.resolve(), repo_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

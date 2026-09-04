"""Build a leak-isolated synthetic v9 pool focused on multi-subject abstention."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_exploration_pool_v4 import HEIGHT, WIDTH, _write_card, _write_json, _write_jsonl
from scripts.build_semantic_robustness_pool_v7 import (
    DIALOGUE_PROMPT,
    PRODUCT_PROMPT,
    _clear_product,
    _conflict_product,
    _dialogue_rows as _v7_dialogue_rows,
    _identities,
    _load_prior_v5_training,
    _negated_product,
    _slice_support,
    _unknown_product,
    build_pool as build_v7_pool,
)
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl


GENERATOR_VERSION = "trip_semantic_robustness_pool_v9"
SPLITS = ("training", "development")
COUNTS = {
    "training": {
        "clear": 96,
        "unknown": 48,
        "multi": 128,
        "negated": 48,
        "conflict": 64,
        "dialogue": 256,
    },
    "development": {
        "clear": 24,
        "unknown": 12,
        "multi": 24,
        "negated": 12,
        "conflict": 12,
        "dialogue": 48,
    },
}
SPLIT_OFFSET = {"training": 11000, "development": 13000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-v5-lock", type=Path, required=True)
    parser.add_argument("--prior-v7-lock", type=Path, required=True)
    parser.add_argument("--expected-lock", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    lock = build_pool(args.output_dir, args.prior_v5_lock, args.prior_v7_lock)
    if args.expected_lock:
        expected = json.loads(args.expected_lock.read_text(encoding="utf-8"))
        if lock != expected:
            raise ValueError("generated v9 robustness pool differs from committed lock")
    print(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True))


def build_pool(
    output_dir: Path,
    prior_v5_lock_path: Path,
    prior_v7_lock_path: Path,
) -> dict[str, Any]:
    prior_v5_rows, prior_v5_lock = _load_prior_v5_training(prior_v5_lock_path)
    prior_v7_lock = json.loads(prior_v7_lock_path.read_text(encoding="utf-8"))
    if prior_v7_lock.get("schema_version") != "semantic_robustness_pool_lock_v7":
        raise ValueError("prior lock is not the v7 semantic-robustness pool lock")
    with tempfile.TemporaryDirectory(prefix="trip-v7-reference-") as temp_dir:
        reference_root = Path(temp_dir) / "v7"
        regenerated_v7_lock = build_v7_pool(reference_root, prior_v5_lock_path)
        if regenerated_v7_lock != prior_v7_lock:
            raise ValueError("regenerated v7 reference differs from its committed lock")
        prior_v7_rows = [
            *load_jsonl(reference_root / "vlm_training_manifest.jsonl"),
            *load_jsonl(reference_root / "vlm_development_manifest.jsonl"),
        ]

    output_dir.mkdir(parents=True)
    registry: list[dict[str, Any]] = []
    manifests: dict[str, list[dict[str, Any]]] = {}
    split_locks: dict[str, Any] = {}
    for split in SPLITS:
        rows = _build_split(output_dir, split, registry)
        manifests[split] = rows
        path = output_dir / f"vlm_{split}_manifest.jsonl"
        _write_jsonl(path, rows)
        split_locks[split] = {
            "sample_support": len(rows),
            "product_support": sum(row["scenario"] == "product" for row in rows),
            "dialogue_support": sum(row["scenario"] == "dialogue" for row in rows),
            "manifest_canonical_sha256": canonical_json_sha256(rows),
            "manifest_file_sha256": file_sha256(path),
            "slice_support": _slice_support(rows),
        }

    isolation = _isolation_evidence(manifests)
    prior_v5_isolation = _prior_cycle_isolation(manifests, prior_v5_rows, "v5_training")
    prior_v7_isolation = _prior_cycle_isolation(manifests, prior_v7_rows, "v7_training_and_development")
    lock = {
        "schema_version": "semantic_robustness_pool_lock_v9",
        "generator_version": GENERATOR_VERSION,
        "evidence_class": "deterministic_synthetic_programmatic_development_only",
        "human_annotation_support": 0,
        "image_encoding": "binary_ppm_p6_rgb_384x256",
        "primary_factor": "multi_subject_counterexample_training_composition_and_support_only",
        "prompt_fixed_within_comparison": True,
        "splits": list(SPLITS),
        "final_policy": "not_defined_not_generated_not_opened",
        "vlm": split_locks,
        "asset_registry_support": len(registry),
        "asset_registry_canonical_sha256": canonical_json_sha256(registry),
        "split_identity_policy": "source_id_image_sha256_sample_id_and_dialogue_text_sha256_disjoint",
        "isolation": isolation,
        "prior_v5_bundle_lock_sha256": canonical_json_sha256(prior_v5_lock),
        "prior_v5_training_isolation": prior_v5_isolation,
        "prior_v7_bundle_lock_sha256": canonical_json_sha256(prior_v7_lock),
        "prior_v7_training_and_development_isolation": prior_v7_isolation,
    }
    _write_json(output_dir / "asset_registry.json", registry)
    _write_json(output_dir / "bundle_lock.json", lock)
    return lock


def _build_split(output_dir: Path, split: str, registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = COUNTS[split]
    offset = SPLIT_OFFSET[split]
    rows: list[dict[str, Any]] = []
    builders = (
        ("clear", _clear_product),
        ("unknown", _unknown_product),
        ("multi", _multi_product_v9),
        ("negated", _negated_product),
        ("conflict", _conflict_product),
    )
    cursor = 0
    for kind, builder in builders:
        for index in range(counts[kind]):
            lines, gold, slices = builder(index, offset)
            lines[0] = "TRIP V9 ROBUST"
            lines.insert(1, f"SPLIT {split.upper()}")
            slices = ["v9_robustness" if name == "v7_robustness" else name for name in slices]
            sample_id = f"v9_{split[:3]}_{kind}_{index:03d}"
            rows.append(
                _product_row(
                    output_dir,
                    registry,
                    split,
                    sample_id,
                    lines,
                    gold,
                    slices,
                    offset + cursor,
                )
            )
            cursor += 1
    rows.extend(_dialogue_rows(split, counts["dialogue"]))
    return rows


def _multi_product_v9(index: int, offset: int) -> tuple[list[str], dict[str, Any], list[str]]:
    categories = ("restaurant", "hotel", "attraction")
    left = categories[(index + offset) % len(categories)]
    right = categories[(index + offset + 1) % len(categories)]
    return (
        [
            "TRIP V9 ROBUST",
            f"LEFT {left.upper()}",
            f"RIGHT {right.upper()}",
            "PRIMARY SUBJECT NONE",
            "FIELDS CONFLICT",
            "DO NOT MERGE",
        ],
        {
            "business_category": "unknown",
            "style_tags": [],
            "visible_facilities": [],
            "price_range": "unknown",
            "unknown_fields": [
                "business_category",
                "style_tags",
                "visible_facilities",
                "price_range",
            ],
        },
        [
            "multi_subject_conflict",
            "insufficient_visual_evidence",
            "price_unknown",
            "unknown_suppression",
            "v7_robustness",
        ],
    )


def _product_row(
    output_dir: Path,
    registry: list[dict[str, Any]],
    split: str,
    sample_id: str,
    lines: list[str],
    gold: dict[str, Any],
    slices: list[str],
    seed: int,
) -> dict[str, Any]:
    relative_path = f"assets/vlm/{split}/{sample_id}.ppm"
    path = output_dir / relative_path
    _write_card(path, lines, seed)
    image_sha = file_sha256(path)
    source_id = f"synthetic:{GENERATOR_VERSION}:{split}:{sample_id}"
    source = {
        "source_id": source_id,
        "generator": GENERATOR_VERSION,
        "split": split,
        "sample_id": sample_id,
    }
    registry.append(
        {
            "record_id": sample_id,
            "source_id": source_id,
            "relative_path": relative_path,
            "sha256": image_sha,
            "width": WIDTH,
            "height": HEIGHT,
            "purpose": "vlm",
            "split": split,
            "contains_image_bytes": False,
        }
    )
    return {
        "sample_id": sample_id,
        "split": split,
        "scenario": "product",
        "source_id": source_id,
        "source_record_sha256": canonical_json_sha256(source),
        "image_relative_path": relative_path,
        "image_sha256": image_sha,
        "slices": slices,
        "label_provenance": "synthetic_programmatic_card_v9_no_human_annotation",
        "prompt": PRODUCT_PROMPT,
        "gold": gold,
        "sample_weight": 1.0,
    }


def _dialogue_rows(split: str, count: int) -> list[dict[str, Any]]:
    rows = _v7_dialogue_rows(split, count)
    transformed: list[dict[str, Any]] = []
    for index, prior in enumerate(rows):
        dialogue = str(prior["dialogue"]).replace("V7", "V9")
        sample_id = f"v9_{split[:3]}_dialogue_{index:04d}"
        source_id = f"synthetic:{GENERATOR_VERSION}:{split}:dialogue:{index:04d}"
        transformed.append(
            {
                **prior,
                "sample_id": sample_id,
                "source_id": source_id,
                "source_record_sha256": canonical_json_sha256(
                    {"source_id": source_id, "dialogue": dialogue}
                ),
                "dialogue_text_sha256": canonical_json_sha256({"dialogue": dialogue}),
                "slices": [
                    "v9_robustness" if name == "v7_robustness" else name
                    for name in prior["slices"]
                ],
                "label_provenance": "synthetic_protocol_case_v9_no_human_annotation",
                "prompt": DIALOGUE_PROMPT,
                "dialogue": dialogue,
            }
        )
    return transformed


def _isolation_evidence(manifests: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    left, right = (_identities(manifests[split]) for split in SPLITS)
    overlaps = {kind: sorted(left[kind] & right[kind]) for kind in left}
    if any(overlaps.values()):
        raise ValueError(f"v9 split leakage: {overlaps}")
    return {"status": "PASS", "training_vs_development": overlaps}


def _prior_cycle_isolation(
    manifests: dict[str, list[dict[str, Any]]],
    prior_rows: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    prior = _identities(prior_rows)
    overlap = {
        split: {kind: sorted(values[kind] & prior[kind]) for kind in values}
        for split, values in ((name, _identities(rows)) for name, rows in manifests.items())
    }
    if any(items for split in overlap.values() for items in split.values()):
        raise ValueError(f"v9 overlaps prior {label}: {overlap}")
    return {"status": "PASS", "compared_prior_row_support": len(prior_rows), "overlaps": overlap}


if __name__ == "__main__":
    main()

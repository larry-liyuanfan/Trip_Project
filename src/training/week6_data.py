"""Deterministically lock Week 5 labels for Week 6 QLoRA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.data.week5_dataset import SCENARIOS, iter_jsonl, load_week5_config
from src.data.week5_workflow import _fewshot_context, _render_preannotation
from src.training.week6_qlora import Week6TrainingError, validate_training_row


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _split_name(sample_id: str, *, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()
    rank = int.from_bytes(digest, "big") / float(1 << 256)
    return "validation" if rank < validation_fraction else "train"


def _latest_human_annotations(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if int(row.get("revision", 0)) >= int(latest.get(sample_id, {}).get("revision", 0)):
            latest[sample_id] = row
    return latest


def _write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
                handle.write("\n")
                count += 1
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return count


def lock_week6_data(root: Path, training_config: dict[str, Any]) -> dict[str, Any]:
    """Create immutable train/validation JSONL plus manifest and split hashes."""
    dataset = training_config.get("dataset", {})
    required = {
        "dataset_version", "week5_config", "source_merge_dir", "output_root",
        "split_seed", "validation_fraction", "model_preannotation_weight",
    }
    missing = sorted(required - set(dataset))
    if missing:
        raise Week6TrainingError(f"Week 6 dataset config missing fields: {missing}")
    validation_fraction = float(dataset["validation_fraction"])
    if not 0 < validation_fraction < 0.5:
        raise Week6TrainingError("validation_fraction must be in (0, 0.5)")
    silver_weight = float(dataset["model_preannotation_weight"])
    if not 0 < silver_weight <= 0.5:
        raise Week6TrainingError("model preannotation weight must be in (0, 0.5]")

    output_dir = root / dataset["output_root"] / dataset["dataset_version"]
    if output_dir.exists():
        raise Week6TrainingError(f"refusing to overwrite locked dataset: {output_dir}")
    week5 = load_week5_config(root, dataset["week5_config"])
    week5_output = root / week5["paths"]["output_dir"]
    merge_dir = root / dataset["source_merge_dir"]
    required_merge = [merge_dir / name for name in ("summary.json", "results.jsonl", "failures.jsonl")]
    if any(not path.is_file() for path in required_merge):
        raise Week6TrainingError("final Week 5 merge artifacts are incomplete")
    summary = json.loads((merge_dir / "summary.json").read_text(encoding="utf-8"))
    if int(summary.get("unique_success", -1)) + int(summary.get("unresolved_failures", -1)) != 80000:
        raise Week6TrainingError("Week 5 merge does not close all 80,000 candidates")

    source_hashes: dict[str, str] = {
        path.relative_to(root).as_posix(): _sha256_file(path) for path in required_merge
    }
    for scenario in SCENARIOS:
        for category in ("pools", "preannotations", "annotations"):
            path = week5_output / category / f"{scenario}.jsonl"
            if not path.is_file():
                raise Week6TrainingError(f"required Week 5 artifact is missing: {path}")
            source_hashes[path.relative_to(root).as_posix()] = _sha256_file(path)

    assignments: dict[str, dict[str, str]] = {scenario: {} for scenario in SCENARIOS}
    split_lines: list[dict[str, str]] = []
    annotations_by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for scenario in SCENARIOS:
        annotations = _latest_human_annotations(
            week5_output / "annotations" / f"{scenario}.jsonl"
        )
        annotations_by_scenario[scenario] = annotations
        split_counts = {"train": 0, "validation": 0}
        for row in iter_jsonl(week5_output / "preannotations" / f"{scenario}.jsonl"):
            sample_id = str(row["sample_id"])
            split = _split_name(
                sample_id,
                seed=int(dataset["split_seed"]),
                validation_fraction=validation_fraction,
            )
            assignments[scenario][sample_id] = split
            split_counts[split] += 1
            split_lines.append({"sample_id": sample_id, "scenario": scenario, "split": split})
        if not set(annotations) <= set(assignments[scenario]):
            raise Week6TrainingError(f"human annotation lacks a valid preannotation: {scenario}")
        counts[scenario] = {
            **split_counts,
            "human_revised": len(annotations),
            "model_preannotation": sum(split_counts.values()) - len(annotations),
        }
    split_lines.sort(key=lambda row: (row["scenario"], row["sample_id"]))
    split_sha256 = hashlib.sha256(
        b"".join(_canonical_bytes(row) + b"\n" for row in split_lines)
    ).hexdigest()
    manifest_payload = {
        "schema_version": "week6_dataset_lock_v1",
        "dataset_version": dataset["dataset_version"],
        "source_week5_merge": dataset["source_merge_dir"],
        "source_hashes": source_hashes,
        "split": {
            "strategy": "sample_id_sha256_threshold_v1",
            "seed": int(dataset["split_seed"]),
            "validation_fraction": validation_fraction,
            "split_sha256": split_sha256,
        },
        "label_policy": {
            "model_preannotation_weight": silver_weight,
            "human_revised_weight": 1.0,
        },
        "counts": counts,
    }
    manifest_sha256 = hashlib.sha256(_canonical_bytes(manifest_payload)).hexdigest()
    dataset_lock = {
        "dataset_version": dataset["dataset_version"],
        "manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
    }

    selection, examples = _fewshot_context(root)
    try:
        output_dir.mkdir(parents=True)
        _write_jsonl_new(output_dir / "split_manifest.jsonl", split_lines)
        for scenario in SCENARIOS:
            if not counts[scenario]["train"] or not counts[scenario]["validation"]:
                raise Week6TrainingError(f"deterministic split is empty for {scenario}")
            candidates = {
                str(row["sample_id"]): row
                for row in iter_jsonl(week5_output / "pools" / f"{scenario}.jsonl")
            }
            annotations = annotations_by_scenario[scenario]
            scenario_dir = output_dir / scenario
            scenario_dir.mkdir(parents=True)
            paths = {
                split: scenario_dir / f"{split}.jsonl"
                for split in ("train", "validation")
            }
            handles = {
                split: path.open("x", encoding="utf-8", newline="\n")
                for split, path in paths.items()
            }
            written = {"train": 0, "validation": 0}
            try:
                for preannotation in iter_jsonl(
                    week5_output / "preannotations" / f"{scenario}.jsonl"
                ):
                    sample_id = str(preannotation["sample_id"])
                    human = annotations.get(sample_id)
                    if human is None:
                        label = preannotation["parsed_output"]
                        label_source = "model_preannotation"
                        sample_weight = silver_weight
                    else:
                        label = human["human_annotation"]
                        label_source = "human_revised"
                        sample_weight = 1.0
                    rendered = _render_preannotation(
                        root, week5, candidates[sample_id], selection, examples
                    )
                    messages = [*rendered["messages"], {
                        "role": "assistant",
                        "content": json.dumps(
                            label, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        ),
                    }]
                    row = {
                        "sample_id": sample_id,
                        "scenario": scenario,
                        "messages": messages,
                        "label_source": label_source,
                        "sample_weight": sample_weight,
                        "dataset_lock": dataset_lock,
                    }
                    validate_training_row(row, scenario=scenario)
                    split = assignments[scenario][sample_id]
                    handles[split].write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                        + "\n"
                    )
                    written[split] += 1
            finally:
                for handle in handles.values():
                    handle.close()
            if written != {
                "train": counts[scenario]["train"],
                "validation": counts[scenario]["validation"],
            }:
                raise Week6TrainingError(f"locked split count mismatch: {scenario}")
        manifest = {**manifest_payload, "manifest_sha256": manifest_sha256}
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except Exception:
        # The version directory is new and incomplete on failure; keep evidence for diagnosis.
        raise
    return {
        "status": "locked",
        "output_dir": output_dir.relative_to(root).as_posix(),
        "dataset_lock": dataset_lock,
        "counts": counts,
    }

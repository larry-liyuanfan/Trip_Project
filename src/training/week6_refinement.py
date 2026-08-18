"""Build an immutable, derived-silver Week 6 itinerary refinement lock."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.training.week6_qlora import Week6TrainingError, iter_training_rows
from src.training.week6_quality import audit_itinerary_target, repair_itinerary_target


REPAIR_VERSION = "itinerary_structural_repair_v1"


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


def build_itinerary_refinement_lock(
    root: Path,
    *,
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    dataset_version: str,
    silver_weight: float = 0.5,
) -> dict[str, Any]:
    """Create new train/validation JSONL without overwriting the source lock."""
    if not dataset_version.strip():
        raise Week6TrainingError("refinement dataset_version is required")
    if not 0 < silver_weight <= 0.5:
        raise Week6TrainingError("derived silver weight must be in (0, 0.5]")
    if output_dir.exists():
        raise Week6TrainingError(f"refusing to overwrite refinement lock: {output_dir}")
    for path in (train_path, validation_path):
        if not path.is_file():
            raise Week6TrainingError(f"refinement source is missing: {path}")

    source_rows = {
        "train": list(iter_training_rows(train_path, scenario="itinerary_planning")),
        "validation": list(
            iter_training_rows(validation_path, scenario="itinerary_planning")
        ),
    }
    source_locks = {
        json.dumps(row["dataset_lock"], sort_keys=True, separators=(",", ":"))
        for rows in source_rows.values()
        for row in rows
    }
    if len(source_locks) != 1:
        raise Week6TrainingError("refinement sources use mixed dataset locks")
    source_lock = json.loads(next(iter(source_locks)))
    sample_ids = [row["sample_id"] for rows in source_rows.values() for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise Week6TrainingError("refinement train and validation samples overlap")

    split_rows = [
        {"sample_id": row["sample_id"], "scenario": "itinerary_planning", "split": split}
        for split, rows in source_rows.items()
        for row in rows
    ]
    split_rows.sort(key=lambda item: item["sample_id"])
    split_sha256 = hashlib.sha256(
        b"".join(_canonical_bytes(row) + b"\n" for row in split_rows)
    ).hexdigest()
    manifest_payload = {
        "schema_version": "week6_itinerary_refinement_lock_v1",
        "dataset_version": dataset_version,
        "scenario": "itinerary_planning",
        "repair_version": REPAIR_VERSION,
        "source_dataset_lock": source_lock,
        "source_files": {
            "train": {"path": train_path.as_posix(), "sha256": _sha256_file(train_path)},
            "validation": {
                "path": validation_path.as_posix(),
                "sha256": _sha256_file(validation_path),
            },
        },
        "label_policy": {
            "label_source": "model_preannotation",
            "sample_weight": silver_weight,
            "human_identity_inherited": False,
        },
        "counts": {split: len(rows) for split, rows in source_rows.items()},
        "split_sha256": split_sha256,
    }
    manifest_sha256 = hashlib.sha256(_canonical_bytes(manifest_payload)).hexdigest()
    dataset_lock = {
        "dataset_version": dataset_version,
        "manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
    }

    output_dir.mkdir(parents=True)
    counts: Counter[str] = Counter()
    try:
        with (output_dir / "split_manifest.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            for item in split_rows:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        for split, rows in source_rows.items():
            path = output_dir / "itinerary_planning" / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                for source_row in rows:
                    row = copy.deepcopy(source_row)
                    repaired = repair_itinerary_target(root, source_row)
                    row["messages"][-1]["content"] = json.dumps(
                        repaired,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    row["dataset_lock"] = dataset_lock
                    row["label_source"] = "model_preannotation"
                    row["sample_weight"] = silver_weight
                    row["target_derivation"] = {
                        "version": REPAIR_VERSION,
                        "source_label_source": source_row["label_source"],
                        "human_identity_inherited": False,
                    }
                    audit = audit_itinerary_target(root, row)
                    if not audit["passed"]:
                        raise Week6TrainingError(
                            f"repaired target failed audit: {row['sample_id']}"
                        )
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                        + "\n"
                    )
                    counts[split] += 1
        manifest = {**manifest_payload, "manifest_sha256": manifest_sha256}
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except Exception:
        # 保留新的不完整目录作为诊断证据，不覆盖或回退源锁。
        raise
    return {
        "status": "locked",
        "output_dir": output_dir.as_posix(),
        "dataset_lock": dataset_lock,
        "repair_version": REPAIR_VERSION,
        "counts": dict(counts),
    }

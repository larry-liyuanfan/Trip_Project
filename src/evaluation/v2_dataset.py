"""Prepare the human-annotation boundary for the curated Week 3 v2 set."""

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.annotation_workflow import export_packet
from src.evaluation.manifests import (
    ManifestValidationError,
    build_exclusion_rows,
    load_configured_manifests,
    load_manifest,
    validate_manifest_record,
    write_jsonl,
)


TARGET_AFTER_SALES_QUOTAS = {
    "hygiene_stain": 38,
    "facility_damage": 38,
    "attraction_closure": 37,
    "transport_delay": 37,
}


def prepare_curated_v2_dataset(
    *,
    root: Path,
    source_config: dict[str, Any],
    target_config: dict[str, Any],
    replacement_after_sales_manifest: Path,
) -> dict[str, Any]:
    """Create v2 manifests without modifying v1 or inventing completed labels."""
    project_root = Path(root)
    if target_config["dataset_version"] != "week3_evaluation_v2":
        raise ManifestValidationError("target config must use week3_evaluation_v2")
    outputs = _output_paths(project_root, target_config)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise ManifestValidationError(
            "refusing to overwrite existing v2 output: " + ", ".join(map(str, existing))
        )

    source = load_configured_manifests(source_config, root=project_root)
    replacements = load_manifest(replacement_after_sales_manifest, root=project_root)
    replacement_counts = Counter(row["sampling_stratum"] for row in replacements)
    if replacement_counts != Counter(TARGET_AFTER_SALES_QUOTAS):
        raise ManifestValidationError(
            f"replacement after-sales strata are invalid: {dict(replacement_counts)}"
        )
    if any(row["annotation_status"] != "pending" for row in replacements):
        raise ManifestValidationError("replacement after-sales records must be pending")

    product = [
        _version_record(row, completed=True, note="Retained v1 human gold; reasonable unknown values remain valid.")
        for row in source["image_product_search"]
    ]
    itinerary = [
        _version_record(
            row,
            completed=False,
            note="V2 supplements only style_preferences; all other v1 human fields are inherited unchanged at submission.",
        )
        for row in source["itinerary_planning"]
    ]
    after_sales, removed_ids = _curate_after_sales(source["after_sales"], replacements)

    manifests = {
        "image_product_search": product,
        "after_sales": after_sales,
        "itinerary_planning": itinerary,
    }
    for scenario, records in manifests.items():
        expected = target_config["scenarios"][scenario]["target_count"]
        if len(records) != expected:
            raise ManifestValidationError(
                f"v2 {scenario} count is {len(records)}, expected {expected}"
            )
        for record in records:
            validate_manifest_record(record, root=project_root)

    exclusion_rows = build_exclusion_rows(
        [record for records in manifests.values() for record in records]
    )
    for scenario, records in manifests.items():
        write_jsonl(
            project_root / target_config["scenarios"][scenario]["manifest_path"],
            records,
        )
    write_jsonl(project_root / target_config["paths"]["exclusion_manifest"], exclusion_rows)

    codings_dir = project_root / target_config["paths"]["codings_dir"] / "week3_v2"
    codings_dir.mkdir(parents=True, exist_ok=True)
    for scenario in ("after_sales", "itinerary_planning"):
        packet = export_packet(
            manifests[scenario],
            scenario=scenario,
            stage="annotation",
            include_suggestions=True,
        )
        write_jsonl(codings_dir / f"{scenario}_annotation_packet.jsonl", packet)

    summary = {
        "status": "pending_human_annotation",
        "dataset_version": target_config["dataset_version"],
        "counts": {
            scenario: {
                "candidate": len(records),
                "annotated": sum(
                    record["annotation_status"] == "completed" for record in records
                ),
                "pending": sum(
                    record["annotation_status"] == "pending" for record in records
                ),
            }
            for scenario, records in manifests.items()
        },
        "after_sales_sources": dict(
            Counter(record["source_type"] for record in after_sales)
        ),
        "after_sales_strata": dict(
            Counter(record["sampling_stratum"] for record in after_sales)
        ),
        "removed_low_evidence_after_sales_count": len(removed_ids),
        "removed_low_evidence_after_sales_sample_ids": sorted(removed_ids),
        "exclusion_count": len(exclusion_rows),
    }
    log_path = (
        project_root
        / target_config["paths"]["sampling_logs_dir"]
        / "week3_v2_dataset_preparation.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _output_paths(root: Path, config: dict[str, Any]) -> list[Path]:
    paths = [
        root / settings["manifest_path"]
        for settings in config["scenarios"].values()
    ]
    paths.append(root / config["paths"]["exclusion_manifest"])
    paths.append(root / config["paths"]["sampling_logs_dir"] / "week3_v2_dataset_preparation.json")
    return paths


def _version_record(
    record: dict[str, Any], *, completed: bool, note: str
) -> dict[str, Any]:
    updated = copy.deepcopy(record)
    updated["dataset_version"] = "week3_evaluation_v2"
    updated["notes"] = " ".join(
        part for part in (record.get("notes"), note) if isinstance(part, str) and part
    )
    updated["review_status"] = "pending"
    updated["reviewer"] = None
    if not completed:
        updated["annotation_status"] = "pending"
        updated["annotator"] = None
        updated["annotation"] = None
    return updated


def _curate_after_sales(
    frozen_records: list[dict[str, Any]],
    replacement_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    retained: list[dict[str, Any]] = []
    removed: list[str] = []
    for record in frozen_records:
        annotation = record.get("annotation") or {}
        issue_type = annotation.get("issue_type")
        keep_public = record["source_type"] == "public_yelp" and issue_type == "hygiene_stain"
        keep_synthetic = (
            record["source_type"] == "business_synthetic"
            and issue_type in {"attraction_closure", "transport_delay"}
        )
        if keep_public or keep_synthetic:
            retained.append(
                _version_record(
                    record,
                    completed=True,
                    note="Retained because the v1 human gold confirms target visual evidence.",
                )
            )
        else:
            removed.append(record["sample_id"])

    counts = Counter(row["sampling_stratum"] for row in retained)
    replacement_index: dict[str, list[dict[str, Any]]] = {}
    for record in replacement_records:
        replacement_index.setdefault(record["sampling_stratum"], []).append(record)
    for stratum, target in TARGET_AFTER_SALES_QUOTAS.items():
        needed = target - counts[stratum]
        if needed < 0 or len(replacement_index.get(stratum, [])) < needed:
            raise ManifestValidationError(f"insufficient v2 replacements for {stratum}")
        retained.extend(
            _version_record(
                row,
                completed=False,
                note="Selected as clear v2 replacement evidence; human annotation required.",
            )
            for row in replacement_index[stratum][:needed]
        )
    final_counts = Counter(row["sampling_stratum"] for row in retained)
    if final_counts != Counter(TARGET_AFTER_SALES_QUOTAS):
        raise ManifestValidationError(
            f"curated after-sales strata are invalid: {dict(final_counts)}"
        )
    return sorted(retained, key=lambda row: row["sample_id"]), removed

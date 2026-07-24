"""Build deterministic pending Week 3 manifests from approved local sources."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_week3_evaluation import rebuild_exclusion_registry
from src.evaluation.candidates import (
    CandidateDeduplicator,
    classify_after_sales_issue,
    classify_product_coverage,
    image_fingerprints,
    render_synthetic_evidence,
    render_synthetic_visual_evidence,
    retain_best_group_row,
)
from src.evaluation.config import load_evaluation_config
from src.evaluation.manifests import ManifestValidationError, write_jsonl
from src.evaluation.sampling import stratified_sample


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_recipe_versions(sources: dict[str, Any]) -> tuple[str, str]:
    """Return independent itinerary and after-sales synthetic recipe versions."""
    itinerary = sources.get("synthetic_recipe_version")
    after_sales = sources.get("after_sales_synthetic_recipe_version")
    if not isinstance(itinerary, str) or not itinerary.strip():
        raise ManifestValidationError("candidate_sources.synthetic_recipe_version is required")
    if not isinstance(after_sales, str) or not after_sales.strip():
        raise ManifestValidationError(
            "candidate_sources.after_sales_synthetic_recipe_version is required"
        )
    return itinerary.strip(), after_sales.strip()


def _rank(seed: int, namespace: str, source_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}|{namespace}|{source_id}".encode("utf-8")).digest(),
        "big",
    )


def _assert_clean_outputs(config: dict[str, Any], root: Path) -> None:
    outputs = [
        root / settings["manifest_path"] for settings in config["scenarios"].values()
    ] + [root / config["paths"]["exclusion_manifest"]]
    for path in outputs:
        if path.exists() and path.stat().st_size:
            raise ManifestValidationError(
                f"refusing to overwrite non-empty evaluation artifact: {path}"
            )


def _copy_source_image(root: Path, source: str, destination: str) -> None:
    source_path = (root / source).resolve()
    destination_path = (root / destination).resolve()
    if not source_path.is_file():
        raise ManifestValidationError(f"source image does not exist: {source}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        if _file_sha256(destination_path) != _file_sha256(source_path):
            raise ManifestValidationError(f"candidate image path already has different bytes: {destination}")
        return
    shutil.copyfile(source_path, destination_path)


def _candidate(
    *,
    scenario: str,
    coverage_group: str,
    source_type: str,
    source_id: str,
    source_license: str,
    image_paths: list[str],
    fingerprints: list[Any],
    source_uri: str | None,
    source_version: str,
    group_id: str,
    pii_review_status: str,
    text_constraints: str | None = None,
    synthetic_recipe_version: str | None = None,
    constraint_template_id: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    images = [
        {
            "path": path,
            "sha256": fingerprint.sha256,
            "perceptual_hash": fingerprint.perceptual_hash,
        }
        for path, fingerprint in zip(image_paths, fingerprints, strict=True)
    ]
    return {
        "scenario": scenario,
        "coverage_group": coverage_group,
        "source_type": source_type,
        "source_id": source_id,
        "source_license": source_license,
        "image_sha256": images[0]["sha256"],
        "input": {"images": images, "text_constraints": text_constraints},
        "file_status": "valid",
        "notes": notes,
        "provenance": {
            "source_uri": source_uri,
            "source_version": source_version,
            "group_id": group_id,
            "synthetic_recipe_version": synthetic_recipe_version,
            "constraint_template_id": constraint_template_id,
            "pii_review_status": pii_review_status,
        },
    }


def _collect_yelp_photo_rows(
    root: Path,
    sources: dict[str, Any],
    *,
    seed: int,
    limit_per_stratum: int,
) -> dict[str, list[dict[str, Any]]]:
    business_path = root / sources["yelp_business_path"]
    eligible_businesses: dict[str, str] = {}
    for batch in pq.ParquetFile(business_path).iter_batches(
        batch_size=50_000, columns=["business_id", "categories"]
    ):
        for row in batch.to_pylist():
            coverage = classify_product_coverage(row["categories"])
            if coverage is not None:
                eligible_businesses[row["business_id"]] = coverage

    grouped_rows: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
    photo_path = root / sources["yelp_photos_path"]
    for batch in pq.ParquetFile(photo_path).iter_batches(
        batch_size=50_000,
        columns=["photo_id", "business_id", "image_path", "caption", "label"],
    ):
        for row in batch.to_pylist():
            coverage = eligible_businesses.get(row["business_id"])
            if coverage is None:
                continue
            retain_best_group_row(
                grouped_rows.setdefault(coverage, {}),
                group_id=row["business_id"],
                row=row,
                rank=_rank(seed, "yelp-photo", row["photo_id"]),
            )
    result: dict[str, list[dict[str, Any]]] = {}
    for stratum, grouped in grouped_rows.items():
        ordered = sorted(grouped.values(), key=lambda item: (item[0], item[1]["photo_id"]))
        result[stratum] = [row for _, row in ordered[:limit_per_stratum]]
    return result


def _select_yelp_candidate(
    *,
    root: Path,
    deduplicator: CandidateDeduplicator,
    row: dict[str, Any],
    scenario: str,
    coverage_group: str,
    source_license: str,
    source_version: str,
    pii_review_status: str,
    text_constraints: str | None = None,
    constraint_template_id: str | None = None,
    notes: str | None = None,
) -> dict[str, Any] | None:
    source_id = f"yelp-photo:{row['photo_id']}"
    group_id = f"yelp-business:{row['business_id']}"
    destination = f"data/eval/images/{scenario}/{row['photo_id']}.jpg"
    source_fingerprint = image_fingerprints(root, row["image_path"])
    if not deduplicator.accept(
        source_id=source_id,
        group_id=group_id,
        image_hashes=[source_fingerprint.sha256],
        perceptual_hashes=[source_fingerprint.perceptual_hash],
    ):
        return None
    _copy_source_image(root, row["image_path"], destination)
    destination_fingerprint = image_fingerprints(root, destination)
    return _candidate(
        scenario=scenario,
        coverage_group=coverage_group,
        source_type="public_yelp",
        source_id=source_id,
        source_license=source_license,
        image_paths=[destination],
        fingerprints=[destination_fingerprint],
        source_uri=f"yelp://photo/{row['photo_id']}",
        source_version=source_version,
        group_id=group_id,
        pii_review_status=pii_review_status,
        text_constraints=text_constraints,
        constraint_template_id=constraint_template_id,
        notes=notes,
    )


def _itinerary_constraint(index: int) -> tuple[str, str]:
    days = 2 + index % 3
    budget = 800 + (index % 8) * 300
    end_hour = 17 + index % 4
    pace = ("慢节奏", "适中节奏", "紧凑节奏")[index % 3]
    transit = ("公共交通优先", "步行与公共交通结合", "避免长距离步行")[index % 3]
    family = f"itinerary-template-{index % 12:02d}"
    text = (
        f"{days}天行程，预算不超过{budget}元，{pace}；{transit}；"
        f"最后一天{end_hour}:00前结束，并包含每日用餐与交通安排。"
    )
    return family, text


def _collect_public_after_sales_rows(
    root: Path,
    sources: dict[str, Any],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Discover bounded public hygiene/facility candidates from weak Yelp pairs."""
    grouped: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {
        "hygiene_stain": {},
        "facility_damage": {},
    }
    weak_path = root / sources["yelp_weak_pairs_path"]
    for batch in pq.ParquetFile(weak_path).iter_batches(
        batch_size=25_000,
        columns=["business_id", "photo_id", "image_path", "review_id", "review_text"],
    ):
        for row in batch.to_pylist():
            issue_type = classify_after_sales_issue(row.get("review_text") or "")
            if issue_type not in grouped:
                continue
            source_id = f"{row['review_id']}:{row['photo_id']}"
            retain_best_group_row(
                grouped[issue_type],
                group_id=row["business_id"],
                row=row,
                rank=_rank(seed, f"after-sales-{issue_type}", source_id),
            )
    return {
        issue_type: [
            row for _, row in sorted(rows.values(), key=lambda item: (item[0], item[1]["photo_id"]))
        ]
        for issue_type, rows in grouped.items()
    }


def _select_public_after_sales_candidate(
    *,
    root: Path,
    deduplicator: CandidateDeduplicator,
    row: dict[str, Any],
    issue_type: str,
    source_license: str,
    source_version: str,
) -> dict[str, Any] | None:
    source_id = f"yelp-review-photo:{row['review_id']}:{row['photo_id']}"
    group_id = f"yelp-business:{row['business_id']}"
    destination = f"data/eval/images/after_sales/{row['photo_id']}.jpg"
    source_fingerprint = image_fingerprints(root, row["image_path"])
    if not deduplicator.accept(
        source_id=source_id,
        group_id=group_id,
        image_hashes=[source_fingerprint.sha256],
        perceptual_hashes=[source_fingerprint.perceptual_hash],
    ):
        return None
    _copy_source_image(root, row["image_path"], destination)
    fingerprint = image_fingerprints(root, destination)
    return _candidate(
        scenario="after_sales",
        coverage_group=issue_type,
        source_type="public_yelp",
        source_id=source_id,
        source_license=source_license,
        image_paths=[destination],
        fingerprints=[fingerprint],
        source_uri=f"yelp://review/{row['review_id']}/photo/{row['photo_id']}",
        source_version=source_version,
        group_id=group_id,
        pii_review_status="pending",
        notes=(
            f"Candidate discovered from business-level review pairing {row['review_id']}; "
            "human annotation is frozen separately from candidate metadata."
        ),
    )


def collect_synthetic_after_sales_candidates(
    *,
    root: Path,
    settings: dict[str, Any],
    recipe_version: str,
    deduplicator: CandidateDeduplicator,
    issue_types: tuple[str, ...] = (
        "hygiene_stain",
        "facility_damage",
        "attraction_closure",
        "transport_delay",
    ),
) -> list[dict[str, Any]]:
    """Build clear, project-owned candidates for all four required issue types."""
    candidates: list[dict[str, Any]] = []
    for stratum in issue_types:
        quota = settings["sampling"]["quotas"][stratum]
        for index in range(quota * 20):
            source_id = f"synthetic:{stratum}:{index:04d}"
            group_id = f"synthetic-event:{stratum}:{index:04d}"
            destination = f"data/eval/images/after_sales/{stratum}_{index:04d}.png"
            if stratum in {"hygiene_stain", "facility_damage"}:
                render_synthetic_visual_evidence(
                    root / destination,
                    issue_type=stratum,
                    index=index,
                )
            else:
                render_synthetic_evidence(
                    root / destination,
                    issue_type=stratum,
                    index=index,
                )
            fingerprint = image_fingerprints(root, destination)
            if not deduplicator.accept(
                source_id=source_id,
                group_id=group_id,
                image_hashes=[fingerprint.sha256],
                perceptual_hashes=[fingerprint.perceptual_hash],
            ):
                continue
            candidates.append(
                _candidate(
                    scenario="after_sales",
                    coverage_group=stratum,
                    source_type="business_synthetic",
                    source_id=source_id,
                    source_license="project_business_synthetic",
                    image_paths=[destination],
                    fingerprints=[fingerprint],
                    source_uri=f"synthetic://week3/{stratum}/{index:04d}",
                    source_version=recipe_version,
                    group_id=group_id,
                    pii_review_status="not_applicable",
                    synthetic_recipe_version=recipe_version,
                    notes=(
                        "Deterministic project-owned business-synthetic evidence; "
                        "human visual annotation required."
                    ),
                )
            )
            if sum(item["coverage_group"] == stratum for item in candidates) == quota:
                break
        accepted = sum(item["coverage_group"] == stratum for item in candidates)
        if accepted != quota:
            raise ManifestValidationError(
                f"synthetic after-sales shortfall for {stratum}: {accepted} of {quota}"
            )
    return candidates


def build_candidate_manifests(config: dict[str, Any], *, root: Path) -> dict[str, Any]:
    """Build exact-size pending manifests without inventing human annotations."""
    root = Path(root)
    sources = config.get("candidate_sources")
    if not isinstance(sources, dict):
        raise ManifestValidationError("candidate_sources configuration is required")
    _assert_clean_outputs(config, root)
    source_paths = [
        root / sources["yelp_business_path"],
        root / sources["yelp_photos_path"],
        root / sources["yelp_weak_pairs_path"],
    ]
    for path in source_paths:
        if not path.is_file():
            raise ManifestValidationError(f"candidate source does not exist: {path}")
    source_version = "yelp-week3:" + hashlib.sha256(
        "|".join(_file_sha256(path) for path in source_paths).encode("ascii")
    ).hexdigest()
    source_license = sources["yelp_source_license"]
    itinerary_recipe_version, after_sales_recipe_version = (
        _candidate_recipe_versions(sources)
    )
    deduplicator = CandidateDeduplicator(max_perceptual_distance=4)

    product_settings = config["scenarios"]["image_product_search"]
    photo_rows = _collect_yelp_photo_rows(
        root,
        sources,
        seed=product_settings["sampling"]["seed"],
        limit_per_stratum=3_000,
    )
    candidate_sets: dict[str, list[dict[str, Any]]] = {
        "image_product_search": [],
        "after_sales": [],
        "itinerary_planning": [],
    }
    for stratum, quota in product_settings["sampling"]["quotas"].items():
        for row in photo_rows.get(stratum, []):
            selected = _select_yelp_candidate(
                root=root,
                deduplicator=deduplicator,
                row=row,
                scenario="image_product_search",
                coverage_group=stratum,
                source_license=source_license,
                source_version=source_version,
                pii_review_status="not_applicable",
            )
            if selected is not None:
                candidate_sets["image_product_search"].append(selected)
            if sum(item["coverage_group"] == stratum for item in candidate_sets["image_product_search"]) == quota:
                break

    itinerary_rows = sorted(
        [row for rows in photo_rows.values() for row in rows],
        key=lambda row: (_rank(20260713, "itinerary", row["photo_id"]), row["photo_id"]),
    )
    itinerary_quota = config["scenarios"]["itinerary_planning"]["target_count"]
    for index, row in enumerate(itinerary_rows):
        template_id, constraints = _itinerary_constraint(index)
        selected = _select_yelp_candidate(
            root=root,
            deduplicator=deduplicator,
            row=row,
            scenario="itinerary_planning",
            coverage_group="paired_reference_and_text",
            source_license=source_license,
            source_version=source_version,
            pii_review_status="not_applicable",
            text_constraints=constraints,
            constraint_template_id=template_id,
            notes="Text constraints are business-synthetic and require human annotation.",
        )
        if selected is not None:
            selected["provenance"]["synthetic_recipe_version"] = (
                itinerary_recipe_version
            )
            candidate_sets["itinerary_planning"].append(selected)
        if len(candidate_sets["itinerary_planning"]) == itinerary_quota:
            break

    after_settings = config["scenarios"]["after_sales"]
    public_rows = _collect_public_after_sales_rows(
        root,
        sources,
        seed=after_settings["sampling"]["seed"],
    )
    for issue_type in ("hygiene_stain", "facility_damage"):
        quota = after_settings["sampling"]["quotas"][issue_type]
        for row in public_rows[issue_type]:
            selected = _select_public_after_sales_candidate(
                root=root,
                deduplicator=deduplicator,
                row=row,
                issue_type=issue_type,
                source_license=source_license,
                source_version=source_version,
            )
            if selected is not None:
                candidate_sets["after_sales"].append(selected)
            if sum(
                item["coverage_group"] == issue_type
                for item in candidate_sets["after_sales"]
            ) == quota:
                break
        accepted = sum(
            item["coverage_group"] == issue_type
            for item in candidate_sets["after_sales"]
        )
        if accepted != quota:
            raise ManifestValidationError(
                f"public after-sales shortfall for {issue_type}: {accepted} of {quota}"
            )
    candidate_sets["after_sales"].extend(
        collect_synthetic_after_sales_candidates(
            root=root,
            settings=after_settings,
            recipe_version=after_sales_recipe_version,
            deduplicator=deduplicator,
            issue_types=("attraction_closure", "transport_delay"),
        )
    )

    output_counts: dict[str, int] = {}
    for scenario, candidates in candidate_sets.items():
        settings = config["scenarios"][scenario]
        requested = settings["target_count"]
        if len(candidates) != requested:
            raise ManifestValidationError(
                f"candidate source shortfall for {scenario}: {len(candidates)} of {requested}"
            )
        records, sampling_log = stratified_sample(
            candidates,
            scenario=scenario,
            dataset_version=config["dataset_version"],
            seed=settings["sampling"]["seed"],
            stratum_field=settings["sampling"]["stratum_field"],
            quotas=settings["sampling"]["quotas"],
            root=root,
        )
        manifest_path = root / settings["manifest_path"]
        write_jsonl(manifest_path, records)
        log_path = root / config["paths"]["sampling_logs_dir"] / f"{scenario}_candidate_build_v1.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {**sampling_log, "source_version": source_version},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        output_counts[scenario] = len(records)
    exclusion = rebuild_exclusion_registry(config, root=root)
    return {
        "status": "pending_human_annotation",
        "dataset_version": config["dataset_version"],
        "source_version": source_version,
        "candidate_counts": output_counts,
        "exclusion_count": exclusion["exclusion_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week3.yaml")
    args = parser.parse_args()
    config = load_evaluation_config(args.config)
    print(
        json.dumps(
            build_candidate_manifests(config, root=Path.cwd()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

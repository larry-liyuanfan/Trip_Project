"""Week 8 product-understanding data lock, Prompt comparison, and one-shot test."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from src.evaluation.prompting import render_standard_prompt
from src.evaluation.product_semantics import audit_product_references
from src.evaluation.schema_validation import SchemaValidationError, validate_output
from src.inference.system_runtime import (
    ModelGenerationError,
    ReleaseSettings,
    TransformersPeftBackend,
    _correction_messages,
    _json_schema_response_format,
)
from src.inference.transport_utils import strip_json_fence
from src.training.week7_data import (
    IDENTITY_FIELDS,
    Week7DataError,
    _collect_repair_public_sources,
    _copy_image,
    _product_target,
    _repair_business_tags,
    _repair_support_flags,
    _row,
    _validate_partition_isolation,
    add_superseded_identities,
    canonical_sha256,
    iter_jsonl,
    load_consumed_identities,
    sha256_file,
    write_jsonl_new,
)
from src.training.week7_evaluation import summarize_raw_records


class Week8ProductError(ValueError):
    """Raised when the Week 8 product contract or immutable evidence is invalid."""


PRODUCT_METRIC_WEIGHTS = {
    "business_category_accuracy": 0.20,
    "price_range_accuracy": 0.15,
    "style_f1": 0.20,
    "facility_f1": 0.20,
    "label_completeness": 0.15,
    "json_compliance": 0.05,
    "schema_pass": 0.05,
}

PRICE_EVIDENCE_RE = re.compile(
    r"(?:[$€£¥]\s*\d|\d\s*(?:dollars?|usd|aud|cny)|price(?:s|d)?|menu board)",
    re.IGNORECASE,
)
MULTI_SUBJECT_RE = re.compile(
    r"\b(?:multiple|several|various|many|crowd|group|people|customers|tables)\b",
    re.IGNORECASE,
)


def load_week8_product_config(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {
        "week8_product_understanding_v1",
        "week8_product_understanding_v2",
        "week8_product_understanding_v3",
        "week8_product_understanding_v4",
        "week8_product_understanding_v5",
        "week8_product_understanding_v6",
        "week8_product_understanding_v7",
    }:
        raise Week8ProductError("unsupported Week 8 product config")
    dataset = payload.get("dataset", {})
    if any(int(dataset.get(key, 0)) <= 0 for key in (
        "development_count", "test_count", "continuation_train_count"
    )):
        raise Week8ProductError("Week 8 split counts must be positive")
    if payload.get("week8", {}).get("label_policy") != "programmatic_silver_only":
        raise Week8ProductError("Week 8 automatic labels must remain silver")
    if set(payload.get("prompts", {})) != {
        "current_release", "compact_field_check", "visual_evidence_guard"
    }:
        raise Week8ProductError("Week 8 must compare exactly the three approved Prompt roles")
    if payload.get("schema_version") != "week8_product_understanding_v1":
        fresh = payload.get("fresh_source", {})
        expected_source_version = (
            "week8_product_fresh_20260827_v3"
            if payload.get("schema_version") in {
                "week8_product_understanding_v6",
                "week8_product_understanding_v7",
            }
            else (
                "week8_product_fresh_20260827_v2"
                if payload.get("schema_version") == "week8_product_understanding_v5"
                else "week8_product_fresh_20260826_v1"
            )
        )
        if (
            payload.get("week8", {}).get("source_version")
            != expected_source_version
            or int(fresh.get("selected_photo_count", 0)) < 520
            or int(fresh.get("minimum_eligible_count", 0)) < 520
            or int(fresh.get("candidate_extract_count", 0))
            < int(fresh.get("selected_photo_count", 0))
            or not str(fresh.get("output_root") or "").startswith(
                f"data/yelp/{expected_source_version}"
            )
        ):
            raise Week8ProductError("Week 8 v2 fresh-source identity is incomplete")
    if payload.get("schema_version") in {
        "week8_product_understanding_v4",
        "week8_product_understanding_v5",
        "week8_product_understanding_v6",
        "week8_product_understanding_v7",
    }:
        support = dataset.get("split_field_support_minimums", {})
        split_sizes = {
            "development": int(dataset.get("development_count", 0)),
            "test": int(dataset.get("test_count", 0)),
            "train": int(dataset.get("continuation_train_count", 0)),
        }
        if any(
            int(support.get(split, {}).get(field, 0)) <= 0
            or int(support[split][field]) > split_sizes[split]
            for split in split_sizes
            for field in ("style", "facility")
        ):
            raise Week8ProductError("Week 8 v4 field support contract is incomplete")
    if payload.get("schema_version") in {
        "week8_product_understanding_v5",
        "week8_product_understanding_v6",
        "week8_product_understanding_v7",
    }:
        expected_categories = {"hotel", "attraction", "restaurant"}
        fresh_minimums = payload.get("fresh_source", {}).get(
            "minimum_per_ota_category_by_category"
        )
        split_minimums = dataset.get("split_category_minimums", {})
        if (
            not isinstance(fresh_minimums, dict)
            or set(fresh_minimums) != expected_categories
            or any(int(fresh_minimums[key]) <= 0 for key in expected_categories)
            or any(
                sum(
                    int(split_minimums.get(split, {}).get(category, 0))
                    for split in ("development", "test", "train")
                )
                > int(fresh_minimums[category])
                for category in expected_categories
            )
        ):
            raise Week8ProductError("Week 8 v5 category support contract is infeasible")
    return payload


def _validate_fresh_source_manifest(
    source_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    fresh = config["fresh_source"]
    relative = fresh.get("manifest_path")
    expected_sha = fresh.get("manifest_sha256")
    if not relative or not expected_sha:
        if config["schema_version"] in {
            "week8_product_understanding_v3",
            "week8_product_understanding_v4",
            "week8_product_understanding_v5",
            "week8_product_understanding_v6",
            "week8_product_understanding_v7",
        }:
            raise Week8ProductError("v3 fresh-source manifest identity is missing")
        return {}
    path = source_root / str(relative)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise Week8ProductError("fresh-source manifest hash changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "COMPLETED"
        or manifest.get("source_version") != config["week8"]["source_version"]
        or manifest.get("config_sha256") != fresh.get("source_config_sha256")
        or int(manifest.get("selected_photo_count", 0))
        != int(fresh["selected_photo_count"])
        or any(
            int(value) != 0
            for value in manifest.get("historical_identity_overlap_counts", {}).values()
        )
    ):
        raise Week8ProductError("fresh-source manifest contract changed")
    return manifest


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _category_from_description(description: str) -> str:
    category_text = (
        description.split("|", 2)[1].casefold()
        if "|" in description
        else description.casefold()
    )
    categories = {item.strip() for item in category_text.split(",") if item.strip()}
    if categories & {
        "hotels", "resorts", "bed & breakfast", "guest houses", "hostels"
    }:
        return "hotel"
    if categories & {
        "museums", "parks", "landmarks & historical buildings",
        "amusement parks", "zoos", "aquariums", "botanical gardens", "tours",
    }:
        return "attraction"
    if categories & {
        "restaurants", "food", "cafes", "coffee & tea", "bars", "bakeries"
    }:
        return "restaurant"
    return "unknown"


def product_silver_target(source: dict[str, Any]) -> dict[str, Any]:
    """Reproduce historical mixed metadata/caption silver, NOT visual ground truth.

    Frozen v1-v7 builds retain this protocol for reproducibility. The reference audit
    and SFT eligibility checker prevent interpreting these targets as visual truth.
    """
    caption_only = dict(source)
    caption_only["repair_mode"] = False
    target = _product_target(caption_only)
    if target["business_category"] == "unknown":
        target["business_category"] = _category_from_description(
            str(source.get("business_description") or "")
        )
        if target["business_category"] != "unknown":
            target["inferred_attributes"].append(
                "业态来自商家元数据弱标签，不作为图片直接证据"
            )
    metadata_styles, metadata_facilities = _repair_business_tags(
        str(source.get("business_description") or "")
    )
    target["style_tags"] = sorted(set(target["style_tags"]) | set(metadata_styles))
    target["visible_facilities"] = sorted(
        set(target["visible_facilities"]) | set(metadata_facilities)
    )
    if metadata_styles or metadata_facilities:
        target["inferred_attributes"].append(
            "风格或设施包含 Yelp 商家元数据弱银标，不作为图片直接证据"
        )
    # Existing source captions do not support a reliable visual price tier.
    target["price_range"] = "unknown"
    target["unknown_fields"] = sorted(set(target["unknown_fields"]) | {"price_range"})
    if target["style_tags"]:
        target["unknown_fields"] = [
            field for field in target["unknown_fields"] if field != "style_tags"
        ]
    if target["visible_facilities"]:
        target["unknown_fields"] = [
            field
            for field in target["unknown_fields"]
            if field != "visible_facilities"
        ]
    target["confidence"] = 0.5
    return target


def product_error_slices(source: dict[str, Any], target: dict[str, Any]) -> list[str]:
    caption = str(source.get("caption") or "")
    slices = ["semantic_label_even_when_schema_complete", "price_without_visual_evidence"]
    if target["business_category"] == "unknown":
        slices.append("business_category_unknown")
    else:
        slices.append(f"business_category_{target['business_category']}")
    if MULTI_SUBJECT_RE.search(caption) or target["business_category"] == "unknown":
        slices.append("multiple_or_ambiguous_subject")
    if len(target["style_tags"]) >= 2:
        slices.append("style_multilabel")
    elif target["style_tags"]:
        slices.append("style_single_label")
    else:
        slices.append("style_should_be_empty")
    if target["visible_facilities"]:
        slices.append("facility_visible")
    else:
        slices.append("facility_should_be_empty")
    slices.append("should_use_unknown")
    return sorted(set(slices))


def _collect_week8_v2_sources(
    source_root: Path,
    compatibility: dict[str, Any],
    consumed: dict[str, set[str]],
    split_needs: dict[str, int],
    split_minimums: dict[str, dict[str, int]],
    field_minimums: dict[str, dict[str, int]],
    source_count: int,
) -> dict[str, list[dict[str, Any]]]:
    """Allocate scarce OTA business categories explicitly without duplicating groups."""

    complete_pool = _collect_repair_public_sources(
        source_root,
        compatibility,
        consumed,
        {"development": source_count, "test": 0, "train": 0},
    )["development"]
    remaining = list(complete_pool)
    selected: dict[str, list[dict[str, Any]]] = {
        split: [] for split in split_needs
    }
    split_order = ("development", "test", "train")
    for split_index, split in enumerate(split_order):
        minimums = split_minimums.get(split, {})
        if sum(int(value) for value in minimums.values()) > split_needs[split]:
            raise Week8ProductError(f"category minimums exceed {split} size")
        for category in ("hotel", "attraction", "restaurant"):
            need = int(minimums.get(category, 0))
            matching = [
                row
                for row in remaining
                if _category_from_description(row["business_description"]) == category
            ][:need]
            if len(matching) != need:
                raise Week8ProductError(
                    f"fresh {category} support shortfall for {split}: "
                    f"{len(matching)}/{need}"
                )
            selected[split].extend(matching)
            matching_ids = {row["source_id"] for row in matching}
            remaining = [
                row for row in remaining if row["source_id"] not in matching_ids
            ]
        for field in ("style", "facility"):
            target_support = int(field_minimums.get(split, {}).get(field, 0))
            if target_support <= 0:
                continue
            current_support = sum(
                bool(_repair_support_flags(row)[field]) for row in selected[split]
            )
            need = max(0, target_support - current_support)
            future_category_minimums = {
                category: sum(
                    int(split_minimums.get(future_split, {}).get(category, 0))
                    for future_split in split_order[split_index + 1 :]
                )
                for category in ("hotel", "attraction", "restaurant")
            }
            available_category_counts = Counter(
                _category_from_description(row["business_description"])
                for row in remaining
            )
            matching = []
            for row in remaining:
                category = _category_from_description(row["business_description"])
                if (
                    len(matching) < need
                    and _repair_support_flags(row)[field]
                    and available_category_counts[category]
                    > future_category_minimums.get(category, 0)
                ):
                    matching.append(row)
                    available_category_counts[category] -= 1
            if len(matching) != need:
                raise Week8ProductError(
                    f"fresh {field} support shortfall for {split}: "
                    f"{current_support + len(matching)}/{target_support}"
                )
            selected[split].extend(matching)
            matching_ids = {row["source_id"] for row in matching}
            remaining = [
                row for row in remaining if row["source_id"] not in matching_ids
            ]
        fill = split_needs[split] - len(selected[split])
        future_minimums = {
            category: sum(
                int(split_minimums.get(future_split, {}).get(category, 0))
                for future_split in split_order[split_index + 1 :]
            )
            for category in ("hotel", "attraction", "restaurant")
        }
        available_counts = Counter(
            _category_from_description(row["business_description"])
            for row in remaining
        )
        fill_rows = []
        deferred_rows = []
        for row in remaining:
            category = _category_from_description(row["business_description"])
            if (
                len(fill_rows) < fill
                and available_counts[category] > future_minimums.get(category, 0)
            ):
                fill_rows.append(row)
                available_counts[category] -= 1
            else:
                deferred_rows.append(row)
        selected[split].extend(fill_rows)
        remaining = deferred_rows
        if len(selected[split]) != split_needs[split]:
            raise Week8ProductError(f"fresh source shortfall for {split}")
    return selected


def build_week8_product_lock(
    root: Path,
    config_path: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    source_root = Path(source_root or root).resolve()
    config_path = Path(config_path).resolve()
    config = load_week8_product_config(config_path)
    dataset = config["dataset"]
    output = root / dataset["output_root"] / config["week8"]["dataset_version"]
    if output.exists():
        raise Week8ProductError(f"refusing to overwrite Week 8 lock: {output}")

    compatibility = {
        "dataset": {
            "seed": int(config["week8"]["seed"]),
            "source_paths": dataset["source_paths"],
        },
        "system_repair": {"enabled": True},
    }
    fresh_source_manifest = (
        _validate_fresh_source_manifest(source_root, config)
        if config["schema_version"] != "week8_product_understanding_v1"
        else {}
    )
    try:
        consumed, exclusion_evidence = load_consumed_identities(source_root, compatibility)
        add_superseded_identities(source_root, compatibility, consumed, exclusion_evidence)
        split_needs = {
            "development": int(dataset["development_count"]),
            "test": int(dataset["test_count"]),
            "train": int(dataset["continuation_train_count"]),
        }
        if config["schema_version"] != "week8_product_understanding_v1":
            sources = _collect_week8_v2_sources(
                source_root,
                compatibility,
                consumed,
                split_needs,
                dataset["split_category_minimums"],
                dataset.get("split_field_support_minimums", {}),
                int(config["fresh_source"]["selected_photo_count"]),
            )
        else:
            sources = _collect_repair_public_sources(
                source_root, compatibility, consumed, split_needs
            )
    except Week7DataError as exc:
        raise Week8ProductError(str(exc)) from exc

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    slice_counts: dict[str, Counter[str]] = {}
    for split in ("train", "development", "test"):
        rows = []
        counts: Counter[str] = Counter()
        for index, source in enumerate(sources[split]):
            digest = source["image_sha256"]
            image_path = _copy_image(root, output, source["source_image"], digest)
            target = product_silver_target(source)
            slices = product_error_slices(source, target)
            counts.update(slices)
            identity = {
                "source_id": source["source_id"],
                "image_sha256": digest,
                "group_id": source["group_id"],
                "constraint_template_id": None,
                "image_path": image_path,
            }
            sample_namespace = (
                "week8-product-v2"
                if config["schema_version"] != "week8_product_understanding_v1"
                else "week8-product"
            )
            row = _row(
                f"{sample_namespace}-{split}-{index:04d}",
                "image_product_search",
                split,
                identity,
                [
                    {
                        "role": "system",
                        "content": "你是专业 OTA 多模态助手。只依据输入证据回答，不确定时标记 unknown。",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": image_path},
                            {"type": "text", "text": "识别图片中的 OTA 商品属性并输出指定 JSON。"},
                        ],
                    },
                ],
                target,
                "programmatic_silver",
                0.5,
            )
            row["error_slices"] = slices
            row["target_provenance"] = {
                "category": "caption_or_business_metadata_silver",
                "style_tags": "caption_lexical_silver",
                "visible_facilities": "caption_lexical_silver",
                "price_range": "visual_evidence_absent_unknown_silver",
                "human_completed": False,
            }
            rows.append(row)
        rows_by_split[split] = rows
        slice_counts[split] = counts

    try:
        isolation = _validate_partition_isolation(rows_by_split, consumed)
    except Week7DataError as exc:
        raise Week8ProductError(str(exc)) from exc
    if isolation["status"] != "PASS":
        raise Week8ProductError("Week 8 five-dimensional isolation failed")

    for split, rows in rows_by_split.items():
        write_jsonl_new(output / split / "image_product_search.jsonl", rows)
    manifest_rows = [
        {field: row.get(field) for field in IDENTITY_FIELDS}
        | {
            "split": split,
            "scenario": row["scenario"],
            "label_source": row["label_source"],
        }
        for split, rows in rows_by_split.items()
        for row in rows
    ]
    write_jsonl_new(output / "identity_manifest.jsonl", manifest_rows)
    file_evidence = {}
    for path in sorted(output.rglob("*.jsonl")):
        file_evidence[path.relative_to(output).as_posix()] = {
            "count": sum(1 for _ in iter_jsonl(path)),
            "sha256": sha256_file(path),
        }
    lock_core = {
        "schema_version": (
            "week8_product_lock_v2"
            if config["schema_version"] != "week8_product_understanding_v1"
            else "week8_product_lock_v1"
        ),
        "dataset_version": config["week8"]["dataset_version"],
        "config_sha256": sha256_file(config_path),
        "git_commit": _git_commit(root),
        "label_source": "programmatic_silver",
        "human_count": 0,
        "split_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "error_slice_counts": {
            split: dict(sorted(counts.items())) for split, counts in slice_counts.items()
        },
        "isolation": isolation,
        "historical_exclusion_evidence": exclusion_evidence,
        "fresh_source_manifest_sha256": (
            config.get("fresh_source", {}).get("manifest_sha256")
            if fresh_source_manifest
            else None
        ),
        "files": file_evidence,
        "test_status": "LOCKED_UNCONSUMED",
    }
    lock = {**lock_core, "lock_sha256": canonical_sha256(lock_core)}
    _write_json_new(output / "dataset_lock.json", lock)
    return lock


def validate_week8_product_lock(
    root: Path, config_path: Path, *, include_test: bool = True
) -> dict[str, Any]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = load_week8_product_config(config_path)
    if config.get("week8", {}).get("development_only") is True:
        include_test = False
    lock_config_path = config_path
    lock_config_relative = config.get("development_lock_config")
    if lock_config_relative:
        if config.get("week8", {}).get("development_only") is not True:
            raise Week8ProductError(
                "alternate lock configs are restricted to development-only overlays"
            )
        lock_config_path = (root / str(lock_config_relative)).resolve()
        try:
            lock_config_path.relative_to(root)
        except ValueError as exc:
            raise Week8ProductError(
                "development lock config must stay inside the repository"
            ) from exc
        lock_config = load_week8_product_config(lock_config_path)
        identity_fields = (
            "output_root", "development_count", "test_count",
            "continuation_train_count",
        )
        if (
            config["week8"]["dataset_version"]
            != lock_config["week8"]["dataset_version"]
            or any(
                config["dataset"].get(field) != lock_config["dataset"].get(field)
                for field in identity_fields
            )
            or config["model"] != lock_config["model"]
        ):
            raise Week8ProductError("development overlay changed the locked identity")
    output = root / config["dataset"]["output_root"] / config["week8"]["dataset_version"]
    lock = json.loads((output / "dataset_lock.json").read_text(encoding="utf-8"))
    failures = []
    if lock.get("config_sha256") != sha256_file(lock_config_path):
        failures.append("config_sha256_changed")
    core = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if lock.get("lock_sha256") != canonical_sha256(core):
        failures.append("lock_sha256_changed")
    expected = {
        "train": int(config["dataset"]["continuation_train_count"]),
        "development": int(config["dataset"]["development_count"]),
        "test": int(config["dataset"]["test_count"]),
    }
    for split, count in expected.items():
        if split == "test" and not include_test:
            continue
        rows = list(iter_jsonl(output / split / "image_product_search.jsonl"))
        if len(rows) != count:
            failures.append(f"{split}_count_changed")
        if any(row.get("label_source") != "programmatic_silver" for row in rows):
            failures.append(f"{split}_non_silver_label")
    for relative, evidence in lock.get("files", {}).items():
        if relative.startswith("test/") and not include_test:
            continue
        path = output / relative
        if not path.is_file() or sha256_file(path) != evidence.get("sha256"):
            failures.append(f"artifact_changed:{relative}")
    isolation = None
    identity_path = output / "identity_manifest.jsonl"
    if "identity_manifest.jsonl" not in lock.get("files", {}) or not identity_path.is_file():
        failures.append("identity_manifest_missing")
    else:
        identities = list(iter_jsonl(identity_path))
        by_split = {split: [row for row in identities if row.get("split") == split] for split in expected}
        if len(identities) != sum(expected.values()) or any(len(by_split[s]) != n for s, n in expected.items()):
            failures.append("identity_manifest_count_changed")
        try:
            isolation = _validate_partition_isolation(by_split, {field: set() for field in IDENTITY_FIELDS})
        except Week7DataError as exc:
            failures.append(f"identity_isolation:{exc}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "dataset_version": lock.get("dataset_version"),
        "lock_sha256": lock.get("lock_sha256"),
        "split_counts": lock.get("split_counts"),
        "test_status": lock.get("test_status"),
        "validation_scope": "all_splits" if include_test else "train_development_and_identity_only",
        "isolation": isolation,
    }


def _scoring_config() -> dict[str, Any]:
    return {
        "evaluation": {
            "metric_weights": {"image_product_search": PRODUCT_METRIC_WEIGHTS},
            "scenario_weights": {"image_product_search": 1.0},
            "dialogue_scoring_protocol": "gate_aligned_v2",
            "dialogue_automatic_gate": {"enabled": False},
        }
    }


def _render_product_messages(root: Path, row: dict[str, Any], prompt_version: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rendered = render_standard_prompt(
        root,
        "image_product_search",
        {"images": [{"path": row["image_path"]}], "text_constraints": None},
        prompt_version,
    )
    return rendered["messages"], rendered.get("response_format") or {"type": "json_object"}


def _run_one_product(
    root: Path,
    backend: TransformersPeftBackend,
    row: dict[str, Any],
    *,
    run_id: str,
    prompt_version: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    messages, response_format = _render_product_messages(root, row, prompt_version)
    attempts = []
    active_messages = messages
    active_response_format = response_format
    total_input_tokens = 0
    total_output_tokens = 0
    for attempt_number in (1, 2):
        started = time.perf_counter()
        raw = ""
        error = None
        try:
            generated = backend.generate_with_usage(
                active_messages,
                response_format=active_response_format,
                max_new_tokens=max_new_tokens,
            )
            raw = generated.content
            total_input_tokens += generated.input_tokens
            total_output_tokens += generated.output_tokens
            parsed = json.loads(strip_json_fence(raw))
            validate_output(root, "image_product_search", parsed, "v1")
        except (ModelGenerationError, json.JSONDecodeError, SchemaValidationError, ValidationError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        attempts.append({
            "attempt": attempt_number,
            "raw_output": raw,
            "error": error,
            "latency_ms": latency_ms,
        })
        if error is None:
            return {
                "run_id": run_id,
                "sample_id": row["sample_id"],
                "scenario": "image_product_search",
                "prompt_version": prompt_version,
                "model_name": "Qwen3-VL-8B+system-repair-checkpoint-87",
                "raw_output": raw,
                "latency_ms": sum(item["latency_ms"] for item in attempts),
                "failed": False,
                "error": None,
                "attempts": attempts,
                "input_token_count": total_input_tokens,
                "generated_token_count": total_output_tokens,
                "generation_max_new_tokens": max_new_tokens,
            }
        if attempt_number == 1:
            active_messages = _correction_messages(
                active_messages, raw, error or "", scenario="image_product_search"
            )
            active_response_format = _json_schema_response_format(
                root, "image_product_search", "v1"
            )
    return {
        "run_id": run_id,
        "sample_id": row["sample_id"],
        "scenario": "image_product_search",
        "prompt_version": prompt_version,
        "model_name": "Qwen3-VL-8B+system-repair-checkpoint-87",
        "raw_output": attempts[-1]["raw_output"],
        "latency_ms": sum(item["latency_ms"] for item in attempts),
        "failed": True,
        "error": attempts[-1]["error"],
        "attempts": attempts,
        "input_token_count": total_input_tokens,
        "generated_token_count": total_output_tokens,
        "generation_max_new_tokens": max_new_tokens,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _unknown_and_slice_metrics(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in rows}
    unknown_scores = []
    price_unknown_scores = []
    known_price_count = 0
    slice_totals: Counter[str] = Counter()
    slice_correct: Counter[str] = Counter()
    for record in records:
        row = by_id[record["sample_id"]]
        try:
            parsed = json.loads(strip_json_fence(str(record.get("raw_output") or "")))
        except json.JSONDecodeError:
            parsed = {}
        if not isinstance(parsed, dict) or record.get("failed"):
            parsed = {}
        expected_unknown = set(row["target"].get("unknown_fields", []))
        def string_set(field: str) -> set[str]:
            value = parsed.get(field)
            return set(value) if isinstance(value, list) and all(isinstance(item, str) for item in value) else set()
        observed_unknown = string_set("unknown_fields")
        score = float(bool(parsed) and not record.get("failed") and expected_unknown == observed_unknown)
        unknown_scores.append(score)
        price_score = float(
            parsed.get("price_range") == "unknown"
            and "price_range" in observed_unknown
        )
        expected_price = row["target"].get("price_range", "unknown")
        if expected_price == "unknown":
            price_unknown_scores.append(price_score)
        else:
            known_price_count += 1
        semantic_correct = float(
            parsed.get("business_category") == row["target"]["business_category"]
            and string_set("style_tags") == set(row["target"]["style_tags"])
            and string_set("visible_facilities") == set(row["target"]["visible_facilities"])
            and parsed.get("price_range") == expected_price
            and (price_score == 1.0 if expected_price == "unknown" else True)
        )
        for name in row.get("error_slices", []):
            slice_totals[name] += 1
            slice_correct[name] += semantic_correct
    return {
        "unknown_usage_accuracy": statistics.fmean(unknown_scores),
        "unknown_usage_support": len(unknown_scores),
        "price_unknown_accuracy": statistics.fmean(price_unknown_scores) if price_unknown_scores else None,
        "price_unknown_support": len(price_unknown_scores),
        "known_price_support": known_price_count,
        "error_slices": {
            name: {
                "support": slice_totals[name],
                "strict_semantic_accuracy": slice_correct[name] / slice_totals[name],
            }
            for name in sorted(slice_totals)
        },
    }


def summarize_product_run(root: Path, rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ids = [row["sample_id"] for row in rows]
    observed_ids = [record["sample_id"] for record in records]
    if (
        not rows or len(expected_ids) != len(set(expected_ids))
        or len(observed_ids) != len(set(observed_ids))
        or set(expected_ids) != set(observed_ids)
    ):
        raise Week8ProductError("product records must cover each fixed sample exactly once")
    # 失败后的占位 JSON 只是传输容器，不能当成模型成功预测参与得分。
    scored_records = [
        {**record, "raw_output": ""} if record.get("failed") else record
        for record in records
    ]
    summary = summarize_raw_records(root, _scoring_config(), rows, scored_records)
    summary["scoring_protocol"] = "week8_product_failure_zero_credit_v2"
    latencies = [float(row.get("latency_ms", 0.0)) for row in records]
    inputs = [int(row.get("input_token_count") or 0) for row in records]
    outputs = [int(row.get("generated_token_count") or 0) for row in records]
    summary.update(_unknown_and_slice_metrics(rows, records))
    summary["reference_semantics"] = audit_product_references(rows)
    summary["latency_ms_p50"] = statistics.median(latencies)
    summary["latency_ms_p95"] = _percentile(latencies, 0.95)
    summary["input_tokens_total"] = sum(inputs)
    summary["output_tokens_total"] = sum(outputs)
    summary["input_tokens_mean"] = statistics.fmean(inputs)
    summary["output_tokens_mean"] = statistics.fmean(outputs)
    summary["retry_count"] = sum(max(0, len(row.get("attempts", [])) - 1) for row in records)
    return summary


def run_prompt_development(
    root: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise Week8ProductError(f"refusing to overwrite development run: {output_dir}")
    config = load_week8_product_config(config_path)
    validation = validate_week8_product_lock(root, config_path, include_test=False)
    if validation["status"] != "PASS" or validation["test_status"] != "LOCKED_UNCONSUMED":
        raise Week8ProductError("Week 8 lock is not eligible for development")
    lock_root = root / config["dataset"]["output_root"] / config["week8"]["dataset_version"]
    rows = list(iter_jsonl(lock_root / "development" / "image_product_search.jsonl"))
    release = ReleaseSettings.load(root=root)
    if release.adapter_model_sha256 != config["model"]["adapter_model_sha256"]:
        raise Week8ProductError("release adapter differs from Week 8 baseline")
    backend = TransformersPeftBackend(release)
    ready, reason = backend.ready()
    if not ready:
        raise Week8ProductError(f"release backend is not ready: {reason}")
    output_dir.mkdir(parents=True)
    summaries = {}
    for role, prompt_version in config["prompts"].items():
        run_id = f"{config['experiment_identity']['development_run_id']}__{role}"
        records = [
            _run_one_product(
                root, backend, row, run_id=run_id, prompt_version=prompt_version,
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
            )
            for row in rows
        ]
        role_dir = output_dir / role
        write_jsonl_new(role_dir / "raw_outputs.jsonl", records)
        summary = summarize_product_run(root, rows, records)
        summary.update({
            "status": "COMPLETED",
            "run_id": run_id,
            "role": role,
            "prompt_version": prompt_version,
            "config_sha256": sha256_file(config_path),
            "dataset_lock_sha256": validation["lock_sha256"],
            "raw_outputs_sha256": sha256_file(role_dir / "raw_outputs.jsonl"),
        })
        _write_json_new(role_dir / "metrics.json", summary)
        summaries[role] = summary
    comparison = select_prompt(config, summaries)
    comparison.update({
        "selection_id": config["experiment_identity"]["selection_id"],
        "config_sha256": sha256_file(config_path),
        "dataset_lock_sha256": validation["lock_sha256"],
        "development_sample_ids_sha256": canonical_sha256([row["sample_id"] for row in rows]),
        "metrics_sha256": {
            role: sha256_file(output_dir / role / "metrics.json") for role in summaries
        },
        "test_consumed": (
            None if config.get("week8", {}).get("development_only") is True else False
        ),
        "test_policy": (
            "DISABLED_DEVELOPMENT_ONLY"
            if config.get("week8", {}).get("development_only") is True
            else "LOCKED_UNCONSUMED"
        ),
    })
    _write_json_new(output_dir / "selection.json", comparison)
    return comparison


def select_prompt(config: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(summaries) != set(config["prompts"]):
        raise Week8ProductError("Prompt summaries do not cover all approved roles")
    for role, summary in summaries.items():
        product = summary["scenarios"]["image_product_search"]
        rates = (
            product["composite"], product["aggregate"]["json_compliance"],
            product["aggregate"]["schema_pass"], summary["failure_rate"],
        )
        latency = summary["latency_ms_mean"]
        # NaN 的大小比较恒为 False，必须先拒绝，不能让损坏指标通过选优。
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in rates):
            raise Week8ProductError(f"invalid bounded selection metrics for {role}")
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise Week8ProductError(f"invalid selection latency for {role}")
    baseline = summaries["current_release"]
    baseline_product = baseline["scenarios"]["image_product_search"]
    eligible = []
    reasons = {}
    for role, summary in summaries.items():
        product = summary["scenarios"]["image_product_search"]
        aggregate = product["aggregate"]
        failures = []
        if role != "current_release" and float(product["composite"]) <= float(baseline_product["composite"]):
            failures.append("composite_not_strictly_above_current_release")
        if float(aggregate["json_compliance"]) < float(config["selection"]["minimum_json_compliance"]):
            failures.append("json_compliance_below_gate")
        if float(aggregate["schema_pass"]) < float(config["selection"]["minimum_schema_pass"]):
            failures.append("schema_pass_below_gate")
        if float(summary["failure_rate"]) > float(config["selection"]["maximum_failure_rate"]):
            failures.append("failure_rate_above_gate")
        if product.get("metric_support") != baseline_product.get("metric_support"):
            failures.append("metric_support_changed")
        reasons[role] = failures
        if role != "current_release" and not failures:
            eligible.append(role)
    if not eligible:
        return {
            "status": "SFT_ALLOWED_NO_PROMPT_WINNER",
            "selected_role": None,
            "selected_prompt_version": None,
            "candidate_failures": reasons,
        }
    selected = min(
        eligible,
        key=lambda role: (
            -float(summaries[role]["scenarios"]["image_product_search"]["composite"]),
            float(summaries[role]["latency_ms_mean"]),
            role,
        ),
    )
    return {
        "status": "PROMPT_LOCKED",
        "selected_role": selected,
        "selected_prompt_version": config["prompts"][selected],
        "candidate_failures": reasons,
    }


def run_final_test_once(
    root: Path,
    config_path: Path,
    development_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    development_dir = Path(development_dir).resolve()
    output_dir = Path(output_dir).resolve()
    config = load_week8_product_config(config_path)
    if config.get("week8", {}).get("development_only") is True:
        raise Week8ProductError(
            "development-only Prompt overlays cannot consume a final test"
        )
    lock_root = root / config["dataset"]["output_root"] / config["week8"]["dataset_version"]
    marker_path = lock_root / "test_consumption.json"
    if marker_path.exists() or output_dir.exists():
        raise Week8ProductError("Week 8 final test has already been consumed or started")
    validation = validate_week8_product_lock(root, config_path, include_test=False)
    if validation["status"] != "PASS":
        raise Week8ProductError("final test requires an intact data lock")
    selection_path = development_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "PROMPT_LOCKED" or selection.get("test_consumed") is not False:
        raise Week8ProductError("Prompt must be locked before final test")
    if selection.get("config_sha256") != sha256_file(config_path) or selection.get("dataset_lock_sha256") != validation["lock_sha256"]:
        raise Week8ProductError("development selection identity changed")
    if set(selection.get("metrics_sha256", {})) != set(config["prompts"]):
        raise Week8ProductError("development metric evidence is incomplete")
    development_summaries = {}
    for role, digest in selection["metrics_sha256"].items():
        path = development_dir / role / "metrics.json"
        if not path.is_file() or sha256_file(path) != digest:
            raise Week8ProductError("development metric evidence changed")
        development_summaries[role] = json.loads(path.read_text(encoding="utf-8"))
    recomputed = select_prompt(config, development_summaries)
    if any(selection.get(key) != value for key, value in recomputed.items()):
        raise Week8ProductError("development selection decision changed")
    release = ReleaseSettings.load(root=root)
    if (
        release.base_model != config["model"]["base_model"]
        or release.base_revision != config["model"]["base_revision"]
        or release.adapter_model_sha256 != config["model"]["adapter_model_sha256"]
    ):
        raise Week8ProductError("final model differs from the locked development identity")
    marker = {
        "status": "STARTED",
        "run_id": config["experiment_identity"]["final_test_run_id"],
        "config_sha256": sha256_file(config_path),
        "dataset_lock_sha256": validation["lock_sha256"],
        "selection_sha256": sha256_file(selection_path),
        "started_at_epoch_seconds": time.time(),
    }
    _write_json_new(marker_path, marker)
    # 先锁定选择并创建消费标记，再允许读取最终标签。
    if validate_week8_product_lock(root, config_path)["status"] != "PASS":
        raise Week8ProductError("final test data changed after selection was locked")
    output_dir.mkdir(parents=True)
    rows = list(iter_jsonl(lock_root / "test" / "image_product_search.jsonl"))
    backend = TransformersPeftBackend(release)
    ready, reason = backend.ready()
    if not ready:
        raise Week8ProductError(f"release backend is not ready: {reason}")
    roles = {
        "current_release": config["prompts"]["current_release"],
        selection["selected_role"]: selection["selected_prompt_version"],
    }
    summaries = {}
    for role, prompt_version in roles.items():
        run_id = f"{config['experiment_identity']['final_test_run_id']}__{role}"
        records = [
            _run_one_product(
                root, backend, row, run_id=run_id, prompt_version=prompt_version,
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
            )
            for row in rows
        ]
        role_dir = output_dir / role
        write_jsonl_new(role_dir / "raw_outputs.jsonl", records)
        summary = summarize_product_run(root, rows, records)
        summary.update({
            "status": "COMPLETED", "run_id": run_id, "role": role,
            "prompt_version": prompt_version,
            "raw_outputs_sha256": sha256_file(role_dir / "raw_outputs.jsonl"),
        })
        _write_json_new(role_dir / "metrics.json", summary)
        summaries[role] = summary
    baseline = summaries["current_release"]
    chosen = summaries[selection["selected_role"]]
    comparison = {
        "status": "COMPLETED",
        "run_id": config["experiment_identity"]["final_test_run_id"],
        "selected_role": selection["selected_role"],
        "sample_count": len(rows),
        "label_source": "programmatic_silver",
        "human_count": 0,
        "composite_before": baseline["scenarios"]["image_product_search"]["composite"],
        "composite_after": chosen["scenarios"]["image_product_search"]["composite"],
        "composite_delta": chosen["scenarios"]["image_product_search"]["composite"] - baseline["scenarios"]["image_product_search"]["composite"],
        "baseline": baseline,
        "selected": chosen,
        "selection_sha256": sha256_file(selection_path),
    }
    _write_json_new(output_dir / "comparison.json", comparison)
    marker["status"] = "COMPLETED"
    marker["completed_at_epoch_seconds"] = time.time()
    marker["comparison_sha256"] = sha256_file(output_dir / "comparison.json")
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return comparison

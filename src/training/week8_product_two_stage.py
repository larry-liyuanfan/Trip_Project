"""Two-stage product evidence extraction, mapping, and silver hard-slice SFT."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.inference.system_runtime import ReleaseSettings, TransformersPeftBackend
from src.inference.transport_utils import strip_json_fence
from src.training.week6_qlora import (
    _trainable_parameter_report,
    environment_report,
    resolve_lora_targets,
)
from src.training.week7_data import (
    canonical_sha256,
    iter_jsonl,
    sha256_file,
    write_jsonl_new,
)
from src.training.week7_qlora import (
    IndexedWeek7Dataset,
    assistant_span_labels,
    decile_evaluation_steps,
    structure_aware_messages,
)
from src.training.week7_runtime import inference_runtime
from src.training.week8_product import (
    load_week8_product_config,
    summarize_product_run,
)


class Week8TwoStageError(ValueError):
    """Raised when two-stage or silver hard-slice identity is invalid."""


SUBJECT_CATEGORIES = {"hotel", "restaurant", "attraction", "unknown"}
SUBJECT_CLARITIES = {"clear", "multiple", "ambiguous", "occluded"}
UNCERTAINTY_REASONS = {
    "no_clear_subject",
    "multiple_subjects",
    "occluded_subject",
    "no_style_evidence",
    "no_facility_evidence",
    "no_price_evidence",
}
STYLE_TERMS = {
    "casual",
    "classy",
    "cozy",
    "historic",
    "modern",
    "romantic",
    "rustic",
    "trendy",
    "upscale",
    "vintage",
}
FACILITY_TERMS = {
    "bar": ("bar", "tap"),
    "outdoor_seating": ("patio", "terrace", "outdoor seating"),
    "pool": ("pool",),
    "front_desk": ("front desk", "reception"),
    "parking": ("parking",),
    "wifi": ("wifi", "wi-fi"),
    "wheelchair_access": ("wheelchair", "accessible"),
    "gym": ("gym", "fitness"),
    "spa": ("spa",),
    "playground": ("playground",),
}
MULTI_SUBJECT_RE = re.compile(
    r"\b(?:multiple|several|various|many|crowd|group|people|customers|tables)\b",
    re.IGNORECASE,
)
PRICE_TEXT_RE = re.compile(
    r"(?:[$€£¥]\s*\d+(?:\.\d{1,2})?|\b(?:budget|mid[- ]?range|premium|luxury)\b)",
    re.IGNORECASE,
)


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_two_stage_config(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "week8_product_two_stage_v1":
        raise Week8TwoStageError("unsupported two-stage product config")
    policy = payload.get("policy", {})
    if (
        policy.get("human_annotation") is not False
        or policy.get("human_review") is not False
        or policy.get("human_acceptance") is not False
        or policy.get("new_label_provenance") != "programmatic_silver"
        or policy.get("final_test_access") != "forbidden"
    ):
        raise Week8TwoStageError("two-stage policy must remain fully automatic silver")
    data = payload.get("hard_slice_data", {})
    if (
        data.get("source_splits") != ["train", "development"]
        or int(data.get("train_count", 0)) != 400
        or int(data.get("development_count", 0)) != 60
        or float(data.get("maximum_silver_weight", 1)) > 0.5
        or data.get("category_resampling")
        != "loss_weighted_without_identity_duplication"
    ):
        raise Week8TwoStageError("hard-slice silver data contract changed")
    continuation = payload.get("continuation", {})
    if (
        int(continuation.get("r", 0)) != 16
        or int(continuation.get("lora_alpha", 0)) != 32
        or float(continuation.get("learning_rate", 1)) > 1e-5
        or not 0 < float(continuation.get("epochs", 0)) <= 1
        or float(continuation.get("evaluation_fraction_steps", 0)) != 0.1
    ):
        raise Week8TwoStageError("hard-slice continuation settings changed")
    two_stage = payload.get("two_stage", {})
    if (
        two_stage.get("evidence_schema_version")
        != "product_observable_evidence_v1"
        or int(two_stage.get("max_schema_retries", -1)) != 1
        or two_stage.get("price_mapping_policy") != "explicit_tier_words_only"
    ):
        raise Week8TwoStageError("two-stage evidence or mapping contract changed")
    return payload


def _schema(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = root / config["two_stage"]["evidence_schema_path"]
    if not path.is_file():
        raise Week8TwoStageError("observable-evidence Schema is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_observable_evidence(
    evidence: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "subject_category",
        "subject_clarity",
        "style_cues",
        "facility_cues",
        "price_text",
        "observable_facts",
        "uncertainty_reasons",
    }
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise Week8TwoStageError("observable evidence keys changed")
    if evidence["subject_category"] not in SUBJECT_CATEGORIES:
        raise Week8TwoStageError("invalid observable subject category")
    if evidence["subject_clarity"] not in SUBJECT_CLARITIES:
        raise Week8TwoStageError("invalid observable subject clarity")
    list_contracts = {
        "style_cues": (set(config["two_stage"]["style_vocabulary"]), 8, 40),
        "facility_cues": (set(config["two_stage"]["facility_vocabulary"]), 10, 40),
        "uncertainty_reasons": (UNCERTAINTY_REASONS, 5, 40),
        "price_text": (None, 4, 40),
        "observable_facts": (None, 8, 80),
    }
    normalized = dict(evidence)
    for field, (vocabulary, maximum, maximum_length) in list_contracts.items():
        values = evidence[field]
        if (
            not isinstance(values, list)
            or len(values) > maximum
            or len(values) != len(set(values))
            or any(
                not isinstance(value, str)
                or not value.strip()
                or len(value) > maximum_length
                for value in values
            )
            or (vocabulary is not None and any(value not in vocabulary for value in values))
        ):
            raise Week8TwoStageError(f"invalid observable evidence field: {field}")
        normalized[field] = sorted(values) if vocabulary is not None else values
    return normalized


def map_evidence_to_product(
    evidence: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Deterministically map observable cues and calibrate unsupported fields to unknown."""

    evidence = validate_observable_evidence(evidence, config)
    clear = evidence["subject_clarity"] == "clear"
    category = evidence["subject_category"] if clear else "unknown"
    styles = evidence["style_cues"] if clear else []
    facilities = evidence["facility_cues"] if clear else []
    price_range = "unknown"
    joined_price = " ".join(evidence["price_text"]).casefold()
    tier_patterns = {
        "budget": r"\bbudget\b",
        "mid_range": r"\bmid[- ]?range\b",
        "premium": r"\bpremium\b",
        "luxury": r"\bluxury\b",
    }
    matches = [name for name, pattern in tier_patterns.items() if re.search(pattern, joined_price)]
    if len(matches) == 1:
        price_range = matches[0]
    unknown = []
    if category == "unknown":
        unknown.append("business_category")
    if not styles:
        unknown.append("style_tags")
    if not facilities:
        unknown.append("visible_facilities")
    if price_range == "unknown":
        unknown.append("price_range")
    confidence = 0.75 if clear and category != "unknown" else 0.35
    if evidence["uncertainty_reasons"]:
        confidence = min(confidence, 0.5)
    facts = []
    for value in evidence["observable_facts"]:
        compact = " ".join(value.split())[:80]
        if compact and compact not in facts:
            facts.append(compact)
    return {
        "business_category": category,
        "style_tags": sorted(styles),
        "visible_facilities": sorted(facilities),
        "price_range": price_range,
        "observed_evidence": facts[:8],
        "inferred_attributes": [],
        "unknown_fields": sorted(unknown),
        "confidence": confidence,
    }


def caption_to_silver_evidence(caption: str) -> dict[str, Any]:
    """Create a caption-lexical silver proxy; it is never represented as human truth."""

    text = str(caption or "").strip()
    folded = text.casefold()
    if any(term in folded for term in ("hotel", "resort", "room", "lobby")):
        category = "hotel"
    elif any(term in folded for term in ("restaurant", "cafe", "bar", "dining")):
        category = "restaurant"
    elif any(term in folded for term in ("museum", "park", "attraction", "landmark")):
        category = "attraction"
    else:
        category = "unknown"
    multi = bool(MULTI_SUBJECT_RE.search(text))
    clarity = "multiple" if multi else ("clear" if category != "unknown" else "ambiguous")
    styles = sorted(term for term in STYLE_TERMS if term in folded)
    facilities = sorted(
        name
        for name, terms in FACILITY_TERMS.items()
        if any(term in folded for term in terms)
    )
    price_text = list(dict.fromkeys(match.group(0) for match in PRICE_TEXT_RE.finditer(text)))[:4]
    facts = []
    if category != "unknown":
        facts.append(f"caption names {category} subject")
    facts.extend(f"caption names {value} style" for value in styles)
    facts.extend(f"caption names {value} facility" for value in facilities)
    uncertainty = []
    if category == "unknown":
        uncertainty.append("no_clear_subject")
    if multi:
        uncertainty.append("multiple_subjects")
    if not styles:
        uncertainty.append("no_style_evidence")
    if not facilities:
        uncertainty.append("no_facility_evidence")
    if not price_text:
        uncertainty.append("no_price_evidence")
    return {
        "subject_category": category,
        "subject_clarity": clarity,
        "style_cues": styles,
        "facility_cues": facilities,
        "price_text": price_text,
        "observable_facts": facts[:8],
        "uncertainty_reasons": uncertainty,
    }


def evidence_messages(
    row: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    image_path = str(row["image_path"]).replace("\\", "/")
    return [
        {"role": "system", "content": config["two_stage"]["system_prompt"]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                {"type": "text", "text": config["two_stage"]["task_prompt"]},
            ],
        },
    ]


def _product_lock_development(
    root: Path, product_config_path: Path, product_config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    lock_root = (
        root
        / product_config["dataset"]["output_root"]
        / product_config["week8"]["dataset_version"]
    )
    lock = json.loads((lock_root / "dataset_lock.json").read_text(encoding="utf-8"))
    core = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if (
        lock.get("lock_sha256") != canonical_sha256(core)
        or lock.get("config_sha256") != sha256_file(product_config_path)
    ):
        raise Week8TwoStageError("product development lock identity changed")
    for split in ("train", "development"):
        relative = f"{split}/image_product_search.jsonl"
        path = lock_root / relative
        if (
            not path.is_file()
            or sha256_file(path)
            != lock.get("files", {}).get(relative, {}).get("sha256")
        ):
            raise Week8TwoStageError(f"product {split} lock identity changed")
    return lock_root, lock


def _generate_evidence(
    root: Path,
    backend: TransformersPeftBackend,
    row: dict[str, Any],
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    messages = evidence_messages(row, config)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": config["two_stage"]["evidence_schema_version"],
            "strict": True,
            "schema": _schema(root, config),
        },
    }
    attempts = []
    input_tokens = 0
    output_tokens = 0
    active_messages = messages
    parsed = None
    for attempt in (1, 2):
        started = time.perf_counter()
        error = None
        raw = ""
        try:
            result = backend.generate_with_usage(
                active_messages,
                response_format=response_format,
                max_new_tokens=int(config["two_stage"]["max_new_tokens"]),
            )
            raw = result.content
            input_tokens += result.input_tokens
            output_tokens += result.output_tokens
            parsed = validate_observable_evidence(
                json.loads(strip_json_fence(raw)), config
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        attempts.append(
            {
                "attempt": attempt,
                "raw_output": raw,
                "error": error,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
        if error is None:
            break
        active_messages = [
            *messages,
            {
                "role": "user",
                "content": "上一次未通过证据 Schema。重新观察原图，只输出完整证据 JSON；不要解释或猜测。",
            },
        ]
    failed = parsed is None
    final = map_evidence_to_product(parsed, config) if parsed is not None else {
        "business_category": "unknown",
        "style_tags": [],
        "visible_facilities": [],
        "price_range": "unknown",
        "observed_evidence": [],
        "inferred_attributes": [],
        "unknown_fields": ["business_category", "price_range", "style_tags", "visible_facilities"],
        "confidence": 0.0,
    }
    return {
        "run_id": run_id,
        "sample_id": row["sample_id"],
        "scenario": "image_product_search",
        "model_name": "Qwen3-VL-8B+two-stage-evidence",
        "raw_output": json.dumps(final, ensure_ascii=False, sort_keys=True),
        "evidence_output": parsed,
        "evidence_raw_output": attempts[-1]["raw_output"],
        "evidence_schema_pass": not failed,
        "failed": failed,
        "error": attempts[-1]["error"] if failed else None,
        "attempts": attempts,
        "latency_ms": sum(item["latency_ms"] for item in attempts),
        "input_token_count": input_tokens,
        "generated_token_count": output_tokens,
        "generation_max_new_tokens": int(config["two_stage"]["max_new_tokens"]),
        "mapping_version": "deterministic_product_mapping_v1",
    }


def run_two_stage_development(
    root: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise Week8TwoStageError("refusing to overwrite two-stage development")
    config = load_two_stage_config(config_path)
    product_config_path = root / config["product_config_path"]
    product_config = load_week8_product_config(product_config_path)
    lock_root, lock = _product_lock_development(
        root, product_config_path, product_config
    )
    rows = list(iter_jsonl(lock_root / "development" / "image_product_search.jsonl"))
    release = ReleaseSettings.load(root=root)
    if (
        release.base_model != config["model"]["base_model"]
        or release.base_revision != config["model"]["base_revision"]
        or release.adapter_model_sha256 != config["model"]["adapter_model_sha256"]
    ):
        raise Week8TwoStageError("release adapter differs from two-stage baseline")
    backend = TransformersPeftBackend(release)
    ready, reason = backend.ready()
    if not ready:
        raise Week8TwoStageError(f"two-stage backend is not ready: {reason}")
    run_id = config["experiment_identity"]["development_run_id"]
    records = [
        _generate_evidence(root, backend, row, config, run_id) for row in rows
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "raw_outputs.jsonl"
    write_jsonl_new(raw_path, records)
    summary = summarize_product_run(root, rows, records)
    summary.update(
        {
            "status": "COMPLETED",
            "run_id": run_id,
            "candidate": "observable_evidence_then_deterministic_mapping_v1",
            "config_sha256": sha256_file(config_path),
            "product_config_sha256": sha256_file(product_config_path),
            "dataset_lock_sha256": lock["lock_sha256"],
            "development_count": len(rows),
            "evidence_schema_pass_rate": sum(
                bool(row["evidence_schema_pass"]) for row in records
            )
            / len(records),
            "label_provenance": "programmatic_silver",
            "human_annotation_count": 0,
            "human_review_count": 0,
            "test_accessed": False,
            "raw_outputs_sha256": sha256_file(raw_path),
        }
    )
    _write_json_new(output_dir / "metrics.json", summary)
    return summary


def _parquet_caption_map(path: Path) -> dict[str, str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise Week8TwoStageError("pyarrow is required for hard-slice data") from exc
    result = {}
    source = parquet.ParquetFile(path)
    for batch in source.iter_batches(batch_size=4096, columns=["photo_id", "caption"]):
        for row in batch.to_pylist():
            result[str(row["photo_id"])] = str(row.get("caption") or "")
    return result


def _hard_slices(evidence: dict[str, Any], mapped: dict[str, Any]) -> list[str]:
    slices = ["unknown_price_negative"]
    if mapped["business_category"] in {"hotel", "attraction"}:
        slices.append(f"rare_category_{mapped['business_category']}")
    if mapped["business_category"] == "unknown":
        slices.append("unknown_category_negative")
    if not mapped["style_tags"]:
        slices.append("unknown_style_negative")
    elif len(mapped["style_tags"]) >= 2:
        slices.append("style_multilabel")
    if not mapped["visible_facilities"]:
        slices.append("unknown_facility_negative")
    else:
        slices.append("facility_positive")
    if evidence["subject_clarity"] != "clear":
        slices.append("multiple_or_ambiguous_subject")
    return sorted(set(slices))


def _hard_slice_row(
    row: dict[str, Any], caption: str, config: dict[str, Any]
) -> dict[str, Any]:
    evidence = validate_observable_evidence(
        caption_to_silver_evidence(caption), config
    )
    mapped = map_evidence_to_product(evidence, config)
    slices = _hard_slices(evidence, mapped)
    original_category = str(row.get("target", {}).get("business_category") or "unknown")
    category_weight = float(
        config["hard_slice_data"]["category_weights"].get(original_category, 0.5)
    )
    has_unknown_negative = any("unknown_" in value for value in slices)
    weight = max(
        category_weight,
        float(config["hard_slice_data"]["unknown_negative_weight"])
        if has_unknown_negative
        else 0.0,
    )
    weight = min(weight, float(config["hard_slice_data"]["maximum_silver_weight"]))
    result = dict(row)
    result.update(
        {
            "sample_id": f"w8-hard-{row['split']}-{row['sample_id']}",
            "source_sample_id": row["sample_id"],
            "label_source": "programmatic_silver",
            "sample_weight": weight,
            "evidence_target": evidence,
            "mapped_silver_target": mapped,
            "hard_slices": slices,
            "target_provenance": {
                "source": "official_yelp_native_caption_lexical_proxy",
                "label_tier": "programmatic_silver",
                "human_annotation": False,
                "human_review": False,
                "human_acceptance": False,
                "mapping_version": "deterministic_product_mapping_v1",
            },
        }
    )
    return result


def build_hard_slice_silver_lock(
    root: Path, config_path: Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = load_two_stage_config(config_path)
    product_config_path = root / config["product_config_path"]
    product_config = load_week8_product_config(product_config_path)
    lock_root, source_lock = _product_lock_development(
        root, product_config_path, product_config
    )
    output = (
        root
        / config["hard_slice_data"]["output_root"]
        / config["hard_slice_data"]["dataset_version"]
    )
    if output.exists():
        raise Week8TwoStageError("refusing to overwrite hard-slice silver lock")
    captions = _parquet_caption_map(
        root / product_config["dataset"]["source_paths"]["photos"]
    )
    rows_by_split = {}
    for split in ("train", "development"):
        source_path = lock_root / split / "image_product_search.jsonl"
        source_rows = list(iter_jsonl(source_path))
        expected = int(config["hard_slice_data"][f"{split}_count"])
        if len(source_rows) != expected:
            raise Week8TwoStageError(f"hard-slice {split} count changed")
        derived = []
        for row in source_rows:
            source_id = str(row.get("source_id") or "")
            if not source_id.startswith("yelp-photo:"):
                raise Week8TwoStageError("hard-slice source is not official Yelp photo")
            photo_id = source_id.removeprefix("yelp-photo:")
            if photo_id not in captions:
                raise Week8TwoStageError(f"missing native caption for {source_id}")
            derived.append(_hard_slice_row(row, captions[photo_id], config))
        rows_by_split[split] = derived
    for field in ("sample_id", "source_id", "image_sha256", "group_id"):
        train_values = {row.get(field) for row in rows_by_split["train"] if row.get(field)}
        dev_values = {row.get(field) for row in rows_by_split["development"] if row.get(field)}
        if train_values & dev_values:
            raise Week8TwoStageError(f"hard-slice train/development {field} overlap")
    files = {}
    for split, rows in rows_by_split.items():
        path = output / split / "image_product_search.jsonl"
        write_jsonl_new(path, rows)
        files[path.relative_to(output).as_posix()] = {
            "count": len(rows),
            "sha256": sha256_file(path),
        }
    lock_core = {
        "schema_version": "week8_product_hard_slice_silver_lock_v1",
        "dataset_version": config["hard_slice_data"]["dataset_version"],
        "build_id": config["experiment_identity"]["hard_slice_build_id"],
        "config_sha256": sha256_file(config_path),
        "product_config_sha256": sha256_file(product_config_path),
        "source_product_lock_sha256": source_lock["lock_sha256"],
        "git_commit": _git_commit(root),
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "files": files,
        "label_provenance": "programmatic_silver",
        "human_annotation_count": 0,
        "human_review_count": 0,
        "human_acceptance_count": 0,
        "category_resampling": config["hard_slice_data"]["category_resampling"],
        "sample_weight_counts": {
            split: dict(sorted(Counter(str(row["sample_weight"]) for row in rows).items()))
            for split, rows in rows_by_split.items()
        },
        "hard_slice_counts": {
            split: dict(sorted(Counter(value for row in rows for value in row["hard_slices"]).items()))
            for split, rows in rows_by_split.items()
        },
        "final_test_included": False,
        "final_test_accessed": False,
    }
    lock = {**lock_core, "lock_sha256": canonical_sha256(lock_core)}
    _write_json_new(output / "dataset_lock.json", lock)
    return lock


def validate_hard_slice_lock(
    root: Path, config_path: Path
) -> tuple[Path, dict[str, Any]]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = load_two_stage_config(config_path)
    output = (
        root
        / config["hard_slice_data"]["output_root"]
        / config["hard_slice_data"]["dataset_version"]
    )
    lock_path = output / "dataset_lock.json"
    if not lock_path.is_file():
        raise Week8TwoStageError("hard-slice silver lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    core = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if (
        lock.get("lock_sha256") != canonical_sha256(core)
        or lock.get("config_sha256") != sha256_file(config_path)
        or lock.get("final_test_included") is not False
        or lock.get("final_test_accessed") is not False
        or lock.get("label_provenance") != "programmatic_silver"
        or any(int(lock.get(field, -1)) != 0 for field in (
            "human_annotation_count", "human_review_count", "human_acceptance_count"
        ))
    ):
        raise Week8TwoStageError("hard-slice silver lock identity changed")
    for split in ("train", "development"):
        relative = f"{split}/image_product_search.jsonl"
        path = output / relative
        evidence = lock.get("files", {}).get(relative, {})
        if (
            not path.is_file()
            or sha256_file(path) != evidence.get("sha256")
            or int(evidence.get("count", -1))
            != int(config["hard_slice_data"][f"{split}_count"])
        ):
            raise Week8TwoStageError(f"hard-slice {split} artifact changed")
    return output, lock


def _evidence_training_messages(
    row: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    from src.inference.system_runtime import _transformers_messages

    messages = _transformers_messages(evidence_messages(row, config))
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                row["evidence_target"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    )
    return messages


def _in_memory_backend(
    model: Any, processor: Any, torch_module: Any
) -> TransformersPeftBackend:
    backend = TransformersPeftBackend.__new__(TransformersPeftBackend)
    backend.settings = None
    backend._model = model
    backend._processor = processor
    backend._torch = torch_module
    return backend


def run_two_stage_continuation_sft(
    root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    resume_from_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Continue checkpoint-87 on evidence targets and select only on development."""

    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = load_two_stage_config(config_path)
    data_root, data_lock = validate_hard_slice_lock(root, config_path)
    adapter_value = os.environ.get(config["model"]["adapter_path_env"], "").strip()
    if not adapter_value:
        raise Week8TwoStageError("TRIP_INITIAL_ADAPTER_DIR is required")
    initial_adapter = Path(adapter_value).resolve()
    try:
        output_dir.relative_to(initial_adapter)
    except ValueError:
        pass
    else:
        raise Week8TwoStageError("two-stage output must not overwrite initial adapter")
    run_id = config["experiment_identity"]["continuation_run_id"]
    declared_run_id = os.environ.get("TRIP_RUN_ID")
    if declared_run_id is not None and declared_run_id != run_id:
        raise Week8TwoStageError("TRIP_RUN_ID differs from two-stage continuation config")
    identity = {
        "run_id": run_id,
        "git_commit": _git_commit(root),
        "config_sha256": sha256_file(config_path),
        "hard_slice_lock_sha256": data_lock["lock_sha256"],
        "source_product_lock_sha256": data_lock["source_product_lock_sha256"],
    }
    if output_dir.exists() and resume_from_checkpoint is None:
        raise Week8TwoStageError("refusing to overwrite two-stage SFT run")
    if resume_from_checkpoint is not None:
        resume_from_checkpoint = Path(resume_from_checkpoint).resolve()
        try:
            resume_from_checkpoint.relative_to(output_dir)
        except ValueError as exc:
            raise Week8TwoStageError("resume checkpoint must stay inside run directory") from exc
        identity_path = output_dir / "run_identity.json"
        if (
            not identity_path.is_file()
            or json.loads(identity_path.read_text(encoding="utf-8")) != identity
        ):
            raise Week8TwoStageError("two-stage resume identity changed")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json_new(output_dir / "run_identity.json", identity)

    environment = environment_report(require_cuda=True)
    if environment["status"] != "ok":
        raise Week8TwoStageError(
            f"two-stage training environment is not ready: {environment['status']}"
        )
    import torch
    from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
    from peft.utils.save_and_load import load_peft_weights
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    adapter_file = initial_adapter / "adapter_model.safetensors"
    if (
        not adapter_file.is_file()
        or sha256_file(adapter_file) != config["model"]["adapter_model_sha256"]
    ):
        raise Week8TwoStageError("initial checkpoint-87 adapter identity changed")
    continuation = config["continuation"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    torch.cuda.reset_peak_memory_stats()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model"]["base_model"],
        revision=config["model"]["base_revision"],
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    resolved_targets = set(
        resolve_lora_targets(
            model,
            {
                "lora": {
                    "language_target_suffixes": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "visual_projection_candidates": [
                        "visual_projection",
                        "visual.merger.linear_fc1",
                        "visual.merger.linear_fc2",
                    ],
                }
            },
        )
    )
    initial_config = PeftConfig.from_pretrained(str(initial_adapter))
    expected_targets = set(continuation["expected_target_modules"])
    if (
        int(initial_config.r) != int(continuation["r"])
        or int(initial_config.lora_alpha) != int(continuation["lora_alpha"])
        or float(initial_config.lora_dropout) != float(continuation["lora_dropout"])
        or str(initial_config.bias) != continuation["bias"]
        or str(initial_config.base_model_name_or_path) != config["model"]["base_model"]
        or set(initial_config.target_modules or []) != expected_targets
        or resolved_targets != expected_targets
    ):
        raise Week8TwoStageError("checkpoint-87 LoRA structure changed")
    model = PeftModel.from_pretrained(
        model, str(initial_adapter), is_trainable=True
    )
    parameter_report = _trainable_parameter_report(model)
    processor = AutoProcessor.from_pretrained(
        config["model"]["base_model"], revision=config["model"]["base_revision"]
    )
    train_dataset = IndexedWeek7Dataset(
        data_root / "train" / "image_product_search.jsonl"
    )
    development_rows = list(
        iter_jsonl(data_root / "development" / "image_product_search.jsonl")
    )
    eval_dataset = IndexedWeek7Dataset(
        data_root / "development" / "image_product_search.jsonl"
    )

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) != 1:
            raise Week8TwoStageError("two-stage multimodal batch size must be one")
        row = batch[0]
        if row.get("label_source") != "programmatic_silver":
            raise Week8TwoStageError("two-stage training row is not silver")
        messages = structure_aware_messages(
            processor,
            _evidence_training_messages(row, config),
            int(continuation["max_length"]),
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
        )
        inputs["labels"] = assistant_span_labels(
            processor, messages, inputs["input_ids"]
        )
        weight = float(row["sample_weight"])
        if not 0 < weight <= float(config["hard_slice_data"]["maximum_silver_weight"]):
            raise Week8TwoStageError("two-stage silver weight exceeds 0.5")
        inputs["_sample_weight"] = torch.tensor(weight)
        return inputs

    evaluation_steps: list[int] = []

    class DecileCallback(TrainerCallback):
        def on_train_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            evaluation_steps[:] = decile_evaluation_steps(int(state.max_steps))
            return control

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            if int(state.global_step) in evaluation_steps:
                control.should_evaluate = True
                control.should_save = True
            return control

    class TwoStageTrainer(Trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._evaluation_cache: dict[int, dict[str, float]] = {}

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            weight = inputs.pop("_sample_weight")
            outputs = model(**inputs)
            loss = outputs.loss * weight.to(outputs.loss.device)
            return (loss, outputs) if return_outputs else loss

        def evaluate(
            self,
            eval_dataset: Any = None,
            ignore_keys: Any = None,
            metric_key_prefix: str = "eval",
        ) -> dict[str, float]:
            del eval_dataset, ignore_keys
            step = int(self.state.global_step)
            if step in self._evaluation_cache:
                return self._evaluation_cache[step]
            started = time.perf_counter()
            backend = _in_memory_backend(self.model, processor, torch)
            with inference_runtime(self.model):
                records = [
                    _generate_evidence(
                        root,
                        backend,
                        row,
                        config,
                        f"{run_id}_development_step_{step:06d}",
                    )
                    for row in development_rows
                ]
            summary = summarize_product_run(root, development_rows, records)
            product = summary["scenarios"]["image_product_search"]
            evaluation_dir = output_dir / "development_evaluations" / f"step-{step:06d}"
            evaluation_dir.mkdir(parents=True, exist_ok=False)
            raw_path = evaluation_dir / "raw_outputs.jsonl"
            write_jsonl_new(raw_path, records)
            summary.update(
                {
                    "status": "COMPLETED",
                    "split": "development",
                    "run_id": f"{run_id}_development_step_{step:06d}",
                    "global_step": step,
                    "config_sha256": sha256_file(config_path),
                    "hard_slice_lock_sha256": data_lock["lock_sha256"],
                    "raw_outputs_sha256": sha256_file(raw_path),
                    "test_accessed": False,
                }
            )
            _write_json_new(evaluation_dir / "metrics.json", summary)
            metrics = {
                f"{metric_key_prefix}_product_composite": float(product["composite"]),
                f"{metric_key_prefix}_failure_rate": float(summary["failure_rate"]),
                f"{metric_key_prefix}_runtime": time.perf_counter() - started,
            }
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics
            )
            self._evaluation_cache[step] = metrics
            return metrics

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=continuation["epochs"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=continuation["gradient_accumulation_steps"],
        learning_rate=continuation["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.03,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=1_000_000_000,
        save_strategy="steps",
        save_steps=1_000_000_000,
        save_total_limit=continuation["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_product_composite",
        greater_is_better=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = TwoStageTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
        callbacks=[
            DecileCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=int(continuation["early_stopping_patience"])
            ),
        ],
    )
    started = time.time()
    result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint else None
        )
    )
    adapter_output = output_dir / "adapter"
    trainer.save_model(str(adapter_output))
    trainer.save_state()
    processor.save_pretrained(str(output_dir / "processor"))
    state = load_peft_weights(str(adapter_output), device="cpu")
    saved_config = PeftConfig.from_pretrained(str(adapter_output))
    if (
        not state
        or not all("lora_" in name for name in state)
        or int(saved_config.r) != int(continuation["r"])
        or int(saved_config.lora_alpha) != int(continuation["lora_alpha"])
        or float(saved_config.lora_dropout) != float(continuation["lora_dropout"])
        or str(saved_config.bias) != continuation["bias"]
        or str(saved_config.base_model_name_or_path) != config["model"]["base_model"]
        or set(saved_config.target_modules or []) != expected_targets
    ):
        raise Week8TwoStageError("saved two-stage adapter failed LoRA-only reload")
    summary = {
        "status": "COMPLETED",
        "run_id": run_id,
        "git_commit": _git_commit(root),
        "config_sha256": sha256_file(config_path),
        "hard_slice_lock_sha256": data_lock["lock_sha256"],
        "source_product_lock_sha256": data_lock["source_product_lock_sha256"],
        "train_samples": len(train_dataset),
        "development_samples": len(eval_dataset),
        "global_step": int(trainer.state.global_step),
        "evaluation_steps": evaluation_steps,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "adapter_model_sha256": sha256_file(adapter_output / "adapter_model.safetensors"),
        "adapter_only": True,
        "adapter_reload_verified": True,
        "label_provenance": "programmatic_silver",
        "human_annotation_count": 0,
        "human_review_count": 0,
        "human_acceptance_count": 0,
        "test_accessed": False,
        "continued_from_adapter_sha256": sha256_file(adapter_file),
        "training_metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "duration_seconds": time.time() - started,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "resumed_from_checkpoint": (
            str(resume_from_checkpoint) if resume_from_checkpoint else None
        ),
        **parameter_report,
    }
    _write_json_new(output_dir / "run_summary.json", summary)
    return summary

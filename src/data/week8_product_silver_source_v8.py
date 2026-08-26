"""Audit untouched Yelp sources for conservative Week 8 product silver evidence."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from src.data.yelp_archives import extract_yelp_photo_files
from src.training.week7_data import (
    IDENTITY_FIELDS,
    add_superseded_identities,
    canonical_sha256,
    load_consumed_identities,
    sha256_file,
)


class Week8SilverSourceAuditError(ValueError):
    """Raised when source identity or conservative evidence rules change."""


AMOUNT_RE = re.compile(
    r"(?:(?P<symbol>[$€£¥])\s*(?P<symbol_amount>\d{1,5}(?:[.,]\d{1,2})?)"
    r"|(?P<word_amount>\d{1,5}(?:[.,]\d{1,2})?)\s*"
    r"(?P<currency>usd|aud|cny|dollars?))",
    re.IGNORECASE,
)
TIER_RE = re.compile(
    r"\b(?:budget|cheap|affordable|mid[- ]range|premium|expensive|luxury)\b",
    re.IGNORECASE,
)
def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week8SilverSourceAuditError(f"invalid JSON artifact: {path}") from exc


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def load_silver_source_audit_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    policy = config.get("policy", {})
    source = config.get("source", {})
    if (
        config.get("schema_version") != "week8_product_silver_source_audit_v8"
        or policy.get("label_provenance") != "programmatic_silver"
        or policy.get("human_annotation") is not False
        or policy.get("human_review") is not False
        or policy.get("human_acceptance") is not False
        or policy.get("read_final_test_rows") is not False
        or policy.get("read_final_test_outputs") is not False
        or policy.get("metadata_is_visual_evidence") is not False
        or policy.get("caption_is_visual_price_confirmation") is not False
        or not 0 < float(policy.get("maximum_silver_weight", 1)) <= 0.5
    ):
        raise Week8SilverSourceAuditError("automatic silver policy changed")
    if any("/test/" in str(value).replace("\\", "/") for value in source.values()):
        raise Week8SilverSourceAuditError("audit config must not reference final-test rows")
    terms = config.get("category_terms", {})
    if terms.get("preserve_existing_classification") is not True:
        raise Week8SilverSourceAuditError("existing Yelp classification must be preserved")
    expected = {"hotel", "attraction", "restaurant"}
    if set(terms.get("existing", {})) != expected or set(
        terms.get("strict_expansion", {})
    ) != expected:
        raise Week8SilverSourceAuditError("category term contract is incomplete")
    price = config.get("visible_price_evidence", {})
    if (
        price.get("require_caption_ocr_exact_token_agreement") is not True
        or price.get("numeric_amount_maps_to_price_range") is not False
        or price.get("ocr_engine") != "tesseract"
    ):
        raise Week8SilverSourceAuditError("visible-price evidence contract changed")
    if int(config.get("candidate_policy", {}).get("maximum_candidate_images", 0)) not in range(
        1, 1001
    ):
        raise Week8SilverSourceAuditError("candidate image audit must remain bounded")
    return config


def strict_ota_category(
    categories: Any, config: dict[str, Any]
) -> tuple[str | None, str | None, str | None]:
    """Preserve current classification, then apply exact expanded Yelp terms."""

    if isinstance(categories, str):
        try:
            values = json.loads(categories)
        except json.JSONDecodeError:
            values = categories.split(",")
    else:
        values = categories or []
    normalized = {str(value).strip().casefold() for value in values if str(value).strip()}
    terms = config["category_terms"]
    for category in ("hotel", "attraction", "restaurant"):
        matches = normalized & {
            str(value).casefold() for value in terms["existing"][category]
        }
        if matches:
            return category, "existing_strict_yelp_category_silver", sorted(matches)[0]
    for category in ("hotel", "attraction", "restaurant"):
        matches = normalized & {
            str(value).casefold() for value in terms["strict_expansion"][category]
        }
        if matches:
            return category, "expanded_strict_yelp_category_silver", sorted(matches)[0]
    return None, None, None


def price_tokens(text: str) -> dict[str, list[str]]:
    """Extract normalized price tokens without converting amounts to tiers."""

    amounts = set()
    for match in AMOUNT_RE.finditer(str(text or "")):
        number = (match.group("symbol_amount") or match.group("word_amount") or "").replace(
            ",", "."
        )
        number = number.rstrip("0").rstrip(".") if "." in number else number
        if match.group("symbol"):
            currency = {"$": "dollar", "€": "eur", "£": "gbp", "¥": "yen"}[
                match.group("symbol")
            ]
        else:
            word = str(match.group("currency") or "").casefold()
            currency = "dollar" if word.startswith("dollar") else word
        amounts.add(f"{currency}:{number}")
    tiers = {
        re.sub(r"\s+", " ", match.group(0).casefold().replace("-", " ")).strip()
        for match in TIER_RE.finditer(str(text or ""))
    }
    return {"amounts": sorted(amounts), "tiers": sorted(tiers)}


def confirmed_visible_price_tokens(caption: str, ocr_text: str) -> dict[str, list[str]]:
    caption_tokens = price_tokens(caption)
    ocr_tokens = price_tokens(ocr_text)
    return {
        "amounts": sorted(set(caption_tokens["amounts"]) & set(ocr_tokens["amounts"])),
        "tiers": sorted(set(caption_tokens["tiers"]) & set(ocr_tokens["tiers"])),
    }


def map_confirmed_tier(
    confirmed: dict[str, list[str]], config: dict[str, Any]
) -> str:
    mapping = {
        re.sub(r"\s+", " ", key.casefold().replace("-", " ")).strip(): value
        for key, value in config["visible_price_evidence"]["explicit_tier_mapping"].items()
    }
    mapped = {mapping[value] for value in confirmed["tiers"] if value in mapping}
    return next(iter(mapped)) if len(mapped) == 1 else "unknown"


def _caption_signals(caption: str, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    text = str(caption or "").casefold()
    styles = [
        value
        for value in config["caption_evidence"]["style_vocabulary"]
        if re.search(rf"\b{re.escape(value)}\b", text)
    ]
    facility_terms = {
        "bar": ("bar",),
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
    facilities = [
        value
        for value in config["caption_evidence"]["facility_vocabulary"]
        if any(term in text for term in facility_terms.get(value, (value,)))
    ]
    return sorted(styles), sorted(facilities)


def _parquet_rows(path: Path, columns: list[str]) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise Week8SilverSourceAuditError("pyarrow is required for source audit") from exc
    source = parquet.ParquetFile(path)
    for batch in source.iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


def _verify_identity(
    root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source = config["source"]
    fresh_path = root / source["fresh_manifest_path"]
    lock_path = root / source["dataset_lock_path"]
    product_config_path = root / source["product_config_path"]
    if (
        sha256_file(fresh_path) != source["fresh_manifest_sha256"]
        or sha256_file(product_config_path) != source["product_config_sha256"]
    ):
        raise Week8SilverSourceAuditError("v7 fresh manifest hash changed")
    fresh = _read_json(fresh_path)
    lock = _read_json(lock_path)
    lock_core = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if (
        fresh.get("status") != "COMPLETED"
        or fresh.get("historical_identity_overlap_counts")
        != {"source_id": 0, "group_id": 0, "image_sha256": 0}
        or lock.get("lock_sha256") != source["dataset_lock_sha256"]
        or lock.get("lock_sha256") != canonical_sha256(lock_core)
        or lock.get("isolation", {}).get("status") != "PASS"
    ):
        raise Week8SilverSourceAuditError("v7 source or lock identity changed")
    identity_path = root / source["identity_manifest_path"]
    declared = lock.get("files", {}).get("identity_manifest.jsonl", {})
    if sha256_file(identity_path) != declared.get("sha256"):
        raise Week8SilverSourceAuditError("v7 identity manifest hash changed")
    fresh_identity_path = root / source["fresh_identity_path"]
    if sha256_file(fresh_identity_path) != fresh.get("identity_manifest", {}).get(
        "sha256"
    ):
        raise Week8SilverSourceAuditError("v7 fresh identity manifest hash changed")
    for relative, evidence in fresh.get("rebuilt_inputs", {}).items():
        # v7 manifest records the canonical rebuilt tables, not generated fresh tables.
        path = root / "data/yelp" / relative
        if not path.is_file() or sha256_file(path) != evidence.get("sha256"):
            raise Week8SilverSourceAuditError(f"rebuilt Yelp input changed: {relative}")
    return fresh, lock, list(_iter_jsonl(identity_path))


def _ocr_image(image_path: Path, config: dict[str, Any]) -> str:
    executable = shutil.which(config["visible_price_evidence"]["ocr_engine"])
    if not executable:
        raise Week8SilverSourceAuditError("configured Tesseract executable is unavailable")
    result = subprocess.run(
        [
            executable,
            str(image_path),
            "stdout",
            "-l",
            str(config["visible_price_evidence"]["ocr_language"]),
            "--psm",
            str(config["visible_price_evidence"]["ocr_page_segmentation_mode"]),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=int(config["visible_price_evidence"]["per_image_timeout_seconds"]),
    )
    if result.returncode != 0:
        raise Week8SilverSourceAuditError("Tesseract failed on a bounded candidate image")
    return result.stdout


def _validate_ocr_runtime(config: dict[str, Any]) -> None:
    executable = shutil.which(config["visible_price_evidence"]["ocr_engine"])
    if not executable:
        raise Week8SilverSourceAuditError("configured Tesseract executable is unavailable")
    result = subprocess.run(
        [executable, "--version"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    expected = str(config["visible_price_evidence"]["ocr_version"])
    if result.returncode != 0 or expected not in result.stdout.splitlines()[0]:
        raise Week8SilverSourceAuditError("Tesseract version differs from the audit lock")


def audit_silver_sources(
    root: Path,
    config_path: Path,
    *,
    output_dir: Path | None = None,
    photos_zip: Path | None = None,
    run_ocr: bool = False,
    verify_zip_sha256: bool = True,
) -> dict[str, Any]:
    """Audit sources without reading any v7 final-test row or model output."""

    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = load_silver_source_audit_config(config_path)
    fresh, lock, locked_identities = _verify_identity(root, config)
    product_config = _read_json(root / config["source"]["product_config_path"])
    compatibility = {
        "dataset": {
            "seed": int(product_config["week8"]["seed"]),
            "source_paths": product_config["dataset"]["source_paths"],
        },
        "system_repair": {"enabled": True},
    }
    consumed, exclusion_evidence = load_consumed_identities(root, compatibility)
    add_superseded_identities(root, compatibility, consumed, exclusion_evidence)
    for row in locked_identities:
        for field in IDENTITY_FIELDS:
            if row.get(field):
                consumed[field].add(str(row[field]))

    fresh_identities = list(_iter_jsonl(root / config["source"]["fresh_identity_path"]))
    locked_sources = {str(row["source_id"]) for row in locked_identities}
    locked_groups = {str(row["group_id"]) for row in locked_identities}
    untouched_fresh = [
        row
        for row in fresh_identities
        if row["source_id"] not in locked_sources and row["group_id"] not in locked_groups
    ]

    businesses: dict[str, dict[str, Any]] = {}
    eligible_business_counts: Counter[str] = Counter()
    expanded_business_terms: Counter[str] = Counter()
    for row in _parquet_rows(
        root / config["source"]["raw_business_path"],
        ["business_id", "categories", "attributes"],
    ):
        business_id = str(row.get("business_id") or "")
        group_id = f"yelp-business:{business_id}"
        if not business_id or group_id in consumed["group_id"]:
            continue
        category, provenance, matched_term = strict_ota_category(row.get("categories"), config)
        if category is None:
            continue
        businesses[business_id] = {
            "category": category,
            "category_provenance": provenance,
            "matched_category_term": matched_term,
            "attributes": row.get("attributes") or "{}",
        }
        eligible_business_counts[category] += 1
        if provenance == "expanded_strict_yelp_category_silver":
            expanded_business_terms[f"{category}:{matched_term}"] += 1

    candidates: list[dict[str, Any]] = []
    photo_upper_bounds: Counter[str] = Counter()
    expanded_photo_terms: Counter[str] = Counter()
    amount_candidate_counts: Counter[str] = Counter()
    tier_candidate_counts: Counter[str] = Counter()
    for row in _parquet_rows(
        root / config["source"]["raw_photos_path"],
        ["photo_id", "business_id", "caption", "label"],
    ):
        photo_id = str(row.get("photo_id") or "")
        business_id = str(row.get("business_id") or "")
        source_id = f"yelp-photo:{photo_id}"
        business = businesses.get(business_id)
        if not photo_id or business is None or source_id in consumed["source_id"]:
            continue
        category = business["category"]
        photo_upper_bounds[category] += 1
        caption = str(row.get("caption") or "").strip()
        tokens = price_tokens(caption)
        if tokens["amounts"]:
            amount_candidate_counts[category] += 1
        if tokens["tiers"]:
            tier_candidate_counts[category] += 1
        category_candidate = category in set(
            config["candidate_policy"]["include_business_categories"]
        )
        price_candidate = bool(tokens["amounts"] or tokens["tiers"])
        if not category_candidate and not price_candidate:
            continue
        if business["category_provenance"] == "expanded_strict_yelp_category_silver":
            expanded_photo_terms[f"{category}:{business['matched_category_term']}"] += 1
        styles, facilities = _caption_signals(caption, config)
        candidates.append(
            {
                "photo_id": photo_id,
                "business_id": business_id,
                "source_id": source_id,
                "group_id": f"yelp-business:{business_id}",
                "business_category": category,
                "category_provenance": business["category_provenance"],
                "matched_category_term": business["matched_category_term"],
                "caption": caption,
                "caption_source": "native" if caption else "absent",
                "caption_style_tags": styles,
                "caption_facility_tags": facilities,
                "caption_price_tokens": tokens,
                "image_sha256": None,
                "ocr_status": "not_run",
                "confirmed_visible_price_tokens": {"amounts": [], "tiers": []},
                "price_range": "unknown",
                "label_provenance": "programmatic_silver",
                "sample_weight_maximum": float(config["policy"]["maximum_silver_weight"]),
                "human_annotation": False,
                "human_review": False,
                "human_acceptance": False,
            }
        )
    maximum_candidates = int(config["candidate_policy"]["maximum_candidate_images"])
    if len(candidates) > maximum_candidates:
        raise Week8SilverSourceAuditError(
            f"bounded audit candidate count changed: {len(candidates)}/{maximum_candidates}"
        )

    post_hash_candidates: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    if photos_zip is not None:
        photos_zip = Path(photos_zip).resolve()
        declared_zip = fresh["official_photos_zip"]
        if photos_zip.stat().st_size != int(declared_zip["size"]):
            raise Week8SilverSourceAuditError("official Yelp Photos ZIP size changed")
        if verify_zip_sha256 and sha256_file(photos_zip) != declared_zip["sha256"]:
            raise Week8SilverSourceAuditError("official Yelp Photos ZIP hash changed")
        if run_ocr:
            _validate_ocr_runtime(config)
        temp_parent = os.environ.get("SLURM_TMPDIR")
        with tempfile.TemporaryDirectory(dir=temp_parent) as directory:
            extract_root = Path(directory)
            extraction = extract_yelp_photo_files(
                photos_zip, extract_root, {row["photo_id"] for row in candidates}
            )
            if int(extraction["extracted_photo_count"]) != len(candidates):
                raise Week8SilverSourceAuditError("candidate image extraction was incomplete")
            seen_hashes: set[str] = set()
            for row in candidates:
                image_path = extract_root / "photos" / f"{row['photo_id']}.jpg"
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (OSError, ValueError):
                    rejection_counts["unreadable_image"] += 1
                    continue
                digest = sha256_file(image_path)
                if digest in consumed["image_sha256"]:
                    rejection_counts["historically_or_v7_consumed_image_sha256"] += 1
                    continue
                if digest in seen_hashes:
                    rejection_counts["candidate_duplicate_image_sha256"] += 1
                    continue
                seen_hashes.add(digest)
                audited = dict(row)
                audited["image_sha256"] = digest
                if run_ocr and (
                    row["caption_price_tokens"]["amounts"]
                    or row["caption_price_tokens"]["tiers"]
                ):
                    ocr_text = _ocr_image(image_path, config)
                    confirmed = confirmed_visible_price_tokens(row["caption"], ocr_text)
                    audited["ocr_status"] = "completed"
                    audited["confirmed_visible_price_tokens"] = confirmed
                    audited["price_range"] = map_confirmed_tier(confirmed, config)
                post_hash_candidates.append(audited)

    final_candidates = post_hash_candidates if photos_zip is not None else candidates
    confirmed_amounts = sum(
        bool(row["confirmed_visible_price_tokens"]["amounts"])
        for row in final_candidates
    )
    confirmed_tiers = sum(
        bool(row["confirmed_visible_price_tokens"]["tiers"])
        for row in final_candidates
    )
    positive_price_ranges = sum(row["price_range"] != "unknown" for row in final_candidates)
    untouched_categories = Counter(row["ota_category"] for row in untouched_fresh)
    untouched_caption_sources = Counter(
        str(row.get("caption_source") or "unknown") for row in untouched_fresh
    )
    untouched_style_support = sum(bool(row.get("caption_styles")) for row in untouched_fresh)
    untouched_facility_support = sum(
        bool(row.get("caption_facilities")) for row in untouched_fresh
    )
    metadata_price_values: Counter[str] = Counter()
    untouched_photo_ids = {str(row["photo_id"]) for row in untouched_fresh}
    untouched_caption_amount_support = 0
    untouched_caption_tier_support = 0
    untouched_caption_any_price_support = 0
    for row in _parquet_rows(
        root / product_config["dataset"]["source_paths"]["photos"],
        ["photo_id", "caption"],
    ):
        if str(row.get("photo_id")) not in untouched_photo_ids:
            continue
        tokens = price_tokens(str(row.get("caption") or ""))
        untouched_caption_amount_support += int(bool(tokens["amounts"]))
        untouched_caption_tier_support += int(bool(tokens["tiers"]))
        untouched_caption_any_price_support += int(
            bool(tokens["amounts"] or tokens["tiers"])
        )
    untouched_business_ids = {str(row["business_id"]) for row in untouched_fresh}
    for business_id in untouched_business_ids:
        value = businesses.get(business_id, {}).get("attributes") or "{}"
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {}
        price_value = str(value.get("RestaurantsPriceRange2") or "").strip()
        if price_value and price_value != "None":
            metadata_price_values[price_value] += 1
    summary = {
        "schema_version": "week8_product_silver_source_audit_result_v8",
        "status": "COMPLETED" if photos_zip is not None else "CANDIDATE_UPPER_BOUND_ONLY",
        "audit_version": config["audit_version"],
        "config_sha256": sha256_file(config_path),
        "v7_fresh_manifest_sha256": source_hash(config, "fresh_manifest_sha256"),
        "v7_dataset_lock_sha256": lock["lock_sha256"],
        "v7_final_rows_read": False,
        "v7_final_outputs_read": False,
        "v7_locked_identity_count": len(locked_identities),
        "untouched_v7_fresh_count": len(untouched_fresh),
        "untouched_v7_fresh_category_counts": dict(sorted(untouched_categories.items())),
        "untouched_v7_caption_source_counts": dict(
            sorted(untouched_caption_sources.items())
        ),
        "untouched_v7_caption_style_support": untouched_style_support,
        "untouched_v7_caption_facility_support": untouched_facility_support,
        "untouched_v7_caption_exact_amount_support": untouched_caption_amount_support,
        "untouched_v7_caption_explicit_tier_support": untouched_caption_tier_support,
        "untouched_v7_unknown_price_silver_upper_bound": (
            len(untouched_fresh) - untouched_caption_any_price_support
        ),
        "untouched_v7_unknown_price_risk": "ocr_false_negative_and_caption_coverage",
        "untouched_v7_metadata_price_values": dict(sorted(metadata_price_values.items())),
        "untouched_v7_metadata_price_support": sum(metadata_price_values.values()),
        "eligible_business_counts": dict(sorted(eligible_business_counts.items())),
        "expanded_business_term_counts": dict(sorted(expanded_business_terms.items())),
        "unconsumed_photo_pre_hash_upper_bounds": dict(sorted(photo_upper_bounds.items())),
        "expanded_photo_term_pre_hash_upper_bounds": dict(sorted(expanded_photo_terms.items())),
        "caption_exact_amount_candidate_upper_bounds": dict(sorted(amount_candidate_counts.items())),
        "caption_explicit_tier_candidate_upper_bounds": dict(sorted(tier_candidate_counts.items())),
        "candidate_count_pre_hash": len(candidates),
        "candidate_count_post_hash": len(post_hash_candidates) if photos_zip is not None else None,
        "candidate_rejection_counts": dict(sorted(rejection_counts.items())),
        "ocr_run": run_ocr,
        "confirmed_visible_amount_support": confirmed_amounts,
        "confirmed_visible_tier_support": confirmed_tiers,
        "positive_price_range_support": positive_price_ranges,
        "numeric_amount_maps_to_price_range": False,
        "metadata_is_visual_evidence": False,
        "label_provenance": "programmatic_silver",
        "human_annotation_count": 0,
        "human_review_count": 0,
        "human_acceptance_count": 0,
        "historical_exclusion_evidence": exclusion_evidence,
        "training_feasibility": {
            "business_category": "bounded_by_post_hash_hotel_attraction_candidates",
            "style": "native_caption_lexical_only",
            "facility": "native_caption_lexical_only",
            "visible_price_text": "requires_exact_caption_ocr_token_agreement",
            "price_range_positive": "requires_confirmed_explicit_tier_word",
            "price_range_unknown": "silver_only_ocr_false_negative_risk",
        },
    }
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        if output_dir.exists():
            raise Week8SilverSourceAuditError("refusing to overwrite v8 source audit")
        output_dir.mkdir(parents=True, exist_ok=False)
        candidate_path = output_dir / "silver_candidate_manifest.jsonl"
        count = _write_jsonl_new(candidate_path, final_candidates)
        summary["candidate_manifest"] = {
            "path": candidate_path.name,
            "count": count,
            "sha256": sha256_file(candidate_path),
        }
        _write_json_new(output_dir / "audit_summary.json", summary)
    return summary


def source_hash(config: dict[str, Any], key: str) -> str:
    """Return a declared source hash while keeping summary construction explicit."""

    value = str(config["source"].get(key) or "")
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise Week8SilverSourceAuditError(f"invalid declared source hash: {key}")
    return value

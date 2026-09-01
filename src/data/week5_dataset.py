"""Week 5 instruction-data sampling, annotation workflow, and quality evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image, ImageDraw

from src.evaluation.schema_validation import SchemaValidationError, validate_output


SCENARIOS = ("image_product_search", "after_sales", "itinerary_planning")
ISSUES = ("hygiene_stain", "facility_damage", "attraction_closure", "transport_delay")
ISSUE_PATTERNS = {
    "hygiene_stain": re.compile(
        r"\b(dirty|filthy|stain|stained|mold|mould|roach|bug|pest|unclean|odor|hair)\b",
        re.IGNORECASE,
    ),
    "facility_damage": re.compile(
        r"\b(broken|damage|damaged|leak|leaking|crack|cracked|out of order|malfunction)\b",
        re.IGNORECASE,
    ),
    "attraction_closure": re.compile(
        r"\b(closed|closure|shut|not open|unavailable)\b", re.IGNORECASE
    ),
    "transport_delay": re.compile(
        r"\b(delay|delayed|late|cancelled|canceled|missed|waiting)\b",
        re.IGNORECASE,
    ),
}
PRICE_MAP = {"1": "budget", "2": "mid_range", "3": "premium", "4": "luxury"}


class Week5DataError(ValueError):
    """Raised when a Week 5 artifact violates an isolation or workflow rule."""


def load_week5_config(root: Path, path: Path | str) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("dataset_version") != "week5_instruction_candidates_v1":
        raise Week5DataError("unsupported Week 5 dataset version")
    if set(payload.get("targets", {})) != {*SCENARIOS, "dialogues"}:
        raise Week5DataError("Week 5 targets must cover three scenarios and dialogues")
    quality = payload.get("quality", {})
    if quality.get("mode") == "single_operator_minimal_review_v1":
        for scope in ("core", "general"):
            cross = float(quality.get(f"{scope}_cross_review_rate", -1))
            audit = float(quality.get(f"{scope}_audit_rate", -1))
            if not 0 <= audit <= cross <= 1:
                raise Week5DataError(
                    "single-operator audit rate must be nested within cross-review rate"
                )
    return payload


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Week5DataError(f"invalid JSONL {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise Week5DataError(f"JSONL row must be an object: {path}:{line_number}")
            yield row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Create a JSONL artifact once; reruns must use a new path or explicit resume flow."""
    if path.exists():
        raise Week5DataError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")
            count += 1
    return count


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
        handle.write("\n")


def load_exclusions(root: Path, config: dict[str, Any]) -> dict[str, set[str]]:
    values = {name: set() for name in ("source_id", "image_sha256", "group_id", "constraint_template_id")}
    for relative in config["paths"]["exclusion_manifests"]:
        path = root / relative
        if not path.is_file():
            raise Week5DataError(f"required evaluation exclusion manifest is missing: {path}")
        for row in read_jsonl(path):
            for name in values:
                value = row.get(name)
                if isinstance(value, str) and value:
                    values[name].add(value)
    return values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _ImageHashCache:
    """Persist verified image hashes so an interrupted large build can resume safely."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS image_hashes "
            "(path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL)"
        )
        self.pending = 0

    def sha256(self, path: Path) -> str:
        stat = path.stat()
        key = str(path.resolve())
        row = self.connection.execute(
            "SELECT sha256 FROM image_hashes WHERE path=? AND size=? AND mtime_ns=?",
            (key, stat.st_size, stat.st_mtime_ns),
        ).fetchone()
        if row:
            return str(row[0])
        value = _sha256_file(path)
        self.connection.execute(
            "INSERT OR REPLACE INTO image_hashes(path,size,mtime_ns,sha256) VALUES(?,?,?,?)",
            (key, stat.st_size, stat.st_mtime_ns, value),
        )
        self.pending += 1
        if self.pending >= 100:
            self.connection.commit()
            self.pending = 0
        return value

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def _stable_rank(seed: int, *values: str) -> str:
    return hashlib.sha256((str(seed) + "\0" + "\0".join(values)).encode("utf-8")).hexdigest()


def _sample_id(scenario: str, source_id: str, suffix: str = "") -> str:
    digest = hashlib.sha256(f"{scenario}\0{source_id}\0{suffix}".encode("utf-8")).hexdigest()[:20]
    return f"week5-{scenario}-{digest}"


def _business_category(categories: Any) -> str | None:
    values = {str(value).lower() for value in (categories or [])}
    if values & {"hotels", "bed & breakfast", "resorts", "hostels", "vacation rentals", "guest houses"}:
        return "hotel"
    if values & {"museums", "parks", "art galleries", "landmarks & historical buildings", "zoos", "botanical gardens", "aquariums", "amusement parks", "tours", "public art", "beaches", "hiking"}:
        return "attraction"
    if "restaurants" in values or values & {"food", "cafes", "coffee & tea", "bars"}:
        return "restaurant"
    return None


def _style_hint(business: dict[str, Any]) -> str:
    for field, value in (
        ("attr_Ambience_classy", "luxury"),
        ("attr_Ambience_trendy", "modern"),
        ("attr_Ambience_romantic", "romantic"),
        ("attr_Ambience_casual", "casual"),
    ):
        if business.get(field) is True:
            return value
    return "unknown"


def _load_businesses(path: Path) -> dict[str, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise Week5DataError("pyarrow is required to build Week 5 pools") from exc
    columns = [
        "business_id", "categories", "city", "attr_RestaurantsPriceRange2",
        "attr_Ambience_casual", "attr_Ambience_classy", "attr_Ambience_trendy",
        "attr_Ambience_romantic",
    ]
    result: dict[str, dict[str, Any]] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            category = _business_category(row.get("categories"))
            if category:
                row["ota_category"] = category
                result[row["business_id"]] = row
    return result


def _candidate_record(
    *, config: dict[str, Any], scenario: str, source_id: str, source_type: str,
    image_path: str, image_sha256: str, group_id: str, text_constraints: str | None,
    sampling_metadata: dict[str, Any], constraint_template_id: str | None = None,
) -> dict[str, Any]:
    """Create a candidate with provenance and an explicitly unfinished workflow."""
    return {
        "sample_id": _sample_id(scenario, source_id, constraint_template_id or ""),
        "scenario": scenario,
        "source_id": source_id,
        "source_type": source_type,
        "source_license": "Yelp Open Dataset Terms of Use" if source_type == "public_yelp" else "project_owned_synthetic",
        "image_sha256": image_sha256,
        "split": "instruction_candidate",
        "dataset_version": config["dataset_version"],
        "input": {"images": [{"path": image_path, "sha256": image_sha256}], "text_constraints": text_constraints},
        "sampling_metadata": sampling_metadata,
        "isolation": {
            "checked_manifests": list(config["paths"]["exclusion_manifests"]),
            "source_id_conflict": False,
            "image_sha256_conflict": False,
            "group_id_conflict": False,
            "constraint_template_id_conflict": False,
        },
        "provenance": {"group_id": group_id, "constraint_template_id": constraint_template_id},
        "workflow": {
            "model_preannotation": "pending",
            "human_correction": "pending",
            "self_review": "pending",
            "cross_review": "pending",
            "core_audit": "pending",
            "final_status": "pending",
        },
    }


def _check_candidate_isolation(
    candidate: dict[str, Any], exclusions: dict[str, set[str]], used_hashes: set[str]
) -> bool:
    """Reject frozen-set identity collisions and duplicate image bytes."""
    provenance = candidate["provenance"]
    checks = (
        ("source_id", candidate["source_id"]),
        ("image_sha256", candidate["image_sha256"]),
        ("group_id", provenance.get("group_id")),
        ("constraint_template_id", provenance.get("constraint_template_id")),
    )
    if candidate["image_sha256"] in used_hashes:
        return False
    return not any(value and value in exclusions[name] for name, value in checks)


def _route_issue(text: str) -> str | None:
    matches = [issue for issue, pattern in ISSUE_PATTERNS.items() if pattern.search(text)]
    if not matches:
        return None
    return min(matches, key=lambda issue: _stable_rank(20260802, text[:200], issue))


def _public_after_sales_candidates(
    root: Path, config: dict[str, Any], exclusions: dict[str, set[str]], used_hashes: set[str],
    hash_cache: _ImageHashCache,
) -> list[dict[str, Any]]:
    """Route weak public evidence into balanced, non-gold after-sales candidates."""
    import pyarrow.parquet as pq

    path = root / config["paths"]["weak_pairs"]
    per_issue: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    seen: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096, columns=["business_id", "photo_id", "image_path", "review_id", "review_text"]):
        for row in batch.to_pylist():
            photo_id = row["photo_id"]
            if photo_id in seen:
                continue
            issue = _route_issue(row.get("review_text") or "")
            if not issue:
                continue
            seen.add(photo_id)
            source_id = f"yelp-photo:{photo_id}"
            group_id = f"yelp-business:{row['business_id']}"
            if source_id in exclusions["source_id"] or group_id in exclusions["group_id"]:
                continue
            rank = _stable_rank(config["seed"], "after_sales", issue, photo_id)
            per_issue[issue].append((rank, row))
    selected: list[dict[str, Any]] = []
    for issue in ISSUES:
        for _, row in sorted(per_issue[issue], key=lambda item: item[0])[:5000]:
            image_path = row["image_path"].replace("\\", "/")
            image_sha = hash_cache.sha256(root / image_path)
            candidate = _candidate_record(
                config=config, scenario="after_sales", source_id=f"yelp-photo:{row['photo_id']}",
                source_type="public_yelp", image_path=image_path, image_sha256=image_sha,
                group_id=f"yelp-business:{row['business_id']}", text_constraints=None,
                sampling_metadata={"issue_route": issue, "severity_hint": "unknown", "route_is_gold": False},
            )
            if _check_candidate_isolation(candidate, exclusions, used_hashes):
                selected.append(candidate)
                used_hashes.add(image_sha)
    return selected


def _synthetic_card(path: Path, issue: str, severity: str, index: int) -> None:
    canvas = Image.new("RGB", (768, 512), "#f7f3ea")
    draw = ImageDraw.Draw(canvas)
    accent = {"hygiene_stain": "#8b5a2b", "facility_damage": "#b33a3a", "attraction_closure": "#555555", "transport_delay": "#245a88"}[issue]
    draw.rectangle((0, 0, 768, 70), fill=accent)
    title = issue.replace("_", " ").upper()
    event = hashlib.sha256(f"{issue}:{index}".encode()).hexdigest()[:10].upper()
    lines = ["OTA SERVICE EVIDENCE", title, f"EVENT ID: W5-{event}", f"IMPACT LEVEL: {severity.upper()}"]
    detail = {
        "hygiene_stain": "VISIBLE STAIN / CONTAMINATION REPORTED",
        "facility_damage": "FACILITY UNAVAILABLE DUE TO DAMAGE",
        "attraction_closure": "ATTRACTION CLOSED - ACCESS UNAVAILABLE",
        "transport_delay": f"TRANSPORT DELAYED {15 + index % 180} MINUTES",
    }[issue]
    lines.append(detail)
    for line_index, line in enumerate(lines):
        draw.text((52, 105 + line_index * 62), line, fill="#111111")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False)


def _fill_synthetic_after_sales(
    root: Path, config: dict[str, Any], rows: list[dict[str, Any]],
    exclusions: dict[str, set[str]], used_hashes: set[str], hash_cache: _ImageHashCache,
) -> None:
    """Fill missing issue strata with traceable synthetic evidence cards."""
    counts = Counter(row["sampling_metadata"]["issue_route"] for row in rows)
    image_dir = root / config["paths"]["output_dir"] / "synthetic_after_sales"
    severities = ("low", "medium", "high", "critical")
    for issue in ISSUES:
        for index in range(counts[issue], 5000):
            severity = severities[index % len(severities)]
            relative = (image_dir / issue / f"{issue}_{index:05d}.png").relative_to(root).as_posix()
            path = root / relative
            if not path.exists():
                _synthetic_card(path, issue, severity, index)
            image_sha = hash_cache.sha256(path)
            source_id = f"week5-synthetic:{issue}:{index:05d}"
            candidate = _candidate_record(
                config=config, scenario="after_sales", source_id=source_id,
                source_type="business_synthetic", image_path=relative, image_sha256=image_sha,
                group_id=f"week5-synthetic-event:{issue}:{index:05d}", text_constraints=None,
                sampling_metadata={"issue_route": issue, "severity_hint": severity, "route_is_gold": False},
            )
            if not _check_candidate_isolation(candidate, exclusions, used_hashes):
                raise Week5DataError(f"unexpected synthetic exclusion collision: {source_id}")
            rows.append(candidate)
            used_hashes.add(image_sha)


def _select_photo_rows(
    root: Path, config: dict[str, Any], businesses: dict[str, dict[str, Any]],
    exclusions: dict[str, set[str]], used_hashes: set[str], scenario: str,
    target: int, hash_cache: _ImageHashCache,
) -> list[dict[str, Any]]:
    """Select deterministic, isolated Yelp photos for one business scenario."""
    import pyarrow.parquet as pq

    ranked: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    parquet = pq.ParquetFile(root / config["paths"]["photos"])
    for batch in parquet.iter_batches(batch_size=8192, columns=["photo_id", "business_id", "image_path"]):
        for photo in batch.to_pylist():
            business = businesses.get(photo["business_id"])
            if not business:
                continue
            source_id = f"yelp-photo:{photo['photo_id']}"
            group_id = f"yelp-business:{photo['business_id']}"
            if source_id in exclusions["source_id"] or group_id in exclusions["group_id"]:
                continue
            ranked.append((_stable_rank(config["seed"], scenario, photo["photo_id"]), photo, business))
    ranked.sort(key=lambda item: item[0])
    selected: list[dict[str, Any]] = []
    product_quota = {
        key: int(value) for key, value in config["sampling"]["product_quotas"].items()
    }
    if sum(product_quota.values()) != config["targets"]["image_product_search"]:
        raise Week5DataError("product quotas must sum to the product target")
    category_counts: Counter[str] = Counter()
    groups = ("solo", "couple", "family", "friends")
    budgets = ("budget", "mid_range", "premium")
    days_values = (2, 3, 4)
    for _, photo, business in ranked:
        if len(selected) >= target:
            break
        category = business["ota_category"]
        if scenario == "image_product_search" and category_counts[category] >= product_quota[category]:
            continue
        image_path = photo["image_path"].replace("\\", "/")
        path = root / image_path
        if not path.is_file():
            continue
        image_sha = hash_cache.sha256(path)
        rank_index = len(selected)
        if scenario == "image_product_search":
            sampling = {
                "business_category": category,
                "style_hint": _style_hint(business),
                "price_hint": PRICE_MAP.get(business.get("attr_RestaurantsPriceRange2"), "unknown"),
                "city": business.get("city") or "unknown",
                "hints_are_gold": False,
            }
            text_constraints = None
            template_id = None
        else:
            crowd = groups[rank_index % len(groups)]
            budget = budgets[(rank_index // len(groups)) % len(budgets)]
            days = days_values[(rank_index // (len(groups) * len(budgets))) % len(days_values)]
            template_id = f"week5-itinerary-{crowd}-{budget}-{days}d"
            city = business.get("city") or "目的地待定"
            text_constraints = f"计划{days}天前往{city}，{crowd}出行，预算档位为{budget}；偏好慢节奏，优先公共交通，每日包含用餐安排。"
            sampling = {"travel_group": crowd, "budget_tier": budget, "trip_days": days, "city": city, "hints_are_gold": False}
        candidate = _candidate_record(
            config=config, scenario=scenario, source_id=f"yelp-photo:{photo['photo_id']}",
            source_type="public_yelp", image_path=image_path, image_sha256=image_sha,
            group_id=f"yelp-business:{photo['business_id']}", text_constraints=text_constraints,
            sampling_metadata=sampling, constraint_template_id=template_id,
        )
        if not _check_candidate_isolation(candidate, exclusions, used_hashes):
            continue
        selected.append(candidate)
        used_hashes.add(image_sha)
        category_counts[category] += 1
    if len(selected) != target:
        raise Week5DataError(f"{scenario} pool shortfall: {len(selected)}/{target}")
    return selected


def build_sample_pools(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Build three immutable local candidate pools after all evaluation exclusions."""
    output_dir = root / config["paths"]["output_dir"]
    pool_dir = output_dir / "pools"
    expected = {scenario: pool_dir / f"{scenario}.jsonl" for scenario in SCENARIOS}
    if any(path.exists() for path in expected.values()):
        raise Week5DataError("pool artifacts already exist; refusing duplicate execution")
    exclusions = load_exclusions(root, config)
    used_hashes: set[str] = set()
    hash_cache = _ImageHashCache(output_dir / "cache" / "image_hashes.sqlite3")
    try:
        after_sales = _public_after_sales_candidates(root, config, exclusions, used_hashes, hash_cache)
        _fill_synthetic_after_sales(root, config, after_sales, exclusions, used_hashes, hash_cache)
        after_sales.sort(key=lambda row: row["sample_id"])
        businesses = _load_businesses(root / config["paths"]["businesses"])
        product = _select_photo_rows(root, config, businesses, exclusions, used_hashes, "image_product_search", config["targets"]["image_product_search"], hash_cache)
        itinerary = _select_photo_rows(root, config, businesses, exclusions, used_hashes, "itinerary_planning", config["targets"]["itinerary_planning"], hash_cache)
    finally:
        hash_cache.close()
    pools = {"image_product_search": product, "after_sales": after_sales, "itinerary_planning": itinerary}
    for scenario, rows in pools.items():
        if len(rows) != config["targets"][scenario]:
            raise Week5DataError(f"{scenario} pool count mismatch: {len(rows)}")
        write_jsonl_new(expected[scenario], rows)
    summary = summarize_pools(pools)
    summary.update({"dataset_version": config["dataset_version"], "generated_at": _now(), "exclusion_rows_loaded": {name: len(values) for name, values in exclusions.items()}})
    summary_path = output_dir / "reports" / "pool_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return summary


def summarize_pools(pools: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"counts": {scenario: len(rows) for scenario, rows in pools.items()}, "distributions": {}}
    for scenario, rows in pools.items():
        fields = {
            "image_product_search": ("business_category", "style_hint", "price_hint", "city"),
            "after_sales": ("issue_route", "severity_hint"),
            "itinerary_planning": ("travel_group", "budget_tier", "trip_days"),
        }[scenario]
        result["distributions"][scenario] = {
            field: dict(sorted(Counter(str(row["sampling_metadata"].get(field, "unknown")) for row in rows).items()))
            for field in fields
        }
        if scenario == "after_sales":
            result["distributions"][scenario]["source_type"] = dict(sorted(Counter(row["source_type"] for row in rows).items()))
    return result


def validate_candidate_record(
    root: Path, config: dict[str, Any], row: dict[str, Any],
    hash_cache: _ImageHashCache | None = None,
) -> None:
    """Recheck candidate identity, image bytes, and recorded isolation claims."""
    scenario = row.get("scenario")
    if scenario not in SCENARIOS:
        raise Week5DataError("invalid candidate scenario")
    if row.get("dataset_version") != config["dataset_version"] or row.get("split") != "instruction_candidate":
        raise Week5DataError("invalid candidate dataset version or split")
    images = row.get("input", {}).get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise Week5DataError("candidate must contain exactly one image")
    image = images[0]
    if image.get("sha256") != row.get("image_sha256"):
        raise Week5DataError("candidate primary image hash mismatch")
    path = root / str(image.get("path", ""))
    actual_sha = hash_cache.sha256(path) if hash_cache and path.is_file() else (_sha256_file(path) if path.is_file() else None)
    if actual_sha != image["sha256"]:
        raise Week5DataError(f"candidate image missing or changed: {path}")
    isolation = row.get("isolation", {})
    if not isolation or any(isolation.get(name) is not False for name in ("source_id_conflict", "image_sha256_conflict", "group_id_conflict", "constraint_template_id_conflict")):
        raise Week5DataError("candidate lacks a passing isolation record")


def load_pools(root: Path, config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    pool_dir = root / config["paths"]["output_dir"] / "pools"
    return {scenario: read_jsonl(pool_dir / f"{scenario}.jsonl") for scenario in SCENARIOS}


def candidate_payload_sha256(row: dict[str, Any]) -> str:
    """绑定候选完整 JSON 语义，sidecar 不复制或改写候选内容。"""
    payload = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidate_manifest_sha256(root: Path, config: dict[str, Any], scenario: str) -> str:
    if scenario not in SCENARIOS:
        raise Week5DataError(f"unsupported scenario: {scenario}")
    path = root / config["paths"]["output_dir"] / "pools" / f"{scenario}.jsonl"
    if not path.is_file():
        raise Week5DataError(f"candidate manifest is missing: {path}")
    return _sha256_file(path)


def initialize_workflow_v2_sidecar(
    root: Path, config: dict[str, Any], scenario: str,
) -> dict[str, Any]:
    """一次性生成 workflow v2；候选文件保持字节级不变。"""
    if scenario not in SCENARIOS:
        raise Week5DataError(f"unsupported scenario: {scenario}")
    pool_path = root / config["paths"]["output_dir"] / "pools" / f"{scenario}.jsonl"
    manifest_sha = candidate_manifest_sha256(root, config, scenario)
    output = (
        root / config["paths"]["output_dir"] / "workflow_v2" / f"{scenario}.jsonl"
    )

    def rows() -> Iterable[dict[str, Any]]:
        for row in iter_jsonl(pool_path):
            yield {
                "schema_version": "week5_workflow_v2",
                "sample_id": row["sample_id"],
                "scenario": scenario,
                "candidate_sha256": candidate_payload_sha256(row),
                "candidate_manifest_sha256": manifest_sha,
                "model_preannotation": {
                    "status": "not_started",
                    "run_id": None,
                    "latest_attempt": None,
                },
                "workflow_status": "awaiting_human_annotation",
                "annotation_revision": 0,
            }

    count = write_jsonl_new(output, rows())
    return {
        "scenario": scenario,
        "records": count,
        "candidate_manifest_sha256": manifest_sha,
        "output": output.relative_to(root).as_posix(),
    }


def validate_workflow_v2_sidecar(
    root: Path, config: dict[str, Any], scenario: str,
) -> dict[str, Any]:
    """Verify sidecar records still bind byte-for-byte to the candidate manifest."""
    if scenario not in SCENARIOS:
        raise Week5DataError(f"unsupported scenario: {scenario}")
    pool_path = root / config["paths"]["output_dir"] / "pools" / f"{scenario}.jsonl"
    path = root / config["paths"]["output_dir"] / "workflow_v2" / f"{scenario}.jsonl"
    manifest_sha = candidate_manifest_sha256(root, config, scenario)
    allowed_model = {"not_started", "running", "completed", "failed"}
    allowed_human = {
        "awaiting_human_annotation", "partial", "awaiting_cross_review",
        "awaiting_core_audit", "accepted", "rejected",
    }
    count = 0
    sentinel = object()
    for candidate, row in zip_longest(
        iter_jsonl(pool_path), iter_jsonl(path), fillvalue=sentinel
    ):
        if candidate is sentinel or row is sentinel:
            raise Week5DataError(f"workflow v2 count mismatch: {scenario}")
        assert isinstance(candidate, dict) and isinstance(row, dict)
        sample_id = row.get("sample_id")
        if sample_id != candidate.get("sample_id"):
            raise Week5DataError(f"workflow v2 candidate order or identity mismatch: {sample_id}")
        if row.get("schema_version") != "week5_workflow_v2":
            raise Week5DataError("invalid workflow v2 schema version")
        if row.get("candidate_manifest_sha256") != manifest_sha:
            raise Week5DataError("workflow v2 candidate manifest hash mismatch")
        if row.get("candidate_sha256") != candidate_payload_sha256(candidate):
            raise Week5DataError(f"workflow v2 candidate hash mismatch: {sample_id}")
        model = row.get("model_preannotation", {})
        if model.get("status") not in allowed_model or row.get("workflow_status") not in allowed_human:
            raise Week5DataError(f"invalid workflow v2 status: {sample_id}")
        if row.get("workflow_status") == "awaiting_human_annotation" and int(row.get("annotation_revision", -1)) != 0:
            raise Week5DataError("awaiting human annotation must have revision zero")
        count += 1
    return {"status": "ok", "scenario": scenario, "records": count, "candidate_manifest_sha256": manifest_sha}


def validate_pools(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    pools = load_pools(root, config)
    exclusions = load_exclusions(root, config)
    ids: set[str] = set()
    hashes: set[str] = set()
    hash_cache = _ImageHashCache(root / config["paths"]["output_dir"] / "cache" / "image_hashes.sqlite3")
    try:
        for scenario, rows in pools.items():
            if len(rows) != config["targets"][scenario]:
                raise Week5DataError(f"{scenario} target mismatch")
            for row in rows:
                validate_candidate_record(root, config, row, hash_cache)
                if row["sample_id"] in ids or row["image_sha256"] in hashes:
                    raise Week5DataError("duplicate sample or image across Week 5 pools")
                ids.add(row["sample_id"])
                hashes.add(row["image_sha256"])
                if not _check_candidate_isolation(row, exclusions, set()):
                    raise Week5DataError(f"evaluation collision: {row['sample_id']}")
    finally:
        hash_cache.close()
    return {"status": "ok", **summarize_pools(pools), "unique_sample_ids": len(ids), "unique_image_sha256": len(hashes)}


def validate_human_annotation(root: Path, scenario: str, annotation: Any) -> None:
    version = "v2" if scenario == "itinerary_planning" else "v1"
    try:
        validate_output(root, scenario, annotation, version)
    except SchemaValidationError as exc:
        raise Week5DataError(f"human annotation Schema failure: {exc}") from exc
    tool = json.loads(
        (root / "configs/week5/annotation_tool.json").read_text(encoding="utf-8")
    )
    vocabularies = tool.get("label_vocabularies", {})
    fields = {
        "image_product_search": ("style_tags", "visible_facilities"),
        "after_sales": (),
        "itinerary_planning": ("style_preferences",),
    }[scenario]
    for field in fields:
        values = annotation.get(field, [])
        allowed = set(vocabularies.get(field, []))
        if not isinstance(values, list) or any(value not in allowed for value in values):
            raise Week5DataError(f"human annotation uses an uncontrolled {field} label")


def export_annotation_packet(root: Path, config: dict[str, Any], scenario: str, output: Path) -> int:
    pools = load_pools(root, config)
    if scenario not in pools:
        raise Week5DataError(f"unsupported scenario: {scenario}")
    preannotations = {row["sample_id"]: row for row in read_jsonl(root / config["paths"]["output_dir"] / "preannotations" / f"{scenario}.jsonl") if row.get("status") == "completed"}
    rows = []
    for candidate in pools[scenario]:
        pre = preannotations.get(candidate["sample_id"])
        rows.append({
            "sample_id": candidate["sample_id"], "scenario": scenario,
            "input": candidate["input"], "sampling_metadata": candidate["sampling_metadata"],
            "isolation": candidate["isolation"],
            "model_preannotation": pre.get("parsed_output") if pre else None,
            "model_preannotation_status": "completed" if pre else "missing",
            "annotator": None, "human_annotation": None, "corrected_at": None,
            "self_review_confirmed": False, "review_session_id": None,
        })
    return write_jsonl_new(output, rows)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _qc_selection_value(sample_id: str) -> float:
    value = int(hashlib.sha256(sample_id.encode()).hexdigest()[:12], 16) / float(16**12)
    return value


def qc_cross_review_selected(
    sample_id: str, scenario: str, config: dict[str, Any],
) -> bool:
    quality = config["quality"]
    rate = (
        quality.get("core_cross_review_rate", 1.0)
        if scenario in quality["core_scenarios"]
        else quality.get("general_cross_review_rate", 1.0)
    )
    return _qc_selection_value(sample_id) < float(rate)


def qc_audit_selected(sample_id: str, scenario: str, config: dict[str, Any]) -> bool:
    quality = config["quality"]
    rate = quality["core_audit_rate"] if scenario in quality["core_scenarios"] else quality["general_audit_rate"]
    value = _qc_selection_value(sample_id)
    return value < float(rate)


def validate_dialogue(dialogue: dict[str, Any]) -> None:
    """验证保留不变的历史 multimodal_dialogue_v1。"""
    required = {"dialogue_id", "scenario", "images", "messages"}
    if set(dialogue) != required:
        raise Week5DataError("dialogue fields do not match the v1 contract")
    if dialogue["scenario"] not in {"image_search_consultation", "itinerary_iteration", "after_sales_negotiation"}:
        raise Week5DataError("invalid dialogue scenario")
    if not isinstance(dialogue["dialogue_id"], str) or not 1 <= len(dialogue["dialogue_id"]) <= 120:
        raise Week5DataError("invalid dialogue ID")
    images = dialogue["images"]
    messages = dialogue["messages"]
    if not isinstance(images, list) or not 1 <= len(images) <= 8:
        raise Week5DataError("dialogue requires 1-8 images")
    for image in images:
        if set(image) != {"image_id", "path", "sha256"}:
            raise Week5DataError("dialogue image fields do not match the v1 contract")
        if not isinstance(image["path"], str) or not 1 <= len(image["path"]) <= 300:
            raise Week5DataError("dialogue image path is invalid")
        if not isinstance(image["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", image["sha256"]) is None:
            raise Week5DataError("dialogue image SHA-256 is invalid")
    image_ids = {image.get("image_id") for image in images}
    if len(image_ids) != len(images) or any(not value for value in image_ids):
        raise Week5DataError("dialogue image IDs must be unique")
    if not isinstance(messages, list) or not 6 <= len(messages) <= 16 or len(messages) % 2:
        raise Week5DataError("dialogue requires 3-8 user/assistant turns")
    referenced: set[str] = set()
    for index, message in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if set(message) != {"role", "content", "image_refs"} or message.get("role") != expected_role:
            raise Week5DataError("dialogue roles must alternate from user to assistant")
        if not isinstance(message.get("content"), str) or not message["content"].strip() or len(message["content"]) > 1000:
            raise Week5DataError("dialogue message content is empty")
        refs = message.get("image_refs")
        if not isinstance(refs, list) or len(refs) > 8 or len(refs) != len(set(refs)) or not set(refs) <= image_ids:
            raise Week5DataError("dialogue contains an invalid image reference")
        referenced.update(refs)
    if not referenced:
        raise Week5DataError("dialogue never references its images")


def validate_dialogue_v2(root: Path, dialogue: dict[str, Any]) -> None:
    """Validate role order, image references, provenance, and review state."""
    required = {
        "schema_version", "dialogue_id", "scenario", "image_resources", "turns",
        "source_sample_ids", "generation", "human_review", "qc",
    }
    if set(dialogue) != required or dialogue.get("schema_version") != "multimodal_dialogue_v2":
        raise Week5DataError("dialogue does not match the explicit v2 field contract")
    if dialogue.get("scenario") not in {"image_search", "itinerary", "after_sales"}:
        raise Week5DataError("invalid dialogue v2 scenario")
    if not isinstance(dialogue.get("dialogue_id"), str) or not 1 <= len(dialogue["dialogue_id"]) <= 120:
        raise Week5DataError("invalid dialogue v2 ID")
    resources = dialogue.get("image_resources")
    if not isinstance(resources, list) or not 1 <= len(resources) <= 8:
        raise Week5DataError("dialogue v2 requires 1-8 image resources")
    image_ids: set[str] = set()
    for image in resources:
        if not isinstance(image, dict) or set(image) != {"image_id", "path", "sha256"}:
            raise Week5DataError("dialogue v2 image fields are invalid")
        image_id = image.get("image_id")
        if not isinstance(image_id, str) or not image_id or image_id in image_ids:
            raise Week5DataError("dialogue v2 image IDs must be unique")
        image_ids.add(image_id)
        path = root / str(image.get("path", ""))
        if not path.is_file() or re.fullmatch(r"[0-9a-f]{64}", str(image.get("sha256", ""))) is None:
            raise Week5DataError("dialogue v2 image path or SHA-256 is invalid")
        if _sha256_file(path) != image["sha256"]:
            raise Week5DataError("dialogue v2 image bytes do not match SHA-256")
    turns = dialogue.get("turns")
    if not isinstance(turns, list) or not 6 <= len(turns) <= 16 or len(turns) % 2:
        raise Week5DataError("dialogue v2 requires 3-8 user/assistant turns")
    referenced: set[str] = set()
    for index, turn in enumerate(turns):
        expected = "user" if index % 2 == 0 else "assistant"
        if not isinstance(turn, dict) or set(turn) != {"role", "content", "image_refs"}:
            raise Week5DataError("dialogue v2 turn fields are invalid")
        if turn.get("role") != expected or not isinstance(turn.get("content"), str) or not turn["content"].strip():
            raise Week5DataError("dialogue v2 roles or content are invalid")
        refs = turn.get("image_refs")
        if not isinstance(refs, list) or len(refs) != len(set(refs)) or not set(refs) <= image_ids:
            raise Week5DataError("dialogue v2 contains an invalid image reference")
        referenced.update(refs)
    if not referenced:
        raise Week5DataError("dialogue v2 never references an image")
    source_ids = dialogue.get("source_sample_ids")
    if not isinstance(source_ids, list) or not source_ids or len(source_ids) != len(set(source_ids)) or any(not isinstance(value, str) or not value for value in source_ids):
        raise Week5DataError("dialogue v2 source_sample_ids are invalid")
    generation = dialogue.get("generation")
    if not isinstance(generation, dict) or set(generation) != {"run_id", "model_name", "prompt_version"} or any(not isinstance(generation.get(name), str) or not generation[name] for name in generation):
        raise Week5DataError("dialogue v2 generation metadata are invalid")
    human = dialogue.get("human_review")
    qc = dialogue.get("qc")
    if not isinstance(human, dict) or not isinstance(qc, dict):
        raise Week5DataError("dialogue v2 human review and QC records are required")
    if human.get("status") not in {"awaiting_human_annotation", "partial", "accepted", "rejected"}:
        raise Week5DataError("invalid dialogue v2 human review status")
    if qc.get("status") not in {"partial", "accepted", "rework", "rejected"}:
        raise Week5DataError("invalid dialogue v2 QC status")
    if human.get("status") == "awaiting_human_annotation":
        if human.get("reviewer") is not None or human.get("reviewed_at") is not None:
            raise Week5DataError("awaiting dialogue review cannot contain a reviewer or time")
        if qc.get("status") != "partial":
            raise Week5DataError("unreviewed dialogue QC must remain partial")


def workflow_summary(
    root: Path,
    config: dict[str, Any],
    *,
    dialogue_run_id: str | None = None,
) -> dict[str, Any]:
    """Count workflow stages without promoting model output to human acceptance."""
    output = root / config["paths"]["output_dir"]
    result: dict[str, Any] = {"scenarios": {}, "dialogues": {"candidate": 0, "human_validated": 0, "final_qualified": 0}}
    for scenario in SCENARIOS:
        pool_count = len(read_jsonl(output / "pools" / f"{scenario}.jsonl"))
        pre = read_jsonl(output / "preannotations" / f"{scenario}.jsonl")
        human = read_jsonl(output / "annotations" / f"{scenario}.jsonl")
        qc = read_jsonl(output / "quality" / f"{scenario}.jsonl")
        latest_human = {row.get("sample_id"): row for row in human}
        stages: dict[str, set[str]] = {name: set() for name in ("self_review", "cross_review", "core_audit")}
        rework = reject = 0
        issues: Counter[str] = Counter()
        for row in qc:
            decision = row.get("decision")
            if decision == "rework": rework += 1
            if decision == "reject": reject += 1
            issues.update(str(item) for item in row.get("issues", []))
            current = latest_human.get(row.get("sample_id"), {})
            if (
                decision == "pass"
                and row.get("stage") in stages
                and row.get("annotation_revision") == current.get("revision")
            ):
                stages[row["stage"]].add(row.get("sample_id"))
        final = 0
        for sample_id in latest_human:
            cross_needed = qc_cross_review_selected(sample_id, scenario, config)
            audit_needed = qc_audit_selected(sample_id, scenario, config)
            if (
                sample_id in stages["self_review"]
                and (not cross_needed or sample_id in stages["cross_review"])
                and (not audit_needed or sample_id in stages["core_audit"])
            ):
                final += 1
        result["scenarios"][scenario] = {
            "pool": pool_count,
            "preannotated": sum(row.get("status") == "completed" for row in pre),
            "preannotation_failed": sum(row.get("status") == "failed" for row in pre),
            "human_corrected": len(latest_human),
            "self_review_passed": len(stages["self_review"]),
            "cross_review_passed": len(stages["cross_review"]),
            "core_audit_passed": len(stages["core_audit"]),
            "rework_records": rework, "rejected_records": reject,
            "final_qualified": final, "issue_distribution": dict(sorted(issues.items())),
        }
    dialogue_dir = output / "dialogues"
    if dialogue_run_id is not None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", dialogue_run_id) is None:
            raise Week5DataError(
                "dialogue_run_id must contain only letters, numbers, dot, underscore, or dash"
            )
        dialogue_dir = output / "runs" / f"dialogue-{dialogue_run_id}"
        if not (dialogue_dir / "candidates.jsonl").is_file():
            raise Week5DataError(
                f"dialogue run candidates do not exist: {dialogue_run_id}"
            )
    candidates = read_jsonl(dialogue_dir / "candidates.jsonl")
    validated = read_jsonl(dialogue_dir / "human_validation.jsonl")
    result["dialogues"] = {
        "run_id": dialogue_run_id,
        "candidate": len(candidates),
        "scenario_distribution": dict(sorted(Counter(row.get("scenario") for row in candidates).items())),
        "average_turns": (
            sum(
                len(row.get("turns", row.get("messages", []))) / 2
                for row in candidates
            ) / len(candidates)
            if candidates else 0
        ),
        "human_validated": len({row.get("dialogue_id") for row in validated}),
        "final_qualified": len({row.get("dialogue_id") for row in validated if row.get("decision") == "pass"}),
    }
    return result

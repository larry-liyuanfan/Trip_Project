"""Build immutable Week 7 train/development/test identities and examples."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image, ImageDraw, ImageFont

from src.evaluation.schema_validation import validate_output


CORE_SCENARIOS = ("image_product_search", "after_sales", "itinerary_planning")
IDENTITY_FIELDS = ("sample_id", "source_id", "image_sha256", "group_id", "constraint_template_id")
DIALOGUE_DIMENSIONS = (
    "historical_image_reference",
    "requirement_update",
    "context_carryover",
    "logical_consistency",
)


class Week7DataError(ValueError):
    """Raised when a Week 7 data or isolation contract is violated."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Week7DataError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise Week7DataError(f"JSONL row must be an object: {path}:{line_number}")
            yield row


def write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            count += 1
    return count


def load_week7_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "week7_multitask_context_v1":
        raise Week7DataError("unsupported Week 7 config")
    if config.get("base_model") != "Qwen/Qwen3-VL-8B-Instruct":
        raise Week7DataError("Week 7 base model changed")
    dataset = config["dataset"]
    total = int(dataset["train_total"])
    if total != 3 * int(dataset["train_per_core_scenario"]) + int(dataset["general_regularization_count"]) + int(dataset["dialogue_count"]):
        raise Week7DataError("training counts do not sum to train_total")
    general_fraction = int(dataset["general_regularization_count"]) / total
    dialogue_fraction = int(dataset["dialogue_count"]) / total
    if general_fraction != float(dataset["general_regularization_fraction"]) or not 0.08 <= general_fraction <= 0.10:
        raise Week7DataError("general regularization ratio is not locked in [8%, 10%]")
    if dialogue_fraction != float(dataset["dialogue_fraction"]) or not 0.14 <= dialogue_fraction <= 0.16:
        raise Week7DataError("dialogue ratio is not approximately 15%")
    training = config["training"]
    if float(training["learning_rate"]) != 0.00015 or float(training["weight_decay"]) <= 0.01:
        raise Week7DataError("Week 7 learning rate or weight decay changed")
    if float(training["evaluation_fraction_steps"]) != 0.1 or int(training["early_stopping_patience"]) != 2:
        raise Week7DataError("Week 7 evaluation cadence or patience changed")
    if config["dynamic_adjustment_policy"]["in_place_changes_allowed"] is not False:
        raise Week7DataError("in-place dynamic adjustment is forbidden")
    return config


def _add_identity(sets: dict[str, set[str]], row: dict[str, Any]) -> None:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    values = {
        "sample_id": row.get("sample_id"),
        "source_id": row.get("source_id"),
        "image_sha256": row.get("image_sha256"),
        "group_id": row.get("group_id") or provenance.get("group_id"),
        "constraint_template_id": row.get("constraint_template_id") or provenance.get("constraint_template_id"),
    }
    for field, value in values.items():
        if isinstance(value, str) and value:
            sets[field].add(value)


def load_consumed_identities(source_root: Path, config: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, Any]]:
    paths = config["dataset"]["source_paths"]
    sets = {field: set() for field in IDENTITY_FIELDS}
    hashes: dict[str, str] = {}
    for relative in paths["week3_exclusions"]:
        path = source_root / relative
        if not path.is_file():
            raise Week7DataError(f"missing Week 3 exclusion: {path}")
        hashes[relative] = sha256_file(path)
        for row in iter_jsonl(path):
            _add_identity(sets, row)
    split_path = source_root / paths["week6_split_manifest"]
    used = {str(row["sample_id"]) for row in iter_jsonl(split_path)}
    if len(used) != 79_936:
        raise Week7DataError(f"Week 6 consumed count changed: {len(used)}")
    hashes[paths["week6_split_manifest"]] = sha256_file(split_path)
    pool_root = source_root / paths["week5_pools"]
    for scenario in CORE_SCENARIOS:
        pool_path = pool_root / f"{scenario}.jsonl"
        hashes[pool_path.relative_to(source_root).as_posix()] = sha256_file(pool_path)
        for row in iter_jsonl(pool_path):
            if str(row.get("sample_id")) in used:
                _add_identity(sets, row)
    return sets, {
        "files": hashes,
        "week6_consumed_sample_count": len(used),
        "dimension_counts": {field: len(values) for field, values in sets.items()},
    }


def audit_week5_dialogues(source_root: Path, config: dict[str, Any], consumed: dict[str, set[str]]) -> dict[str, Any]:
    paths = config["dataset"]["source_paths"]
    candidates = source_root / paths["week5_dialogue_candidates"]
    validations = source_root / paths["week5_dialogue_human_validation"]
    accepted = {str(row["dialogue_id"]) for row in iter_jsonl(validations) if row.get("decision") == "pass"}
    counts = Counter()
    for row in iter_jsonl(candidates):
        counts["candidate_count"] += 1
        collision = any(str(value) in consumed["sample_id"] for value in row.get("source_sample_ids", []))
        counts["source_collision_count" if collision else "eligible_count"] += 1
        if str(row.get("dialogue_id")) in accepted:
            counts["human_accepted_count"] += 1
            counts["human_accepted_collision_count" if collision else "human_accepted_eligible_count"] += 1
    result = {key: int(counts[key]) for key in (
        "candidate_count", "source_collision_count", "eligible_count", "human_accepted_count",
        "human_accepted_collision_count", "human_accepted_eligible_count",
    )}
    expected = {
        "candidate_count": 10_000, "source_collision_count": 10_000, "eligible_count": 0,
        "human_accepted_count": 100, "human_accepted_collision_count": 100,
        "human_accepted_eligible_count": 0,
    }
    if result != expected:
        raise Week7DataError(f"Week 5 dialogue audit changed: {result}")
    return {
        **result,
        "candidates_sha256": sha256_file(candidates),
        "human_validation_sha256": sha256_file(validations),
        "disposition": "excluded_from_week7_due_to_week6_source_collision",
    }


def _parquet_rows(path: Path, columns: list[str]) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise Week7DataError("pyarrow is required to build Week 7 locks") from exc
    source = parquet.ParquetFile(path)
    for batch in source.iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


def _business_fields(description: str) -> dict[str, Any]:
    parts = [part.strip() for part in description.split(" | ")]
    categories = parts[1].casefold() if len(parts) > 1 else ""
    category = "other"
    if any(term in categories for term in ("hotel", "resort", "hostel")):
        category = "hotel"
    elif any(term in categories for term in ("museum", "landmark", "attraction", "parks", "tours", "arts & entertainment", "historic")):
        category = "attraction"
    elif any(term in categories for term in ("restaurant", "food", "cafe", "bar", "bakery")):
        category = "restaurant"
    match = re.search(r"RestaurantsPriceRange2:\s*(?:u)?['\"]?(\d)", description)
    price = {"1": "budget", "2": "mid_range", "3": "premium", "4": "luxury"}.get(match.group(1) if match else "", "unknown")
    ambience: list[str] = []
    match = re.search(r"Ambience:\s*(\{.*?\})(?:,\s*[A-Z]|\s*\|)", description)
    if match:
        try:
            ambience = sorted(str(key) for key, value in ast.literal_eval(match.group(1)).items() if value is True)
        except (SyntaxError, ValueError, AttributeError):
            pass
    return {
        "name": (parts[0] if parts else "OTA POI")[:120],
        "city": (parts[2] if len(parts) > 2 else "unknown city")[:80],
        "category": category,
        "price_range": price,
        "ambience": ambience[:8],
    }


def _caption_tags(caption: str) -> tuple[list[str], list[str]]:
    text = caption.casefold()
    styles = [term for term in ("casual", "classy", "cozy", "historic", "modern", "romantic", "rustic", "trendy", "upscale", "vintage") if term in text]
    mapping = {"bar": ("bar", "tap"), "outdoor_seating": ("patio", "terrace"), "pool": ("pool",), "front_desk": ("front desk", "reception"), "parking": ("parking",), "wheelchair_access": ("wheelchair", "accessible")}
    facilities = [key for key, terms in mapping.items() if any(term in text for term in terms)]
    return sorted(styles), sorted(facilities)


def _collect_public_sources(source_root: Path, config: dict[str, Any], consumed: dict[str, set[str]], count: int) -> list[dict[str, Any]]:
    paths = config["dataset"]["source_paths"]
    captions = {}
    for row in _parquet_rows(source_root / paths["strong_pairs"], ["photo_id", "image_path", "caption", "label"]):
        caption = str(row.get("caption") or "").strip()
        if caption:
            captions[str(row["photo_id"])] = {"caption": caption[:120], "image_path": str(row["image_path"]), "label": str(row.get("label") or "unknown")}
    candidates = []
    seen_businesses: set[str] = set()
    seed = int(config["dataset"]["seed"])
    for row in _parquet_rows(source_root / paths["medium_pairs"], ["photo_id", "business_id", "image_path", "business_description"]):
        photo_id, business_id = str(row.get("photo_id") or ""), str(row.get("business_id") or "")
        if photo_id not in captions or not business_id or business_id in seen_businesses:
            continue
        source_id, group_id = f"yelp-photo:{photo_id}", f"yelp-business:{business_id}"
        if source_id in consumed["source_id"] or group_id in consumed["group_id"]:
            continue
        image = source_root / Path(str(row.get("image_path") or captions[photo_id]["image_path"]))
        if not image.is_file():
            continue
        seen_businesses.add(business_id)
        candidates.append((hashlib.sha256(f"{seed}\0{photo_id}".encode()).hexdigest(), {
            "photo_id": photo_id, "business_id": business_id, "source_id": source_id,
            "group_id": group_id, "source_image": image, "caption": captions[photo_id]["caption"],
            "business_description": str(row.get("business_description") or ""),
        }))
    selected = []
    for _, row in sorted(candidates, key=lambda item: item[0]):
        digest = sha256_file(row["source_image"])
        if digest in consumed["image_sha256"]:
            continue
        row["image_sha256"] = digest
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise Week7DataError(f"fresh public source shortfall: {len(selected)}/{count}")
    return selected


def _copy_image(root: Path, output: Path, source: Path, digest: str) -> str:
    suffix = source.suffix.lower() if source.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
    target = output / "images" / "public" / f"{digest}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    if sha256_file(target) != digest:
        raise Week7DataError("copied image hash mismatch")
    return target.relative_to(root).as_posix()


def _system() -> dict[str, str]:
    return {"role": "system", "content": "你是专业 OTA 多模态助手。只依据输入证据回答，不确定时标记 unknown。"}


def _user(image_path: str, text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "image", "path": image_path}, {"type": "text", "text": text}]}


def _row(sample_id: str, scenario: str, split: str, identity: dict[str, Any], messages: list[dict[str, Any]], target: Any, label_source: str, weight: float) -> dict[str, Any]:
    return {
        "sample_id": sample_id, "scenario": scenario, "split": split,
        "source_id": identity["source_id"], "image_sha256": identity["image_sha256"],
        "group_id": identity["group_id"], "constraint_template_id": identity.get("constraint_template_id"),
        "image_path": identity["image_path"], "messages": messages, "target": target,
        "label_source": label_source, "sample_weight": weight,
    }


def _category(attributes: dict[str, Any], label: str) -> str:
    categories = " ".join(attributes.get("categories", [])).casefold()
    text = f"{categories} {label.casefold()}"
    if any(term in text for term in ("hotel", "resort", "lodging")):
        return "hotel"
    if any(term in text for term in ("restaurant", "food", "cafe", "bar")):
        return "restaurant"
    if any(term in text for term in ("museum", "park", "attraction", "landmark")):
        return "attraction"
    return "other"


def _product_target(source: dict[str, Any]) -> dict[str, Any]:
    attrs = _business_fields(source["business_description"])
    styles, facilities = _caption_tags(source["caption"])
    unknown = []
    if not styles:
        unknown.append("style_tags")
    if not facilities:
        unknown.append("visible_facilities")
    if attrs["price_range"] == "unknown":
        unknown.append("price_range")
    return {
        "business_category": attrs["category"],
        "style_tags": sorted(set(styles + attrs["ambience"])),
        "visible_facilities": facilities,
        "price_range": attrs["price_range"],
        "observed_evidence": [source["caption"]],
        "inferred_attributes": [f"business_name:{attrs['name']}", f"city:{attrs['city']}"],
        "unknown_fields": unknown,
        "confidence": 0.65,
    }


def _itinerary_target(source: dict[str, Any], template_index: int) -> tuple[str, dict[str, Any]]:
    attrs = _business_fields(source["business_description"])
    days = 2 + template_index % 2
    hard = [f"行程共{days}天", "每日活动在19:00前结束", "使用公共交通"]
    soft = ["偏好图片所示风格", "每日包含一处用餐地点"]
    constraints = "；".join(hard + soft)
    itinerary = []
    for day in range(1, days + 1):
        itinerary.append({
            "day_index": day,
            "date": None,
            "summary": f"第{day}天按约束安排",
            "activities": [{
                "start_time": "10:00", "end_time": "17:00", "place_name": None,
                "activity": "参观图片风格地点并就近用餐", "transport": "公共交通",
                "source_evidence": [source["caption"]],
            }],
        })
    target = {
        "style_preferences": _caption_tags(source["caption"])[0] or ["图片所示风格"],
        "hard_constraints": hard,
        "soft_constraints": soft,
        "required_itinerary_elements": ["activities", "meals", "transport", "end_time"],
        "itinerary": itinerary,
        "constraint_check": [
            {"constraint": item, "constraint_type": "hard" if item in hard else "soft", "status": "satisfied", "evidence": "行程结构已覆盖"}
            for item in hard + soft
        ],
        "observed_evidence": [source["caption"], f"category:{attrs['category']}"],
        "unknown_fields": ["exact_place_name", "exact_date"],
        "confidence": 0.60,
    }
    return constraints, target


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _after_sales_identity(root: Path, output: Path, split: str, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    issue_types = ("transport_delay", "attraction_closure", "facility_damage", "hygiene_stain")
    issue = issue_types[index % len(issue_types)]
    severity = ("medium", "high", "low", "critical")[index % 4]
    reference = f"W7-{split.upper()}-{index:04d}"
    lines = {
        "transport_delay": ["SERVICE DELAY", reference, "Departure delayed 95 minutes"],
        "attraction_closure": ["TEMPORARILY CLOSED", reference, "Closed for maintenance today"],
        "facility_damage": ["ROOM NOTICE", reference, "Broken bathroom fixture"],
        "hygiene_stain": ["GUEST EVIDENCE", reference, "Visible stain on bed linen"],
    }[issue]
    image = Image.new("RGB", (1024, 640), (244, 246, 249))
    draw = ImageDraw.Draw(image)
    draw.rectangle((55, 55, 969, 585), outline=(40, 75, 115), width=6)
    for line_index, line in enumerate(lines):
        draw.text((110, 135 + line_index * 125), line, fill=(25, 35, 45), font=_font(42 if line_index else 54))
    target_path = output / "images" / "synthetic_after_sales" / f"{split}_{index:04d}.png"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(target_path, format="PNG")
    digest = sha256_file(target_path)
    identity = {
        "source_id": f"week7-synthetic-after-sales:{split}:{index}",
        "group_id": f"week7-synthetic-after-sales-group:{split}:{index}",
        "constraint_template_id": None,
        "image_sha256": digest,
        "image_path": target_path.relative_to(root).as_posix(),
    }
    target = {
        "issue_type": issue, "severity": severity, "issue_location": "evidence card",
        "key_information": [reference, lines[2]], "ocr_text": lines,
        "observed_evidence": lines, "unknown_fields": [], "confidence": 1.0,
    }
    return identity, target


def _public_identity(root: Path, output: Path, source: dict[str, Any], template_id: str | None = None) -> dict[str, Any]:
    return {
        "source_id": source["source_id"], "group_id": source["group_id"],
        "constraint_template_id": template_id, "image_sha256": source["image_sha256"],
        "image_path": _copy_image(root, output, source["source_image"], source["image_sha256"]),
    }


def _core_public_row(root: Path, output: Path, scenario: str, split: str, source: dict[str, Any], ordinal: int, weight: float) -> dict[str, Any]:
    if scenario == "image_product_search":
        target = _product_target(source)
        identity = _public_identity(root, output, source)
        prompt = "识别图片中的 OTA 商品属性并严格输出指定 JSON Schema；不可猜测不可见字段。"
    elif scenario == "itinerary_planning":
        template_id = f"week7-itinerary-template:{split}:{ordinal}"
        constraints, target = _itinerary_target(source, ordinal)
        identity = _public_identity(root, output, source, template_id)
        prompt = f"依据图片和以下约束规划行程，严格输出指定 JSON Schema：{constraints}"
    else:
        raise Week7DataError(f"unexpected public scenario: {scenario}")
    validate_output(root, scenario, target, "v1")
    sample_id = f"week7-{split}-{scenario}-{ordinal:04d}"
    return _row(sample_id, scenario, split, identity, [_system(), _user(identity["image_path"], prompt)], target, "programmatic_silver", weight)


def _after_sales_row(root: Path, output: Path, split: str, ordinal: int, weight: float) -> dict[str, Any]:
    identity, target = _after_sales_identity(root, output, split, ordinal)
    validate_output(root, "after_sales", target, "v1")
    return _row(
        f"week7-{split}-after_sales-{ordinal:04d}", "after_sales", split, identity,
        [_system(), _user(identity["image_path"], "读取视觉证据，判断问题类型、严重度和关键信息，严格输出指定 JSON Schema。")],
        target, "programmatic_silver", weight,
    )


def _dialogue_row(parent: dict[str, Any], split: str, ordinal: int, tool_fraction: float, weight: float) -> dict[str, Any]:
    rounds = 5 + ordinal % 4
    target = parent["target"]
    messages = list(parent["messages"])
    followups = [
        ("请明确引用刚才那张图片中的证据。", "我会继续只引用首次用户轮的图片证据。"),
        ("把偏好调整为更重视预算，但保留硬约束。", "已更新当前需求，历史硬约束保持不变。"),
        ("总结目前累计的限制，不要遗漏。", "已承接图片证据、预算调整和原有硬约束。"),
        ("检查前后回答是否存在逻辑冲突。", "上下文逻辑一致；不确定信息仍标记 unknown。"),
        ("给出最终结构化结果。", json.dumps(target, ensure_ascii=False, sort_keys=True)),
        ("再次确认没有引入图片外事实。", "确认：回答仅使用首次图片及对话内明确提供的信息。"),
        ("说明本轮相对最初需求的变化。", "新增预算优先级，未删除原有硬约束。"),
    ]
    for user_text, assistant_text in followups[: rounds - 1]:
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
    messages.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False, sort_keys=True)})
    tool_call = ordinal < round(tool_fraction * 1000) and ordinal % 10 == 0
    if tool_call:
        messages.insert(-1, {"role": "assistant", "content": "<tool_call>{\"name\":\"check_constraints\",\"arguments\":{\"scope\":\"conversation\"}}</tool_call>"})
        messages.insert(-1, {"role": "tool", "content": "{\"status\":\"ok\"}"})
    identity = {
        "source_id": parent["source_id"], "group_id": parent["group_id"],
        "constraint_template_id": parent.get("constraint_template_id"),
        "image_sha256": parent["image_sha256"], "image_path": parent["image_path"],
    }
    row = _row(f"week7-{split}-dialogue-{ordinal:04d}", "dialogue", split, identity, messages, target, "programmatic_silver", weight)
    row.update({"parent_sample_id": parent["sample_id"], "dialogue_rounds": rounds, "evaluation_dimensions": list(DIALOGUE_DIMENSIONS), "contains_tool_call": tool_call})
    return row


def _validate_partition_isolation(rows_by_split: dict[str, list[dict[str, Any]]], consumed: dict[str, set[str]]) -> dict[str, Any]:
    split_sets: dict[str, dict[str, set[str]]] = {}
    for split, rows in rows_by_split.items():
        split_sets[split] = {field: set() for field in IDENTITY_FIELDS}
        sample_ids: set[str] = set()
        for row in rows:
            if row["sample_id"] in sample_ids:
                raise Week7DataError(f"duplicate sample_id in {split}: {row['sample_id']}")
            sample_ids.add(row["sample_id"])
            for field in IDENTITY_FIELDS:
                value = row.get(field)
                if value and value in consumed[field]:
                    raise Week7DataError(f"consumed {field} entered {split}: {value}")
                if value:
                    split_sets[split][field].add(value)
    collisions = []
    splits = tuple(rows_by_split)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1:]:
            for field in IDENTITY_FIELDS:
                overlap = split_sets[left][field] & split_sets[right][field]
                if overlap:
                    collisions.append({"left": left, "right": right, "field": field, "count": len(overlap), "examples": sorted(overlap)[:3]})
    if collisions:
        raise Week7DataError(f"cross-split identity collisions: {collisions[:3]}")
    return {"status": "PASS", "dimensions": list(IDENTITY_FIELDS), "split_unique_counts": {split: {field: len(values) for field, values in fields.items()} for split, fields in split_sets.items()}, "cross_split_collisions": collisions}


def _write_split_files(output: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for split, rows in rows_by_split.items():
        path = output / f"{split}.jsonl"
        write_jsonl_new(path, rows)
        files[path.name] = {"count": len(rows), "sha256": sha256_file(path)}
        for scenario in (*CORE_SCENARIOS, "dialogue", "general_multimodal"):
            scenario_rows = [row for row in rows if row["scenario"] == scenario]
            if not scenario_rows:
                continue
            scenario_path = output / split / f"{scenario}.jsonl"
            write_jsonl_new(scenario_path, scenario_rows)
            files[scenario_path.relative_to(output).as_posix()] = {"count": len(scenario_rows), "sha256": sha256_file(scenario_path)}
    return files


def build_week7_lock(root: Path, source_root: Path, config_path: Path) -> Path:
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    config = load_week7_config(config_path)
    dataset = config["dataset"]
    output = root / dataset["output_root"] / dataset["dataset_version"]
    if output.exists():
        raise Week7DataError(f"immutable lock path already exists: {output}")
    output.mkdir(parents=True)
    consumed, exclusion_evidence = load_consumed_identities(source_root, config)
    dialogue_audit = audit_week5_dialogues(source_root, config, consumed)

    train_core = int(dataset["train_per_core_scenario"])
    dev_core = int(dataset["development_per_core_scenario"])
    test_core = int(dataset["test_per_core_scenario"])
    public_needed = 2 * (train_core + dev_core + test_core) + int(dataset["general_regularization_count"])
    sources = iter(_collect_public_sources(source_root, config, consumed, public_needed))
    core: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "development", "test")}
    split_counts = {"train": train_core, "development": dev_core, "test": test_core}
    silver_weight = float(dataset["silver_weight"])
    for split, count in split_counts.items():
        for scenario in ("image_product_search", "itinerary_planning"):
            for ordinal in range(count):
                core[split].append(_core_public_row(root, output, scenario, split, next(sources), ordinal, silver_weight))
        for ordinal in range(count):
            core[split].append(_after_sales_row(root, output, split, ordinal, silver_weight))

    general = []
    for ordinal in range(int(dataset["general_regularization_count"])):
        source = next(sources)
        identity = _public_identity(root, output, source)
        target = {"caption": source["caption"], "evidence_policy": "visible_only"}
        general.append(_row(
            f"week7-train-general-{ordinal:04d}", "general_multimodal", "train", identity,
            [_system(), _user(identity["image_path"], "用一句话描述图片中可见内容，不补充不可见事实。")],
            target, "programmatic_silver", silver_weight,
        ))

    dialogue_counts = {"train": int(dataset["dialogue_count"]), "development": int(dataset["development_dialogue_count"]), "test": int(dataset["test_dialogue_count"])}
    dialogues: dict[str, list[dict[str, Any]]] = {}
    for split, count in dialogue_counts.items():
        parents = core[split]
        dialogues[split] = [_dialogue_row(parents[index % len(parents)], split, index, float(config["sampling"]["tool_call_dialogue_fraction"]), silver_weight) for index in range(count)]
    rows_by_split = {
        "train": core["train"] + general + dialogues["train"],
        "development": core["development"] + dialogues["development"],
        "test": core["test"] + dialogues["test"],
    }
    isolation = _validate_partition_isolation(rows_by_split, consumed)
    if len(rows_by_split["train"]) != int(dataset["train_total"]):
        raise Week7DataError("training row count changed")
    manifest_rows = []
    for split, rows in rows_by_split.items():
        for row in rows:
            manifest_rows.append({field: row.get(field) for field in IDENTITY_FIELDS} | {"split": split, "scenario": row["scenario"], "label_source": row["label_source"]})
    write_jsonl_new(output / "identity_manifest.jsonl", manifest_rows)
    files = _write_split_files(output, rows_by_split)
    files["identity_manifest.jsonl"] = {"count": len(manifest_rows), "sha256": sha256_file(output / "identity_manifest.jsonl")}
    queue = [
        {"queue_id": f"week7-dialogue-human-{index:03d}", "sample_id": row["sample_id"], "required_dimensions": list(DIALOGUE_DIMENSIONS), "human_reviewer": None, "human_scores": None, "decision": "PENDING_REAL_HUMAN_INPUT"}
        for index, row in enumerate(dialogues["development"][: int(config["evaluation"]["human_review_queue_size"])])
    ]
    write_jsonl_new(output / "dialogue_human_review_queue.jsonl", queue)
    files["dialogue_human_review_queue.jsonl"] = {"count": len(queue), "sha256": sha256_file(output / "dialogue_human_review_queue.jsonl")}
    counts = {split: dict(Counter(row["scenario"] for row in rows)) for split, rows in rows_by_split.items()}
    lock = {
        "schema_version": "week7_dataset_lock_v1", "dataset_version": dataset["dataset_version"],
        "seed": dataset["seed"], "config_sha256": sha256_file(config_path),
        "source_project_root_recorded": str(source_root), "exclusion_evidence": exclusion_evidence,
        "week5_dialogue_audit": dialogue_audit, "counts": counts,
        "actual_train_ratios": {"general_multimodal": len(general) / len(rows_by_split["train"]), "dialogue": len(dialogues["train"]) / len(rows_by_split["train"])},
        "isolation": isolation, "files": files,
        "test_policy": {"status": "LOCKED_UNCONSUMED", "may_read_only_after_parameter_lock": True, "maximum_evaluations": 1},
        "label_policy": {"week7_rows": "programmatic_silver", "human_accepted_rows": 0, "human_queue_status": "PENDING_REAL_HUMAN_INPUT"},
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    (output / "dataset_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def validate_week7_lock(root: Path, config_path: Path, *, include_test: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    config = load_week7_config(config_path)
    output = root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    lock = json.loads((output / "dataset_lock.json").read_text(encoding="utf-8"))
    lock_hash = lock.pop("lock_sha256")
    if canonical_sha256(lock) != lock_hash:
        raise Week7DataError("dataset_lock.json canonical hash mismatch")
    lock["lock_sha256"] = lock_hash
    splits = ["train", "development"] + (["test"] if include_test else [])
    validated = Counter()
    for split in splits:
        path = output / f"{split}.jsonl"
        if sha256_file(path) != lock["files"][path.name]["sha256"]:
            raise Week7DataError(f"split hash mismatch: {split}")
        for row in iter_jsonl(path):
            image = root / row["image_path"]
            if sha256_file(image) != row["image_sha256"]:
                raise Week7DataError(f"image hash mismatch: {row['sample_id']}")
            if row["scenario"] in CORE_SCENARIOS:
                validate_output(root, row["scenario"], row["target"], "v1")
            if row["scenario"] == "dialogue":
                image_blocks = [item for message in row["messages"] for item in (message.get("content") if isinstance(message.get("content"), list) else []) if isinstance(item, dict) and item.get("type") == "image"]
                if len(image_blocks) != 1 or row["dialogue_rounds"] not in range(5, 9):
                    raise Week7DataError(f"dialogue template violation: {row['sample_id']}")
            validated[split] += 1
    return {"status": "PASS", "dataset_version": lock["dataset_version"], "lock_sha256": lock_hash, "validated_splits": dict(validated), "test_consumed": include_test, "isolation": lock["isolation"], "actual_train_ratios": lock["actual_train_ratios"]}

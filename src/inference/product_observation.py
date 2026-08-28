"""Image observation followed by deterministic public-product contract mapping."""
import json
import hashlib
from pathlib import Path
import re
import time

from src.evaluation.schema_validation import _validate_instance
from src.inference.schemas import ModelAttempt
from src.inference.transport_utils import strip_json_fence


def canonical_config_sha256(config):
    return hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_observation_config(path: Path, expected_sha256: str):
    config = json.loads(path.read_text(encoding="utf-8"))
    if canonical_config_sha256(config) != expected_sha256:
        raise ValueError("product observation config hash mismatch")
    if (config.get("protocol") not in {"product_visual_observation_v1", "product_visual_observation_v2", "product_visual_observation_v3", "product_visual_observation_v4"}
            or config.get("max_attempts") != 2 or type(config.get("max_new_tokens")) is not int
            or not 1 <= config["max_new_tokens"] <= 4096
            or config.get("price_policy") != "unknown_without_verified_comparison_scale"):
        raise ValueError("unsupported product observation contract")
    for field in ("style_vocabulary", "facility_vocabulary"):
        values = config.get(field)
        if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
            raise ValueError("invalid observation vocabulary")
    mapping = config.get("subject_categories")
    if not isinstance(mapping, dict) or not mapping or any(value not in {"hotel", "restaurant", "attraction", "other", "unknown"} for value in mapping.values()):
        raise ValueError("invalid observation category mapping")
    if any(not isinstance(config.get(key), str) or not config[key].strip() for key in ("system_prompt", "task_prompt")):
        raise ValueError("observation prompt missing")
    return config


def observation_schema(config):
    def cues(values):
        if config["protocol"] == "product_visual_observation_v4":
            # label -> fact 去掉重复键名，不删除正标签或缩减事实支持。
            return {"type": "object", "additionalProperties": False, "properties": {
                label: {"type": "string", "minLength": 1, "maxLength": 80} for label in values}}
        return {"type": "array", "maxItems": len(values), "items": {
            "type": "object", "additionalProperties": False, "required": ["label", "fact"],
            "properties": {"label": {"type": "string", "enum": values},
                           "fact": {"type": "string", "minLength": 1, "maxLength": 80}}}}
    return {"type": "object", "additionalProperties": False,
            "required": ["subject_kind", "subject_fact", "style_evidence", "facility_evidence", "price_text"],
            "properties": {
                "subject_kind": {"type": "string", "enum": list(config["subject_categories"])},
                "subject_fact": {"type": "string", "minLength": 1, "maxLength": 80},
                "style_evidence": cues(config["style_vocabulary"]),
                "facility_evidence": cues(config["facility_vocabulary"]),
                "price_text": {"type": "array", "maxItems": 4, "uniqueItems": True,
                               "items": {"type": "string", "minLength": 1, "maxLength": 80}}}}


def validate_observation(value, config):
    _validate_instance(observation_schema(config), value, path="$")
    original = value
    compact = config["protocol"] == "product_visual_observation_v4"
    value = _canonical_observation(value, config)
    for field in ("style_evidence", "facility_evidence"):
        labels = [item["label"] for item in value[field]]
        if len(labels) != len(set(labels)):
            raise ValueError(f"duplicate labels in {field}")
        for item in value[field]:
            if compact and re.search(r"\b(?:impl(?:y|ies|ied)|infer(?:red)?|suggest(?:s|ed)?|likely|presumably|must have)\b|意味着|应当有|推断|暗示", item["fact"], re.I):
                raise ValueError(f"inferred rather than observable evidence in {field}")
            if re.search(r"\b(?:not visible|not shown|cannot see|assum\w*|probably)\b|未见|看不到|推测|可能存在|photo type", item["fact"], re.I):
                raise ValueError(f"non-observational evidence in {field}")
            if negated_label_fact(item["label"], item["fact"]):
                raise ValueError(f"negated positive label evidence in {field}")
    if value["subject_kind"] == "food_closeup" and (value["style_evidence"] or value["facility_evidence"]):
        raise ValueError("food closeup cannot establish venue style or facilities")
    return original


def _canonical_observation(value, config):
    if config["protocol"] != "product_visual_observation_v4":
        return value
    return {**value, **{field: [{"label": label, "fact": fact} for label, fact in value[field].items()]
                        for field in ("style_evidence", "facility_evidence")}}


def parse_observation(raw, config):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate observation key: " + key)
            result[key] = value
        return result
    options = {"object_pairs_hook": unique_object} if config["protocol"] == "product_visual_observation_v4" else {}
    return json.loads(strip_json_fence(raw), **options)


def negated_label_fact(label, fact):
    # 否定须紧邻该标签或可见物件；不把“椅子旁没有杂物”误当作没有椅子。
    aliases = {"seating": ("chairs", "chair", "benches", "bench", "stools", "stool"),
               "dining_tables": ("tables", "table"), "front_desk": ("reception",),
               "outdoor_seating": ("patio", "terrace"), "wifi_sign": ("wifi", "wi-fi"),
               "parking": ("停车", "停车场")}
    terms = (label.replace("_", " "), *aliases.get(label, ()))
    target = "(?:" + "|".join(re.escape(term) for term in terms) + ")"
    before = r"\b(?:no|not|without|neither|lack(?:s|ing)?|absent)\s+(?:(?:any|visible|accessible|a|an|the)\s+){0,3}" + target + r"\b"
    after = r"\b" + target + r"\s+(?:(?:is|are|was|were)\s+)?(?:not|absent|unavailable|missing)\b"
    return bool(re.search(before + "|" + after + r"|(?:没有|无|缺少)" + target, fact, re.I))


def map_observation(value, config):
    value = _canonical_observation(validate_observation(value, config), config)
    category = config["subject_categories"][value["subject_kind"]]
    styles = sorted(item["label"] for item in value["style_evidence"])
    facilities = sorted(item["label"] for item in value["facility_evidence"])
    facts = list(dict.fromkeys([value["subject_fact"]] + [item["fact"] for key in ("style_evidence", "facility_evidence") for item in value[key]]))
    if len(facts) > 10:
        raise ValueError("at most ten distinct facts; combine related visible facts without dropping labels")
    # 价位需要币种、单位和明确比较口径；可见数字不自动变成价位标签。
    if config["price_policy"] != "unknown_without_verified_comparison_scale":
        raise ValueError("unsupported price evidence policy")
    unknown = ["price_range"]
    if category == "unknown":
        unknown.append("business_category")
    if not styles:
        unknown.append("style_tags")
    if not facilities:
        unknown.append("visible_facilities")
    return {"business_category": category, "style_tags": styles, "visible_facilities": facilities,
            "price_range": "unknown", "observed_evidence": facts, "inferred_attributes": [],
            "unknown_fields": sorted(unknown), "confidence": None}


def observation_messages(image, config):
    image_url = str(image) if str(image).startswith(("data:", "https://", "http://", "file://")) else "file://" + str(image).replace("\\", "/")
    schema = json.dumps(observation_schema(config), ensure_ascii=False, separators=(",", ":"))
    return [{"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}},
             {"type": "text", "text": config["task_prompt"] + "\nJSON Schema: " + schema}]}]


def generate_observation(backend, image, config):
    messages = observation_messages(image, config)
    attempts = []
    active = messages
    for attempt in range(1, config["max_attempts"] + 1):
        started = time.perf_counter()
        generated = backend.generate_with_usage(active, response_format=None, max_new_tokens=config["max_new_tokens"])
        error = None
        observation = product = None
        try:
            observation = parse_observation(generated.content, config)
            product = map_observation(observation, config)
        except ValueError as exc:
            # 不把错误对象的完整 instance/schema 重复塞回 Prompt。
            error = str(exc)
        attempts.append(ModelAttempt(attempt=attempt, raw_output=generated.content, error=error,
                                     input_tokens=generated.input_tokens, output_tokens=generated.output_tokens,
                                     latency_ms=(time.perf_counter() - started) * 1000))
        if error is None:
            return {"passed": True, "result": product, "observation": observation, "attempts": attempts}
        previous = [{"role": "assistant", "content": generated.content}] if config["protocol"] == "product_visual_observation_v4" else []
        active = [*messages, *previous, {"role": "user", "content": "Validation error: " + error + ". Reobserve the original image and return a complete corrected JSON. Do not invent facts."}]
    return {"passed": False, "result": None, "observation": None, "attempts": attempts}

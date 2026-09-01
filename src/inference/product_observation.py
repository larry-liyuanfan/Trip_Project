"""Image observation followed by deterministic public-product contract mapping."""
import json
import hashlib
from pathlib import Path
import re
import time

from src.evaluation.schema_validation import _validate_instance
from src.inference.schemas import ModelAttempt
from src.inference.transport_utils import strip_json_fence


FOOD_SUBJECT_CONFLICT = "food closeup cannot establish venue style or facilities"


def canonical_config_sha256(config):
    return hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_observation_config(path: Path, expected_sha256: str):
    """Load the selected observation protocol only when its canonical hash matches."""
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
    if config.get("prompt_schema_style", "expanded") not in {"expanded", "property_names"}:
        raise ValueError("unsupported prompt schema presentation")
    if config.get("prompt_schema_style") == "property_names" and config["protocol"] != "product_visual_observation_v4":
        raise ValueError("compact schema presentation requires compact evidence objects")
    if "style_refinement" in config:
        from src.inference.product_style_refinement import validate_refinement_config
        validate_refinement_config(config)
    if "category_refinement" in config:
        from src.inference.product_category_refinement import validate_category_refinement
        validate_category_refinement(config)
    if "facility_refinement" in config:
        from src.inference.product_facility_refinement import validate_facility_refinement
        validate_facility_refinement(config)
    if "style_scope_policy" in config:
        from src.inference.product_style_scope import validate_style_scope
        validate_style_scope(config)
    validate_correction_protocol(config)
    validate_prompt_schema_annotations(config)
    return config


def observation_schema(config):
    """Build the intermediate visible-facts Schema from versioned cue lists."""
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
    """Reject malformed visible facts before they reach business-label mapping."""
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
        raise ValueError(FOOD_SUBJECT_CONFLICT)
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
    """Map supported visible cues conservatively without inventing labels."""
    value = _canonical_observation(validate_observation(value, config), config)
    category = config["subject_categories"][value["subject_kind"]]
    from src.inference.product_style_scope import venue_style_evidence
    style_evidence, _ = venue_style_evidence(value, config)
    styles = sorted(item["label"] for item in style_evidence)
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


def validate_prompt_schema_annotations(config):
    if "prompt_schema_annotations" not in config:
        return
    annotations = config["prompt_schema_annotations"]
    fields = {"subject_kind", "subject_fact", "style_evidence", "facility_evidence", "price_text"}
    if (config.get("protocol") != "product_visual_observation_v3"
            or not isinstance(annotations, dict) or not annotations
            or any(key not in fields or not isinstance(value, str)
                   or not value.strip() or len(value) > 400
                   for key, value in annotations.items())):
        raise ValueError("invalid observation prompt schema annotations")


def observation_prompt_schema(config):
    validate_prompt_schema_annotations(config)
    schema_value = observation_schema(config)
    # 注释只进入模型提示，不改变实际 Schema、字段支持或语义校验。
    for field, description in config.get("prompt_schema_annotations", {}).items():
        schema_value["properties"][field]["description"] = description
    if config.get("prompt_schema_style") == "property_names":
        # 与展开 Schema 等价；提示词不逐项列出可选属性，避免诱导把缺席设施也填满。
        for field, vocabulary in (("style_evidence", "style_vocabulary"), ("facility_evidence", "facility_vocabulary")):
            schema_value["properties"][field] = {"type": "object", "propertyNames": {"enum": config[vocabulary]},
                "additionalProperties": {"type": "string", "minLength": 1, "maxLength": 80}}
    return schema_value


def observation_messages(image, config):
    image_url = str(image) if str(image).startswith(("data:", "https://", "http://", "file://")) else "file://" + str(image).replace("\\", "/")
    schema = json.dumps(observation_prompt_schema(config), ensure_ascii=False, separators=(",", ":"))
    return [{"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}},
             {"type": "text", "text": config["task_prompt"] + "\nJSON Schema: " + schema}]}]


def validate_correction_protocol(config):
    protocol = config.get("correction_protocol", "legacy_v1")
    if protocol not in {"legacy_v1", "bounded_history_v1", "subject_schema_v1", "subject_schema_v2", "food_conflict_schema_v1"}:
        raise ValueError("unsupported observation correction protocol")
    if protocol != "legacy_v1" and (config.get("protocol") != "product_visual_observation_v3" or config.get("max_attempts") != 2):
        raise ValueError("bounded correction requires v3 with exactly one correction")


def observation_correction_response_format(config, validation_error=None):
    validate_correction_protocol(config)
    selected = config.get("correction_protocol")
    if selected == "food_conflict_schema_v1":
        if not isinstance(validation_error, str) or not validation_error:
            raise ValueError("food conflict decoder requires the previous validation error")
        # 只约束明确的主体/场所字段矛盾；其他错误保留原纠错，不扰动其语义标签。
        if validation_error != FOOD_SUBJECT_CONFLICT:
            return None
    elif selected not in {"subject_schema_v1", "subject_schema_v2"}:
        return None
    from src.inference.observation_constraints import PROTOCOL, SHARED_SERIALIZATION_PROTOCOL
    protocol = PROTOCOL if selected == "subject_schema_v1" else SHARED_SERIALIZATION_PROTOCOL
    return {"type": "json_schema", "constraint_protocol": protocol,
            "json_schema": {"name": "product_observation_correction", "schema": observation_schema(config)}}


def observation_correction_messages(messages, raw, error, config):
    validate_correction_protocol(config)
    history = config.get("correction_protocol") == "bounded_history_v1"
    previous = [{"role": "assistant", "content": raw}] if history or config["protocol"] == "product_visual_observation_v4" else []
    instruction = "Validation error: " + error + ". Reobserve the original image and return a complete corrected JSON. Do not invent facts."
    if history:
        # 只带本请求的一条失败答复；不注入参考标签、不修改映射规则或增加纠错次数。
        instruction += (" Your previous response is invalid; do not repeat the contradiction or return a partial patch. "
                        "Check subject_kind against the actual visible scene and make all evidence fields consistent with it. "
                        "A food_closeup has no venue style or facility evidence; those arrays must be empty. "
                        "Only use a venue subject_kind when the image itself establishes a venue. "
                        "Unsupported evidence must remain absent.")
    return [*messages, *previous, {"role": "user", "content": instruction}]


def generate_observation(backend, image, config):
    """Generate visible facts, allow one bounded correction, then map labels."""
    validate_correction_protocol(config)
    if config.get("facility_refinement") is not None:
        from src.inference.product_facility_refinement import generate_facility_refined_observation
        return generate_facility_refined_observation(backend, image, config)
    if config.get("category_refinement") is not None:
        from src.inference.product_category_refinement import generate_category_refined_observation
        return generate_category_refined_observation(backend, image, config)
    if config.get("style_refinement") is not None:
        from src.inference.product_style_refinement import generate_refined_observation
        return generate_refined_observation(backend, image, config)
    messages = observation_messages(image, config)
    attempts = []
    active = messages
    for attempt in range(1, config["max_attempts"] + 1):
        started = time.perf_counter()
        response_format = observation_correction_response_format(config, attempts[-1].error) if attempt > 1 else None
        generated = backend.generate_with_usage(active, response_format=response_format, max_new_tokens=config["max_new_tokens"])
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
            result = {"passed": True, "result": product, "observation": observation, "attempts": attempts}
            if config.get("style_scope_policy") is not None:
                from src.inference.product_style_scope import venue_style_evidence
                result["style_scope_exclusions"] = venue_style_evidence(observation, config)[1]
            return result
        active = observation_correction_messages(messages, generated.content, error, config)
    return {"passed": False, "result": None, "observation": None, "attempts": attempts}

"""Image observation followed by deterministic public-product contract mapping."""
import json
import re
import time

from src.evaluation.schema_validation import _validate_instance
from src.inference.schemas import ModelAttempt
from src.inference.transport_utils import strip_json_fence


def observation_schema(config):
    def cues(values):
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
    for field in ("style_evidence", "facility_evidence"):
        labels = [item["label"] for item in value[field]]
        if len(labels) != len(set(labels)):
            raise ValueError(f"duplicate labels in {field}")
        for item in value[field]:
            if re.search(r"\b(?:not visible|not shown|cannot see|assum\w*|probably)\b|未见|看不到|推测|可能存在|photo type", item["fact"], re.I):
                raise ValueError(f"non-observational evidence in {field}")
    if value["subject_kind"] == "food_closeup" and (value["style_evidence"] or value["facility_evidence"]):
        raise ValueError("food closeup cannot establish venue style or facilities")
    return value


def map_observation(value, config):
    value = validate_observation(value, config)
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
            observation = json.loads(strip_json_fence(generated.content))
            product = map_observation(observation, config)
        except ValueError as exc:
            # 不把错误对象的完整 instance/schema 重复塞回 Prompt。
            error = str(exc)
        attempts.append(ModelAttempt(attempt=attempt, raw_output=generated.content, error=error,
                                     input_tokens=generated.input_tokens, output_tokens=generated.output_tokens,
                                     latency_ms=(time.perf_counter() - started) * 1000))
        if error is None:
            return {"passed": True, "result": product, "observation": observation, "attempts": attempts}
        active = [*messages, {"role": "user", "content": "Validation error: " + error + ". Reobserve the original image and return a complete corrected JSON. Do not invent facts."}]
    return {"passed": False, "result": None, "observation": None, "attempts": attempts}

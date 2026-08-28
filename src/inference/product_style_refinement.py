"""Versioned visual style review without changing category, facilities or price."""
import copy
import json
import re
import time

from src.evaluation.schema_validation import _validate_instance
from src.inference.product_observation import (
    map_observation, observation_schema, parse_observation,
)
from src.inference.schemas import ModelAttempt


def validate_refinement_config(config):
    refinement = config.get("style_refinement")
    if refinement is None:
        return
    if (config.get("protocol") != "product_visual_observation_v3"
            or not isinstance(refinement, dict)
            or refinement.get("protocol") != "visual_style_review_v1"
            or refinement.get("mode") not in {"replace", "add_only", "repair_invalid"}
            or refinement.get("eligibility") not in {"non_food_scene", "nonempty_style", "out_of_scope_fact"}
            or refinement.get("max_attempts") != 2
            or type(refinement.get("max_new_tokens")) is not int
            or not 1 <= refinement["max_new_tokens"] <= 1024
            or any(not isinstance(refinement.get(key), str) or not refinement[key].strip()
                   for key in ("system_prompt", "task_prompt"))):
        raise ValueError("invalid bounded visual style refinement config")
    if refinement["eligibility"] == "out_of_scope_fact":
        from src.inference.product_style_scope import validate_style_scope
        if config.get("style_scope_policy") is None:
            raise ValueError("scope-triggered review requires its explicit scope policy")
        validate_style_scope(config)
    if refinement["mode"] == "repair_invalid" and refinement["eligibility"] != "out_of_scope_fact":
        raise ValueError("targeted style repair requires scope-triggered eligibility")
    action = refinement.get("unsupported_scope_action", "reject")
    if action not in {"reject", "abstain"} or (action == "abstain" and refinement["mode"] != "repair_invalid"):
        raise ValueError("scope abstention is only allowed for targeted inference hypotheses")


def should_refine(observation, config):
    refinement = config["style_refinement"]
    # 只按当前图像输出路由；不读取样本身份、教师或商家 metadata。
    if observation["subject_kind"] == "food_closeup":
        return False
    if refinement["eligibility"] == "out_of_scope_fact":
        from src.inference.product_style_scope import venue_style_evidence
        return bool(venue_style_evidence(observation, config)[1])
    return (refinement["eligibility"] == "non_food_scene"
            or bool(observation["style_evidence"]))


def refinement_schema(config):
    return {"type": "object", "additionalProperties": False, "required": ["style_evidence"],
            "properties": {"style_evidence": observation_schema(config)["properties"]["style_evidence"]}}


def refinement_messages(image, primary, config):
    refinement = config["style_refinement"]
    image_url = str(image)
    if not image_url.startswith(("data:", "https://", "http://", "file://")):
        image_url = "file://" + image_url.replace("\\", "/")
    text = refinement["task_prompt"]
    if refinement["mode"] == "add_only":
        # 这只是同一候选先前生成的标签，不是评测参考；替换模式完全独立看图。
        text += "\nAlready recorded styles (do not repeat): " + json.dumps(
            primary["style_evidence"], ensure_ascii=False, separators=(",", ":"))
    elif refinement["mode"] == "repair_invalid":
        from src.inference.product_style_scope import venue_style_evidence
        invalid = venue_style_evidence(primary, config)[1]
        text += "\nUnverified style hypotheses to check (not reference labels): " + json.dumps(
            [cue["label"] for cue in invalid], separators=(",", ":"))
    text += "\nJSON Schema: " + json.dumps(refinement_schema(config), separators=(",", ":"))
    return [{"role": "system", "content": refinement["system_prompt"]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}},
                                         {"type": "text", "text": text}]}]


def apply_refinement(primary, raw, config):
    # 新复查对象拒绝重复键；保留旧观察协议的历史解析行为不变。
    parsed = parse_observation(raw, {"protocol": "product_visual_observation_v4"})
    _validate_instance(refinement_schema(config), parsed, path="$")
    proposed = parsed["style_evidence"]
    # 先独立验证全部提议，不能靠合并去掉重复或矛盾标签以伪造有效输出。
    validation = {**copy.deepcopy(primary), "style_evidence": proposed}
    map_observation(validation, config)
    if config["style_refinement"]["mode"] == "repair_invalid":
        from src.inference.product_style_scope import venue_style_evidence
        permitted = {cue["label"] for cue in venue_style_evidence(primary, config)[1]}
        if any(cue["label"] not in permitted for cue in proposed):
            raise ValueError("targeted review cannot add unrequested style labels")
    for cue in proposed:
        if re.search(r"\b(?:impl(?:y|ies|ied)|infer(?:red)?|suggest(?:s|ed)?|likely|presumably|must have)\b|意味着|应当有|推断|暗示", cue["fact"], re.I):
            raise ValueError("style review requires observable facts, not inference")
    if config.get("style_scope_policy") is not None:
        from src.inference.product_style_scope import venue_style_evidence
        kept, excluded = venue_style_evidence(validation, config)
        if excluded:
            if config["style_refinement"].get("unsupported_scope_action", "reject") != "abstain":
                raise ValueError("style review still uses nonvenue object facts; reobserve venue decor")
            # 结构有效但只找到非场所物件：按显式未知策略弃权，不把玻璃杯等当作装修。
            # 原始答复仍完整保留；结构、重复标签、推断句和长度错误仍然失败。
            proposed = kept
    merged = copy.deepcopy(primary)
    if config["style_refinement"]["mode"] == "replace":
        merged["style_evidence"] = proposed
    elif config["style_refinement"]["mode"] == "add_only":
        known = {cue["label"] for cue in primary["style_evidence"]}
        merged["style_evidence"].extend(cue for cue in proposed if cue["label"] not in known)
    else:
        from src.inference.product_style_scope import venue_style_evidence
        kept, _ = venue_style_evidence(primary, config)
        # 范围正确的标签保留；只修复或否定原先依据越界的假设，不作全面扩展。
        merged["style_evidence"] = copy.deepcopy(kept) + proposed
    # 仍用原十事实上限和完整公共 Schema，不截断正标签。
    product = map_observation(merged, config)
    return merged, product


def generate_refined_observation(backend, image, config):
    from src.inference.product_observation import generate_observation
    from src.inference.system_runtime import ModelGenerationError

    validate_refinement_config(config)
    base_config = {key: value for key, value in config.items() if key != "style_refinement"}
    primary = generate_observation(backend, image, base_config)
    primary["primary_attempt_count"] = len(primary["attempts"])
    primary["refinement_applied"] = False
    if not primary["passed"] or not should_refine(primary["observation"], config):
        return primary
    attempts = list(primary["attempts"])
    messages = refinement_messages(image, primary["observation"], config)
    active = messages
    for _ in range(config["style_refinement"]["max_attempts"]):
        started = time.perf_counter()
        try:
            generated = backend.generate_with_usage(active, response_format=None,
                max_new_tokens=config["style_refinement"]["max_new_tokens"])
        except ModelGenerationError as exc:
            attempts.extend(exc.attempts or [ModelAttempt(attempt=len(attempts) + 1,
                raw_output="", error=str(exc), latency_ms=(time.perf_counter() - started) * 1000)])
            break
        error = None
        try:
            observation, product = apply_refinement(primary["observation"], generated.content, config)
        except ValueError as exc:
            error = str(exc)
        attempts.append(ModelAttempt(attempt=len(attempts) + 1, raw_output=generated.content, error=error,
            input_tokens=generated.input_tokens, output_tokens=generated.output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000))
        if error is None:
            result = {"passed": True, "result": product, "observation": observation, "attempts": attempts,
                    "primary_attempt_count": primary["primary_attempt_count"], "refinement_applied": True,
                    "primary_scope_exclusions": primary.get("style_scope_exclusions", [])}
            if config["style_refinement"].get("unsupported_scope_action") == "abstain":
                from src.inference.product_style_scope import venue_style_evidence
                proposal = parse_observation(generated.content, {"protocol": "product_visual_observation_v4"})
                result["style_evidence_abstentions"] = venue_style_evidence(proposal, config)[1]
            return result
        active = [*messages, {"role": "assistant", "content": generated.content},
                  {"role": "user", "content": "Validation error: " + error +
                   ". Reinspect the same image and return the complete style_evidence JSON, without guesses."}]
    # 不以主阶段成功掩盖已要求的复查失败。
    return {"passed": False, "result": None, "observation": None, "attempts": attempts,
            "primary_attempt_count": primary["primary_attempt_count"], "refinement_applied": True,
            "primary_scope_exclusions": primary.get("style_scope_exclusions", [])}


def replay_refined_observation(record, config):
    validate_refinement_config(config)
    attempts = record.get("attempts", [])
    primary = None
    boundary = None
    for index, attempt in enumerate(attempts[:config["max_attempts"]]):
        if attempt.get("error") is None:
            primary = parse_observation(attempt["raw_output"], config)
            map_observation(primary, config)
            boundary = index + 1
            break
    if primary is None:
        raise ValueError("passing style pipeline has no valid primary observation")
    remaining = attempts[boundary:]
    if not should_refine(primary, config):
        if remaining:
            raise ValueError("unexpected style stage for ineligible primary observation")
        return map_observation(primary, config)
    if (not 1 <= len(remaining) <= config["style_refinement"]["max_attempts"]
            or remaining[-1].get("error") is not None
            or any(item.get("error") is None for item in remaining[:-1])):
        raise ValueError("passing style pipeline has incomplete or extra refinement attempts")
    _, product = apply_refinement(primary, remaining[-1]["raw_output"], config)
    return product

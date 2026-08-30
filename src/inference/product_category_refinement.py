"""Bounded image-only subject review; preserve style, facilities and price."""
import copy
import hashlib
import json
import re
import time

from src.evaluation.schema_validation import _validate_instance
from src.data.product_labels import affirmative_term
from src.inference.product_observation import map_observation, parse_observation
from src.inference.schemas import ModelAttempt


def category_review_source_hashes(root, generation_config):
    paths = set(generation_config.get("observation_profile_configs", {}).values())
    if generation_config.get("observation_config"):
        paths.add(generation_config["observation_config"])
    if not any(json.loads((root / path).read_text(encoding="utf-8")).get("category_refinement") for path in paths):
        return {}
    # 新组合阶段的源码随生成身份绑定，跨主机统一换行，不追改历史运行。
    sources = ("src/inference/product_category_refinement.py", "src/inference/product_observation.py",
               "src/inference/product_style_refinement.py", "src/inference/system_runtime.py",
               "src/inference/schemas.py", "scripts/review_week8_contracts.py",
               "src/data/product_labels.py")
    return {path: hashlib.sha256((root / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in sources}


def validate_category_review_identity(root, generation_config, identity):
    expected = category_review_source_hashes(root, generation_config)
    if expected and identity.get("category_review_source_lf_sha256") != expected:
        raise ValueError("subject review generation implementation changed or was not bound")


def validate_category_refinement(config):
    review = config.get("category_refinement")
    if review is None:
        return
    if (config.get("protocol") != "product_visual_observation_v3"
            or not isinstance(review, dict)
            or review.get("protocol") != "visual_subject_review_v1"
            or review.get("max_attempts") != 2
            or type(review.get("max_new_tokens")) is not int
            or not 1 <= review["max_new_tokens"] <= 512
            or any(not isinstance(review.get(key), str) or not review[key].strip()
                   for key in ("system_prompt", "task_prompt"))):
        raise ValueError("invalid bounded subject review config")
    kinds = review.get("eligible_subject_kinds")
    if (not isinstance(kinds, list) or not kinds or any(not isinstance(kind, str) for kind in kinds)
            or len(kinds) != len(set(kinds)) or not set(kinds) <= set(config["subject_categories"])):
        raise ValueError("invalid subject review eligibility")
    eligibility = review.get("eligibility", "subject_kind")
    if eligibility not in {"subject_kind", "visible_function_conflict"}:
        raise ValueError("unknown subject review eligibility rule")
    if eligibility == "visible_function_conflict":
        for key in ("visible_function_terms", "venue_context_terms"):
            terms = review.get(key)
            if (not isinstance(terms, list) or not 1 <= len(terms) <= 64
                    or any(not isinstance(term, str) or not term.strip() or len(term) > 60 for term in terms)
                    or len(terms) != len(set(terms))):
                raise ValueError("subject conflict terms must be explicit and bounded")


def should_review_subject(observation, config):
    # 只用当前模型输出决定复查；不使用样本ID、参考类别或商家属性。
    review = config["category_refinement"]
    if observation["subject_kind"] not in review["eligible_subject_kinds"]:
        return False
    if review.get("eligibility") == "visible_function_conflict":
        fact = observation["subject_fact"]
        # 词匹配仅触发独立看图，不直接产生新类别；否定和完整词沿用现有解析。
        return (any(affirmative_term(fact, term) for term in review["visible_function_terms"])
                and any(affirmative_term(fact, term) for term in review["venue_context_terms"]))
    return True


def subject_review_schema(config):
    return {"type": "object", "additionalProperties": False,
            "required": ["subject_kind", "subject_fact"], "properties": {
                "subject_kind": {"type": "string", "enum": list(config["subject_categories"])},
                "subject_fact": {"type": "string", "minLength": 1, "maxLength": 80}}}


def subject_review_messages(image, config):
    image_url = str(image)
    if not image_url.startswith(("data:", "https://", "http://", "file://")):
        image_url = "file://" + image_url.replace("\\", "/")
    review = config["category_refinement"]
    text = review["task_prompt"] + "\nJSON Schema: " + json.dumps(subject_review_schema(config), separators=(",", ":"))
    # 独立重看原图，不向复查模型提供旧类别、教师或候选设施标签。
    return [{"role": "system", "content": review["system_prompt"]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}},
                                         {"type": "text", "text": text}]}]


def apply_subject_review(primary, raw, config):
    proposed = parse_observation(raw, {"protocol": "product_visual_observation_v4"})
    _validate_instance(subject_review_schema(config), proposed, path="$")
    if re.search(r"\b(?:infer(?:red)?|likely|presumably|must have|metadata)\b|推断|暗示|商家属性", proposed["subject_fact"], re.I):
        raise ValueError("subject review requires directly observable evidence")
    refined = {**copy.deepcopy(primary), **proposed}
    # 完整原契约仍生效；不能为新类别删掉风格/设施，也不能掩盖食品与场所字段的矛盾。
    return refined, map_observation(refined, config)


def generate_category_refined_observation(backend, image, config):
    from src.inference.product_observation import generate_observation
    from src.inference.system_runtime import ModelGenerationError

    validate_category_refinement(config)
    base = {key: value for key, value in config.items() if key != "category_refinement"}
    primary = generate_observation(backend, image, base)
    primary["category_refinement_applied"] = False
    if not primary["passed"] or not should_review_subject(primary["observation"], config):
        return primary
    attempts = list(primary["attempts"])
    messages = subject_review_messages(image, config)
    active = messages
    for _ in range(config["category_refinement"]["max_attempts"]):
        started = time.perf_counter()
        try:
            generated = backend.generate_with_usage(active, response_format=None,
                max_new_tokens=config["category_refinement"]["max_new_tokens"])
        except ModelGenerationError as exc:
            attempts.extend(exc.attempts or [ModelAttempt(attempt=len(attempts) + 1,
                raw_output="", error=str(exc), latency_ms=(time.perf_counter() - started) * 1000)])
            break
        error = None
        try:
            observation, product = apply_subject_review(primary["observation"], generated.content, config)
        except ValueError as exc:
            error = str(exc)
        attempts.append(ModelAttempt(attempt=len(attempts) + 1, raw_output=generated.content, error=error,
            input_tokens=generated.input_tokens, output_tokens=generated.output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000))
        if error is None:
            return {**primary, "observation": observation, "result": product,
                    "attempts": attempts, "category_refinement_applied": True}
        active = [*messages, {"role": "assistant", "content": generated.content},
                  {"role": "user", "content": "Validation error: " + error +
                   ". Reinspect the original image and return the complete subject JSON; do not guess."}]
    return {**primary, "passed": False, "result": None, "observation": None,
            "attempts": attempts, "category_refinement_applied": True}


def replay_category_refined_observation(record, config):
    from src.inference.product_style_refinement import (
        apply_refinement, replay_refined_observation, should_refine,
    )

    validate_category_refinement(config)
    attempts = record.get("attempts", [])
    base = {key: value for key, value in config.items() if key != "category_refinement"}
    boundary = next((index + 1 for index, attempt in enumerate(attempts[:config["max_attempts"]])
                     if attempt.get("error") is None), None)
    if boundary is None:
        raise ValueError("subject review has no successful primary observation")
    primary = parse_observation(attempts[boundary - 1]["raw_output"], base)
    map_observation(primary, base)
    if base.get("style_refinement") is not None:
        if should_refine(primary, base):
            count = next((index + 1 for index, attempt in enumerate(
                attempts[boundary:boundary + base["style_refinement"]["max_attempts"]])
                if attempt.get("error") is None), None)
            if count is None:
                raise ValueError("subject review has no successful style stage")
            boundary += count
            primary, _ = apply_refinement(primary, attempts[boundary - 1]["raw_output"], base)
        # 复用旧复查的严格重放，不依赖自报阶段计数或已映射的最终结果。
        if replay_refined_observation({"attempts": attempts[:boundary]}, base) != map_observation(primary, base):
            raise ValueError("subject review primary replay mismatch")
    remaining = attempts[boundary:]
    if not should_review_subject(primary, config):
        if remaining:
            raise ValueError("unexpected subject review for ineligible observation")
        return map_observation(primary, config)
    if (not 1 <= len(remaining) <= config["category_refinement"]["max_attempts"]
            or remaining[-1].get("error") is not None
            or any(attempt.get("error") is None for attempt in remaining[:-1])):
        raise ValueError("subject review is incomplete or has extra attempts")
    return apply_subject_review(primary, remaining[-1]["raw_output"], config)[1]

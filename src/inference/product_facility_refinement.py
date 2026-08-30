"""Bounded image-only facility review that preserves every other product field."""
import copy
import hashlib
import json
import re
import time

from src.evaluation.schema_validation import _validate_instance
from src.inference.product_observation import map_observation, observation_schema, parse_observation
from src.inference.schemas import ModelAttempt


def validate_facility_refinement(config):
    review = config.get("facility_refinement")
    if review is None:
        return
    if (config.get("protocol") != "product_visual_observation_v3"
            or not isinstance(review, dict)
            or review.get("protocol") != "visual_facility_review_v1"
            or review.get("mode") != "replace"
            or review.get("eligibility") != "non_food_scene"
            or review.get("max_attempts") != 2
            or type(review.get("max_new_tokens")) is not int
            or not 1 <= review["max_new_tokens"] <= 512
            or any(not isinstance(review.get(key), str) or not review[key].strip()
                   for key in ("system_prompt", "task_prompt"))):
        raise ValueError("invalid bounded visual facility refinement config")


def should_review_facilities(observation, config):
    validate_facility_refinement(config)
    # 只按当前图像输出路由；食品特写不能建立场所设施。
    return observation["subject_kind"] != "food_closeup"


def facility_review_schema(config):
    return {"type": "object", "additionalProperties": False, "required": ["facility_evidence"],
            "properties": {"facility_evidence": observation_schema(config)["properties"]["facility_evidence"]}}


def facility_review_messages(image, config):
    image_url = str(image)
    if not image_url.startswith(("data:", "https://", "http://", "file://")):
        image_url = "file://" + image_url.replace("\\", "/")
    review = config["facility_refinement"]
    text = review["task_prompt"] + "\nJSON Schema: " + json.dumps(
        facility_review_schema(config), ensure_ascii=False, separators=(",", ":"))
    # 独立重看原图，不提供旧设施、参考标签、样本身份或商家 metadata。
    return [{"role": "system", "content": review["system_prompt"]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}},
                                         {"type": "text", "text": text}]}]


def apply_facility_review(primary, raw, config):
    proposed = parse_observation(raw, {"protocol": "product_visual_observation_v4"})
    _validate_instance(facility_review_schema(config), proposed, path="$")
    for cue in proposed["facility_evidence"]:
        if re.search(r"\b(?:impl(?:y|ies|ied)|infer(?:red)?|suggest(?:s|ed)?|likely|presumably|must have|metadata)\b|"
                     r"意味着|应当有|推断|暗示|商家属性", cue["fact"], re.I):
            raise ValueError("facility review requires directly observable evidence")
    refined = copy.deepcopy(primary)
    refined["facility_evidence"] = proposed["facility_evidence"]
    # 完整原契约继续检查重复、否定、食品冲突和十事实上限。
    return refined, map_observation(refined, config)


def generate_facility_refined_observation(backend, image, config):
    from src.inference.product_observation import generate_observation
    from src.inference.system_runtime import ModelGenerationError

    validate_facility_refinement(config)
    base = {key: value for key, value in config.items() if key != "facility_refinement"}
    primary = generate_observation(backend, image, base)
    primary["facility_refinement_applied"] = False
    primary["facility_primary_attempt_count"] = len(primary["attempts"])
    if not primary["passed"] or not should_review_facilities(primary["observation"], config):
        return primary
    attempts = list(primary["attempts"])
    messages = facility_review_messages(image, config)
    active = messages
    for _ in range(config["facility_refinement"]["max_attempts"]):
        started = time.perf_counter()
        try:
            generated = backend.generate_with_usage(
                active, response_format=None,
                max_new_tokens=config["facility_refinement"]["max_new_tokens"])
        except ModelGenerationError as exc:
            attempts.extend(exc.attempts or [ModelAttempt(
                attempt=len(attempts) + 1, raw_output="", error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000)])
            break
        error = None
        try:
            observation, product = apply_facility_review(primary["observation"], generated.content, config)
        except ValueError as exc:
            error = str(exc)
        attempts.append(ModelAttempt(
            attempt=len(attempts) + 1, raw_output=generated.content, error=error,
            input_tokens=generated.input_tokens, output_tokens=generated.output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000))
        if error is None:
            return {**primary, "observation": observation, "result": product,
                    "attempts": attempts, "facility_refinement_applied": True}
        active = [*messages, {"role": "assistant", "content": generated.content},
                  {"role": "user", "content": "Validation error: " + error +
                   ". Reinspect the original image and return the complete facility_evidence JSON; do not guess."}]
    # 已要求的设施复查失败时不能悄悄回退为主阶段成功。
    return {**primary, "passed": False, "result": None, "observation": None,
            "attempts": attempts, "facility_refinement_applied": True}


def _replay_base(record, base):
    from src.inference.product_category_refinement import replay_category_refined_observation
    from src.inference.product_style_refinement import replay_refined_observation

    if base.get("category_refinement") is not None:
        return replay_category_refined_observation(record, base)
    if base.get("style_refinement") is not None:
        return replay_refined_observation(record, base)
    attempts = record.get("attempts", [])
    if not attempts or attempts[-1].get("error") is not None:
        raise ValueError("facility review has no successful primary observation")
    observation = parse_observation(attempts[-1]["raw_output"], base)
    return map_observation(observation, base)


def _rebuild_base_observation(record, base):
    """Rebuild nested raw stages after their existing strict replayers accept the prefix."""
    from src.inference.product_category_refinement import apply_subject_review, should_review_subject
    from src.inference.product_style_refinement import apply_refinement, should_refine

    attempts = record["attempts"]
    boundary = next((index + 1 for index, attempt in enumerate(attempts[:base["max_attempts"]])
                     if attempt.get("error") is None), None)
    if boundary is None:
        raise ValueError("facility review has no successful primary observation")
    primary_config = {key: value for key, value in base.items()
                      if key not in {"category_refinement", "style_refinement"}}
    primary = parse_observation(attempts[boundary - 1]["raw_output"], primary_config)
    map_observation(primary, primary_config)
    if base.get("style_refinement") is not None and should_refine(primary, base):
        count = next((index + 1 for index, attempt in enumerate(
            attempts[boundary:boundary + base["style_refinement"]["max_attempts"]])
            if attempt.get("error") is None), None)
        if count is None:
            raise ValueError("facility review has no successful style stage")
        boundary += count
        primary, _ = apply_refinement(primary, attempts[boundary - 1]["raw_output"], base)
    if base.get("category_refinement") is not None and should_review_subject(primary, base):
        count = next((index + 1 for index, attempt in enumerate(
            attempts[boundary:boundary + base["category_refinement"]["max_attempts"]])
            if attempt.get("error") is None), None)
        if count is None:
            raise ValueError("facility review has no successful subject stage")
        boundary += count
        primary, _ = apply_subject_review(primary, attempts[boundary - 1]["raw_output"], base)
    if boundary != len(attempts):
        raise ValueError("facility review primary stage boundary mismatch")
    return primary


def replay_facility_refined_observation(record, config):
    """Replay the nested primary stages, then the independent facility stage."""
    validate_facility_refinement(config)
    attempts = record.get("attempts", [])
    base = {key: value for key, value in config.items() if key != "facility_refinement"}
    boundary = primary_product = None
    # 旧阶段各自拒绝额外 attempt，因此最短成功前缀就是不可伪造的阶段边界。
    for index in range(1, len(attempts) + 1):
        try:
            primary_product = _replay_base({"attempts": attempts[:index]}, base)
        except ValueError:
            continue
        boundary = index
        break
    if boundary is None:
        raise ValueError("facility review has no replayable primary pipeline")
    # 从相同成功前缀重建原始观察，而不是从公共标签反推 subject_kind。
    primary = _rebuild_base_observation({"attempts": attempts[:boundary]}, base)
    remaining = attempts[boundary:]
    if not should_review_facilities(primary, config):
        if remaining:
            raise ValueError("unexpected facility stage for food closeup")
        return map_observation(primary, config)
    if (not 1 <= len(remaining) <= config["facility_refinement"]["max_attempts"]
            or remaining[-1].get("error") is not None
            or any(attempt.get("error") is None for attempt in remaining[:-1])):
        raise ValueError("facility review is incomplete or has extra attempts")
    return apply_facility_review(primary, remaining[-1]["raw_output"], config)[1]


def facility_review_source_hashes(root, generation_config):
    paths = set(generation_config.get("observation_profile_configs", {}).values())
    if generation_config.get("observation_config"):
        paths.add(generation_config["observation_config"])
    if not any(json.loads((root / path).read_text(encoding="utf-8")).get("facility_refinement") for path in paths):
        return {}
    sources = (
        "src/inference/product_facility_refinement.py", "src/inference/product_category_refinement.py",
        "src/inference/product_style_refinement.py", "src/inference/product_observation.py",
        "src/inference/system_runtime.py", "src/inference/schemas.py",
        "scripts/review_week8_contracts.py", "src/data/product_labels.py",
    )
    return {path: hashlib.sha256((root / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in sources}


def validate_facility_review_identity(root, generation_config, identity):
    expected = facility_review_source_hashes(root, generation_config)
    if expected and identity.get("facility_review_source_lf_sha256") != expected:
        raise ValueError("facility review generation implementation changed or was not bound")

"""Reject invalid visual reference evidence instead of silently filtering labels."""
from src.inference.product_observation import map_observation
from src.inference.product_style_scope import venue_style_evidence


def map_teacher_observation(observation, config):
    target = map_observation(observation, config)
    if config.get("style_scope_policy") and venue_style_evidence(observation, config)[1]:
        # 参考不能用推理端过滤器悄悄减少正标签；要求教师重新看图或明确失败。
        raise ValueError("teacher style evidence uses nonvenue objects; reobserve venue decor or omit unsupported style")
    return target

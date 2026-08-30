"""Keep venue style separate from food, drinks and people's clothing."""
from src.data.product_labels import affirmative_term


def validate_style_scope(config):
    policy = config.get("style_scope_policy")
    if policy is None:
        return
    if (not isinstance(policy, dict) or policy.get("protocol") != "venue_style_evidence_scope_v1"
            or config.get("protocol") != "product_visual_observation_v3"):
        raise ValueError("unsupported venue style scope policy")
    for field in ("nonvenue_terms", "venue_context_terms"):
        terms = policy.get(field)
        if (not isinstance(terms, list) or not terms
                or any(not isinstance(term, str) or not term.strip() for term in terms)
                or len(set(terms)) != len(terms)):
            raise ValueError("style scope requires explicit unique literal terms")


def _present(text, term):
    return affirmative_term(text, term.casefold())


def venue_style_evidence(observation, config):
    cues = observation["style_evidence"]
    policy = config.get("style_scope_policy")
    if policy is None:
        return cues, []
    validate_style_scope(config)
    kept, excluded = [], []
    for cue in cues:
        nonvenue = [term for term in policy["nonvenue_terms"] if _present(cue["fact"], term)]
        venue = [term for term in policy["venue_context_terms"] if _present(cue["fact"], term)]
        # 只排除明确的对象范围错误；存在场所装修上下文或范围不清时保留原判断。
        if nonvenue and not venue:
            excluded.append({**cue, "reason": "nonvenue_object_fact", "matched_terms": nonvenue})
        else:
            kept.append(cue)
    return kept, excluded

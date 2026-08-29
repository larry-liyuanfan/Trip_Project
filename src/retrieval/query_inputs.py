"""Query-side signals only: reference labels must never enter ranking."""

import re

from src.data.product_labels import affirmative_term


QUERY_TERMS = {
    "business_category": {"hotel": ("hotel", "hotels", "酒店"), "restaurant": ("restaurant", "restaurants", "cafe", "cafes", "餐厅", "咖啡馆"), "attraction": ("museum", "museums", "park", "parks", "景点", "博物馆")},
    "price_range": {"budget": ("budget", "cheap", "便宜", "经济"), "mid_range": ("mid range", "适中"), "premium": ("premium", "高档"), "luxury": ("luxury", "奢华")},
}


def _term_pattern(term):
    return r"(?<!\w)" + re.escape(term) + r"(?!\w)" if term.isascii() else re.escape(term)


def _affirmative_query_term(text, term):
    if term.isascii():
        return affirmative_term(text, term)
    for clause in re.split(r"[，。；,;.!?]|但是|而是", text):
        for match in re.finditer(re.escape(term), clause):
            if not re.search(r"不要|不去|排除|不是|不想|无需|没有|不推荐", clause[:match.start()]):
                return True
    return False


def user_query_attributes(text, explicit=None):
    explicit_fields = {key for key, value in (explicit or {}).items()
                       if key in {"city", "business_category", "price_range"}
                       and value not in (None, "", "unknown")}
    attributes = {key: value for key, value in (explicit or {}).items()
                  if key in {"city", "business_category", "price_range"} and value not in (None, "", "unknown")}
    for field, mapping in QUERY_TERMS.items():
        values = [value for value, terms in mapping.items() if any(
            _affirmative_query_term(text, term) for term in terms)]
        if field not in attributes and len(values) == 1:
            attributes[field] = values[0]
        elif (field not in attributes and len(values) > 1
              and re.search(r"(?:\bor\b|或|或者)", text, re.I)):
            # 同一字段的显式析取可由既有 Milvus IN 过滤完整执行；没有连接词仍保持歧义。
            attributes[field] = values
    # “酒店或便宜餐厅”是跨字段析取，不能错误实现成“便宜的酒店或餐厅”的字段交集。
    # 显式 API filter 由调用者独立提供，可安全地作用于整个析取结果。
    if (isinstance(attributes.get("business_category"), list)
            and "price_range" in attributes and "price_range" not in explicit_fields):
        attributes.pop("business_category", None)
        attributes.pop("price_range", None)
    return attributes


def ranking_query_attributes(query):
    inputs = query.get("query_inputs", {})
    if inputs.get("source") not in {"user", "model_prediction"}:
        return {}
    return user_query_attributes(inputs.get("query_text", ""), inputs.get("attributes", {}))


def unapplied_query_text(text, attributes):
    """公开当前结构化检索无法应用的文字，不声称满足安静/氛围等未建模约束。"""
    if re.search(r"不要|不去|排除|不是|不想|without|\bnot\b", text, re.I):
        return text.strip()
    remainder = text
    for value in attributes.values():
        for item in value if isinstance(value, (list, tuple)) else (value,):
            remainder = re.sub(_term_pattern(str(item)), " ", remainder, flags=re.I)
    # 只消费真正应用的条件；多候选或与显式过滤冲突的文字不能被洗成 COMPLETED。
    for field, mapping in QUERY_TERMS.items():
        values = attributes.get(field)
        for value in values if isinstance(values, (list, tuple)) else (values,):
            for term in mapping.get(value, ()):
                remainder = re.sub(_term_pattern(term), " ", remainder, flags=re.I)
    if any(isinstance(value, (list, tuple)) for value in attributes.values()):
        remainder = re.sub(r"\b(?:or|either)\b|或者|或", " ", remainder, flags=re.I)
    remainder = re.sub(r"推荐|查找|搜索|帮我|请|一个|一些|的|\b(?:find|recommend|search|for|a|an|in|please)\b", " ", remainder, flags=re.I)
    return re.sub(r"[\s，。、；,;.!?：:]+", " ", remainder).strip()

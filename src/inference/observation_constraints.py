"""Decoder-side product subject contract; all normal post-validation still applies."""
import copy


PROTOCOL = "product_observation_subject_v1"
SHARED_SERIALIZATION_PROTOCOL = "product_observation_subject_v2"
PROTOCOLS = {PROTOCOL, SHARED_SERIALIZATION_PROTOCOL}


def observation_constraint_schemas(schema):
    required = ["subject_kind", "subject_fact", "style_evidence", "facility_evidence", "price_text"]
    if (not isinstance(schema, dict) or schema.get("type") != "object"
            or schema.get("required") != required or schema.get("additionalProperties") is not False
            or set(schema.get("properties", {})) != set(required)):
        raise ValueError("subject constraint requires the complete observation schema")
    properties = schema["properties"]
    kinds = properties["subject_kind"].get("enum", [])
    if ("food_closeup" not in kinds or len(kinds) < 2
            or properties["subject_kind"].get("type") != "string"
            or properties["subject_fact"].get("type") != "string"
            or any(properties[field].get("type") != "array" for field in required[2:])):
        raise ValueError("subject constraint requires expanded evidence and food/nonfood alternatives")
    other = copy.deepcopy(schema)
    other["properties"]["subject_kind"]["enum"] = [kind for kind in kinds if kind != "food_closeup"]
    return other, copy.deepcopy(properties["subject_fact"]), copy.deepcopy(properties["price_text"])


def build_observation_constraint_parser(schema, protocol=PROTOCOL):
    from lmformatenforcer import JsonSchemaParser, SequenceParser, StringParser, UnionParser

    other, fact, price = observation_constraint_schemas(schema)
    if protocol == SHARED_SERIALIZATION_PROTOCOL:
        # 两类主体共享键顺序和分隔符；不能让空格或先输出其他键排除 food 分支。
        def branch(kinds, food):
            parsers = [StringParser('{"subject_kind":'), JsonSchemaParser(kinds),
                       StringParser(',"subject_fact":'), JsonSchemaParser(fact)]
            for field in ("style_evidence", "facility_evidence", "price_text"):
                parsers.append(StringParser(',"' + field + '":'))
                parsers.append(StringParser('[]') if food and field != "price_text"
                               else JsonSchemaParser(copy.deepcopy(schema["properties"][field])))
            return SequenceParser([*parsers, StringParser('}')])
        food_kind = {**copy.deepcopy(schema["properties"]["subject_kind"]), "enum": ["food_closeup"]}
        return UnionParser([branch(food_kind, True), branch(other["properties"]["subject_kind"], False)])
    if protocol != PROTOCOL:
        raise ValueError("unsupported observation subject grammar")
    # 当前 LMFE 的 maxItems=0 仍允许首项；空数组改用字面语法，不改依赖或吞掉错误标签。
    # 模型仍可在 food/nonfood 两个分支中选择，不能根据先前错误强行确定主体。
    food = SequenceParser([
        StringParser('{"subject_kind":"food_closeup","subject_fact":'),
        JsonSchemaParser(fact),
        StringParser(',"style_evidence":[],"facility_evidence":[],"price_text":'),
        JsonSchemaParser(price),
        StringParser('}'),
    ])
    return UnionParser([food, JsonSchemaParser(other)])

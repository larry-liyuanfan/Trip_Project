"""Exercise the installed real decoder grammar on synthetic JSON contracts, without a GPU."""
import argparse
import copy
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.week8_visual_holdout import read_json, write_json_new
from src.inference.observation_constraints import build_observation_constraint_parser
from src.inference.product_observation import observation_schema, map_observation, observation_correction_response_format, FOOD_SUBJECT_CONFLICT
from src.training.week7_data import sha256_file


def accepts(parser, text):
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


def run(config_path):
    from lmformatenforcer import JsonSchemaParser

    config = read_json(config_path)
    schema = observation_schema(config)
    protocol = observation_correction_response_format(config, FOOD_SUBJECT_CONFLICT)["constraint_protocol"]
    basic = {"subject_kind": "food_closeup", "subject_fact": "Bowl of noodles", "style_evidence": [], "facility_evidence": [], "price_text": []}
    cases = []
    for kind in config["subject_categories"]:
        value = {**copy.deepcopy(basic), "subject_kind": kind}
        if kind != "food_closeup":
            value["style_evidence"] = [{"label": "modern", "fact": "Geometric wall"}]
            value["facility_evidence"] = [{"label": "seating", "fact": "Wooden chairs"}]
        cases.append((kind, value, True, True))
    for label, changes in (("visible_price", {"price_text": ["$12.50"]}),
                           ("escaped_fact", {"subject_fact": 'A "red" bowl'}),
                           ("maximum_fact", {"subject_fact": "x" * 80})):
        cases.append((label, {**copy.deepcopy(basic), **changes}, True, True))
    for field in ("style_evidence", "facility_evidence"):
        value = copy.deepcopy(basic)
        value[field] = [{"label": "modern" if field == "style_evidence" else "seating", "fact": "Visible object"}]
        cases.append(("food_conflict_" + field, value, False, False))
    cases.extend([
        ("missing_price", {key: value for key, value in basic.items() if key != "price_text"}, False, False),
        ("empty_fact", {**basic, "subject_fact": ""}, False, False),
        ("long_fact", {**basic, "subject_fact": "x" * 81}, False, False),
        ("invalid_kind", {**basic, "subject_kind": "invented"}, False, False),
        ("negation_still_post_validated", {**basic, "subject_kind": "dining_space", "facility_evidence": [{"label": "parking", "fact": "No parking"}]}, True, False),
    ])
    results = []
    for name, value, decoder_expected, post_expected in cases:
        decoder_actual = accepts(build_observation_constraint_parser(schema, protocol), json.dumps(value, separators=(",", ":")))
        try:
            map_observation(value, config)
            post_actual = True
        except ValueError:
            post_actual = False
        results.append({"case": name, "decoder_accepted": decoder_actual, "post_validation_accepted": post_actual,
                        "passed": decoder_actual == decoder_expected and post_actual == post_expected})
    formatting = []
    for style in ("compact", "spaced", "leading_space", "reverse_keys"):
        accepted = {}
        for kind in config["subject_categories"]:
            value = {**copy.deepcopy(basic), "subject_kind": kind}
            if style == "reverse_keys":
                value = dict(reversed(list(value.items())))
            raw = json.dumps(value, separators=(",", ":")) if style == "compact" else json.dumps(value)
            if style == "leading_space":
                raw = " " + raw
            accepted[kind] = accepts(build_observation_constraint_parser(schema, protocol), raw)
        formatting.append({"serialization": style, "accepted_by_subject": accepted,
                           "passed": len(set(accepted.values())) == 1})
    return {"status": "PASS" if all(item["passed"] for item in [*results, *formatting]) else "FAIL", "cases": results,
            "formatting_symmetry": formatting, "constraint_protocol": protocol,
            "correction_validation_error": FOOD_SUBJECT_CONFLICT,
            "decoder_version": version("lm-format-enforcer"), "config_sha256": sha256_file(config_path),
            "implementation_sha256": sha256_file(ROOT / "src/inference/observation_constraints.py"),
            "stock_maxitems_zero_accepts_nonempty": accepts(JsonSchemaParser({"type": "array", "maxItems": 0, "items": {"type": "string"}}), '["x"]'),
            "model_requests": 0, "human_annotation_count": 0, "test_rows_read": False,
            "scope": "synthetic_decoder_contracts_not_visual_quality"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve())
    write_json_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)

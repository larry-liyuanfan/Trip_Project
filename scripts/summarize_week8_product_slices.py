"""Count product error slices from immutable paired raw outputs and image silver."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.week8_visual_silver import _labels, _unknown, FIELDS, replay_record
from src.training.week7_data import sha256_file, iter_jsonl
from src.data.week8_visual_holdout import read_json, write_json_new


def summarize(references, predictions):
    counts = {}
    categories = Counter()
    details = []
    for reference in references:
        if reference.get("label_source") != "model_generated_silver" or reference.get("error"):
            raise ValueError("requires complete image-only silver")
        target = reference["target"]
        categories[target["business_category"]] += 1
        subject = reference.get("observation", {}).get("subject_kind")
        memberships = {"all_samples": True, "business_category": target["business_category"] != "unknown",
            "multiple_or_ambiguous_subjects": subject in {"mixed_subjects", "unidentified_space"},
            "food_closeup": subject == "food_closeup", "style_multilabel": len(target["style_tags"]) > 1,
            "style_missing_or_extra": True, "facility_missing_or_extra": True,
            "price_without_evidence": target["price_range"] == "unknown",
            "should_use_unknown": any(_unknown(target, field) for field in FIELDS), "schema_valid_semantic_error": True}
        per_role = {}
        for role, values in predictions.items():
            prediction = values[reference["sample_id"]]
            ok = prediction is not None
            prediction = prediction or {}
            field_errors = {"business_category": not ok or prediction.get("business_category") != target["business_category"],
                            "price": not ok or prediction.get("price_range") != target["price_range"]}
            for name, field in (("style", "style_tags"), ("facility", "visible_facilities")):
                truth, proposed = _labels(target[field]), _labels(prediction.get(field, []))
                field_errors[name + "_missing"] = sorted(truth - proposed)
                field_errors[name + "_extra"] = sorted(proposed - truth)
            any_error = any(bool(value) for value in field_errors.values()) or not ok
            unknown_guesses = [field for field in FIELDS if _unknown(target, field) and not _unknown(prediction, field)]
            failures = {"all_samples": any_error, "business_category": field_errors["business_category"],
                "multiple_or_ambiguous_subjects": any_error, "food_closeup": any_error,
                "style_multilabel": bool(field_errors["style_missing"] or field_errors["style_extra"]),
                "style_missing_or_extra": bool(field_errors["style_missing"] or field_errors["style_extra"]),
                "facility_missing_or_extra": bool(field_errors["facility_missing"] or field_errors["facility_extra"]),
                "price_without_evidence": field_errors["price"], "should_use_unknown": bool(unknown_guesses) or not ok,
                "schema_valid_semantic_error": ok and any_error}
            for name, included in memberships.items():
                counter = counts.setdefault(name, {}).setdefault(role, {"support": 0, "errors": 0})
                counter["support"] += int(included)
                counter["errors"] += int(included and failures[name])
            per_role[role] = {"field_errors": field_errors, "unknown_guesses": unknown_guesses, "request_failed": not ok}
        details.append({"sample_id": reference["sample_id"], "subject_kind": subject, "roles": per_role})
    return {"label_source": "model_generated_silver", "human_accuracy_claim": False,
            "category_support": dict(categories), "error_slices": counts, "details": details}


def run(args):
    references = list(iter_jsonl(args.references))
    expected = {row["sample_id"] for row in references}
    predictions = {}
    for role, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        records = list(iter_jsonl(path))
        if len(records) != len(references) or {row["sample_id"] for row in records} != expected:
            raise ValueError("slice inputs must retain all paired samples")
        observation = read_json(args.observation) if role == "candidate" else None
        predictions[role] = {row["sample_id"]: replay_record(ROOT, row, observation) for row in records}
    result = summarize(references, predictions)
    result["source_sha256"] = {key: sha256_file(getattr(args, key)) for key in ("references", "baseline", "candidate", "observation")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"details", "source_sha256"}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("references", "baseline", "candidate", "observation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())

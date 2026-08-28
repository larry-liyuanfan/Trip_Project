"""Paired, image-teacher silver agreement; explicitly not human visual accuracy."""
import json
import math
import statistics

from src.evaluation.product_semantics import product_consistency_errors
from src.evaluation.schema_validation import validate_output
from src.inference.product_observation import map_observation, parse_observation
from src.inference.transport_utils import strip_json_fence


FIELDS = ("business_category", "style_tags", "visible_facilities", "price_range")
ALIASES = {"parking lot": "parking", "outdoor seating": "outdoor_seating",
           "swimming pool": "pool", "chairs": "seating", "chair": "seating", "stools": "seating",
           "dining table": "dining_tables", "dining tables": "dining_tables", "tables": "dining_tables",
           "front desk": "front_desk", "modern style": "modern", "现代": "modern", "现代风": "modern"}


def _labels(value):
    return {ALIASES.get(str(item).strip().casefold(), str(item).strip().casefold()) for item in value}


def _unknown(target, field):
    return target.get(field) in (None, "unknown", [])


def replay_record(root, record, observation=None):
    """Recompute accepted output from raw bytes, not self-reported metric flags."""
    if not record.get("passed"):
        return None
    attempts = record.get("attempts", [])
    if not attempts or attempts[-1].get("error") is not None:
        raise ValueError("passing record has no successful raw attempt")
    if observation is not None and observation.get("style_refinement") is not None:
        from src.inference.product_style_refinement import replay_refined_observation
        value = replay_refined_observation(record, observation)
    else:
        raw = attempts[-1]["raw_output"]
        value = parse_observation(raw, observation) if observation is not None else json.loads(strip_json_fence(raw))
        if observation is not None:
            value = map_observation(value, observation)
    validate_output(root, "image_product_search", value, "v1")
    if value != record.get("result"):
        raise ValueError("reported product differs from raw generation/mapping")
    return value


def _validate_reference_audit(audit, phase="development"):
    if phase not in {"development", "final"}:
        raise ValueError("unknown evaluation phase")
    if (not isinstance(audit, dict)
            or audit.get("protocol") != "independent_image_model_observation_silver_v3"
            or audit.get("metadata_supplied") is not False
            or audit.get("candidate_outputs_supplied") is not False
            or audit.get("model_independent") is not True
            or audit.get("test_rows_read") is not (phase == "final")
            or not isinstance(audit.get("reference_raw_sha256"), str)
            or len(audit["reference_raw_sha256"]) != 64):
        raise ValueError("missing or invalid independent visual silver audit")
    if phase == "final" and (not isinstance(audit.get("candidate_lock_sha256"), str) or len(audit["candidate_lock_sha256"]) != 64):
        raise ValueError("final scoring requires a pre-existing candidate lock")


def score_paired(root, references, records, observation=None, *, reference_audit=None, phase="development"):
    _validate_reference_audit(reference_audit, phase)
    reference_ids = [row["sample_id"] for row in references]
    if len(reference_ids) != len(set(reference_ids)) or len(records) != len(references):
        raise ValueError("all fixed reference samples must be retained exactly once")
    by_id = {row["sample_id"]: row for row in records}
    if len(by_id) != len(records) or set(by_id) != set(reference_ids):
        raise ValueError("prediction identity coverage differs from references")
    supports = {"samples": len(references), "business_category": 0, "style_positive_samples": 0,
                "style_positive_labels": 0, "facility_positive_samples": 0, "facility_positive_labels": 0,
                "price_range": 0, "unknown_decisions": len(references) * len(FIELDS)}
    counts = {field: {"tp": 0, "fp": 0, "fn": 0} for field in ("style", "facility")}
    category_hits = category_all_hits = price_hits = unknown_hits = consistent = json_ok = passed = 0
    latencies, input_tokens, output_tokens, completeness, errors = [], [], [], [], []
    for reference in references:
        if reference.get("error") or reference.get("label_source") != "model_generated_silver" or not reference.get("target"):
            raise ValueError("invalid or missing image-only silver reference")
        target = reference["target"]
        validate_output(root, "image_product_search", target, "v1")
        if product_consistency_errors(target):
            raise ValueError("inconsistent silver reference")
        record = by_id[reference["sample_id"]]
        prediction = replay_record(root, record, observation)
        ok = prediction is not None
        passed += int(ok)
        prediction = prediction or {}
        raw_attempts = record.get("attempts", [])
        try:
            json.loads(strip_json_fence(raw_attempts[-1]["raw_output"]))
            json_ok += 1
        except (ValueError, KeyError, IndexError):
            pass
        consistent += int(ok and not product_consistency_errors(prediction))
        known_category = not _unknown(target, "business_category")
        known_price = not _unknown(target, "price_range")
        supports["business_category"] += int(known_category)
        supports["price_range"] += int(known_price)
        category_hit = int(ok and prediction.get("business_category") == target["business_category"])
        category_hits += category_hit * known_category
        category_all_hits += category_hit
        price_hit = int(ok and prediction.get("price_range") == target["price_range"])
        price_hits += price_hit * known_price
        sample_errors = []
        if not category_hit:
            sample_errors.append("business_category")
        numerator, denominator = category_hit * known_category + price_hit * known_price, int(known_category) + int(known_price)
        for metric, field in (("style", "style_tags"), ("facility", "visible_facilities")):
            truth, predicted = _labels(target[field]), _labels(prediction.get(field, []))
            supports[metric + "_positive_samples"] += int(bool(truth))
            supports[metric + "_positive_labels"] += len(truth)
            tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
            counts[metric]["tp"] += tp
            counts[metric]["fp"] += fp
            counts[metric]["fn"] += fn
            numerator += tp
            denominator += len(truth)
            if fp or fn:
                sample_errors.append(metric)
        # 空参考也保留在 precision 分母中，猜测不可因没有正支持而免费。
        unknown_hits += sum(int(ok and _unknown(target, field) == _unknown(prediction, field)) for field in FIELDS)
        if denominator:
            completeness.append(numerator / denominator)
        latency = record["elapsed_ms"]
        if not isinstance(latency, (float, int)) or not math.isfinite(latency) or latency < 0:
            raise ValueError("invalid latency")
        latencies.append(latency)
        input_tokens.append(sum(attempt.get("input_tokens") or 0 for attempt in raw_attempts))
        output_tokens.append(sum(attempt.get("output_tokens") or 0 for attempt in raw_attempts))
        if sample_errors or not ok:
            errors.append({"sample_id": reference["sample_id"], "fields": sample_errors, "request_failed": not ok})
    total = len(references)
    if not total:
        raise ValueError("empty evaluation")
    metrics = {"business_category_accuracy": category_hits / supports["business_category"] if supports["business_category"] else None,
               "business_category_including_unknown_accuracy": category_all_hits / total,
               "price_range_accuracy": price_hits / supports["price_range"] if supports["price_range"] else None,
               "unknown_accuracy": unknown_hits / supports["unknown_decisions"],
               "label_completeness": statistics.fmean(completeness) if completeness else None,
               "json_compliance": json_ok / total, "schema_pass_rate": passed / total,
               "request_failure_rate": 1 - passed / total, "internal_consistency_rate": consistent / total}
    for metric, values in counts.items():
        tp, fp, fn = (values[key] for key in ("tp", "fp", "fn"))
        metrics[metric + "_precision"] = tp / (tp + fp) if tp + fp else 0.0
        metrics[metric + "_recall"] = tp / (tp + fn) if tp + fn else None
        metrics[metric + "_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    primary = [metrics[key] for key in ("business_category_accuracy", "style_f1", "facility_f1", "price_range_accuracy") if metrics[key] is not None]
    metrics["composite"] = statistics.fmean(primary) if primary else None
    return {"protocol": "independent_visual_silver_agreement_v1", "phase": phase, "human_visual_accuracy_claim": False,
            "reference_audit": reference_audit,
            "metrics": metrics, "supports": supports, "multilabel_counts": counts, "errors": errors,
            "latency_ms": {"mean": statistics.fmean(latencies), "p50": statistics.median(latencies),
                           "p95": sorted(latencies)[math.ceil(0.95 * total) - 1]},
            "tokens": {"input_total": sum(input_tokens), "output_total": sum(output_tokens),
                       "input_mean": statistics.fmean(input_tokens), "output_mean": statistics.fmean(output_tokens)}}


def select_development_candidate(summaries):
    baseline = summaries["formal_adapter"]
    reasons, eligible = {}, []
    for role, summary in summaries.items():
        metrics = summary["metrics"]
        problems = []
        _validate_reference_audit(summary.get("reference_audit"))
        if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1) for value in metrics.values()):
            raise ValueError("invalid bounded comparison metric")
        if metrics.get("composite") is None:
            raise ValueError("composite requires supported fields")
        if not isinstance(summary["latency_ms"]["mean"], (int, float)) or not math.isfinite(summary["latency_ms"]["mean"]) or summary["latency_ms"]["mean"] < 0:
            raise ValueError("invalid candidate latency")
        if summary.get("protocol") != "independent_visual_silver_agreement_v1" or summary.get("human_visual_accuracy_claim") is not False:
            problems.append("invalid_reference_claim")
        if summary["supports"] != baseline["supports"]:
            problems.append("support_changed")
        if summary["reference_audit"] != baseline["reference_audit"]:
            problems.append("reference_identity_changed")
        if metrics["json_compliance"] != 1 or metrics["schema_pass_rate"] != 1 or metrics["request_failure_rate"] != 0:
            problems.append("format_or_request_failure")
        if metrics["internal_consistency_rate"] != 1:
            problems.append("inconsistent_product_fields")
        for field in ("business_category_accuracy", "style_precision", "style_recall", "style_f1", "facility_precision", "facility_recall", "facility_f1", "price_range_accuracy", "unknown_accuracy", "label_completeness"):
            before, after = baseline["metrics"][field], metrics[field]
            if before is not None and (after is None or after < before):
                problems.append(field + "_regressed")
        if role != "formal_adapter" and metrics["composite"] <= baseline["metrics"]["composite"]:
            problems.append("no_composite_improvement")
        reasons[role] = problems
        if role != "formal_adapter" and not problems:
            eligible.append(role)
    selected = min(eligible, key=lambda role: (-summaries[role]["metrics"]["composite"], summaries[role]["latency_ms"]["mean"])) if eligible else None
    return {"status": "DEVELOPMENT_CANDIDATE" if selected else "NO_ELIGIBLE_CANDIDATE",
            "selected_role": selected, "failures": reasons, "promotion_allowed": False,
            "interpretation": "Image-teacher silver agreement only; independent final and system validation remain required."}


def validate_locked_final(baseline, candidate):
    """验收锁定方案，不使用 test 排序、选 Prompt 或调整阈值。"""
    for summary in (baseline, candidate):
        _validate_reference_audit(summary["reference_audit"], "final")
        if summary.get("phase") != "final" or summary.get("human_visual_accuracy_claim") is not False:
            raise ValueError("final silver phase/claim mismatch")
        if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1)
               for value in summary["metrics"].values()):
            raise ValueError("invalid final metric")
    failures = []
    if baseline["supports"] != candidate["supports"] or baseline["reference_audit"] != candidate["reference_audit"]:
        failures.append("paired_support_or_reference_changed")
    for name in ("json_compliance", "schema_pass_rate", "internal_consistency_rate"):
        if candidate["metrics"][name] != 1:
            failures.append(name + "_not_complete")
    if candidate["metrics"]["request_failure_rate"] != 0:
        failures.append("request_failure")
    for name in ("business_category_accuracy", "business_category_including_unknown_accuracy", "style_precision", "style_recall", "style_f1",
                 "facility_precision", "facility_recall", "facility_f1", "price_range_accuracy", "unknown_accuracy", "label_completeness"):
        before, after = baseline["metrics"][name], candidate["metrics"][name]
        if before is not None and (after is None or after < before):
            failures.append(name + "_regressed")
    if candidate["metrics"]["composite"] is None or candidate["metrics"]["composite"] <= baseline["metrics"]["composite"]:
        failures.append("no_composite_improvement")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures,
            "candidate_reselection_allowed": False, "human_visual_accuracy_claim": False}

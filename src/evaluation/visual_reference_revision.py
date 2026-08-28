"""Minimal, traceable development silver correction; never a human acceptance."""
import base64
import copy
import hashlib
import json
import mimetypes
import time

import requests

from src.inference.product_observation import map_observation
from src.inference.product_style_refinement import apply_refinement, refinement_messages
from src.inference.product_style_scope import venue_style_evidence
from src.training.week7_data import sha256_file


IDENTITY_FIELDS = ("sample_id", "source_id", "image_sha256", "group_id", "constraint_template_id")
PROTOCOL = "development_visual_style_reference_revision_v1"


def record_sha256(record):
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_config(config):
    if (config.get("protocol") != PROTOCOL or config.get("final_test_access") is not False
            or config.get("human_annotation_count") != 0 or config.get("sample_weight") != 0.5
            or config.get("label_source") != "model_generated_silver"
            or config.get("development_indices") != "all"
            or config.get("repair_rule") != "reobserve_style_only_for_clear_nonvenue_evidence"
            or config.get("inheritance_rule") != "unchanged_non_style_fields_and_unflagged_rows"
            or type(config.get("max_attempts")) is not int or config["max_attempts"] not in (1, 2, 3)
            or type(config.get("concurrency")) is not int or config["concurrency"] not in (1, 2)
            or type(config.get("max_tokens")) is not int or not 1 <= config["max_tokens"] <= 1200
            or type(config.get("timeout_seconds")) is not int or not 1 <= config["timeout_seconds"] <= 120
            or any(config.get(key) is not False for key in (
                "prior_targets_supplied", "candidate_outputs_supplied", "metadata_supplied"))):
        raise ValueError("only bounded independent development style reference repair is allowed")


def validate_sources(rows, references, original_observation, repair_observation):
    if repair_observation.get("style_refinement", {}).get("mode") != "replace":
        raise ValueError("reference reobservation must not expose prior labels")
    if repair_observation.get("style_scope_policy") is None:
        raise ValueError("reference repair requires an explicit evidence scope audit")
    row_ids = [row["sample_id"] for row in rows]
    if (not rows or len(set(row_ids)) != len(rows) or len(references) != len(rows)
            or [row["sample_id"] for row in references] != row_ids):
        raise ValueError("all fixed development rows must be retained in manifest order")
    audit = []
    for row, reference in zip(rows, references):
        if (row.get("split") != "development" or reference.get("error") is not None
                or reference.get("label_source") != "model_generated_silver"
                or reference.get("sample_weight") != 0.5
                or any(row.get(key) != reference.get(key) for key in IDENTITY_FIELDS)
                or any(not row.get(key) for key in IDENTITY_FIELDS[:-1])):
            raise ValueError("reference five-dimensional development identity mismatch")
        attempts = reference.get("attempts", [])
        if (not attempts or attempts[-1].get("error") is not None
                or json.loads(attempts[-1]["raw_content"]) != reference["observation"]
                or map_observation(reference["observation"], original_observation) != reference["target"]):
            raise ValueError("original reference raw replay differs from its target")
        excluded = venue_style_evidence(reference["observation"], repair_observation)[1]
        audit.append({"sample_id": row["sample_id"], "source_record_sha256": record_sha256(reference),
                      "reobserve_style": bool(excluded), "scope_errors": excluded})
    return audit


def style_payload(row, config, observation, root):
    image = (root / row["image_path"]).resolve()
    image.relative_to(root.resolve())
    if sha256_file(image) != row["image_sha256"]:
        raise ValueError("repair image identity mismatch")
    mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
    data = base64.b64encode(image.read_bytes()).decode("ascii")
    # replace 模式不接收旧标签、候选或商家属性；只发送图像和统一字段协议。
    messages = refinement_messages(f"data:{mime};base64,{data}", {}, observation)
    return {"model": config["model"], "temperature": 0, "max_tokens": config["max_tokens"],
            "enable_thinking": False, "response_format": {"type": "json_object"}, "messages": messages}


def repair_record(row, source, audit, config, observation, root, base_url, key, post=None):
    record = copy.deepcopy(source)
    record["reference_revision"] = {"protocol": PROTOCOL, "source_record_sha256": record_sha256(source),
                                    "style_reobserved": audit["reobserve_style"]}
    if not audit["reobserve_style"]:
        return record
    record.update(source_attempts=copy.deepcopy(source["attempts"]), attempts=[], target=None,
                  observation=None, error="style_reobservation_incomplete")
    payload = style_payload(row, config, observation, root)
    original_messages = copy.deepcopy(payload["messages"])
    post = post or requests.post
    for number in range(1, config["max_attempts"] + 1):
        started = time.perf_counter()
        attempt = {"attempt": number}
        try:
            response = post(base_url + "/chat/completions", headers={"Authorization": "Bearer " + key},
                            json=payload, timeout=config["timeout_seconds"])
            if response.status_code != 200:
                raise ValueError(f"model_http_status_{response.status_code}")
            data = response.json()
            attempt.update(response_model=data.get("model"), usage=data.get("usage"),
                           raw_content=data["choices"][0]["message"]["content"])
            if attempt["response_model"] != config["model"]:
                raise ValueError("response model identity mismatch")
            merged, target = apply_refinement(source["observation"], attempt["raw_content"], observation)
            record.update(observation=merged, target=target)
            attempt["error"] = None
        except requests.RequestException:
            attempt["error"] = "model_transport_error"
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            attempt["error"] = str(exc)
        attempt["latency_ms"] = (time.perf_counter() - started) * 1000
        record["attempts"].append(attempt)
        record.update({name: value for name, value in attempt.items() if name != "attempt"})
        if attempt["error"] is None:
            break
        # 纠错只见本次教师原始答复，不泄漏旧参考或候选。
        payload["messages"] = [*original_messages]
        if attempt.get("raw_content"):
            payload["messages"].append({"role": "assistant", "content": attempt["raw_content"]})
        payload["messages"].append({"role": "user", "content": "Validation error: " + attempt["error"] +
            ". Reobserve venue decor in the image; return complete style_evidence JSON with short visible facts."})
    record["latency_ms"] = sum(item["latency_ms"] for item in record["attempts"])
    return record


def replay_revision(source, revised, audit, config, observation):
    expected = {"protocol": PROTOCOL, "source_record_sha256": record_sha256(source),
                "style_reobserved": audit["reobserve_style"]}
    if revised.get("reference_revision") != expected:
        raise ValueError("reference revision lineage mismatch")
    if not audit["reobserve_style"]:
        if revised != {**source, "reference_revision": expected}:
            raise ValueError("unflagged reference must remain exactly unchanged")
        return
    attempts = revised.get("attempts", [])
    if (not 1 <= len(attempts) <= config["max_attempts"] or revised.get("error") is not None
            or attempts[-1].get("error") is not None
            or any(item.get("error") is None for item in attempts[:-1])
            or attempts[-1].get("response_model") != config["model"]
            or revised.get("source_attempts") != source["attempts"]):
        raise ValueError("incomplete or invalid silver correction attempts")
    merged, target = apply_refinement(source["observation"], attempts[-1]["raw_content"], observation)
    if revised.get("observation") != merged or revised.get("target") != target:
        raise ValueError("revised silver target differs from raw style-only replay")
    for key in (*IDENTITY_FIELDS, "label_source", "sample_weight", "visual_accuracy_claim_supported"):
        if revised.get(key) != source.get(key):
            raise ValueError("reference repair changed identity or label authority")


def supports(references):
    return {"samples": len(references), "business_category": sum(r["target"]["business_category"] != "unknown" for r in references),
            "style_positive_samples": sum(bool(r["target"]["style_tags"]) for r in references),
            "style_positive_labels": sum(len(r["target"]["style_tags"]) for r in references),
            "facility_positive_samples": sum(bool(r["target"]["visible_facilities"]) for r in references),
            "facility_positive_labels": sum(len(r["target"]["visible_facilities"]) for r in references),
            "price_range": sum(r["target"]["price_range"] != "unknown" for r in references)}

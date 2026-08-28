"""Bounded image-silver self-correction; never accepts an invalid target."""
import json
import time

import requests

from src.evaluation.schema_validation import validate_output
from src.evaluation.product_semantics import product_consistency_errors
from src.evaluation.visual_reference_validation import map_teacher_observation
from src.inference.transport_utils import strip_json_fence


def collect_with_history(row, config, observation, root, base_url, key):
    from scripts.collect_week8_visual_silver import teacher_payload
    if config.get("retry_protocol") != "bounded_history_correction_v1" or config.get("max_attempts") != 4:
        raise ValueError("teacher retry protocol must be explicitly versioned and bounded")
    record = {name: row.get(name) for name in ("sample_id", "source_id", "group_id", "image_sha256", "constraint_template_id")}
    record.update(label_source="model_generated_silver", sample_weight=0.5,
                  visual_accuracy_claim_supported=False, target=None, attempts=[], retry_protocol=config["retry_protocol"])
    payload = teacher_payload(config, root / row["image_path"], observation)
    for number in range(1, config["max_attempts"] + 1):
        started = time.perf_counter()
        attempt = {"attempt": number}
        try:
            response = requests.post(base_url + "/chat/completions", headers={"Authorization": "Bearer " + key},
                                     json=payload, timeout=config["timeout_seconds"])
            if response.status_code != 200:
                raise ValueError(f"model_http_status_{response.status_code}")
            data = response.json()
            attempt.update(response_model=data.get("model"), usage=data.get("usage"),
                           raw_content=data["choices"][0]["message"]["content"])
            if data.get("model") != config["model"]:
                raise ValueError("response model identity mismatch")
            attempt["observation"] = json.loads(strip_json_fence(attempt["raw_content"]))
            target = map_teacher_observation(attempt["observation"], observation)
            validate_output(root, "image_product_search", target, "v1")
            if product_consistency_errors(target):
                raise ValueError("silver internally inconsistent")
            record["target"] = target
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
        # 仅回传教师自己的失败输出及原有验证规则；从不注入候选答案或商家标签。
        if attempt.get("raw_content"):
            payload["messages"].append({"role": "assistant", "content": attempt["raw_content"]})
        payload["messages"].append({"role": "user", "content":
            "The previous JSON failed the unchanged observation rules: " + attempt["error"]
            + ". Reobserve the original image and correct the inconsistency. Return complete JSON only; no explanation. "
              "Keep short facts and do not guess. No reference or candidate labels are provided."})
    record["latency_ms"] = sum(item["latency_ms"] for item in record["attempts"])
    return record

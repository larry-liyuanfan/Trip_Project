"""Collect blind image-only model silver; never human gold or a final-test runner."""
import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference.client import _read_api_key
from src.evaluation.schema_validation import validate_output
from src.evaluation.product_semantics import product_consistency_errors
from src.inference.transport_utils import strip_json_fence
from src.training.week7_data import sha256_file
from scripts.review_week8_contracts import write_new
from src.inference.product_observation import observation_messages
from src.evaluation.visual_reference_validation import map_teacher_observation


def teacher_payload(config, image, observation=None):
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
    payload = {"model": config["model"], "temperature": 0, "max_tokens": config["max_tokens"],
            "enable_thinking": False, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": config["system_prompt"]},
                         {"role": "user", "content": [
                             {"type": "text", "text": config["task_prompt"]},
                             {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}]}]}
    if observation:
        payload["messages"] = observation_messages(f"data:{mime};base64,{encoded}", observation)
    return payload


def load_inputs(config, root):
    if (config["final_test_access"] is not False or config["human_annotation_count"] != 0
            or config["label_source"] != "model_generated_silver" or config["sample_weight"] != 0.5):
        raise ValueError("only development model silver allowed")
    manifest = (root / config["manifest"]).resolve()
    manifest.relative_to(root.resolve())
    if "development" not in manifest.parts or "test" in manifest.parts:
        raise ValueError("only an explicit development manifest is allowed")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row["split"] != "development" for row in rows):
        raise ValueError("non-development row")
    indices = config["development_indices"]
    if indices != "all":
        if not isinstance(indices, list) or len(indices) != len(set(indices)):
            raise ValueError("indices must be unique")
        rows = [rows[index] for index in indices]
    for row in rows:
        image = (root / row["image_path"]).resolve()
        image.relative_to(root.resolve())
        if sha256_file(image) != row["image_sha256"]:
            raise ValueError("image identity mismatch")
    return rows, manifest


def collect_row(row, config, observation, root, base_url, key):
    record = {name: row.get(name) for name in ("sample_id", "source_id", "group_id", "image_sha256", "constraint_template_id")}
    record.update(label_source="model_generated_silver", sample_weight=0.5,
                  visual_accuracy_claim_supported=False, target=None, attempts=[])
    payload = teacher_payload(config, root / row["image_path"], observation)
    original_messages = payload["messages"]
    for number in range(1, config.get("max_attempts", 1) + 1):
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
            target = json.loads(strip_json_fence(attempt["raw_content"]))
            if observation:
                attempt["observation"] = target
                target = map_teacher_observation(target, observation)
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
        payload["messages"] = [*original_messages, {"role": "user", "content":
            "Validation failed: " + attempt["error"] + ". Return a complete corrected JSON; keep each fact short. Reobserve the original image, do not guess or explain."}]
    record["latency_ms"] = sum(item["latency_ms"] for item in record["attempts"])
    return record


def run(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    rows, manifest = load_inputs(config, ROOT)
    if config.get("max_attempts", 1) not in (1, 2) or config.get("concurrency", 1) not in (1, 2):
        raise ValueError("teacher requests must remain bounded")
    observation = json.loads((ROOT / config["observation_config"]).read_text(encoding="utf-8")) if config.get("observation_config") else None
    key = _read_api_key()
    if not key:
        raise ValueError("model API key is not configured")
    base_url = os.environ.get("MODEL_API_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").rstrip("/")
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    identity = {"protocol": config["protocol"], "config_sha256": sha256_file(path),
                "manifest_sha256": sha256_file(manifest), "runner_sha256": sha256_file(Path(__file__)),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "selected_sample_ids": [row["sample_id"] for row in rows],
                "test_rows_read": False, "human_annotation_count": 0,
                "candidate_outputs_supplied": False, "metadata_supplied": False,
                "observation_config_sha256": sha256_file(ROOT / config["observation_config"]) if observation else None,
                "interpretation": "Independent image-only model silver, not human visual accuracy."}
    write_new(output / "identity.json", identity)
    failures = 0
    with ThreadPoolExecutor(max_workers=config.get("concurrency", 1)) as executor, (output / "raw_outputs.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for record in executor.map(lambda row: collect_row(row, config, observation, ROOT, base_url, key), rows):
            failures += int(bool(record.get("error")))
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"sample_id": record["sample_id"], "error": record.get("error")}), flush=True)
    summary = {"status": "COMPLETED", "rows": len(rows), "failed": failures,
               "raw_sha256": sha256_file(output / "raw_outputs.jsonl"),
               "model": config["model"], "label_source": "model_generated_silver",
               "human_annotation_count": 0, "test_rows_read": False,
               "release_selection": "NOT_PERFORMED"}
    write_new(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week8/visual_teacher_v1.json")
    run(parser.parse_args().config.resolve())

"""Real production service probe, business checks and bounded-cache performance."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_week8_contracts import write_new
from scripts.run_system_model_smoke import run_model_smoke
from src.training.week7_data import sha256_file
from src.inference.schemas import TaskRequest, DialogueRequest
from src.inference.system_runtime import ReleaseSettings, ScenarioService, TransformersPeftBackend, ModelGenerationError
from src.inference.product_observation import canonical_config_sha256
from src.evaluation.week8_visual_silver import replay_record
from src.inference.product_observation import parse_observation
from src.inference.product_style_scope import venue_style_evidence


def validate_product_scope_probe(response, observation, require_abstention=False, root=ROOT,
                                 expected_category=None, require_subject_review=False):
    if not isinstance(observation, dict):
        raise ValueError("product scope probe requires a visual observation configuration")
    product = replay_record(root, response, observation)
    if product is None:
        raise ValueError("product scope probe request failed")
    excluded = []
    reviewed_subject = False
    for attempt in response["attempts"]:
        if attempt.get("error") is not None:
            continue
        raw = parse_observation(attempt["raw_output"], {"protocol": "product_visual_observation_v4"})
        # 先由完整raw重放确认阶段合法，再按各阶段的专用字段辨认；末条可能是主体复查。
        if isinstance(raw, dict) and set(raw) == {"style_evidence"} and observation.get("style_refinement", {}).get("unsupported_scope_action") == "abstain":
            excluded = venue_style_evidence(raw, observation)[1]
        if isinstance(raw, dict) and set(raw) == {"subject_kind", "subject_fact"}:
            reviewed_subject = True
    if require_abstention and not excluded:
        raise ValueError("fixed product scope probe did not exercise the abstention branch")
    if any(item["label"] in product["style_tags"] for item in excluded):
        raise ValueError("nonvenue style hypothesis leaked into the public product result")
    if expected_category is not None and product["business_category"] != expected_category:
        raise ValueError("fixed product subject probe has the wrong category")
    if require_subject_review and (not observation.get("category_refinement") or not reviewed_subject):
        raise ValueError("fixed product subject probe did not execute subject review")
    return excluded


def validate_probe_config(config):
    if config["test_rows_read"] is not False or config["human_annotation_count"] != 0:
        raise ValueError("runtime probe cannot read a final test")
    if (not isinstance(config["latency_repetitions_per_mode"], int)
            or not 5 <= config["latency_repetitions_per_mode"] <= 100
            or set(config["cache_modes"]) != {"uncached", "processor_cached", "prepared_cached"}
            or len(config["cache_modes"]) != 3
            or not config["itinerary_requests"]):
        raise ValueError("probe requires repeated paired cache modes and business requests")


def attempt_metrics(responses):
    """Count bounded correction use without treating a corrected task as first-pass."""
    attempts = [item for response in responses for item in response.get("attempts", [])]
    return {
        "requests": len(responses),
        "passed": sum(bool(response.get("passed", True)) for response in responses),
        "first_attempt_pass": sum(
            bool(response.get("attempts")) and response["attempts"][0].get("error") is None
            and bool(response.get("passed", True))
            for response in responses
        ),
        "attempts_total": len(attempts),
        "input_tokens_total": sum(item.get("input_tokens") or 0 for item in attempts),
        "output_tokens_total": sum(item.get("output_tokens") or 0 for item in attempts),
        "latency_ms_total": sum(item.get("latency_ms") or 0 for item in attempts),
    }


def run(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_probe_config(config)
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    image, comparison = ROOT / config["image"], ROOT / config["comparison_image"]
    if sha256_file(image) != config["image_sha256"] or sha256_file(comparison) != config["comparison_image_sha256"]:
        raise ValueError("probe image identity mismatch")
    settings = ReleaseSettings.load(ROOT, ROOT / config["release_config"])
    identity = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "runner_sha256": sha256_file(Path(__file__)), "config_sha256": sha256_file(config_path),
                "release_config_sha256": sha256_file(ROOT / config["release_config"]),
                "base_model": settings.base_model, "base_revision": settings.base_revision,
                "adapter_sha256": sha256_file(settings.adapter_path / "adapter_model.safetensors"),
                "adapter_disabled_scenarios": settings.adapter_disabled_scenarios,
                "observation_canonical_sha256": canonical_config_sha256(settings.product_observation),
                "images": [config["image_sha256"], config["comparison_image_sha256"]],
                "test_rows_read": False, "human_annotation_count": 0}
    write_new(output / "identity.json", identity)
    backend = TransformersPeftBackend(settings)
    started = time.perf_counter()
    ok, detail = backend.ready()
    if not ok:
        raise RuntimeError(detail)
    cold_start_ms = (time.perf_counter() - started) * 1000
    service = ScenarioService(settings, backend)
    smoke = run_model_smoke(service, image)
    write_new(output / "model_smoke.json", smoke)
    print(json.dumps({"smoke": smoke["status"]}), flush=True)
    itineraries = []
    for index, text in enumerate(config["itinerary_requests"]):
        try:
            result = service.run_task("itinerary_planning", TaskRequest(image_urls=[str(image)], text_context=text)).model_dump()
            result["passed"] = True
        except ModelGenerationError as exc:
            result = {"passed": False, "error": str(exc), "attempts": [item.model_dump() for item in exc.attempts]}
        write_new(output / f"itinerary_{index}.json", {"request": text, "response": result})
        itineraries.append(result)
        print(json.dumps({"itinerary": index, "passed": result["passed"]}), flush=True)
    dialogues = {}
    specs = {
        "product": DialogueRequest(messages=[{"role": "user", "content": "识别这张图片中的商品"}], image_urls=[str(image)]),
        "comparison": DialogueRequest(messages=[
            {"role": "user", "content": "这是第一张图片", "image_urls": [str(image)]},
            {"role": "assistant", "content": "已收到第一张图片。"},
            {"role": "user", "content": "第二张图片与第一张相比，有什么不同？", "image_urls": [str(comparison)]}]),
    }
    for name, request in specs.items():
        response = service.run_dialogue(request).model_dump()
        write_new(output / f"dialogue_{name}.json", response)
        dialogues[name] = response["task_status"]
        print(json.dumps({"dialogue": name, "status": response["task_status"]}), flush=True)
    scope_probes = []
    for index, spec in enumerate(config.get("product_scope_probes", [])):
        probe_image = ROOT / spec["image"]
        if sha256_file(probe_image) != spec["image_sha256"]:
            raise ValueError("product scope probe image identity mismatch")
        response = None
        try:
            response = service.run_task("image_product_search", TaskRequest(image_urls=[str(probe_image)])).model_dump()
            response["passed"] = True
            excluded = validate_product_scope_probe(response, settings.product_observation, spec.get("require_abstention", False),
                expected_category=spec.get("expected_category"), require_subject_review=spec.get("require_subject_review", False))
            value = {"passed": True, "response": response, "scope_abstentions": excluded, "specification": spec}
        except ModelGenerationError as exc:
            value = {"passed": False, "error": str(exc), "attempts": [item.model_dump() for item in exc.attempts], "specification": spec}
        except ValueError as exc:
            value = {"passed": False, "response": response, "error": str(exc), "specification": spec}
        write_new(output / f"product_scope_{index}.json", value)
        scope_probes.append(value["passed"])
        print(json.dumps({"product_scope_probe": index, "passed": value["passed"]}), flush=True)
    timings = {}
    reference_result = None
    for mode in config["cache_modes"]:
        if mode not in {"uncached", "processor_cached", "prepared_cached"}:
            raise ValueError("unknown cache mode")
        backend.configure_processor_cache(config["cache_max_entries"] if mode == "processor_cached" else 0)
        backend.configure_prepared_input_cache(config["cache_max_entries"] if mode == "prepared_cached" else 0)
        values = []
        for index in range(config["latency_repetitions_per_mode"]):
            started = time.perf_counter()
            try:
                response = service.run_task("image_product_search", TaskRequest(image_urls=[str(image)])).model_dump()
                response["passed"] = True
            except ModelGenerationError as exc:
                response = {"passed": False, "error": str(exc), "attempts": [item.model_dump() for item in exc.attempts]}
            response["elapsed_ms"] = (time.perf_counter() - started) * 1000
            write_new(output / f"{mode}_{index}.json", response)
            values.append(response)
            if reference_result is None and response["passed"]:
                reference_result = response["result"]
        latencies = [row["elapsed_ms"] for row in values]
        tokens = [sum(attempt["output_tokens"] or 0 for attempt in row["attempts"]) for row in values]
        timings[mode] = {"count": len(values), "mean_ms": statistics.fmean(latencies), "p50_ms": statistics.median(latencies),
                         "p95_ms": sorted(latencies)[math.ceil(0.95 * len(values)) - 1],
                         "first_request_ms": latencies[0], "failure_count": sum(not row["passed"] for row in values),
                         "all_labels_equal": all(row.get("result") == reference_result for row in values),
                         "input_tokens": sum(sum(item["input_tokens"] or 0 for item in row["attempts"]) for row in values),
                         "output_tokens": sum(tokens), "generated_tokens_per_second": sum(tokens) / (sum(latencies) / 1000),
                         "processor_cache": backend.processor_cache_snapshot(), "prepared_cache": backend.prepared_input_cache_snapshot()}
        print(json.dumps({"cache_mode": mode, "metrics": timings[mode]}), flush=True)
    artifact_sha256 = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    summary = {"status": "PASS" if smoke["status"] == "PASS" and all(row["passed"] for row in itineraries)
               and all(value == "COMPLETED" for value in dialogues.values())
               and all(scope_probes)
               and all(value["failure_count"] == 0 and value["all_labels_equal"] for value in timings.values()) else "FAIL",
               "model_smoke_status": smoke["status"], "itinerary_business_pass": sum(row["passed"] for row in itineraries),
               "itinerary_count": len(itineraries), "itinerary_attempts": attempt_metrics(itineraries),
               "dialogue_itinerary_attempts": attempt_metrics([{**smoke["dialogue"], "passed": smoke["dialogue"]["task_status"] == "COMPLETED"}]),
               "dialogue_tasks": dialogues, "latency": timings,
               "product_scope_probes": scope_probes,
               "cold_start_ms": cold_start_ms, "hardware": backend._torch.cuda.get_device_name(0),
               "peak_gpu_allocated_bytes": backend._torch.cuda.max_memory_allocated(),
               "peak_gpu_reserved_bytes": backend._torch.cuda.max_memory_reserved(),
               "test_rows_read": False, "visual_accuracy_claim": "not_established_by_smoke_or_repeated_inputs"}
    summary["artifact_sha256"] = artifact_sha256
    write_new(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week8/candidate_runtime_probe_v1.json")
    run(parser.parse_args().config.resolve())

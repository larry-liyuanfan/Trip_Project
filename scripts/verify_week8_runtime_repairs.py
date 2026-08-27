"""Real-model business smoke and fixed-photo timings; never a final-test run."""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_system_model_smoke import run_model_smoke, _sha256
from src.inference.schemas import DialogueRequest, TaskRequest
from src.inference.system_runtime import ReleaseSettings, ScenarioService, TransformersPeftBackend, ModelGenerationError


def _write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def run(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["final_test_labels_access"] is not False:
        raise ValueError("final-test access forbidden")
    output = ROOT / config["output_root"] / "runtime"
    output.mkdir(parents=True, exist_ok=False)
    image = ROOT / config["smoke_image"]
    if _sha256(image) != config["smoke_image_sha256"]:
        raise ValueError("real-image identity mismatch")
    settings = ReleaseSettings.load(ROOT, ROOT / config["release_config"])
    valid, detail = settings.validate_adapter()
    if not valid:
        raise ValueError(detail)
    backend = TransformersPeftBackend(settings)
    started = time.perf_counter()
    valid, detail = backend.ready()
    if not valid:
        raise ValueError(detail)
    cold_start_ms = (time.perf_counter() - started) * 1000
    service = ScenarioService(settings, backend)
    smoke = run_model_smoke(service, image)
    _write(output / "model_smoke.json", smoke)
    print(json.dumps({"model_smoke_status": smoke["status"], "technical_status": smoke["technical_status"]}), flush=True)
    dialogue = service.run_dialogue(DialogueRequest(messages=[{"role": "user", "content": "识别这张图中的商品"}], image_urls=[str(image)]))
    _write(output / "product_dialogue.json", dialogue.model_dump())
    timings, responses = [], []
    for index in range(config["product_repetitions"]):
        started = time.perf_counter()
        try:
            result = service.run_task("image_product_search", TaskRequest(image_urls=[str(image)])).model_dump()
        except ModelGenerationError as exc:
            result = {"schema_valid": False, "error": str(exc), "attempts": [item.model_dump() for item in exc.attempts]}
        timings.append((time.perf_counter() - started) * 1000)
        responses.append(result)
        _write(output / f"product_repeat_{index}.json", result)
    summary = {
        "status": "COMPLETED", "business_smoke_status": smoke["status"],
        "technical_smoke_status": smoke["technical_status"], "dialogue_product_task_status": dialogue.task_status,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": _sha256(config_path), "release_config_sha256": _sha256(ROOT / config["release_config"]),
        "adapter_model_sha256": _sha256(settings.adapter_path / "adapter_model.safetensors"),
        "image_sha256": _sha256(image), "hardware": backend._torch.cuda.get_device_name(0),
        "cold_start_ms": cold_start_ms, "product_repetitions": len(timings),
        "latency_ms_mean": statistics.fmean(timings), "latency_ms_p50": statistics.median(timings),
        "latency_ms_p95_nearest_rank": sorted(timings)[-1],
        "request_failure_rate": sum(not row["schema_valid"] for row in responses) / len(responses),
        "input_tokens": [sum(item["input_tokens"] or 0 for item in row["attempts"]) for row in responses],
        "output_tokens": [sum(item["output_tokens"] or 0 for item in row["attempts"]) for row in responses],
        "peak_gpu_allocated_bytes": backend._torch.cuda.max_memory_allocated(),
        "outputs_equal": all(row.get("result") == responses[0].get("result") for row in responses),
        "final_test_labels_read": False, "visual_accuracy_assessed": False,
        "latency_interpretation": "Five warm repetitions after smoke; P95 is the observed maximum, not a stable tail estimate."
    }
    _write(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week8/audit_repair_v1.json")
    run(parser.parse_args().config)

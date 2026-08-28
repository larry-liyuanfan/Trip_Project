"""Isolated development ablation of prompt conflicts and adapter behaviour."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_week8_product import load_review_inputs
from src.inference.schemas import TaskRequest
from src.inference.system_runtime import (
    ModelGenerationError, ReleaseSettings, ScenarioService, TransformersPeftBackend,
)
from src.training.week8_product import sha256_file
from src.inference.product_observation import generate_observation


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def build_requests(root, chosen, texts):
    requests = [("image_product_search", row["sample_id"], TaskRequest(
        image_urls=[str(root / row["image_path"])])) for row in chosen]
    image_path = requests[0][2].image_urls[0]
    requests.extend(("itinerary_planning", f"itinerary_{index}", TaskRequest(
        image_urls=[image_path], text_context=text)) for index, text in enumerate(texts))
    return requests


def run(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["final_test_access"] is not False or config["human_annotation_count"] != 0:
        raise ValueError("development only; no human labels")
    _, _, rows, validation, development_sha = load_review_inputs(ROOT, ROOT / config["source_review_config"])
    chosen = rows if config["development_indices"] == "all" else [rows[index] for index in config["development_indices"]]
    if len({row["sample_id"] for row in chosen}) != len(chosen):
        raise ValueError("duplicate development samples")
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    settings = ReleaseSettings.load(ROOT, ROOT / config["release_config"])
    observation = json.loads((ROOT / config["observation_config"]).read_text(encoding="utf-8")) if config.get("observation_config") else None
    prompt_paths = [ROOT / "configs/evaluation/prompts" / config[key] / filename
                    for key, filename in (("product_prompt", "common.yaml"),
                                          ("product_prompt", "image_product_search.yaml"),
                                          ("itinerary_prompt", "common.yaml"),
                                          ("itinerary_prompt", "itinerary_planning.yaml"))]
    identity = {
        "run_id": config["run_id"], "config_sha256": sha256_file(path),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runner_sha256": sha256_file(Path(__file__)), "development_sha256": development_sha,
        "dataset_validation": validation, "test_rows_read": False,
        "adapter_sha256": sha256_file(settings.adapter_path / "adapter_model.safetensors"),
        "base_model": settings.base_model, "base_revision": settings.base_revision,
        "prompt_hashes": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in prompt_paths},
        "selected_sample_ids": [row["sample_id"] for row in chosen],
        "human_annotation_count": 0, "visual_accuracy_claim_supported": False,
        "observation_config_sha256": sha256_file(ROOT / config["observation_config"]) if observation else None,
        "observation_profile_config_hashes": {key: sha256_file(ROOT / value) for key, value in config.get("observation_profile_configs", {}).items()},
    }
    write_new(output / "identity.json", identity)
    backend = TransformersPeftBackend(settings)
    started = time.perf_counter()
    ok, detail = backend.ready()
    if not ok:
        raise RuntimeError(detail)
    cold_start_ms = (time.perf_counter() - started) * 1000
    summaries = {}
    for profile in config["profiles"]:
        active = settings
        if profile not in {"release_adapter", "formal_adapter"}:
            active = replace(settings,
                schema_constrained_retry=config.get("schema_constrained_retry", settings.schema_constrained_retry),
                itinerary_structured_request=config.get("itinerary_structured_request", False),
                prompt_versions={**settings.prompt_versions,
                    "image_product_search": config["product_prompt"],
                    "itinerary_planning": config["itinerary_prompt"]},
                max_new_tokens_by_scenario={**settings.max_new_tokens_by_scenario,
                    "image_product_search": config["product_max_new_tokens"],
                    "itinerary_planning": config["itinerary_max_new_tokens"]})
        if profile == "formal_adapter":
            active = replace(settings, prompt_versions={**settings.prompt_versions, "image_product_search": "system_repair_product_compact_v3"})
        active_observation = observation
        if profile in config.get("observation_profile_configs", {}):
            active_observation = json.loads((ROOT / config["observation_profile_configs"][profile]).read_text(encoding="utf-8"))
        if profile not in {"release_adapter", "formal_adapter", "repaired_adapter", "repaired_base", "observation_base", "observation_enhanced_base"}:
            raise ValueError("unsupported ablation profile")
        service = ScenarioService(active, backend)
        requests = build_requests(ROOT, chosen, config["itinerary_requests"])
        results = []
        # 整段消融与生成共用锁，避免 adapter 状态泄漏到其他请求。
        with backend._execution_lock:
            context = backend._model.disable_adapter() if profile.endswith("_base") else nullcontext()
            with context, (output / f"{profile}.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
                for scenario, sample_id, request in requests:
                    started = time.perf_counter()
                    try:
                        if profile.startswith("observation_") and scenario == "image_product_search":
                            if active_observation is None:
                                raise ValueError("observation profile requires its config")
                            record = generate_observation(backend, request.image_urls[0], active_observation)
                            record["attempts"] = [item.model_dump() for item in record["attempts"]]
                        else:
                            response = service.run_task(scenario, request)
                            record = {"passed": True, **response.model_dump()}
                    except ModelGenerationError as exc:
                        record = {"passed": False, "error": str(exc),
                                  "attempts": [item.model_dump() for item in exc.attempts]}
                    record.update(sample_id=sample_id, scenario=scenario, profile=profile,
                                  adapter_enabled=not profile.endswith("_base"),
                                  elapsed_ms=(time.perf_counter() - started) * 1000)
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    results.append(record)
                    print(json.dumps({"profile": profile, "sample": sample_id, "passed": record["passed"]}), flush=True)
        summaries[profile] = {
            "technical_or_business_pass": sum(row["passed"] for row in results),
            "requests": len(results), "raw_sha256": sha256_file(output / f"{profile}.jsonl"),
            "itinerary_business_pass": sum(row["passed"] for row in results if row["scenario"] == "itinerary_planning"),
            "visual_accuracy_claim_supported": False,
        }
    summary = {"status": "COMPLETED", "profiles": summaries, "cold_start_ms": cold_start_ms,
               "hardware": backend._torch.cuda.get_device_name(0),
               "peak_gpu_allocated_bytes": backend._torch.cuda.max_memory_allocated(),
               "test_rows_read": False, "release_changed": False}
    write_new(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week8/contract_ablation_v1.json")
    run(parser.parse_args().config.resolve())

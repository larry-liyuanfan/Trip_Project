"""CPU-only verification of effective pixel limits with the pinned real processor."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.week8_visual_holdout import write_json_new
from src.inference.product_observation import observation_messages
from src.inference.system_runtime import _transformers_messages
from src.inference.visual_limits import configure_visual_pixel_limit
from src.training.week7_data import sha256_file


def run(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["final_test_access"] is not False or config["human_annotation_count"] != 0:
        raise ValueError("processor probe must not consume final or human labels")
    image = ROOT / config["image_path"]
    if sha256_file(image) != config["image_sha256"]:
        raise ValueError("processor probe image identity mismatch")
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from transformers import AutoProcessor
    torch.set_num_threads(config["cpu_threads"])
    observation = json.loads((ROOT / config["observation_config"]).read_text(encoding="utf-8"))
    messages = _transformers_messages(observation_messages(str(image), observation))
    rows = []
    for mode in config["modes"]:
        processor = AutoProcessor.from_pretrained(config["base_model"], revision=config["base_revision"],
                                                 local_files_only=True, trust_remote_code=False)
        if mode["mode"] == "legacy_attribute":
            processor.image_processor.max_pixels = mode["max_pixels"]
        elif mode["mode"] == "effective_limit":
            configure_visual_pixel_limit(processor, mode["max_pixels"])
        elif mode["mode"] != "baseline":
            raise ValueError("unsupported pixel probe mode")
        started = time.perf_counter()
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              return_dict=True, return_tensors="pt", truncation=False)
        grid = inputs["image_grid_thw"][0].tolist()
        patch, merge = processor.image_processor.patch_size, processor.image_processor.merge_size
        pixels = grid[1] * grid[2] * patch ** 2
        rows.append({**mode, "actual_size_setting": dict(processor.image_processor.size),
                     "image_grid_thw": grid, "resized_pixels": pixels,
                     "visual_tokens": grid[0] * grid[1] * grid[2] // merge ** 2,
                     "input_tokens": int(inputs["input_ids"].shape[1]),
                     "processor_ms": (time.perf_counter() - started) * 1000,
                     "bound_applied": pixels <= mode["max_pixels"] if mode.get("max_pixels") else None})
    result = {"status": "PASS" if all(row["bound_applied"] for row in rows if row["mode"] == "effective_limit") else "FAIL",
              "profiles": rows, "model_weights_loaded": False, "gpu_used": False,
              "model_quality_assessed": False, "final_test_access": False, "human_annotation_count": 0,
              "config_sha256": sha256_file(config_path), "image_sha256": config["image_sha256"],
              "base_model": config["base_model"], "base_revision": config["base_revision"],
              "transformers_version": transformers.__version__, "torch_version": torch.__version__,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "runner_sha256": sha256_file(Path(__file__)),
              "implementation_sha256": sha256_file(ROOT / "src/inference/visual_limits.py")}
    write_json_new(output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config.resolve())

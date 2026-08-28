"""Development-only validation of bounded teacher correction reliability."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_week8_visual_silver import load_inputs
from src.data.week8_visual_holdout import read_json, write_json_new
from src.evaluation.visual_teacher_retry import collect_with_history
from src.inference.client import _read_api_key
from src.training.week7_data import sha256_file


def run(config_path):
    config = read_json(config_path)
    rows, manifest = load_inputs(config, ROOT)
    observation = read_json(ROOT / config["observation_config"])
    key = _read_api_key()
    if not key:
        raise ValueError("teacher credential missing")
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    identity = {"config_sha256": sha256_file(config_path), "manifest_sha256": sha256_file(manifest),
                "runner_sha256": sha256_file(Path(__file__)), "retry_sha256": sha256_file(ROOT / "src/evaluation/visual_teacher_retry.py"),
                "reference_validation_sha256": sha256_file(ROOT / "src/evaluation/visual_reference_validation.py"),
                "observation_sha256": sha256_file(ROOT / config["observation_config"]),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "test_rows_read": False, "candidate_outputs_supplied": False, "metadata_supplied": False,
                "human_annotation_count": 0, "evaluation": "reference_reliability_only_not_candidate_selection"}
    write_json_new(output / "identity.json", identity)
    base_url = os.environ.get("MODEL_API_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").rstrip("/")
    failures = retries = 0
    with ThreadPoolExecutor(max_workers=2) as executor, (output / "raw_outputs.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for record in executor.map(lambda row: collect_with_history(row, config, observation, ROOT, base_url, key), rows):
            failures += int(bool(record.get("error")))
            retries += len(record["attempts"]) - 1
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"sample_id": record["sample_id"], "error": record.get("error")}), flush=True)
    result = {"status": "COMPLETED", "count": len(rows), "failures": failures, "extra_attempts": retries,
              "raw_sha256": sha256_file(output / "raw_outputs.jsonl"), "test_rows_read": False,
              "label_source": "model_generated_silver", "candidate_selection": "NOT_PERFORMED"}
    write_json_new(output / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week8/visual_teacher_v4.json")
    run(parser.parse_args().config.resolve())

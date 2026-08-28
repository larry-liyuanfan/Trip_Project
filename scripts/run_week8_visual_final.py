"""Seal a development-selected candidate; execute each final role at most once."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_week8_visual_silver import collect_row
from src.data.week8_visual_holdout import read_json, write_json_new, within, validate_holdout
from src.evaluation.week8_visual_silver import score_paired, validate_locked_final, select_development_candidate
from src.inference.business_validation import itinerary_business_errors
from src.inference.client import _read_api_key
from src.inference.product_observation import canonical_config_sha256, map_observation
from src.inference.schemas import TaskRequest
from src.inference.system_runtime import ReleaseSettings, ScenarioService, TransformersPeftBackend, ModelGenerationError
from src.inference.transport_utils import strip_json_fence
from src.training.week7_data import IDENTITY_FIELDS, sha256_file, iter_jsonl
from src.evaluation.visual_teacher_retry import collect_with_history
from src.evaluation.visual_reference_validation import map_teacher_observation
from src.inference.visual_limits import temporary_visual_pixel_limit
from scripts.compare_week8_development_revision import compare as compare_revision


def source_hash(path):
    """源代码按 LF 规范化；图片/模型/raw 输出始终按原始字节哈希。"""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def protocol_files(root, config):
    paths = {config[key] for key in ("candidate_release", "formal_release", "teacher_config", "teacher_observation", "candidate_observation")}
    paths.update({"scripts/run_week8_visual_final.py", "scripts/collect_week8_visual_silver.py",
                  "src/inference/system_runtime.py", "src/inference/product_observation.py", "src/inference/schemas.py",
                  "src/inference/processor_cache.py", "src/inference/business_validation.py",
                  "src/evaluation/week8_visual_silver.py", "src/evaluation/product_semantics.py",
                  "src/evaluation/schema_validation.py", "src/evaluation/prompting.py", "src/data/week8_visual_holdout.py",
                  "src/evaluation/visual_teacher_retry.py", "scripts/compare_week8_development_revision.py",
                  "src/evaluation/visual_reference_validation.py", "src/inference/visual_limits.py",
                  "src/inference/product_style_scope.py", "src/data/product_labels.py",
                  "scripts/verify_week8_candidate_acceptance.py"})
    if config.get("development_config"):
        paths.add(config["development_config"])
    if config.get("development_reference_revision"):
        paths.add(config["development_reference_revision"])
        revision = read_json(within(root, config["development_reference_revision"]))
        paths.update({revision["observation_config"], revision["source_teacher_config"],
                      "scripts/score_week8_reference_revision.py", "scripts/repair_week8_visual_reference.py",
                      "scripts/compare_week8_incumbent.py", "src/evaluation/visual_reference_revision.py",
                      "src/inference/product_style_refinement.py", "src/inference/product_style_scope.py",
                      "src/inference/visual_limits.py", "src/data/product_labels.py"})
    for key in ("candidate_release", "formal_release"):
        release = read_json(root / config[key])
        for scenario, version in release["prompts"].items():
            directory = root / "configs/evaluation/prompts" / version
            if directory.is_dir():
                paths.update(p.relative_to(root).as_posix() for p in directory.glob("*.yaml"))
    paths.update(p.relative_to(root).as_posix() for p in (root / "configs/evaluation/schemas").glob("*.json"))
    return {path: source_hash(within(root, path)) for path in sorted(paths)}


def validate_development_identity(root, config, comparison):
    """把选优分数绑定到真正生成该分数的商品配置，不能只验证候选自己的哈希。"""
    specification = read_json(within(root, config["development_config"]))
    directory = within(root, specification["output_root"])
    identity = read_json(directory / "identity.json")
    candidate = read_json(within(root, config["candidate_release"]))
    role = comparison["selection"]["selected_role"]
    if config.get("development_reference_revision"):
        from scripts.score_week8_reference_revision import build_comparison
        verified = build_comparison(within(root, config["development_config"]),
                                    within(root, config["development_reference_revision"]), root)
        if comparison != verified:
            raise ValueError("development comparison differs from raw reference revision replay")
        improvement = verified["incumbent_comparison"]
        if (improvement["status"] != "IMPROVED_DEVELOPMENT_CANDIDATE"
                or improvement["selected_role"] != role):
            raise ValueError("new candidate has not improved over the incumbent on development")
        limit = specification.get("profile_visual_max_pixels", {}).get(role)
        if (candidate["generation"].get("visual_max_pixels") != limit
                or identity.get("profile_visual_max_pixels", {}) != specification.get("profile_visual_max_pixels", {})):
            raise ValueError("candidate visual limits differ from tested development settings")
    if (comparison["generation_identity_sha256"] != sha256_file(directory / "identity.json")
            or identity["config_sha256"] != sha256_file(root / config["development_config"])
            or identity["test_rows_read"] is not False or specification["final_test_access"] is not False
            or identity["development_sha256"] != specification["reference_manifest_sha256"]
            or comparison["reference_raw_sha256"] != specification["reference_raw_sha256"]):
        raise ValueError("development generation identity mismatch")
    observation_path = specification["observation_profile_configs"][role]
    if (observation_path != config["candidate_observation"]
            or identity["observation_profile_config_hashes"][role] != sha256_file(within(root, observation_path))
            or candidate["product_pipeline"]["config"] != observation_path
            or "image_product_search" not in candidate["model"].get("adapter_disabled_scenarios", [])
            or role != "observation_enhanced_base"):
        raise ValueError("candidate is not the development-tested observation and adapter route")
    for key, identity_key in (("base_model", "base_model"), ("base_revision", "base_revision"), ("adapter_model_sha256", "adapter_sha256")):
        if candidate["model"][key] != identity[identity_key]:
            raise ValueError("candidate model differs from development")
    if config.get("previous_final"):
        previous = within(root, config["previous_final"])
        failure = read_json(previous / "acceptance_failure.json")
        if failure["classification"] != "INVALID_REFERENCE" or failure["semantic_scores_computed"] is not False:
            raise ValueError("previous final outcomes cannot be used for candidate revision")
        if not any(item["path"] == config["previous_final"] and item["lock_sha256"] == failure["data_lock_sha256"]
                   for item in config.get("previous_visual_holdouts", [])):
            raise ValueError("consumed previous final must be explicitly excluded")
        previous_comparison = read_json(within(root, config["previous_development_comparison"]))
        revision = compare_revision(previous_comparison, comparison)
        if not revision["new_final_allowed"]:
            raise ValueError("no substantive development revision; do not replace the final test")
    return sha256_file(directory / "identity.json")


def validate_runtime_probe(root, config):
    directory = within(root, config["runtime_probe"])
    summary = read_json(directory / "summary.json")
    identity = read_json(directory / "identity.json")
    probe_config = read_json(root / config["runtime_probe_config"])
    if summary["status"] != "PASS" or summary["test_rows_read"] is not False:
        raise ValueError("runtime probe has not passed")
    if len(list(directory.glob("itinerary_*.json"))) != len(probe_config["itinerary_requests"]):
        raise ValueError("runtime business evidence incomplete")
    for path in directory.glob("itinerary_*.json"):
        value = read_json(path)
        if not value["response"].get("passed") or itinerary_business_errors(value["response"]["result"], value["request"]):
            raise ValueError("runtime itinerary failed independent business replay")
    smoke = read_json(directory / "model_smoke.json")
    if smoke["status"] != "PASS" or smoke["dialogue"]["task_status"] != "COMPLETED":
        raise ValueError("runtime smoke incomplete")
    if itinerary_business_errors(smoke["scenarios"]["itinerary_planning"]["result"], "规划上海两日行程，预算适中，偏好安静的文化体验。"):
        raise ValueError("smoke itinerary failed replay")
    if itinerary_business_errors(smoke["dialogue"]["task_result"]["result"], "城市：Shanghai；2天；推荐安静行程"):
        raise ValueError("dialogue itinerary failed replay")
    probe_release = read_json(root / probe_config["release_config"])
    candidate = read_json(root / config["candidate_release"])
    if sha256_file(root / probe_config["release_config"]) != identity["release_config_sha256"]:
        raise ValueError("tested release identity changed")
    for key in ("model", "product_pipeline", "prompts", "schemas", "dialogue"):
        if probe_release[key] != candidate[key]:
            raise ValueError("candidate differs from tested production settings")
    generation = dict(candidate["generation"])
    tested = dict(probe_release["generation"])
    for key in ("processor_cache_max_entries", "prepared_input_cache_max_entries"):
        generation.pop(key, None)
        tested.pop(key, None)
    if generation != tested:
        raise ValueError("untested candidate generation settings")
    selected_cache = config["selected_cache_mode"]
    benchmark = summary["latency"][selected_cache]
    if benchmark["failure_count"] != 0 or not benchmark["all_labels_equal"]:
        raise ValueError("cache optimization has changed product quality")
    expected = {"processor_cache_max_entries": probe_config["cache_max_entries"] if selected_cache == "processor_cached" else 0,
                "prepared_input_cache_max_entries": probe_config["cache_max_entries"] if selected_cache == "prepared_cached" else 0}
    if any(candidate["generation"].get(key, 0) != value for key, value in expected.items()):
        raise ValueError("candidate cache does not match benchmark")
    return {path.name: sha256_file(path) for path in directory.glob("*.json")}


def seal(root, config_path):
    config = read_json(config_path)
    directory = within(root, config["output_root"])
    if (directory / "candidate_lock.json").exists():
        raise FileExistsError("candidate lock already exists")
    _, data_lock = validate_holdout(root, config)
    comparison = read_json(root / config["development_comparison"])
    selection = select_development_candidate(comparison["summaries"])
    if selection["selected_role"] != "observation_enhanced_base" or selection != comparison["selection"]:
        raise ValueError("candidate not selected by fixed development")
    development_identity = validate_development_identity(root, config, comparison)
    probe = validate_runtime_probe(root, config)
    candidate = read_json(root / config["candidate_release"])
    observation = read_json(root / config["candidate_observation"])
    if candidate["product_pipeline"]["config_canonical_sha256"] != canonical_config_sha256(observation):
        raise ValueError("selected observation identity mismatch")
    lock = {"protocol": config["protocol"], "config_canonical_sha256": canonical_config_sha256(config),
            "data_lock_sha256": data_lock["lock_sha256"], "development_comparison_sha256": sha256_file(root / config["development_comparison"]),
            "selected_role": selection["selected_role"], "runtime_probe_files": probe,
            "development_generation_identity_sha256": development_identity,
            "source_files_lf_sha256": protocol_files(root, config),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "selection_only_used_development": True, "final_roles": ["teacher", "inference"],
            "test_results_read_before_lock": False, "human_annotation_count": 0}
    lock["lock_sha256"] = canonical_config_sha256(lock)
    write_json_new(directory / "candidate_lock.json", lock)
    return {"status": "CANDIDATE_LOCKED_FOR_ONCE_ONLY_FINAL", "lock_sha256": lock["lock_sha256"]}


def final_context(root, config_path, role):
    config = read_json(config_path)
    directory = within(root, config["output_root"])
    lock = read_json(directory / "candidate_lock.json")
    if (lock["lock_sha256"] != canonical_config_sha256({k: v for k, v in lock.items() if k != "lock_sha256"})
            or lock["config_canonical_sha256"] != canonical_config_sha256(config)
            or lock["source_files_lf_sha256"] != protocol_files(root, config)
            or lock["selection_only_used_development"] is not True):
        raise ValueError("locked final protocol changed")
    if role not in lock["final_roles"]:
        raise ValueError("unknown once-only final role")
    rows, data_lock = validate_holdout(root, config)
    if data_lock["lock_sha256"] != lock["data_lock_sha256"]:
        raise ValueError("final data identity changed after candidate lock")
    output = directory / role
    output.mkdir(exist_ok=False)
    # 创建即消费，不按最终结果重跑；出错仍保存这次尝试和失败身份。
    write_json_new(output / "consumed.json", {"role": role, "candidate_lock_sha256": lock["lock_sha256"],
        "manifest_sha256": data_lock["manifest_sha256"], "sample_ids": [row["sample_id"] for row in rows], "test_rows_read": True})
    return config, rows, lock, output


def teacher(root, config_path):
    key = _read_api_key()
    if not key:
        raise ValueError("teacher credential is missing")
    config, rows, lock, output = final_context(root, config_path, "teacher")
    specification = read_json(root / config["teacher_config"])
    observation = read_json(root / config["teacher_observation"])
    base_url = os.environ.get("MODEL_API_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").rstrip("/")
    failures = 0
    with ThreadPoolExecutor(max_workers=specification["concurrency"]) as executor, (output / "raw_outputs.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        collector = collect_with_history if specification.get("retry_protocol") == "bounded_history_correction_v1" else collect_row
        for record in executor.map(lambda row: collector(row, specification, observation, root, base_url, key), rows):
            failures += int(bool(record.get("error")))
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"sample_id": record["sample_id"], "error": record.get("error")}), flush=True)
    summary = {"status": "COMPLETED", "count": len(rows), "failures": failures, "model": specification["model"],
               "metadata_supplied": False, "candidate_outputs_supplied": False, "label_source": "model_generated_silver",
               "candidate_lock_sha256": lock["lock_sha256"], "raw_sha256": sha256_file(output / "raw_outputs.jsonl")}
    write_json_new(output / "summary.json", summary)
    return summary


def inference(root, config_path):
    config, rows, lock, output = final_context(root, config_path, "inference")
    candidate = ReleaseSettings.load(root, root / config["candidate_release"])
    baseline = ReleaseSettings.load(root, root / config["formal_release"])
    if (candidate.base_model, candidate.base_revision, candidate.adapter_model_sha256) != (baseline.base_model, baseline.base_revision, baseline.adapter_model_sha256):
        raise ValueError("paired base/adapter identities differ")
    # 从正式处理器开始，每个组显式应用自己的像素参数；不能把候选上限泄漏给基线。
    backend = TransformersPeftBackend(baseline)
    started = time.perf_counter()
    ok, detail = backend.ready()
    if not ok:
        raise RuntimeError(detail)
    cold_start = (time.perf_counter() - started) * 1000
    roles = {}
    for role, settings in (("formal_adapter", baseline), ("locked_candidate", candidate)):
        backend.configure_processor_cache(settings.processor_cache_max_entries)
        backend.configure_prepared_input_cache(settings.prepared_input_cache_max_entries)
        service = ScenarioService(settings, backend)
        failures = 0
        with backend._execution_lock, temporary_visual_pixel_limit(backend._processor, settings.visual_max_pixels), (output / (role + ".jsonl")).open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                started = time.perf_counter()
                try:
                    response = service.run_task("image_product_search", TaskRequest(image_urls=[str(root / row["image_path"])]))
                    record = {"passed": True, **response.model_dump()}
                except ModelGenerationError as exc:
                    record = {"passed": False, "error": str(exc), "attempts": [item.model_dump() for item in exc.attempts]}
                    failures += 1
                record.update({key: row[key] for key in IDENTITY_FIELDS})
                record.update(role=role, elapsed_ms=(time.perf_counter() - started) * 1000)
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps({"role": role, "sample_id": row["sample_id"], "passed": record["passed"]}), flush=True)
        roles[role] = {"count": len(rows), "failures": failures, "raw_sha256": sha256_file(output / (role + ".jsonl"))}
    summary = {"status": "COMPLETED", "roles": roles, "candidate_lock_sha256": lock["lock_sha256"],
               "hardware": backend._torch.cuda.get_device_name(0), "cold_start_ms": cold_start,
               "peak_gpu_allocated_bytes": backend._torch.cuda.max_memory_allocated(),
               "adapter_sha256": sha256_file(candidate.adapter_path / "adapter_model.safetensors")}
    write_json_new(output / "summary.json", summary)
    return summary


def replay_final(root, config_path):
    config = read_json(config_path)
    directory = within(root, config["output_root"])
    lock = read_json(directory / "candidate_lock.json")
    if (lock["lock_sha256"] != canonical_config_sha256({key: value for key, value in lock.items() if key != "lock_sha256"})
            or lock["config_canonical_sha256"] != canonical_config_sha256(config)
            or lock["source_files_lf_sha256"] != protocol_files(root, config)):
        raise ValueError("cannot score with an unlocked implementation")
    rows, data_lock = validate_holdout(root, config)
    if lock["data_lock_sha256"] != data_lock["lock_sha256"]:
        raise ValueError("final dataset differs from candidate lock")
    summaries = {role: read_json(directory / role / "summary.json") for role in ("teacher", "inference")}
    for role, summary in summaries.items():
        consumed = read_json(directory / role / "consumed.json")
        if (summary["status"] != "COMPLETED" or summary["candidate_lock_sha256"] != lock["lock_sha256"]
                or consumed["candidate_lock_sha256"] != lock["lock_sha256"] or consumed["manifest_sha256"] != data_lock["manifest_sha256"]):
            raise ValueError("final attempt is incomplete or has wrong lock")
    teacher_path = directory / "teacher/raw_outputs.jsonl"
    if sha256_file(teacher_path) != summaries["teacher"]["raw_sha256"]:
        raise ValueError("teacher raw bytes changed")
    references = list(iter_jsonl(teacher_path))
    teacher_observation = read_json(root / config["teacher_observation"])
    for reference in references:
        if reference.get("error") or map_teacher_observation(json.loads(strip_json_fence(reference["attempts"][-1]["raw_content"])), teacher_observation) != reference["target"]:
            raise ValueError("invalid final reference; never drop or replace it")
    audit = {"protocol": read_json(root / config["teacher_config"])["protocol"],
             "metadata_supplied": summaries["teacher"]["metadata_supplied"], "candidate_outputs_supplied": summaries["teacher"]["candidate_outputs_supplied"],
             "model_independent": summaries["teacher"]["model"] != read_json(root / config["candidate_release"])["model"]["base_model"],
             "test_rows_read": True, "candidate_lock_sha256": lock["lock_sha256"], "reference_raw_sha256": sha256_file(teacher_path)}
    expected = {row["sample_id"]: {key: row[key] for key in IDENTITY_FIELDS} for row in rows}
    scores = {}
    for role in ("formal_adapter", "locked_candidate"):
        path = directory / "inference" / (role + ".jsonl")
        if sha256_file(path) != summaries["inference"]["roles"][role]["raw_sha256"]:
            raise ValueError("inference raw bytes changed")
        records = list(iter_jsonl(path))
        for collection in (records, references):
            if len(collection) != len(expected) or {row["sample_id"] for row in collection} != set(expected):
                raise ValueError("final sample coverage changed")
            if any({key: row[key] for key in IDENTITY_FIELDS} != expected[row["sample_id"]] for row in collection):
                raise ValueError("final five-dimension identity mismatch")
        observation = read_json(root / config["candidate_observation"]) if role == "locked_candidate" else None
        scores[role] = score_paired(root, references, records, observation, reference_audit=audit, phase="final")
    result = {"acceptance": validate_locked_final(scores["formal_adapter"], scores["locked_candidate"]), "scores": scores,
              "candidate_lock_sha256": lock["lock_sha256"], "human_annotation_count": 0, "test_used_for_tuning": False}
    return result


def score(root, config_path):
    result = replay_final(root, config_path)
    directory = within(root, read_json(config_path)["output_root"])
    write_json_new(directory / "final_comparison.json", result)
    return result["acceptance"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["seal", "teacher", "inference", "score"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week8/visual_final_v2.json")
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](ROOT, args.config.resolve()), ensure_ascii=False), flush=True)

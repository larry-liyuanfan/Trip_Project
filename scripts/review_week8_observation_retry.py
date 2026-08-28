"""Compare one correction on every first-attempt error from fixed development runs."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.week8_visual_holdout import read_json, write_json_new, within
from src.training.week7_data import IDENTITY_FIELDS, iter_jsonl, sha256_file
from src.inference.product_observation import (
    canonical_config_sha256, load_observation_config, observation_messages,
    observation_correction_messages, parse_observation, map_observation,
    observation_correction_response_format,
)
from src.inference.system_runtime import ReleaseSettings, TransformersPeftBackend, ModelGenerationError


def previous_wire(raw, source_config):
    """Only translate compact evidence objects; never repair or drop their content."""
    if source_config["protocol"] != "product_visual_observation_v4":
        return raw, "unchanged"
    try:
        parsed = parse_observation(raw, source_config)
    except ValueError:
        return raw, "unparsed_raw_preserved"
    if not isinstance(parsed, dict):
        return raw, "nonobject_raw_preserved"
    converted = copy.deepcopy(parsed)
    for field in ("style_evidence", "facility_evidence"):
        if isinstance(converted.get(field), dict):
            converted[field] = [{"label": label, "fact": fact} for label, fact in converted[field].items()]
    return json.dumps(converted, ensure_ascii=False), "lossless_evidence_object_to_array"


def load_cases(root, config):
    if config.get("final_test_access") is not False or config.get("human_annotation_count") != 0:
        raise ValueError("correction diagnostics are development-only without human labels")
    manifest = within(root, config["development_manifest"])
    if sha256_file(manifest) != config["development_manifest_sha256"]:
        raise ValueError("development manifest identity changed")
    rows = list(iter_jsonl(manifest))
    by_id = {row["sample_id"]: row for row in rows}
    if len(rows) != config["development_count"] or len(by_id) != len(rows):
        raise ValueError("fixed development coverage mismatch")
    cases, sources = [], {}
    for specification in config["sources"]:
        generation_path = within(root, specification["generation_config"])
        generation = read_json(generation_path)
        if generation.get("final_test_access") is not False or generation.get("development_indices") != "all":
            raise ValueError("cannot import final or selectively sampled generation")
        directory = within(root, generation["output_root"])
        identity, summary = read_json(directory / "identity.json"), read_json(directory / "summary.json")
        if (identity["config_sha256"] != sha256_file(generation_path) or identity["test_rows_read"] is not False
                or identity["development_sha256"] != config["development_manifest_sha256"]
                or identity["selected_sample_ids"] != [row["sample_id"] for row in rows]
                or summary["status"] != "COMPLETED"):
            raise ValueError("source is not the complete fixed development generation")
        for role in specification["profiles"]:
            raw_path = directory / (role + ".jsonl")
            raw_sha = sha256_file(raw_path)
            if raw_sha != summary["profiles"][role]["raw_sha256"]:
                raise ValueError("source raw bytes differ from immutable summary")
            source_observation_path = generation.get("observation_profile_configs", {}).get(role, generation["observation_config"])
            source_observation = read_json(within(root, source_observation_path))
            expected_sha = identity.get("observation_profile_config_hashes", {}).get(role, identity.get("observation_config_sha256"))
            if sha256_file(within(root, source_observation_path)) != expected_sha:
                raise ValueError("source observation configuration changed")
            records = list(iter_jsonl(raw_path))
            if len(records) != len(rows) or {record["sample_id"] for record in records} != set(by_id):
                raise ValueError("source raw coverage differs from fixed development")
            sources[raw_path.relative_to(root).as_posix()] = raw_sha
            for record in records:
                attempts = record.get("attempts", [])
                if not attempts:
                    raise ValueError("source request has no preserved first attempt")
                if not attempts[0].get("error"):
                    continue
                row = by_id[record["sample_id"]]
                if sha256_file(within(root, row["image_path"])) != row["image_sha256"]:
                    raise ValueError("development image identity changed")
                raw, translation = previous_wire(attempts[0]["raw_output"], source_observation)
                cases.append({"case_id": generation["run_id"] + ":" + role + ":" + record["sample_id"],
                    **{key: row.get(key) for key in IDENTITY_FIELDS}, "image_path": row["image_path"],
                    "source_raw_path": raw_path.relative_to(root).as_posix(), "source_raw_sha256": raw_sha,
                    "source_first_attempt": attempts[0], "previous_raw": raw, "wire_translation": translation,
                    "validation_error": attempts[0]["error"]})
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("diagnostic cases must be nonempty and unique")
    return cases, {"sources": sources, "case_count": len(cases), "unique_development_images": len({c["image_sha256"] for c in cases}),
                   "first_error_counts": dict(Counter(case["validation_error"] for case in cases)),
                   "case_selection": "all_first_attempt_errors_without_final_or_reference_labels"}


def load_continuation(root, config, cases, source_audit, generation_root=None):
    specification = config.get("continue_incomplete")
    if specification is None:
        return {}, None
    directory = within(root, specification["output_root"])
    original_path = within(root, specification["config_path"])
    original = read_json(original_path)
    identity_path = directory / "identity.json"
    identity = read_json(identity_path)
    if ((directory / "summary.json").exists() or original["output_root"] != specification["output_root"]
            or original.get("final_test_access") is not False or original.get("human_annotation_count") != 0
            or specification["output_root"] == config["output_root"]
            or sha256_file(identity_path) != specification["identity_sha256"]
            or identity["config_sha256"] != sha256_file(original_path)
            or identity["final_test_access"] is not False or identity["reference_targets_supplied"] is not False
            or identity["human_annotation_count"] != 0
            or specification["interruption"]["state"] != "TIMEOUT"
            or any(original[key] != config[key] for key in (
                "development_manifest", "development_manifest_sha256", "development_count", "release_config", "profiles", "sources"))
            or any(identity.get(key) != value for key, value in source_audit.items())
            or list(iter_jsonl(directory / "cases.jsonl")) != cases):
        raise ValueError("continuation requires the exact incomplete development execution")
    for key, path in (("correction_implementation_sha256", "src/inference/product_observation.py"),
                      ("decoder_implementation_sha256", "src/inference/observation_constraints.py"),
                      ("backend_implementation_sha256", "src/inference/system_runtime.py")):
        if identity[key] != sha256_file(root / path):
            raise ValueError("cannot change model execution while continuing partial evidence")
    if not re.fullmatch(r"[0-9a-f]{40}", identity["git_commit"]):
        raise ValueError("invalid original runner commit")
    original_source = subprocess.check_output(["git", "show", identity["git_commit"] + ":scripts/review_week8_observation_retry.py"], cwd=root)
    if hashlib.sha256(original_source).hexdigest() != identity["runner_sha256"]:
        raise ValueError("original runner does not match its immutable Git source")
    from scripts.verify_week8_observation_retry import replay_records
    prefixes, audit = {}, {}
    if set(specification["prefixes"]) != set(config["profiles"]):
        raise ValueError("continuation must retain every original profile")
    for name, path in config["profiles"].items():
        if identity["profile_config_hashes"][name] != sha256_file(within(root, path)):
            raise ValueError("continuation observation configuration changed")
        raw_path = directory / (name + ".jsonl")
        declared = specification["prefixes"][name]
        rows = list(iter_jsonl(raw_path)) if raw_path.exists() else []
        digest = sha256_file(raw_path) if raw_path.exists() else None
        if (len(rows) > len(cases) or len(rows) != declared["count"] or digest != declared["sha256"]
                or [row["case_id"] for row in rows] != [case["case_id"] for case in cases[:len(rows)]]):
            raise ValueError("continuation prefix changed or skipped a difficult case")
        if rows:
            replay_records(root, cases[:len(rows)], rows, read_json(within(root, path)), generation_root)
        prefixes[name] = rows
        audit[name] = {"count": len(rows), "sha256": digest}
    return prefixes, {"source_output_root": specification["output_root"], "source_identity_sha256": sha256_file(identity_path),
                      "prefixes": audit, "interruption": specification["interruption"],
                      "prior_continuation": identity.get("continuation"),
                      "model_identity": {key: identity[key] for key in ("base_model", "base_revision", "adapter_sha256")},
                      "inflight_attempt_may_have_been_interrupted": True, "old_files_modified": False}


def run(config_path, audit_only=False):
    config = read_json(config_path)
    cases, source_audit = load_cases(ROOT, config)
    if audit_only:
        return {"status": "DEVELOPMENT_CASES_VERIFIED", **source_audit, "final_test_access": False}
    profiles = {name: load_observation_config(within(ROOT, path), canonical_config_sha256(read_json(within(ROOT, path))))
                for name, path in config["profiles"].items()}
    prefixes, continuation = load_continuation(ROOT, config, cases, source_audit)
    first_messages = [observation_messages("identity-only-placeholder", value) for value in profiles.values()]
    if any(value != first_messages[0] for value in first_messages[1:]):
        raise ValueError("correction-only experiment cannot change the first-stage prompt")
    output = within(ROOT, config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    settings = ReleaseSettings.load(ROOT, within(ROOT, config["release_config"]))
    if continuation is not None and continuation["model_identity"] != {
            "base_model": settings.base_model, "base_revision": settings.base_revision,
            "adapter_sha256": sha256_file(settings.adapter_path / "adapter_model.safetensors")}:
        raise ValueError("cannot change model or adapter during diagnostic continuation")
    write_json_new(output / "identity.json", {**source_audit, "config_sha256": sha256_file(config_path),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "adapter_sha256": sha256_file(settings.adapter_path / "adapter_model.safetensors"),
        "base_model": settings.base_model, "base_revision": settings.base_revision,
        "profile_config_hashes": {name: sha256_file(within(ROOT, path)) for name, path in config["profiles"].items()},
        "runner_sha256": sha256_file(Path(__file__)), "correction_implementation_sha256": sha256_file(ROOT / "src/inference/product_observation.py"),
        "decoder_implementation_sha256": sha256_file(ROOT / "src/inference/observation_constraints.py"),
        "backend_implementation_sha256": sha256_file(ROOT / "src/inference/system_runtime.py"),
        "continuation": continuation,
        "reference_targets_supplied": False, "final_test_access": False, "human_annotation_count": 0})
    with (output / "cases.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    backend = TransformersPeftBackend(settings)
    started = time.perf_counter()
    ok, detail = backend.ready()
    if not ok:
        raise RuntimeError(detail)
    cold_start_ms = (time.perf_counter() - started) * 1000
    results = {}
    for name, observation in profiles.items():
        preserved = prefixes.get(name, [])
        failed = sum(not record["passed"] for record in preserved)
        with backend._execution_lock, backend._model.disable_adapter(), (output / (name + ".jsonl")).open("x", encoding="utf-8", newline="\n") as handle:
            for record in preserved:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            for case in cases[len(preserved):]:
                messages = observation_correction_messages(observation_messages(str(ROOT / case["image_path"]), observation),
                    case["previous_raw"], case["validation_error"], observation)
                started = time.perf_counter()
                response_format = observation_correction_response_format(observation)
                record = {"case_id": case["case_id"], "sample_id": case["sample_id"], "input_messages_sha256": canonical_config_sha256(messages),
                          "response_format_sha256": canonical_config_sha256(response_format)}
                try:
                    generated = backend.generate_with_usage(messages, response_format=response_format, max_new_tokens=observation["max_new_tokens"])
                    record.update(raw_output=generated.content, input_tokens=generated.input_tokens, output_tokens=generated.output_tokens)
                    record["result"] = map_observation(parse_observation(generated.content, observation), observation)
                    record.update(passed=True, error=None)
                except (ValueError, ModelGenerationError) as exc:
                    record.update(passed=False, error=str(exc))
                    failed += 1
                record["elapsed_ms"] = (time.perf_counter() - started) * 1000
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps({"profile": name, "case_id": case["case_id"], "passed": record["passed"]}), flush=True)
        results[name] = {"count": len(cases), "failures": failed, "raw_sha256": sha256_file(output / (name + ".jsonl"))}
    summary = {"status": "COMPLETED", "profiles": results, "cold_start_ms": cold_start_ms,
               "continuation": continuation, "execution_interruptions": int(continuation is not None),
               "case_manifest_sha256": sha256_file(output / "cases.jsonl"), "final_test_access": False,
               "scope": "one_model_correction_per_historical_development_error_not_end_to_end_visual_accuracy"}
    write_json_new(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.audit_only), ensure_ascii=False))

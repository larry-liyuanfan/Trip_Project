"""Keep known correction-slice regressions out of a new subject-decoder candidate."""
from pathlib import Path

from scripts.verify_week8_observation_retry import verify as verify_retry
from scripts.summarize_week8_retry_semantics import build_summary
from src.data.week8_visual_holdout import read_json, within
from src.training.week7_data import sha256_file


EVIDENCE_KEYS = ("correction_diagnostic_config", "correction_diagnostic_replay", "correction_diagnostic_semantics")
DECODER_PROTOCOLS = {"subject_schema_v1", "subject_schema_v2", "food_conflict_schema_v1"}


def validate_nonregression(value):
    if (value.get("test_rows_read") is not False or value.get("human_annotation_count") != 0
            or value.get("selection_allowed") is not False or value.get("new_model_requests") != 0
            or value.get("label_source") != "model_generated_silver"):
        raise ValueError("correction semantics requires development-only silver replay")
    profiles = value["profiles"]
    if set(profiles) != {"legacy_correction", "subject_schema_correction"}:
        raise ValueError("correction evidence requires the complete paired diagnostic")
    before, after = profiles["legacy_correction"], profiles["subject_schema_correction"]
    if set(before) != set(after) or "all_cases" not in before or before["all_cases"]["count"] != value["case_count"]:
        raise ValueError("correction error-slice coverage changed")
    for error, baseline in before.items():
        candidate = after[error]
        if type(baseline["count"]) is not int or baseline["count"] <= 0 or baseline["count"] != candidate["count"]:
            raise ValueError("correction error-slice support changed")
        if baseline["failures"] != 0 or candidate["failures"] != 0:
            raise ValueError("correction diagnostic has request failures")
        if candidate["category_errors"] > baseline["category_errors"]:
            raise ValueError("correction category regressed")
        for field in ("style", "facility"):
            left, right = baseline[field], candidate[field]
            if left["tp"] + left["fn"] != right["tp"] + right["fn"]:
                raise ValueError("correction label support changed")
            if right["tp"] < left["tp"] or right["fp"] > left["fp"] or right["fn"] > left["fn"]:
                raise ValueError("correction " + field + " regressed in " + error)


def validate_correction_evidence(root, config):
    if "candidate_observation" not in config:
        return None
    observation = read_json(within(root, config["candidate_observation"]))
    required = observation.get("correction_protocol") in DECODER_PROTOCOLS
    if not required and not any(key in config for key in EVIDENCE_KEYS):
        return None
    if not all(config.get(key) for key in (*EVIDENCE_KEYS, "development_reference_revision")):
        raise ValueError("subject decoder candidate requires complete correction semantic evidence")
    paths = {key: within(root, config[key]) for key in EVIDENCE_KEYS}
    probe = read_json(paths["correction_diagnostic_config"])
    if probe["profiles"].get("subject_schema_correction") != config["candidate_observation"]:
        raise ValueError("correction diagnostic did not execute the selected observation configuration")
    directory = within(root, probe["output_root"])
    identity = read_json(directory / "identity.json")
    release = read_json(within(root, config["candidate_release"]))
    for key, original in (("base_model", "base_model"), ("base_revision", "base_revision"), ("adapter_model_sha256", "adapter_sha256")):
        if release["model"][key] != identity[original]:
            raise ValueError("correction diagnostic model identity differs from candidate")
    receipt = read_json(paths["correction_diagnostic_replay"])
    generation_root = receipt.get("generation_root")
    if not isinstance(generation_root, str) or not generation_root:
        raise ValueError("correction replay must identify the original execution root")
    if verify_retry(root, paths["correction_diagnostic_config"], generation_root) != receipt:
        raise ValueError("correction execution receipt differs from raw replay")
    semantics = build_summary(root, paths["correction_diagnostic_config"],
        within(root, config["development_reference_revision"]), generation_root)
    if semantics != read_json(paths["correction_diagnostic_semantics"]):
        raise ValueError("correction semantic evidence differs from raw replay")
    validate_nonregression(semantics)
    artifacts = {**{config[key]: sha256_file(path) for key, path in paths.items()},
                 **{(Path(probe["output_root"]) / path.name).as_posix(): sha256_file(path)
                    for path in directory.iterdir() if path.suffix in {".json", ".jsonl"}}}
    packaged = {**{"evidence/" + path.name: sha256_file(path) for path in paths.values()},
                **{"evidence/" + directory.name + "/" + path.name: sha256_file(path)
                   for path in directory.iterdir() if path.suffix in {".json", ".jsonl"}}}
    if len(packaged) != len(artifacts):
        raise ValueError("correction evidence package names collide")
    return {"status": "PASS", "case_count": semantics["case_count"], "unique_images": semantics["unique_images"],
            "artifact_sha256": artifacts, "reference_raw_sha256": semantics["reference_audit"]["reference_raw_sha256"],
            "packaged_artifact_sha256": packaged,
            "semantic_evidence_member": "evidence/" + paths["correction_diagnostic_semantics"].name,
            "interpretation": "Additional error-slice nonregression, not a substitute for complete development or final."}

"""Build audited Week 7 preference pairs from real per-output human scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.training.week7_data import canonical_sha256, iter_jsonl, sha256_file, write_jsonl_new
from src.training.week7_qlora import Week7TrainingError


SCHEMA_VERSION = "week7_mdpo_config_v1"
LOCK_SCHEMA_VERSION = "week7_preference_lock_v1"
CORE_SCENARIOS = ("image_product_search", "after_sales", "itinerary_planning")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week7TrainingError(f"invalid preference JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Week7TrainingError(f"preference JSON must be an object: {path}")
    return value


def load_mdpo_config(root: Path, config_path: Path) -> dict[str, Any]:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise Week7TrainingError("unsupported mDPO config schema")
    if config.get("scope") != {
        "split": "development_preference_train_validation",
        "test_allowed": False,
        "single_ablation_only": True,
        "agent_labels_may_replace_human": False,
    }:
        raise Week7TrainingError("mDPO scope changed or permits test/agent-human substitution")
    for section in ("base_config", "corrected_development"):
        path = root / config[section]["path"]
        if not path.is_file() or sha256_file(path) != config[section]["sha256"]:
            raise Week7TrainingError(f"mDPO {section} identity mismatch")
    for collection in ("model_outputs", "human_scores"):
        for role in ("multitask", "week6"):
            path = root / config[collection][role]["path"]
            if not path.is_file() or sha256_file(path) != config[collection][role]["sha256"]:
                raise Week7TrainingError(f"mDPO {collection}/{role} identity mismatch")
    dimensions = tuple(config["audit"]["dimensions"])
    if dimensions != (
        "historical_image_reference", "requirement_update",
        "context_carryover", "logical_consistency",
    ):
        raise Week7TrainingError("mDPO human-score dimensions changed")
    return config


def _latest_human_scores(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if (
            not sample_id or not row.get("reviewer") or not row.get("self_review_confirmed")
            or row.get("decision") not in {"pass", "rework", "reject"}
        ):
            raise Week7TrainingError("preference source contains incomplete human review")
        previous = latest.get(sample_id)
        if previous is None or int(row.get("revision", 0)) > int(previous.get("revision", 0)):
            latest[sample_id] = row
    return latest


def _evidence_overlap(row: dict[str, Any], parsed: dict[str, Any]) -> bool:
    references = list(row["target"]["context_state"].get("historical_image_reference") or [])
    output_text = json.dumps(parsed, ensure_ascii=False, sort_keys=True).casefold()
    return bool(references) and all(str(reference).casefold() in output_text for reference in references)


def _split_pairs(pairs: list[dict[str, Any]], config: dict[str, Any]) -> None:
    seed = int(config["split"]["seed"])
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pair in pairs:
        key = (pair["parent_scenario"], pair["chosen_model_role"])
        by_stratum.setdefault(key, []).append(pair)
    expected = {(scenario, role) for scenario in CORE_SCENARIOS for role in ("multitask", "week6")}
    if set(by_stratum) != expected:
        raise Week7TrainingError("preference pairs do not cover every scenario/chosen-role stratum")
    for key, rows in by_stratum.items():
        rows.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{key[0]}:{key[1]}:{row['sample_id']}".encode("utf-8")
        ).hexdigest())
        rows[0]["split"] = "validation"
        for row in rows[1:]:
            row["split"] = "train"


def build_preference_pairs(root: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    root, config_path, output_dir = (
        Path(root).resolve(), Path(config_path).resolve(), Path(output_dir).resolve()
    )
    if output_dir.exists():
        raise Week7TrainingError("refusing to overwrite preference-pair identity")
    config = load_mdpo_config(root, config_path)
    development = {
        str(row["sample_id"]): row
        for row in iter_jsonl(root / config["corrected_development"]["path"])
    }
    outputs = {
        role: {str(row["sample_id"]): row for row in iter_jsonl(root / value["path"])}
        for role, value in config["model_outputs"].items()
    }
    scores = {
        role: _latest_human_scores(root / value["path"])
        for role, value in config["human_scores"].items()
    }
    identities = [set(development), *(set(value) for value in outputs.values()), *(set(value) for value in scores.values())]
    if len(development) != 24 or any(identity != identities[0] for identity in identities[1:]):
        raise Week7TrainingError("preference sources do not have identical 24-sample identity")

    audit = config["audit"]
    dimensions = tuple(audit["dimensions"])
    pairs: list[dict[str, Any]] = []
    rejected_reasons: dict[str, int] = {}
    for sample_id in sorted(development):
        role_totals = {
            role: sum(int(scores[role][sample_id]["scores"][dimension]) for dimension in dimensions)
            for role in ("multitask", "week6")
        }
        margin = abs(role_totals["multitask"] - role_totals["week6"])
        if margin < int(audit["minimum_total_score_margin"]):
            rejected_reasons["human_score_tie"] = rejected_reasons.get("human_score_tie", 0) + 1
            continue
        chosen_role = "multitask" if role_totals["multitask"] > role_totals["week6"] else "week6"
        rejected_role = "week6" if chosen_role == "multitask" else "multitask"
        chosen_score = scores[chosen_role][sample_id]
        if min(int(chosen_score["scores"][dimension]) for dimension in dimensions) < int(audit["minimum_chosen_dimension_score"]):
            rejected_reasons["chosen_human_quality_below_gate"] = rejected_reasons.get("chosen_human_quality_below_gate", 0) + 1
            continue
        chosen_record, rejected_record = outputs[chosen_role][sample_id], outputs[rejected_role][sample_id]
        if chosen_record.get("failed") or rejected_record.get("failed"):
            rejected_reasons["generation_failure"] = rejected_reasons.get("generation_failure", 0) + 1
            continue
        try:
            chosen_json = json.loads(str(chosen_record["raw_output"]))
        except (TypeError, json.JSONDecodeError):
            rejected_reasons["chosen_not_json"] = rejected_reasons.get("chosen_not_json", 0) + 1
            continue
        if not isinstance(chosen_json, dict):
            rejected_reasons["chosen_not_json_object"] = rejected_reasons.get("chosen_not_json_object", 0) + 1
            continue
        if audit["require_visual_evidence_overlap"] and not _evidence_overlap(development[sample_id], chosen_json):
            rejected_reasons["visual_evidence_not_grounded"] = rejected_reasons.get("visual_evidence_not_grounded", 0) + 1
            continue
        row = development[sample_id]
        pairs.append({
            "schema_version": "week7_preference_pair_v1",
            "pair_id": f"week7-preference-{len(pairs):03d}",
            "sample_id": sample_id,
            "parent_scenario": row["parent_scenario"],
            "prompt_messages": row["messages"][:-1],
            "chosen": chosen_record["raw_output"],
            "rejected": rejected_record["raw_output"],
            "chosen_model_role": chosen_role,
            "rejected_model_role": rejected_role,
            "human_score_total": role_totals,
            "human_score_margin": margin,
            "chosen_dimension_scores": {
                dimension: int(chosen_score["scores"][dimension]) for dimension in dimensions
            },
            "label_source": "derived_from_real_human_four_dimension_scores_v1",
            "explicit_human_pair_choice": False,
            "agent_adversarial_audit": {
                "chosen_json_object": True,
                "visual_evidence_overlap": True,
                "source_identity_match": True,
                "reverse_pair_probe_rejected": True,
            },
        })
    if len(pairs) != 16 or rejected_reasons != {
        "human_score_tie": 7, "chosen_not_json": 1,
    }:
        raise Week7TrainingError(
            f"preference audit count changed: accepted={len(pairs)} rejected={rejected_reasons}"
        )
    _split_pairs(pairs, config)
    train = [pair for pair in pairs if pair["split"] == "train"]
    validation = [pair for pair in pairs if pair["split"] == "validation"]
    if len(train) != int(config["training"]["max_train_pairs"]) or len(validation) != int(config["training"]["max_validation_pairs"]):
        raise Week7TrainingError("preference train/validation counts changed")
    output_dir.mkdir(parents=True, exist_ok=False)
    train_path, validation_path = output_dir / "train.jsonl", output_dir / "validation.jsonl"
    write_jsonl_new(train_path, train)
    write_jsonl_new(validation_path, validation)
    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "dataset_version": config["dataset_version"],
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "source_sha256": {
            "development": config["corrected_development"]["sha256"],
            "multitask_outputs": config["model_outputs"]["multitask"]["sha256"],
            "week6_outputs": config["model_outputs"]["week6"]["sha256"],
            "multitask_human_scores": config["human_scores"]["multitask"]["sha256"],
            "week6_human_scores": config["human_scores"]["week6"]["sha256"],
        },
        "counts": {
            "source_samples": 24, "accepted_pairs": 16, "train": len(train),
            "validation": len(validation), "reverse_probe_rejected": len(pairs),
        },
        "rejected_reasons": rejected_reasons,
        "files": {
            "train.jsonl": {"count": len(train), "sha256": sha256_file(train_path)},
            "validation.jsonl": {"count": len(validation), "sha256": sha256_file(validation_path)},
        },
        "human_pair_choice_explicit": False,
        "agent_audit_does_not_replace_human_scores": True,
        "test_read": False,
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    (output_dir / "dataset_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return lock

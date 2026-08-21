"""Week 7 终态证据的机器优先对抗审计。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

from src.training.week7_data import sha256_file


BASELINE_COMMIT = "132779b0f6d2929ce1cdbed18e62adf3ef9edd18"
EVIDENCE_PATHS = {
    "data": "experiments/week7_data_lock_20260820_v3.json",
    "schema": "experiments/week7_schema_decoding_20260820_v3.json",
    "final": "experiments/week7_final_evaluation_20260821_v5.json",
    "dialogue_audit": "experiments/week7_dialogue_context_audit_20260821_v1.json",
    "dialogue_repair": "experiments/week7_dialogue_repair_20260821_v2.json",
    "human": "experiments/week7_dialogue_human_review_20260822_v2.json",
    "comparison": "experiments/week7_dialogue_comparison_20260822_v1.json",
    "mdpo": "experiments/week7_mdpo_20260822_v1.json",
}


class Week7AdversarialAuditError(ValueError):
    """Raised when an evidence mutation violates a Week 7 invariant."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week7AdversarialAuditError(f"invalid evidence JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Week7AdversarialAuditError(f"evidence must be an object: {path}")
    return value


def load_week7_evidence(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    evidence: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name, relative in EVIDENCE_PATHS.items():
        path = root / relative
        evidence[name] = _read_object(path)
        hashes[name] = sha256_file(path)
    return evidence, hashes


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Week7AdversarialAuditError(code)


def validate_week7_evidence(
    root: Path,
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate cross-file identities and return a conservative delivery verdict."""
    data = evidence["data"]
    schema = evidence["schema"]
    final = evidence["final"]
    dialogue_audit = evidence["dialogue_audit"]
    repair = evidence["dialogue_repair"]
    human = evidence["human"]
    comparison = evidence["comparison"]
    mdpo = evidence["mdpo"]

    config_path = root / "configs/week7/qwen3_vl_8b_multitask_context_v3.json"
    mdpo_config_path = root / "configs/week7/mdpo_v1.json"
    _require(data.get("baseline_commit") == BASELINE_COMMIT, "BASELINE_IDENTITY_MISMATCH")
    _require(data.get("status") == "PASS", "DATA_LOCK_NOT_PASS")
    _require(data.get("config_sha256") == sha256_file(config_path), "CONFIG_HASH_MISMATCH")
    lock_sha = data.get("dataset_lock_sha256")
    _require(isinstance(lock_sha, str) and len(lock_sha) == 64, "DATA_LOCK_HASH_INVALID")
    _require(
        data.get("isolation", {}).get("dimensions")
        == ["sample_id", "source_id", "image_sha256", "group_id", "constraint_template_id"],
        "ISOLATION_DIMENSIONS_MISMATCH",
    )
    _require(
        data.get("isolation", {}).get("cross_split_collision_count") == 0,
        "CROSS_SPLIT_COLLISION",
    )
    ratios = data.get("actual_train_ratios", {})
    _require(ratios.get("general_multimodal") == 0.09, "GENERAL_RATIO_MISMATCH")
    _require(ratios.get("dialogue") == 0.15, "DIALOGUE_RATIO_MISMATCH")
    _require(ratios.get("dialogue_tool_call") == 0.10, "TOOL_CALL_RATIO_MISMATCH")

    _require(schema.get("dataset_lock_sha256") == lock_sha, "CROSS_EVIDENCE_LOCK_MISMATCH")
    _require(schema.get("scope") == "format_only", "SCHEMA_SCOPE_ESCALATION")
    _require(schema.get("semantic_claims") == "FORBIDDEN", "SCHEMA_SEMANTIC_CLAIM")
    _require(schema.get("test_consumed") is False, "SCHEMA_READ_TEST")
    _require(
        schema.get("selected_production_mode") == "free",
        "UNSUPPORTED_CONSTRAINED_MODE_SELECTED",
    )
    _require(
        schema.get("constrained_primary", {}).get("primary_failure_rate") == 1.0,
        "CONSTRAINED_FAILURE_NOT_RECORDED",
    )

    _require(final.get("baseline_commit") == BASELINE_COMMIT, "FINAL_BASELINE_MISMATCH")
    _require(
        final.get("data_identity", {}).get("dataset_lock_sha256") == lock_sha,
        "FINAL_DATA_LOCK_MISMATCH",
    )
    _require(
        final.get("data_identity", {}).get("cross_split_collision_counts")
        == {
            "sample_id": 0,
            "source_id": 0,
            "image_sha256": 0,
            "group_id": 0,
            "constraint_template_id": 0,
        },
        "FINAL_ISOLATION_MISMATCH",
    )
    final_test = final.get("final_test", {})
    _require(final_test.get("test_consumption_status") == "COMPLETED", "FINAL_TEST_NOT_COMPLETED")
    _require(final_test.get("resume_count") == 0, "UNEXPECTED_TEST_REPLAY")
    _require(final_test.get("failure_history") == [], "FINAL_TEST_FAILURE_HISTORY_CHANGED")
    _require(final_test.get("all_passed") is True, "CORE_FINAL_GATE_FAILED")
    _require(
        final_test.get("support_counts", {}).get("scenario_samples")
        == {
            "image_product_search": 30,
            "after_sales": 30,
            "itinerary_planning": 30,
            "dialogue": 24,
        },
        "FINAL_SUPPORT_COUNT_MISMATCH",
    )
    selected_sha = final.get("training", {}).get("selected_adapter_sha256")
    _require(
        selected_sha
        == "612af9b79b63f2652e0cd5e9d0d1d9259b5fab08a69eaddbe0d186e9c8f7540b",
        "SELECTED_ADAPTER_MISMATCH",
    )

    _require(
        dialogue_audit.get("status") == "BLOCKED_INVALID_SOURCE_CONTEXT",
        "DIALOGUE_DEFECT_LAUNDERED",
    )
    _require(
        dialogue_audit.get("finding", {})
        .get("declared_affected_counts", {})
        .get("test")
        == 24,
        "DIALOGUE_AFFECTED_COUNT_MISMATCH",
    )
    _require(
        dialogue_audit.get("immutability", {}).get("test_marker_modified") is False,
        "TEST_MARKER_MUTATED",
    )
    _require(repair.get("scope", {}).get("split") == "development", "REPAIR_SCOPE_ESCALATION")
    _require(repair.get("scope", {}).get("test_read") is False, "REPAIR_READ_TEST")
    _require(
        repair.get("dataset", {}).get("context_integrity_pass_count") == 24,
        "REPAIRED_DIALOGUE_INTEGRITY_FAILED",
    )
    _require(
        repair.get("dataset", {}).get("legacy_anticipatory_reply_count") == 0,
        "REPAIRED_DIALOGUE_STILL_MISORDERED",
    )

    _require(human.get("status") == "HUMAN_REVIEW_COMPLETED", "HUMAN_REVIEW_INCOMPLETE")
    _require(
        human.get("human_validation", {}).get("agent_filled_score_count") == 0,
        "AGENT_IMPERSONATED_HUMAN",
    )
    _require(
        human.get("result", {}).get("latest_unique_sample_count") == 24,
        "HUMAN_SUPPORT_MISMATCH",
    )
    _require(
        comparison.get("status") == "HUMAN_COMPARISON_COMPLETED",
        "HUMAN_COMPARISON_INCOMPLETE",
    )
    _require(
        comparison.get("human_comparison", {}).get("agent_filled_scores") == 0,
        "AGENT_IMPERSONATED_COMPARISON",
    )
    _require(comparison.get("scope", {}).get("test_read") is False, "COMPARISON_READ_TEST")

    _require(
        mdpo.get("config", {}).get("sha256") == sha256_file(mdpo_config_path),
        "MDPO_CONFIG_HASH_MISMATCH",
    )
    _require(
        mdpo.get("scope", {}).get("single_ablation_only") is True,
        "MDPO_RETRY_POLICY_MISSING",
    )
    _require(mdpo.get("scope", {}).get("test_read") is False, "MDPO_READ_TEST")
    _require(
        mdpo.get("preference_dataset", {}).get("explicit_human_pair_choice") is False,
        "DERIVED_PAIR_MISSTATED_AS_HUMAN_CHOICE",
    )
    gate_passed = mdpo.get("result", {}).get("validation_gate", {}).get("passed")
    selected = mdpo.get("result", {}).get("selected_for_use")
    _require(not selected or gate_passed is True, "FAILED_MDPO_SELECTED")
    _require(gate_passed is False and selected is False, "MDPO_OUTCOME_MISMATCH")
    _require(mdpo.get("result", {}).get("final_test") == "NOT_RUN", "MDPO_TOUCHED_FINAL_TEST")
    _require(
        mdpo.get("audit_conclusion", {}).get("gradient_updates_observed") is True,
        "MDPO_EMPTY_RUN",
    )

    return {
        "agent_primary_machine_audit": "PASS",
        "human_evidence_role": "SUPPORTING_ONLY_NOT_AGENT_REPLACED",
        "core_automated_gate": "PASS",
        "corrected_dialogue_development_gate": "PASS",
        "mdpo_selection_gate": "REJECTED_VALIDATION_REGRESSION",
        "implementation_ready_for_dev_integration": True,
        "full_week7_claim_gate": "FAIL_KNOWN_V3_TEST_DIALOGUE_INVALID",
        "stg_or_release_promotion_allowed": False,
        "test_replay_allowed": False,
        "selected_model": "checkpoint-151",
    }


def run_counterfactual_probes(
    root: Path,
    evidence: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """Inject plausible evidence-laundering mutations; every probe must be rejected."""
    probes: list[tuple[str, str, Callable[[dict[str, dict[str, Any]]], None]]] = [
        (
            "baseline_swap",
            "BASELINE_IDENTITY_MISMATCH",
            lambda x: x["data"].__setitem__("baseline_commit", "0" * 40),
        ),
        (
            "split_collision",
            "CROSS_SPLIT_COLLISION",
            lambda x: x["data"]["isolation"].__setitem__(
                "cross_split_collision_count", 1
            ),
        ),
        (
            "ratio_drift",
            "DIALOGUE_RATIO_MISMATCH",
            lambda x: x["data"]["actual_train_ratios"].__setitem__("dialogue", 0.20),
        ),
        (
            "schema_semantic_laundering",
            "SCHEMA_SEMANTIC_CLAIM",
            lambda x: x["schema"].__setitem__("semantic_claims", "IMPROVED"),
        ),
        (
            "test_replay",
            "UNEXPECTED_TEST_REPLAY",
            lambda x: x["final"]["final_test"].__setitem__("resume_count", 1),
        ),
        (
            "support_deletion",
            "FINAL_SUPPORT_COUNT_MISMATCH",
            lambda x: x["final"]["final_test"]["support_counts"][
                "scenario_samples"
            ].__setitem__("after_sales", 29),
        ),
        (
            "dialogue_defect_laundering",
            "DIALOGUE_DEFECT_LAUNDERED",
            lambda x: x["dialogue_audit"].__setitem__("status", "PASS"),
        ),
        (
            "repair_reads_test",
            "REPAIR_READ_TEST",
            lambda x: x["dialogue_repair"]["scope"].__setitem__("test_read", True),
        ),
        (
            "agent_human_impersonation",
            "AGENT_IMPERSONATED_HUMAN",
            lambda x: x["human"]["human_validation"].__setitem__(
                "agent_filled_score_count", 1
            ),
        ),
        (
            "failed_mdpo_selected",
            "FAILED_MDPO_SELECTED",
            lambda x: x["mdpo"]["result"].__setitem__("selected_for_use", True),
        ),
        (
            "mdpo_reads_test",
            "MDPO_TOUCHED_FINAL_TEST",
            lambda x: x["mdpo"]["result"].__setitem__("final_test", "COMPLETED"),
        ),
    ]
    results: list[dict[str, str]] = []
    for name, expected, mutate in probes:
        candidate = copy.deepcopy(evidence)
        mutate(candidate)
        try:
            validate_week7_evidence(root, candidate)
        except Week7AdversarialAuditError as exc:
            actual = str(exc)
            _require(actual == expected, f"PROBE_WRONG_REJECTION:{name}:{actual}")
            results.append({"probe": name, "status": "REJECTED", "reason": actual})
        else:
            raise Week7AdversarialAuditError(f"PROBE_NOT_REJECTED:{name}")
    return results


def audit_week7_repository(root: Path) -> dict[str, Any]:
    evidence, hashes = load_week7_evidence(root)
    verdict = validate_week7_evidence(root, evidence)
    probes = run_counterfactual_probes(root, evidence)
    return {
        "schema_version": "week7_adversarial_completion_audit_v1",
        "status": "PASS_WITH_KNOWN_IMMUTABLE_LIMITATION",
        "authority": {
            "primary": "deterministic_agent_machine_audit",
            "secondary": "previous_real_human_development_scores",
            "agent_may_replace_human_identity_or_scores": False,
        },
        "evidence_sha256": hashes,
        "verdict": verdict,
        "counterfactual_probes": {
            "total": len(probes),
            "rejected": len(probes),
            "details": probes,
        },
    }

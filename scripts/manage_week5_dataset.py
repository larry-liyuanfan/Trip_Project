"""Command-line entry points for the Week 5 annotation and QC workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.week5_dataset import (
    SCENARIOS,
    Week5DataError,
    build_sample_pools,
    export_annotation_packet,
    initialize_workflow_v2_sidecar,
    load_week5_config,
    validate_pools,
    validate_workflow_v2_sidecar,
    workflow_summary,
)
from src.data.week5_workflow import (
    apply_dialogue_validation,
    apply_human_corrections,
    apply_quality_records,
    export_audited_pilot_annotation_packet,
    export_quality_packet,
    generate_dialogue_candidates,
    run_itinerary_paired_prompt_pilot,
    run_full_preannotation,
    run_preannotation,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Manage Week 5 instruction data")
    result.add_argument("--config", default="configs/week5_dataset.json")
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build-pools")
    subparsers.add_parser("validate-pools")
    pre = subparsers.add_parser("preannotate")
    pre.add_argument("--scenario", choices=SCENARIOS, required=True)
    pre.add_argument("--limit", type=int)
    pre.add_argument("--retry-failures", action="store_true")
    pre_all = subparsers.add_parser("preannotate-all")
    pre_all.add_argument("--run-id", required=True)
    pre_all.add_argument("--resume", action="store_true")
    pre_all.add_argument("--retry-failures", action="store_true")
    workflow = subparsers.add_parser("init-workflow-v2")
    workflow.add_argument("--scenario", choices=SCENARIOS, required=True)
    workflow_validate = subparsers.add_parser("validate-workflow-v2")
    workflow_validate.add_argument("--scenario", choices=SCENARIOS, required=True)
    pilot = subparsers.add_parser("pilot-itinerary-prompts")
    pilot.add_argument("--run-id", required=True)
    pilot.add_argument("--limit", type=int, default=30)
    pilot.add_argument("--resume", action="store_true")
    pilot_export = subparsers.add_parser("export-pilot-annotations")
    pilot_export.add_argument("--run-id", required=True)
    pilot_export.add_argument("--output", type=Path, required=True)
    export = subparsers.add_parser("export-annotations")
    export.add_argument("--scenario", choices=SCENARIOS, required=True)
    export.add_argument("--output", type=Path, required=True)
    human = subparsers.add_parser("apply-human")
    human.add_argument("--scenario", choices=SCENARIOS, required=True)
    human.add_argument("--input", type=Path, required=True)
    quality_export = subparsers.add_parser("export-quality")
    quality_export.add_argument("--scenario", choices=SCENARIOS, required=True)
    quality_export.add_argument(
        "--stage", choices=("cross_review", "core_audit"), required=True
    )
    quality_export.add_argument("--output", type=Path, required=True)
    quality = subparsers.add_parser("apply-quality")
    quality.add_argument("--scenario", choices=SCENARIOS, required=True)
    quality.add_argument("--input", type=Path, required=True)
    dialogues = subparsers.add_parser("generate-dialogues")
    dialogues.add_argument("--limit", type=int)
    dialogues.add_argument("--run-id", required=True)
    dialogue_qc = subparsers.add_parser("apply-dialogue-quality")
    dialogue_qc.add_argument("--input", type=Path, required=True)
    dialogue_qc.add_argument("--run-id", required=True)
    subparsers.add_parser("report")
    return result


def main() -> None:
    args = parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_week5_config(root, args.config)
    try:
        if args.command == "build-pools":
            payload = build_sample_pools(root, config)
        elif args.command == "validate-pools":
            payload = validate_pools(root, config)
        elif args.command == "preannotate":
            payload = run_preannotation(root, config, args.scenario, limit=args.limit, retry_failures=args.retry_failures)
        elif args.command == "preannotate-all":
            payload = run_full_preannotation(
                root, config, args.run_id, resume=args.resume,
                retry_failures=args.retry_failures,
            )
        elif args.command == "init-workflow-v2":
            payload = initialize_workflow_v2_sidecar(root, config, args.scenario)
        elif args.command == "validate-workflow-v2":
            payload = validate_workflow_v2_sidecar(root, config, args.scenario)
        elif args.command == "pilot-itinerary-prompts":
            payload = run_itinerary_paired_prompt_pilot(
                root, config, args.run_id, limit=args.limit, resume=args.resume
            )
        elif args.command == "export-pilot-annotations":
            payload = export_audited_pilot_annotation_packet(
                root, config, args.run_id, args.output
            )
        elif args.command == "export-annotations":
            payload = {"exported": export_annotation_packet(root, config, args.scenario, args.output)}
        elif args.command == "apply-human":
            payload = apply_human_corrections(root, config, args.scenario, args.input)
        elif args.command == "export-quality":
            payload = export_quality_packet(
                root, config, args.scenario, args.stage, args.output
            )
        elif args.command == "apply-quality":
            payload = apply_quality_records(root, config, args.scenario, args.input)
        elif args.command == "generate-dialogues":
            payload = generate_dialogue_candidates(
                root, config, run_id=args.run_id, limit=args.limit
            )
        elif args.command == "apply-dialogue-quality":
            payload = apply_dialogue_validation(
                root, config, args.input, run_id=args.run_id
            )
        else:
            payload = workflow_summary(root, config)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    except Week5DataError as exc:
        raise SystemExit(f"Week 5 workflow error: {exc}") from exc


if __name__ == "__main__":
    main()

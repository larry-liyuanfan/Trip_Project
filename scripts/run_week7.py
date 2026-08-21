#!/usr/bin/env python3
"""Run locked Week 7 development experiments and multitask training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.week6_qlora import environment_report
from src.training.week7_inference import (
    combine_week6_development_baseline,
    run_schema_experiment,
    run_transformers_development,
    run_week6_dialogue_development,
)
from src.training.week7_final_evaluation import (
    create_parameter_lock,
    recover_interrupted_final_test,
    run_final_test_suite,
)
from src.training.week7_dialogue_repair import (
    build_dialogue_review_v2,
    run_dialogue_week6_baseline_v1,
    run_dialogue_review_v2,
)
from src.training.week7_qlora import Week7TrainingError, run_multitask_training
from src.training.week7_latency_protocol import run_latency_protocol_v4
from src.training.week7_selection import select_development_checkpoint


def build_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v3.json")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("check-environment")
    train = commands.add_parser("train-multitask")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--confirm-dataset-lock", action="store_true")
    train.add_argument("--resume-from-checkpoint", type=Path)
    infer = commands.add_parser("evaluate-development")
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--run-id", required=True)
    infer.add_argument("--adapter-dir", type=Path)
    infer.add_argument("--model-role", default="zero_shot")
    infer.add_argument("--scenario", choices=("image_product_search", "after_sales", "itinerary_planning"))
    infer.add_argument("--max-new-tokens", type=int, default=2048)
    dialogue = commands.add_parser("evaluate-week6-dialogue-development")
    dialogue.add_argument("--output-dir", type=Path, required=True)
    dialogue.add_argument("--product-adapter", type=Path, required=True)
    dialogue.add_argument("--after-sales-adapter", type=Path, required=True)
    dialogue.add_argument("--itinerary-adapter", type=Path, required=True)
    dialogue.add_argument("--max-new-tokens", type=int, default=2048)
    schema = commands.add_parser("schema-experiment")
    schema.add_argument("--output-dir", type=Path, required=True)
    schema.add_argument("--endpoint", required=True)
    schema.add_argument("--served-model", required=True)
    schema.add_argument("--timeout", type=int, default=300)
    combine = commands.add_parser("combine-week6-development")
    combine.add_argument("--product-metrics", type=Path, required=True)
    combine.add_argument("--after-sales-metrics", type=Path, required=True)
    combine.add_argument("--itinerary-metrics", type=Path, required=True)
    combine.add_argument("--dialogue-metrics", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    select = commands.add_parser("select-checkpoint")
    select.add_argument("--training-dir", type=Path, required=True)
    select.add_argument("--training-summary", type=Path, required=True)
    select.add_argument("--week6-baseline", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--latency-protocol", type=Path)
    for command in ("latency-protocol-v4", "latency-protocol-v5"):
        latency = commands.add_parser(command)
        latency.add_argument("--output-dir", type=Path, required=True)
        latency.add_argument("--protocol-config", type=Path, required=True)
        latency.add_argument("--training-dir", type=Path, required=True)
        latency.add_argument("--training-summary", type=Path, required=True)
        latency.add_argument("--week6-baseline", type=Path, required=True)
        latency.add_argument("--week6-baseline-evidence", type=Path, required=True)
        latency.add_argument("--week6-product-adapter", type=Path, required=True)
        latency.add_argument("--week6-after-sales-adapter", type=Path, required=True)
        latency.add_argument("--week6-itinerary-adapter", type=Path, required=True)
    lock = commands.add_parser("lock-parameters")
    lock.add_argument("--output", type=Path, required=True)
    lock.add_argument("--training-summary", type=Path, required=True)
    lock.add_argument("--selection", type=Path, required=True)
    lock.add_argument("--selected-checkpoint", type=Path, required=True)
    lock.add_argument("--week6-product-adapter", type=Path, required=True)
    lock.add_argument("--week6-product-sha256", required=True)
    lock.add_argument("--week6-after-sales-adapter", type=Path, required=True)
    lock.add_argument("--week6-after-sales-sha256", required=True)
    lock.add_argument("--week6-itinerary-adapter", type=Path, required=True)
    lock.add_argument("--week6-itinerary-sha256", required=True)
    lock.add_argument("--development-week6-baseline", type=Path, required=True)
    lock.add_argument("--development-zero-shot", type=Path, required=True)
    lock.add_argument("--development-multitask", type=Path, required=True)
    lock.add_argument("--schema-comparison", type=Path, required=True)
    lock.add_argument("--schema-decoding-mode", choices=("free", "constrained"), required=True)
    lock.add_argument("--max-new-tokens", type=int, default=2048)
    final_test = commands.add_parser("final-test")
    final_test.add_argument("--parameter-lock", type=Path, required=True)
    final_test.add_argument("--output-dir", type=Path, required=True)
    final_test.add_argument("--resume", action="store_true")
    recover_test = commands.add_parser("recover-final-test")
    recover_test.add_argument("--parameter-lock", type=Path, required=True)
    recover_test.add_argument("--output-dir", type=Path, required=True)
    recover_test.add_argument("--slurm-job-id", required=True)
    recover_test.add_argument("--slurm-job-state", required=True)
    build_dialogue = commands.add_parser("build-dialogue-review-v2")
    build_dialogue.add_argument(
        "--review-config", type=Path,
        default=ROOT / "configs/week7/dialogue_review_v2.json",
    )
    build_dialogue.add_argument("--output-dir", type=Path)
    run_dialogue = commands.add_parser("dialogue-review-v2")
    run_dialogue.add_argument(
        "--review-config", type=Path,
        default=ROOT / "configs/week7/dialogue_review_v2.json",
    )
    run_dialogue.add_argument("--dataset-dir", type=Path, required=True)
    run_dialogue.add_argument("--adapter-dir", type=Path, required=True)
    run_dialogue.add_argument("--output-dir", type=Path, required=True)
    compare_dialogue = commands.add_parser("dialogue-week6-baseline-v1")
    compare_dialogue.add_argument(
        "--comparison-config", type=Path,
        default=ROOT / "configs/week7/dialogue_comparison_v1.json",
    )
    compare_dialogue.add_argument("--dataset-dir", type=Path, required=True)
    compare_dialogue.add_argument("--product-adapter", type=Path, required=True)
    compare_dialogue.add_argument("--after-sales-adapter", type=Path, required=True)
    compare_dialogue.add_argument("--itinerary-adapter", type=Path, required=True)
    compare_dialogue.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "check-environment":
            payload = environment_report(require_cuda=True)
        elif args.command == "train-multitask":
            payload = run_multitask_training(
                ROOT, args.config, args.output_dir,
                confirm_dataset_lock=args.confirm_dataset_lock,
                resume_from_checkpoint=args.resume_from_checkpoint,
            )
        elif args.command == "evaluate-development":
            payload = run_transformers_development(
                ROOT, args.config, args.output_dir, run_id=args.run_id,
                adapter_dir=args.adapter_dir, model_role=args.model_role,
                max_new_tokens=args.max_new_tokens, scenario=args.scenario,
            )
        elif args.command == "schema-experiment":
            payload = run_schema_experiment(
                ROOT, args.config, args.output_dir, endpoint=args.endpoint,
                served_model=args.served_model, timeout=args.timeout,
            )
        elif args.command == "evaluate-week6-dialogue-development":
            payload = run_week6_dialogue_development(
                ROOT, args.config, args.output_dir,
                adapter_dirs={
                    "image_product_search": args.product_adapter,
                    "after_sales": args.after_sales_adapter,
                    "itinerary_planning": args.itinerary_adapter,
                },
                max_new_tokens=args.max_new_tokens,
            )
        elif args.command == "combine-week6-development":
            payload = combine_week6_development_baseline(
                args.config,
                {
                    "image_product_search": args.product_metrics,
                    "after_sales": args.after_sales_metrics,
                    "itinerary_planning": args.itinerary_metrics,
                },
                args.dialogue_metrics,
                args.output,
            )
        elif args.command == "select-checkpoint":
            payload = select_development_checkpoint(
                args.config, args.training_dir, args.training_summary,
                args.week6_baseline, args.output,
                latency_protocol_path=args.latency_protocol,
            )
        elif args.command in {"latency-protocol-v4", "latency-protocol-v5"}:
            payload = run_latency_protocol_v4(
                ROOT, args.config, args.output_dir,
                protocol_config_path=args.protocol_config,
                training_dir=args.training_dir,
                training_summary_path=args.training_summary,
                week6_baseline_path=args.week6_baseline,
                week6_baseline_evidence_path=args.week6_baseline_evidence,
                week6_adapters={
                    "image_product_search": args.week6_product_adapter,
                    "after_sales": args.week6_after_sales_adapter,
                    "itinerary_planning": args.week6_itinerary_adapter,
                },
            )
        elif args.command == "lock-parameters":
            payload = create_parameter_lock(
                ROOT, args.config, args.output,
                training_summary_path=args.training_summary,
                selection_path=args.selection,
                selected_checkpoint=args.selected_checkpoint,
                week6_adapters={
                    "image_product_search": (args.week6_product_adapter, args.week6_product_sha256),
                    "after_sales": (args.week6_after_sales_adapter, args.week6_after_sales_sha256),
                    "itinerary_planning": (args.week6_itinerary_adapter, args.week6_itinerary_sha256),
                },
                development_evidence={
                    "week6_development_baseline": args.development_week6_baseline,
                    "zero_shot_development": args.development_zero_shot,
                    "multitask_development": args.development_multitask,
                    "schema_decoding": args.schema_comparison,
                },
                schema_decoding_mode=args.schema_decoding_mode,
                max_new_tokens=args.max_new_tokens,
            )
        elif args.command == "final-test":
            payload = run_final_test_suite(
                ROOT, args.config, args.parameter_lock, args.output_dir,
                resume=args.resume,
            )
        elif args.command == "recover-final-test":
            payload = recover_interrupted_final_test(
                ROOT, args.config, args.parameter_lock, args.output_dir,
                slurm_job_id=args.slurm_job_id,
                slurm_job_state=args.slurm_job_state,
            )
        elif args.command == "build-dialogue-review-v2":
            payload = build_dialogue_review_v2(
                ROOT, args.review_config, args.output_dir,
            )
        elif args.command == "dialogue-review-v2":
            payload = run_dialogue_review_v2(
                ROOT, args.config, args.review_config, args.dataset_dir,
                args.adapter_dir, args.output_dir,
            )
        else:
            payload = run_dialogue_week6_baseline_v1(
                ROOT, args.config, args.comparison_config, args.dataset_dir,
                {
                    "image_product_search": args.product_adapter,
                    "after_sales": args.after_sales_adapter,
                    "itinerary_planning": args.itinerary_adapter,
                },
                args.output_dir,
            )
    except (OSError, ValueError, Week7TrainingError) as exc:
        raise SystemExit(f"Week 7 execution error: {exc}") from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

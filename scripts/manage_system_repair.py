#!/usr/bin/env python3
"""Operate immutable system-repair data, pilots, training, and gates."""

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.system_prompt_pilot import (
    load_completed_prompt_pilot,
    run_prompt_pilot,
)
from src.inference.system_runtime import (
    ReleaseSettings,
    ScenarioService,
    TransformersPeftBackend,
)
from src.training.system_repair import (
    build_week5_repair_v2,
    evaluate_system_release_gates,
    merge_week5_repair_results,
    run_week5_repair_queue,
    select_system_repair_candidate,
)
from src.training.week7_data import (
    build_week7_lock,
    clone_week7_lock_for_runtime_config,
    load_week7_config,
    sha256_file,
)
from src.training.week7_inference import run_system_repair_test_once
from src.training.week7_qlora import run_multitask_training


DEFAULT_REPAIR = ROOT / "configs/system_repair/qwen3_vl_8b_system_repair_v1.json"
DEFAULT_WEEK5 = ROOT / "configs/system_repair/week5_preannotation_repair_v2.json"
DEFAULT_PROMPTS = ROOT / "configs/system_repair/prompt_candidates_v1.json"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", type=Path, default=DEFAULT_REPAIR)
    sub = command.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-week5-repair")
    run = sub.add_parser("run-week5-repair")
    run.add_argument("--run-id", required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--resume", action="store_true")
    merge = sub.add_parser("merge-week5-repair")
    merge.add_argument("--run-id", required=True)
    lock = sub.add_parser("build-data-lock")
    lock.add_argument("--source-root", type=Path, default=ROOT)
    clone = sub.add_parser("clone-data-lock")
    clone.add_argument("--source-config", type=Path, required=True)
    pilot = sub.add_parser("prompt-pilot")
    pilot.add_argument("--output-dir", type=Path, required=True)
    pilot.add_argument("--endpoint", required=True)
    pilot.add_argument("--served-model", required=True)
    combined = sub.add_parser("run-inference-repair")
    combined.add_argument("--output-dir", type=Path, required=True)
    combined.add_argument("--run-id", required=True)
    combined.add_argument("--adapter-dir", type=Path, required=True)
    combined.add_argument("--resume", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--resume-from-checkpoint", type=Path)
    select = sub.add_parser("select-candidate")
    select.add_argument("--training-dir", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    gate = sub.add_parser("evaluate-gates")
    gate.add_argument("--candidate", type=Path, required=True)
    gate.add_argument("--existing", type=Path, required=True)
    gate.add_argument("--zero-shot", type=Path, required=True)
    gate.add_argument("--week6-routed", type=Path, required=True)
    gate.add_argument("--selection", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    final_test = sub.add_parser("run-final-test")
    final_test.add_argument("--selection", type=Path, required=True)
    final_test.add_argument("--gate", type=Path, required=True)
    final_test.add_argument("--output-dir", type=Path, required=True)
    sub.add_parser("validate-config")
    return command


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare-week5-repair":
        result = build_week5_repair_v2(ROOT, DEFAULT_WEEK5)
    elif args.command == "run-week5-repair":
        result = run_week5_repair_queue(
            ROOT,
            DEFAULT_WEEK5,
            run_id=args.run_id,
            base_url=args.base_url,
            resume=args.resume,
        )
    elif args.command == "merge-week5-repair":
        result = merge_week5_repair_results(ROOT, DEFAULT_WEEK5, run_id=args.run_id)
    elif args.command == "build-data-lock":
        output = build_week7_lock(ROOT, args.source_root, args.config)
        result = {"status": "COMPLETED", "output": str(output)}
    elif args.command == "clone-data-lock":
        result = clone_week7_lock_for_runtime_config(
            ROOT,
            args.source_config,
            args.config,
        )
    elif args.command == "prompt-pilot":
        result = run_prompt_pilot(
            ROOT,
            args.config,
            DEFAULT_PROMPTS,
            args.output_dir,
            endpoint=args.endpoint,
            served_model=args.served_model,
        )
    elif args.command == "run-inference-repair":
        os.environ["TRIP_ADAPTER_DIR"] = str(args.adapter_dir.resolve())
        settings = ReleaseSettings.load(root=ROOT)
        backend = TransformersPeftBackend(settings)
        service = ScenarioService(settings, backend)
        if (
            args.output_dir.exists()
            and args.resume
            and (args.output_dir / "selection.json").is_file()
        ):
            pilot_result = load_completed_prompt_pilot(
                args.config,
                DEFAULT_PROMPTS,
                args.output_dir,
            )
        else:
            pilot_result = run_prompt_pilot(
                ROOT,
                args.config,
                DEFAULT_PROMPTS,
                args.output_dir,
                endpoint="in-process://transformers-peft",
                served_model=settings.adapter_name,
                generator=backend.generate_with_usage,
                resume=args.resume,
            )
        repair_result = run_week5_repair_queue(
            ROOT,
            DEFAULT_WEEK5,
            run_id=args.run_id,
            resume=args.resume,
            service=service,
        )
        final_summary = (
            ROOT
            / "outputs/system_repair/week5_preannotation_repair_v2/final_summary.json"
        )
        if final_summary.exists() and args.resume:
            merge_result = json.loads(final_summary.read_text(encoding="utf-8"))
            final_result = final_summary.with_name("schema_valid_silver_80000.jsonl")
            if (
                merge_result.get("status") != "COMPLETED"
                or merge_result.get("schema_valid") != 80000
                or merge_result.get("unresolved") != 0
                or not final_result.is_file()
                or merge_result.get("result_sha256") != sha256_file(final_result)
            ):
                raise SystemExit("resumed Week 5 final evidence is invalid")
        else:
            merge_result = merge_week5_repair_results(
                ROOT,
                DEFAULT_WEEK5,
                run_id=args.run_id,
            )
        result = {
            "status": "COMPLETED",
            "prompt_pilot": pilot_result,
            "week5_repair": repair_result,
            "week5_merge": merge_result,
        }
    elif args.command == "train":
        result = run_multitask_training(
            ROOT,
            args.config,
            args.output_dir,
            confirm_dataset_lock=True,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    elif args.command == "select-candidate":
        result = select_system_repair_candidate(
            ROOT,
            args.config,
            args.training_dir,
            args.output,
        )
    elif args.command == "evaluate-gates":
        config = load_week7_config(args.config)
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        if (
            selection.get("status") != "COMPLETED"
            or selection.get("development_metrics", {}).get("sha256")
            != sha256_file(args.candidate)
        ):
            raise SystemExit("candidate selection does not bind the supplied metrics")
        result = evaluate_system_release_gates(
            config,
            json.loads(args.candidate.read_text(encoding="utf-8")),
            json.loads(args.existing.read_text(encoding="utf-8")),
            json.loads(args.zero_shot.read_text(encoding="utf-8")),
            json.loads(args.week6_routed.read_text(encoding="utf-8")),
        )
        result["evidence"] = {
            "config_sha256": sha256_file(args.config),
            "selection_sha256": sha256_file(args.selection),
            "candidate_metrics_sha256": sha256_file(args.candidate),
            "existing_metrics_sha256": sha256_file(args.existing),
            "zero_shot_metrics_sha256": sha256_file(args.zero_shot),
            "week6_routed_metrics_sha256": sha256_file(args.week6_routed),
            "adapter_model_sha256": selection["adapter_model_sha256"],
        }
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    elif args.command == "run-final-test":
        result = run_system_repair_test_once(
            ROOT,
            args.config,
            args.selection,
            args.gate,
            args.output_dir,
        )
    else:
        config = load_week7_config(args.config)
        result = {
            "status": "ok",
            "repair_id": config["system_repair"]["repair_id"],
            "train_total": config["dataset"]["train_total"],
            "learning_rate": config["training"]["learning_rate"],
            "epochs": config["training"]["epochs"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"FAIL", "failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

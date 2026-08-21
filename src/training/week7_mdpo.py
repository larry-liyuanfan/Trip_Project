"""Memory-bounded multimodal DPO-style ablation for audited Week 7 pairs."""

from __future__ import annotations

import gc
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from src.training.week6_qlora import (
    _trainable_parameter_report,
    environment_report,
)
from src.training.week7_data import iter_jsonl, load_week7_config, sha256_file
from src.training.week7_preference import LOCK_SCHEMA_VERSION, SCHEMA_VERSION
from src.training.week7_qlora import Week7TrainingError, structure_aware_messages


RUN_SCHEMA_VERSION = "week7_mdpo_run_v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week7TrainingError(f"invalid mDPO JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Week7TrainingError(f"mDPO JSON must be an object: {path}")
    return value


def dpo_loss_and_coefficient(
    policy_chosen: float,
    policy_rejected: float,
    reference_chosen: float,
    reference_rejected: float,
    beta: float,
) -> tuple[float, float, float]:
    """Return loss, d(loss)/d(policy chosen logp), and policy-reference margin."""
    margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    z = beta * margin
    loss = math.log1p(math.exp(-abs(z))) + max(-z, 0.0)
    coefficient = -beta / (1.0 + math.exp(z))
    return loss, coefficient, margin


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _training_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise Week7TrainingError("unsupported mDPO training config")
    if config.get("scope") != {
        "split": "development_preference_train_validation",
        "test_allowed": False,
        "single_ablation_only": True,
        "agent_labels_may_replace_human": False,
    }:
        raise Week7TrainingError("mDPO training scope changed")
    base_config_path = root / config["base_config"]["path"]
    if not base_config_path.is_file() or sha256_file(base_config_path) != config["base_config"]["sha256"]:
        raise Week7TrainingError("mDPO base config identity mismatch")
    training = config["training"]
    if (
        int(training["epochs"]) != 1
        or training["reference_mode"] != "precomputed_initial_adapter_logprobs"
        or training["logprob_reduction"] != "mean"
    ):
        raise Week7TrainingError("mDPO must remain a single mean-logprob ablation")
    return config


def _sequence_logprob(
    model: Any,
    processor: Any,
    prompt_messages: list[dict[str, Any]],
    response: str,
    max_length: int,
) -> Any:
    import torch.nn.functional as functional

    messages = structure_aware_messages(
        processor,
        [*prompt_messages, {"role": "assistant", "content": response}],
        max_length,
    )
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt", truncation=False,
    )
    prompt = processor.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", truncation=False,
    )
    labels = inputs["input_ids"].clone()
    labels[:, : prompt["input_ids"].shape[1]] = -100
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    labels = labels.to(device)
    outputs = model(**inputs)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    if not mask.any():
        raise Week7TrainingError("mDPO response has no trainable token after truncation")
    shifted_logits = outputs.logits[:, :-1, :][mask]
    token_targets = shifted_labels[mask]
    return -functional.cross_entropy(shifted_logits.float(), token_targets, reduction="mean")


def _evaluate_pairs(
    model: Any,
    processor: Any,
    pairs: list[dict[str, Any]],
    references: dict[str, dict[str, float]],
    max_length: int,
) -> dict[str, Any]:
    import torch

    rows = []
    with torch.no_grad():
        for pair in pairs:
            chosen = float(_sequence_logprob(
                model, processor, pair["prompt_messages"], pair["chosen"], max_length,
            ).item())
            rejected = float(_sequence_logprob(
                model, processor, pair["prompt_messages"], pair["rejected"], max_length,
            ).item())
            reference = references[pair["pair_id"]]
            margin = (chosen - rejected) - (reference["chosen"] - reference["rejected"])
            rows.append({
                "pair_id": pair["pair_id"], "sample_id": pair["sample_id"],
                "parent_scenario": pair["parent_scenario"],
                "chosen_model_role": pair["chosen_model_role"],
                "policy_chosen_logp": chosen, "policy_rejected_logp": rejected,
                "reference_chosen_logp": reference["chosen"],
                "reference_rejected_logp": reference["rejected"],
                "policy_reference_margin": margin,
                "preference_correct": margin > 0.0,
            })
    return {
        "count": len(rows),
        "preference_accuracy": sum(row["preference_correct"] for row in rows) / len(rows),
        "mean_policy_reference_margin": sum(row["policy_reference_margin"] for row in rows) / len(rows),
        "rows": rows,
    }


def run_mdpo_ablation(
    root: Path,
    config_path: Path,
    dataset_dir: Path,
    initial_adapter_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    root, config_path, dataset_dir, initial_adapter_dir, output_dir = (
        Path(root).resolve(), Path(config_path).resolve(), Path(dataset_dir).resolve(),
        Path(initial_adapter_dir).resolve(), Path(output_dir).resolve(),
    )
    if output_dir.exists():
        raise Week7TrainingError("refusing to overwrite mDPO ablation")
    config = _training_config(root, config_path)
    lock = _read_json(dataset_dir / "dataset_lock.json")
    if (
        lock.get("schema_version") != LOCK_SCHEMA_VERSION
        or lock.get("dataset_version") != config["dataset_version"]
        or lock.get("config_sha256") != sha256_file(config_path)
        or lock.get("test_read") is not False
    ):
        raise Week7TrainingError("mDPO preference lock mismatch")
    train_path, validation_path = dataset_dir / "train.jsonl", dataset_dir / "validation.jsonl"
    if (
        sha256_file(train_path) != lock["files"]["train.jsonl"]["sha256"]
        or sha256_file(validation_path) != lock["files"]["validation.jsonl"]["sha256"]
    ):
        raise Week7TrainingError("mDPO preference files changed")
    train_pairs, validation_pairs = list(iter_jsonl(train_path)), list(iter_jsonl(validation_path))
    training = config["training"]
    if len(train_pairs) != int(training["max_train_pairs"]) or len(validation_pairs) != int(training["max_validation_pairs"]):
        raise Week7TrainingError("mDPO pair counts changed")
    adapter_model = initial_adapter_dir / "adapter_model.safetensors"
    if not adapter_model.is_file() or sha256_file(adapter_model) != config["selected_checkpoint"]["adapter_sha256"]:
        raise Week7TrainingError("mDPO initial adapter identity mismatch")
    report = environment_report(require_cuda=True)
    if report["status"] != "ok" or not os.environ.get("SLURM_JOB_ID"):
        raise Week7TrainingError("mDPO must run in the verified Slurm CUDA environment")

    import torch
    from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    base_config = load_week7_config(root / config["base_config"]["path"])
    quant = base_config["quantization"]
    torch.manual_seed(int(training["seed"]))
    torch.cuda.manual_seed_all(int(training["seed"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    run_identity = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": training["run_id"],
        "git_commit": _git_commit(root),
        "config_sha256": sha256_file(config_path),
        "preference_lock_sha256": lock["lock_sha256"],
        "initial_adapter_sha256": sha256_file(adapter_model),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "test_read": False,
    }
    (output_dir / "run_identity.json").write_text(
        json.dumps(run_identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_config["base_model"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        attn_implementation=base_config["training"]["attn_implementation"],
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(model, str(initial_adapter_dir), is_trainable=True)
    model.train()
    # 保持 gradient checkpointing 的训练路径，同时关闭随机 dropout，确保分离的
    # chosen/rejected 前向与系数重算严格对应同一确定性目标。
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    parameter_report = _trainable_parameter_report(model)
    processor = AutoProcessor.from_pretrained(base_config["base_model"])
    max_length = int(training["max_length"])
    all_pairs = [*train_pairs, *validation_pairs]
    references: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for pair in all_pairs:
            references[pair["pair_id"]] = {
                "chosen": float(_sequence_logprob(
                    model, processor, pair["prompt_messages"], pair["chosen"], max_length,
                ).item()),
                "rejected": float(_sequence_logprob(
                    model, processor, pair["prompt_messages"], pair["rejected"], max_length,
                ).item()),
            }
    reference_path = output_dir / "reference_logprobs.jsonl"
    with reference_path.open("x", encoding="utf-8", newline="\n") as handle:
        for pair in all_pairs:
            handle.write(json.dumps({"pair_id": pair["pair_id"], **references[pair["pair_id"]]}, sort_keys=True) + "\n")

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
    )
    beta = float(training["beta"])
    accumulation = int(training["gradient_accumulation_steps"])
    order = list(train_pairs)
    random.Random(int(training["seed"])).shuffle(order)
    logs = []
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for index, pair in enumerate(order, start=1):
        with torch.no_grad():
            policy_chosen = float(_sequence_logprob(
                model, processor, pair["prompt_messages"], pair["chosen"], max_length,
            ).item())
            policy_rejected = float(_sequence_logprob(
                model, processor, pair["prompt_messages"], pair["rejected"], max_length,
            ).item())
        reference = references[pair["pair_id"]]
        loss, coefficient, margin = dpo_loss_and_coefficient(
            policy_chosen, policy_rejected, reference["chosen"], reference["rejected"], beta,
        )
        chosen_logp = _sequence_logprob(
            model, processor, pair["prompt_messages"], pair["chosen"], max_length,
        )
        (chosen_logp * (coefficient / accumulation)).backward()
        del chosen_logp
        rejected_logp = _sequence_logprob(
            model, processor, pair["prompt_messages"], pair["rejected"], max_length,
        )
        (rejected_logp * (-coefficient / accumulation)).backward()
        del rejected_logp
        should_step = index % accumulation == 0 or index == len(order)
        gradient_norm = None
        if should_step:
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(training["max_grad_norm"]),
            ).item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        logs.append({
            "pair_id": pair["pair_id"], "sample_id": pair["sample_id"],
            "loss": loss, "policy_reference_margin_before_update": margin,
            "gradient_coefficient": coefficient, "optimizer_step": should_step,
            "gradient_norm": gradient_norm,
        })
        gc.collect()
        torch.cuda.empty_cache()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(output_dir / "processor"))
    saved = adapter_dir / "adapter_model.safetensors"
    if not saved.is_file() or not PeftConfig.from_pretrained(str(adapter_dir)).base_model_name_or_path:
        raise Week7TrainingError("mDPO adapter reload metadata is incomplete")
    train_eval = _evaluate_pairs(model, processor, train_pairs, references, max_length)
    validation_eval = _evaluate_pairs(model, processor, validation_pairs, references, max_length)
    gate = config["validation_gate"]
    gate_passed = (
        validation_eval["preference_accuracy"] >= float(gate["minimum_preference_accuracy"])
        and validation_eval["mean_policy_reference_margin"] > float(gate["minimum_mean_policy_reference_margin"])
    )
    metrics_path = output_dir / "preference_metrics.json"
    metrics_path.write_text(
        json.dumps({"train": train_eval, "validation": validation_eval}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    log_path = output_dir / "training_log.jsonl"
    with log_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in logs:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        **run_identity,
        "status": "COMPLETED_GATE_PASSED" if gate_passed else "COMPLETED_GATE_FAILED",
        "method": "memory_bounded_mean_logprob_mdpo_style_v1",
        "train_pairs": len(train_pairs), "validation_pairs": len(validation_pairs),
        "optimizer_updates": math.ceil(len(train_pairs) / accumulation),
        "duration_seconds": time.time() - started,
        "adapter_sha256": sha256_file(saved),
        "reference_logprobs_sha256": sha256_file(reference_path),
        "training_log_sha256": sha256_file(log_path),
        "preference_metrics_sha256": sha256_file(metrics_path),
        "train_metrics": {key: value for key, value in train_eval.items() if key != "rows"},
        "validation_metrics": {key: value for key, value in validation_eval.items() if key != "rows"},
        "validation_gate_passed": gate_passed,
        **parameter_report,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return summary

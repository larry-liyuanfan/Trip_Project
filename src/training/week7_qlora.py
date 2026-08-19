"""Unified Week 7 multitask QLoRA/SFT with full-development validation."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from src.training.week6_qlora import (
    _normalize_processor_messages,
    _trainable_parameter_report,
    environment_report,
    resolve_lora_targets,
)
from src.training.week7_data import canonical_sha256, iter_jsonl, load_week7_config, sha256_file
from src.training.week7_evaluation import summarize_raw_records


class Week7TrainingError(ValueError):
    """Raised when Week 7 training violates a locked contract."""


def decile_evaluation_steps(max_steps: int) -> list[int]:
    if max_steps <= 0:
        raise Week7TrainingError("max_steps must be positive")
    return sorted({math.ceil(max_steps * index / 10) for index in range(1, 11)})


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def training_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = [dict(item) for item in row["messages"]]
    target = json.dumps(row["target"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if messages and messages[-1].get("role") == "assistant":
        messages[-1] = {"role": "assistant", "content": target}
    else:
        messages.append({"role": "assistant", "content": target})
    return messages


def structure_aware_messages(processor: Any, messages: list[dict[str, Any]], max_length: int) -> list[dict[str, Any]]:
    """Keep system/first image turn/final target and remove oldest complete middle pairs."""
    normalized = _normalize_processor_messages(messages)

    def length(value: list[dict[str, Any]]) -> int:
        encoded = processor.apply_chat_template(
            value, tokenize=True, add_generation_prompt=False, return_dict=True,
            return_tensors="pt", truncation=False,
        )
        return int(encoded["input_ids"].shape[1])

    if length(normalized) <= max_length:
        return normalized
    if len(normalized) <= 3:
        raise Week7TrainingError("structure-aware truncation cannot preserve required messages")
    middle = normalized[2:-1]
    while middle:
        remove = 2 if len(middle) >= 2 else 1
        middle = middle[remove:]
        candidate = [normalized[0], normalized[1], *middle, normalized[-1]]
        if length(candidate) <= max_length:
            return candidate
    candidate = [normalized[0], normalized[1], normalized[-1]]
    if length(candidate) > max_length:
        raise Week7TrainingError("required system/image/target structure exceeds max_length")
    return candidate


class IndexedWeek7Dataset:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.offsets = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    row = json.loads(line.decode("utf-8"))
                    if row.get("scenario") not in {"image_product_search", "after_sales", "itinerary_planning", "dialogue", "general_multimodal"}:
                        raise Week7TrainingError("unknown Week 7 scenario")
                    self.offsets.append(offset)
        if not self.offsets:
            raise Week7TrainingError("empty Week 7 dataset")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline().decode("utf-8"))


def _generate_record(root: Path, model: Any, processor: Any, row: dict[str, Any], run_id: str, max_new_tokens: int) -> dict[str, Any]:
    import torch

    normalized = structure_aware_messages(processor, training_messages(row), 8192)[:-1]
    started = time.perf_counter()
    failed = False
    error = None
    raw = ""
    try:
        inputs = processor.apply_chat_template(
            normalized, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", truncation=True, max_length=8192,
        )
        device = next(model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        suffix = generated[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(suffix, skip_special_tokens=True)[0].strip()
    except (RuntimeError, ValueError) as exc:
        failed, error = True, f"{type(exc).__name__}: {exc}"
    return {
        "run_id": run_id, "sample_id": row["sample_id"], "model_name": "Qwen3-VL-8B+Week7-QLoRA",
        "raw_output": raw, "latency_ms": (time.perf_counter() - started) * 1000,
        "failed": failed, "error": error,
    }


def run_multitask_training(root: Path, config_path: Path, output_dir: Path, *, confirm_dataset_lock: bool, resume_from_checkpoint: Path | None = None) -> dict[str, Any]:
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if not confirm_dataset_lock:
        raise Week7TrainingError("explicit dataset-lock confirmation is required")
    config = load_week7_config(config_path)
    lock_root = root / config["dataset"]["output_root"] / config["dataset"]["dataset_version"]
    lock = json.loads((lock_root / "dataset_lock.json").read_text(encoding="utf-8"))
    config_sha256 = sha256_file(config_path)
    if lock.get("config_sha256") != config_sha256:
        raise Week7TrainingError("training config SHA-256 does not match the dataset lock")
    run_id = config["experiment_identity"]["multitask_sft_run_id"]
    run_identity = {
        "run_id": run_id,
        "config_sha256": config_sha256,
        "dataset_config_sha256": canonical_sha256(config["dataset"]),
        "dataset_lock_sha256": lock["lock_sha256"],
        "git_commit": _git_commit(root),
    }
    if output_dir.exists() and resume_from_checkpoint is None:
        raise Week7TrainingError("refusing to overwrite an existing run")
    if resume_from_checkpoint is not None:
        try:
            resume_from_checkpoint.resolve().relative_to(output_dir)
        except ValueError as exc:
            raise Week7TrainingError("resume checkpoint must be inside output_dir") from exc
        identity_path = output_dir / "run_identity.json"
        if not identity_path.is_file() or json.loads(identity_path.read_text(encoding="utf-8")) != run_identity:
            raise Week7TrainingError("resume identity differs from the locked config, data, run, or commit")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "run_identity.json").write_text(
            json.dumps(run_identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
    report = environment_report(require_cuda=True)
    if report["status"] != "ok":
        raise Week7TrainingError(f"training environment is not ready: {report['status']}")

    import torch
    from peft import LoraConfig, PeftConfig, get_peft_model, prepare_model_for_kbit_training
    from peft.utils.save_and_load import load_peft_weights
    from transformers import (
        AutoProcessor, BitsAndBytesConfig, EarlyStoppingCallback,
        Qwen3VLForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments,
    )

    train_config = config["training"]
    quant = config["quantization"]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    torch.cuda.reset_peak_memory_stats()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["base_model"], quantization_config=quant_config, torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        attn_implementation=train_config["attn_implementation"],
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = resolve_lora_targets(model, config)
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["lora_alpha"], lora_dropout=lora["lora_dropout"],
        bias=lora["bias"], target_modules=targets, task_type="CAUSAL_LM",
    ))
    parameter_report = _trainable_parameter_report(model)
    processor = AutoProcessor.from_pretrained(config["base_model"])
    train_dataset = IndexedWeek7Dataset(lock_root / "train.jsonl")
    development_rows = list(iter_jsonl(lock_root / "development.jsonl"))
    eval_dataset = IndexedWeek7Dataset(lock_root / "development.jsonl")

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) != 1:
            raise Week7TrainingError("multimodal collator requires batch size 1")
        messages = structure_aware_messages(processor, training_messages(batch[0]), int(train_config["max_length"]))
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True,
            return_tensors="pt", truncation=False,
        )
        prompt = processor.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", truncation=False,
        )
        labels = inputs["input_ids"].clone()
        labels[:, : prompt["input_ids"].shape[1]] = -100
        if not (labels != -100).any():
            raise Week7TrainingError("assistant target was truncated")
        inputs["labels"] = labels
        inputs["_sample_weight"] = torch.tensor(float(batch[0]["sample_weight"]))
        return inputs

    total_updates = math.ceil(len(train_dataset) / int(train_config["gradient_accumulation_steps"])) * int(train_config["epochs"])
    decile_steps: list[int] = []

    class DecileEvaluationCallback(TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            decile_steps[:] = decile_evaluation_steps(int(state.max_steps))
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if int(state.global_step) in decile_steps:
                control.should_evaluate = True
                control.should_save = True
            return control

    class Week7Trainer(Trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._week7_evaluation_cache: dict[int, dict[str, float]] = {}

        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, num_items_in_batch: Any = None) -> Any:
            weight = inputs.pop("_sample_weight")
            outputs = model(**inputs)
            loss = outputs.loss * weight.to(outputs.loss.device)
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, eval_dataset: Any = None, ignore_keys: Any = None, metric_key_prefix: str = "eval") -> dict[str, float]:
            step = int(self.state.global_step)
            if step in self._week7_evaluation_cache:
                return self._week7_evaluation_cache[step]
            evaluation_started = time.perf_counter()
            was_training = self.model.training
            self.model.eval()
            records = [_generate_record(root, self.model, processor, row, run_id, 2048) for row in development_rows]
            summary = summarize_raw_records(root, config, development_rows, records)
            summary.update({
                "status": "COMPLETED",
                "model_role": "multitask_checkpoint",
                "split": "development",
                "run_id": f"{run_id}_development_step_{step:06d}",
                "config_sha256": config_sha256,
                "dataset_lock_sha256": lock["lock_sha256"],
                "global_step": step,
            })
            evaluation_dir = output_dir / "development_evaluations" / f"step-{step:06d}"
            evaluation_dir.mkdir(parents=True, exist_ok=False)
            with (evaluation_dir / "raw_outputs.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            (evaluation_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            if was_training:
                self.model.train()
            metrics = {
                f"{metric_key_prefix}_weighted_composite": float(summary["weighted_composite"]),
                f"{metric_key_prefix}_failure_rate": float(summary["failure_rate"]),
                f"{metric_key_prefix}_runtime": time.perf_counter() - evaluation_started,
            }
            for scenario, payload in summary["scenarios"].items():
                metrics[f"{metric_key_prefix}_{scenario}_composite"] = float(payload["composite"])
            if summary["dialogue"]:
                metrics[f"{metric_key_prefix}_dialogue_format_compliance"] = float(summary["dialogue"]["format_compliance"])
                metrics[f"{metric_key_prefix}_dialogue_context_recall"] = float(summary["dialogue"]["context_recall"])
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics,
            )
            self._week7_evaluation_cache[step] = metrics
            return metrics

    arguments = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=train_config["epochs"],
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"], lr_scheduler_type=train_config["lr_scheduler_type"],
        warmup_ratio=train_config["warmup_ratio"], weight_decay=train_config["weight_decay"],
        max_grad_norm=train_config["max_grad_norm"], optim=train_config["optimizer"],
        bf16=True, gradient_checkpointing=True, logging_steps=train_config["logging_steps"],
        eval_strategy="steps", eval_steps=1_000_000_000,
        save_strategy="steps", save_steps=1_000_000_000,
        save_total_limit=train_config["save_total_limit"], load_best_model_at_end=True,
        metric_for_best_model="eval_weighted_composite", greater_is_better=True,
        report_to=[], remove_unused_columns=False,
    )
    trainer = Week7Trainer(
        model=model, args=arguments, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=collate,
        callbacks=[
            DecileEvaluationCallback(),
            EarlyStoppingCallback(early_stopping_patience=int(train_config["early_stopping_patience"])),
        ],
    )
    started = time.time()
    result = trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    trainer.save_state()
    processor.save_pretrained(str(output_dir / "processor"))
    state = load_peft_weights(str(adapter_dir), device="cpu")
    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    if not state or not all("lora_" in name for name in state) or peft_config.base_model_name_or_path != config["base_model"]:
        raise Week7TrainingError("saved adapter failed LoRA-only reload verification")
    checkpoints = sorted(path for path in output_dir.glob("checkpoint-*") if path.is_dir())
    summary = {
        "status": "COMPLETED", "run_id": run_id, "git_commit": _git_commit(root),
        "config_sha256": config_sha256, "dataset_lock_sha256": lock["lock_sha256"],
        "train_samples": len(train_dataset), "development_samples": len(eval_dataset),
        "total_update_steps_planned": total_updates, "evaluation_steps": decile_steps,
        "global_step": int(trainer.state.global_step), "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric, "checkpoints": [path.name for path in checkpoints],
        "checkpoint_hashes": {path.name: sha256_file(path / "adapter_model.safetensors") for path in checkpoints if (path / "adapter_model.safetensors").is_file()},
        "adapter_hashes": {path.name: sha256_file(path) for path in adapter_dir.iterdir() if path.is_file()},
        "adapter_only": True, "adapter_reload_verified": True, "lora_targets": targets,
        **parameter_report, "training_metrics": result.metrics, "log_history": trainer.state.log_history,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "duration_seconds": time.time() - started, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "resumed_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return summary

"""Continue checkpoint-87 on the locked targeted synthetic training split only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl
from src.training.week6_qlora import _trainable_parameter_report, environment_report
from src.training.week7_qlora import assistant_span_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--confirm-training-lock", action="store_true")
    args = parser.parse_args()
    if not args.confirm_training_lock:
        raise ValueError("--confirm-training-lock is required")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_lock = json.loads(Path(config["pool"]["committed_lock"]).read_text(encoding="utf-8"))
    actual_lock = json.loads((args.bundle_dir / "bundle_lock.json").read_text(encoding="utf-8"))
    if actual_lock != expected_lock:
        raise ValueError("generated bundle lock differs from committed lock")
    adapter_file = args.initial_adapter / "adapter_model.safetensors"
    if file_sha256(adapter_file) != config["training"]["initial_adapter_model_sha256"]:
        raise ValueError("checkpoint-87 initial adapter SHA-256 mismatch")
    training_manifest = args.bundle_dir / "vlm_training_manifest.jsonl"
    if file_sha256(training_manifest) != expected_lock["vlm"]["training"]["manifest_file_sha256"]:
        raise ValueError("training manifest SHA-256 mismatch")
    rows = load_jsonl(training_manifest)
    if any(row.get("split") != "training" for row in rows):
        raise ValueError("training command may only load the training split")
    for row in rows:
        if row.get("scenario") == "product":
            image_path = args.bundle_dir / row["image_relative_path"]
            if not image_path.is_file() or file_sha256(image_path) != row["image_sha256"]:
                raise ValueError(f"training image mismatch: {row.get('sample_id')}")

    readiness = environment_report(require_cuda=True)
    if readiness["status"] != "ok":
        raise RuntimeError(f"training environment is not ready: {readiness['status']}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    identity = {
        "schema_version": "targeted_exploration_training_identity_v4",
        "run_id": config["training"]["run_id"],
        "git_commit": _implementation_commit(args.implementation_commit),
        "config_sha256": file_sha256(args.config),
        "pool_lock_sha256": canonical_json_sha256(actual_lock),
        "training_manifest_sha256": file_sha256(training_manifest),
        "initial_adapter_model_sha256": file_sha256(adapter_file),
        "development_or_final_opened": False,
    }
    _write_json(args.output_dir / "run_identity.json", identity)
    summary = _train(args, config, rows, identity)
    _write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _train(
    args: argparse.Namespace,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    identity: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    train_config = config["training"]
    set_seed(int(train_config["seed"]))
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    torch.cuda.reset_peak_memory_stats()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["vlm"]["base_model"],
        revision=config["vlm"]["base_revision"],
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        attn_implementation=train_config["attn_implementation"],
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = PeftConfig.from_pretrained(str(args.initial_adapter))
    if (
        int(peft_config.r) != 16
        or int(peft_config.lora_alpha) != 32
        or str(peft_config.base_model_name_or_path) != config["vlm"]["base_model"]
    ):
        raise ValueError("checkpoint-87 LoRA identity differs from the locked continuation")
    model = PeftModel.from_pretrained(model, str(args.initial_adapter), is_trainable=True)
    processor = AutoProcessor.from_pretrained(
        config["vlm"]["base_model"],
        revision=config["vlm"]["base_revision"],
        trust_remote_code=False,
    )
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = min(int(image_processor.max_pixels or 1024 * 1024), 1024 * 1024)

    class Dataset:
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return rows[index]

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) != 1:
            raise ValueError("multimodal training requires batch size 1")
        row = batch[0]
        content: list[dict[str, Any]] = []
        prompt = str(row["prompt"])
        if row["scenario"] == "product":
            content.append({
                "type": "image",
                "image": str((args.bundle_dir / row["image_relative_path"]).resolve()),
            })
        else:
            prompt = f"{prompt}\n\nCASE:\n{row['dialogue']}"
        content.append({"type": "text", "text": prompt})
        target = {key: value for key, value in row["gold"].items() if key != "unknown_fields"}
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": json.dumps(
                target, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )},
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
        )
        if int(inputs["input_ids"].shape[1]) > int(train_config["max_length"]):
            raise ValueError(f"training row exceeds max_length: {row['sample_id']}")
        inputs["labels"] = assistant_span_labels(processor, messages, inputs["input_ids"])
        inputs["_sample_weight"] = torch.tensor(float(row.get("sample_weight", 1.0)))
        return inputs

    class WeightedTrainer(Trainer):
        def compute_loss(
            self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            weight = inputs.pop("_sample_weight")
            outputs = model(**inputs)
            loss = outputs.loss * weight.to(outputs.loss.device)
            return (loss, outputs) if return_outputs else loss

    arguments = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=float(train_config["epochs"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        learning_rate=float(train_config["learning_rate"]),
        lr_scheduler_type=train_config["lr_scheduler_type"],
        warmup_ratio=float(train_config["warmup_ratio"]),
        weight_decay=float(train_config["weight_decay"]),
        max_grad_norm=float(train_config["max_grad_norm"]),
        optim=train_config["optimizer"],
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=int(train_config["logging_steps"]),
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=int(train_config["seed"]),
        data_seed=int(train_config["seed"]),
    )
    trainer = WeightedTrainer(model=model, args=arguments, train_dataset=Dataset(), data_collator=collate)
    started = time.time()
    result = trainer.train()
    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(args.output_dir / "processor"))
    saved_adapter = adapter_dir / "adapter_model.safetensors"
    saved_config = PeftConfig.from_pretrained(str(adapter_dir))
    if not saved_adapter.is_file() or str(saved_config.base_model_name_or_path) != config["vlm"]["base_model"]:
        raise RuntimeError("saved targeted adapter failed reload validation")
    return {
        "schema_version": "targeted_exploration_training_summary_v4",
        "status": "COMPLETED",
        **identity,
        "training_support": len(rows),
        "product_support": sum(row["scenario"] == "product" for row in rows),
        "dialogue_support": sum(row["scenario"] == "dialogue" for row in rows),
        "global_step": int(trainer.state.global_step),
        "training_metrics": result.metrics,
        "duration_seconds": time.time() - started,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "adapter_model_sha256": file_sha256(saved_adapter),
        "adapter_config_sha256": file_sha256(adapter_dir / "adapter_config.json"),
        "adapter_only": True,
        "development_or_final_opened": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        **_trainable_parameter_report(model),
    }


def _implementation_commit(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 40 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("implementation commit must be a full 40-character Git SHA")
    return normalized


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

"""Validated Qwen3-VL QLoRA configuration and guarded small-sample runner."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Iterator


class Week6TrainingError(ValueError):
    """Raised when Week 6 data or environment violates the accepted boundary."""


SCENARIOS = ("image_product_search", "after_sales", "itinerary_planning")


def load_training_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "week6_qwen3_vl_qlora_v1":
        raise Week6TrainingError("unsupported Week 6 training config")
    if config.get("base_model") != "Qwen/Qwen3-VL-8B-Instruct":
        raise Week6TrainingError("Week 6 primary base must be Qwen3-VL-8B-Instruct")
    quantization = config.get("quantization", {})
    expected_quantization = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }
    if any(quantization.get(key) != value for key, value in expected_quantization.items()):
        raise Week6TrainingError("Week 6 quantization must use NF4 double-quant bf16")
    lora = config.get("lora", {})
    if (lora.get("r"), lora.get("lora_alpha"), lora.get("lora_dropout"), lora.get("bias")) != (
        16,
        32,
        0.05,
        "none",
    ):
        raise Week6TrainingError("Week 6 LoRA parameters do not match the accepted standard")
    training = config.get("training", {})
    effective = (
        int(training.get("per_device_train_batch_size", 0))
        * int(training.get("gradient_accumulation_steps", 0))
    )
    if effective != int(training.get("effective_global_batch_size_one_gpu", -1)) or effective != 16:
        raise Week6TrainingError("single-GPU effective global batch size must be 16")
    if set(config.get("scenarios", {})) != set(SCENARIOS):
        raise Week6TrainingError("Week 6 config must define all three scenarios")
    return config


def iter_training_rows(path: Path, *, scenario: str | None = None) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Week6TrainingError(f"invalid training JSONL at line {line_number}") from exc
            validate_training_row(row, scenario=scenario)
            yield row


def validate_training_row(row: dict[str, Any], *, scenario: str | None = None) -> None:
    required = {"sample_id", "scenario", "messages", "label_source", "sample_weight", "dataset_lock"}
    missing = sorted(required - set(row))
    if missing:
        raise Week6TrainingError(f"training row missing fields: {missing}")
    if row["scenario"] not in SCENARIOS or (scenario and row["scenario"] != scenario):
        raise Week6TrainingError("training row scenario mismatch")
    if row["label_source"] not in {"model_preannotation", "human_revised"}:
        raise Week6TrainingError("unsupported training label source")
    weight = row["sample_weight"]
    if not isinstance(weight, (int, float)) or not 0 < float(weight) <= 1:
        raise Week6TrainingError("sample_weight must be in (0, 1]")
    if row["label_source"] == "model_preannotation" and float(weight) > 0.5:
        raise Week6TrainingError("model preannotation weight cannot exceed 0.5")
    lock = row["dataset_lock"]
    if not isinstance(lock, dict) or not all(
        isinstance(lock.get(key), str) and lock[key]
        for key in ("dataset_version", "manifest_sha256", "split_sha256")
    ):
        raise Week6TrainingError("training row requires a complete dataset lock")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise Week6TrainingError("training messages must contain user and assistant turns")
    if messages[-1].get("role") != "assistant":
        raise Week6TrainingError("final training message must be assistant output")
    if not any(message.get("role") == "user" for message in messages[:-1]):
        raise Week6TrainingError("training row requires a user message")


def environment_report(*, require_cuda: bool = True) -> dict[str, Any]:
    minimums = {
        "torch": "2.6.0",
        "transformers": "4.57.0",
        "accelerate": "1.10.0",
        "peft": "0.17.0",
        "bitsandbytes": "0.47.0",
    }
    versions: dict[str, str | None] = {}
    missing: list[str] = []
    incompatible: list[str] = []
    for name, minimum in minimums.items():
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", None) or importlib.metadata.version(name)
            versions[name] = version
            from packaging.version import Version

            if Version(version) < Version(minimum):
                incompatible.append(f"{name}>={minimum}")
        except ImportError:
            versions[name] = None
            missing.append(name)
    report: dict[str, Any] = {
        "versions": versions,
        "minimum_versions": minimums,
        "missing": missing,
        "incompatible": incompatible,
    }
    if "torch" not in missing:
        torch = importlib.import_module("torch")
        report.update(
            {
                "cuda_available": bool(torch.cuda.is_available()),
                "bf16_supported": bool(
                    torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                ),
                "gpu_count": int(torch.cuda.device_count()),
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    if missing:
        report["status"] = "missing_dependencies"
    elif incompatible:
        report["status"] = "incompatible_dependencies"
    elif require_cuda and not report.get("cuda_available"):
        report["status"] = "cuda_unavailable"
    elif require_cuda and not report.get("bf16_supported"):
        report["status"] = "bf16_unavailable"
    else:
        report["status"] = "ok"
    return report


def resolve_lora_targets(model: Any, config: dict[str, Any]) -> list[str]:
    names = {name for name, _ in model.named_modules()}
    lora = config["lora"]
    targets: list[str] = []
    for suffix in lora["language_target_suffixes"]:
        if not any(name.endswith(suffix) for name in names):
            raise Week6TrainingError(f"model is missing LoRA language target: {suffix}")
        targets.append(suffix)
    visual = [
        candidate
        for candidate in lora["visual_projection_candidates"]
        if any(name.endswith(candidate) for name in names)
    ]
    if not visual:
        raise Week6TrainingError("model exposes no accepted visual projection target")
    return [*targets, *visual]


def run_small_sample_training(
    config: dict[str, Any],
    *,
    scenario: str,
    train_path: Path,
    eval_path: Path,
    output_dir: Path,
    dataset_lock_confirmed: bool,
) -> dict[str, Any]:
    """Run the explicit GPU pilot; never called during import or validation."""
    if not dataset_lock_confirmed:
        raise Week6TrainingError("formal training requires an explicit locked Week 5 dataset")
    if output_dir.exists():
        raise Week6TrainingError("refusing to overwrite an existing training output directory")
    rows = list(iter_training_rows(train_path, scenario=scenario))
    eval_rows = list(iter_training_rows(eval_path, scenario=scenario))
    if not rows or not eval_rows:
        raise Week6TrainingError("training and evaluation inputs must be non-empty")
    max_samples = int(config["pilot"]["max_samples"])
    rows = rows[:max_samples]
    eval_rows = eval_rows[:max_samples]
    report = environment_report(require_cuda=True)
    if report["status"] != "ok":
        raise Week6TrainingError(f"training environment is not ready: {report['status']}")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    quant = config["quantization"]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["base_model"],
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config["training"]["gradient_checkpointing"]
    )
    targets = resolve_lora_targets(model, config)
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["r"],
            lora_alpha=lora["lora_alpha"],
            lora_dropout=lora["lora_dropout"],
            bias=lora["bias"],
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    processor = AutoProcessor.from_pretrained(config["base_model"])

    class MessageDataset:
        def __init__(self, values: list[dict[str, Any]]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.values[index]

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) != 1:
            raise Week6TrainingError("multimodal pilot collator requires per-device batch size 1")
        messages = batch[0]["messages"]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=scenario_config["max_length"],
        )
        prompt = processor.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=scenario_config["max_length"],
        )
        labels = inputs["input_ids"].clone()
        labels[:, : prompt["input_ids"].shape[1]] = -100
        if not (labels != -100).any():
            raise Week6TrainingError("max_length truncation removed the assistant target")
        inputs["labels"] = labels
        inputs["_sample_weight"] = torch.tensor(float(batch[0]["sample_weight"]))
        return inputs

    class WeightedTrainer(Trainer):
        def compute_loss(
            self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            weight = inputs.pop("_sample_weight")
            outputs = model(**inputs)
            loss = outputs.loss * weight.to(outputs.loss.device)
            return (loss, outputs) if return_outputs else loss

    train = config["training"]
    scenario_config = config["scenarios"][scenario]
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=scenario_config["epochs"],
        max_steps=int(config["pilot"]["max_steps"]),
        per_device_train_batch_size=train["per_device_train_batch_size"],
        per_device_eval_batch_size=train["per_device_eval_batch_size"],
        gradient_accumulation_steps=train["gradient_accumulation_steps"],
        learning_rate=train["learning_rate"],
        lr_scheduler_type=train["lr_scheduler_type"],
        warmup_ratio=train["warmup_ratio"],
        weight_decay=train["weight_decay"],
        optim=train["optimizer"],
        bf16=train["bf16"],
        gradient_checkpointing=train["gradient_checkpointing"],
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=train["save_total_limit"],
        load_best_model_at_end=train["load_best_model_at_end"],
        metric_for_best_model=train["metric_for_best_model"],
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=MessageDataset(rows),
        eval_dataset=MessageDataset(eval_rows),
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    processor.save_pretrained(str(output_dir / "processor"))
    return {
        "status": "completed",
        "scenario": scenario,
        "train_samples": len(rows),
        "eval_samples": len(eval_rows),
        "lora_targets": targets,
        "adapter_only": True,
    }

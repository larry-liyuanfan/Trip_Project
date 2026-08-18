"""Validated Qwen3-VL QLoRA configuration and guarded small-sample runner."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import time
from itertools import islice
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
    expected_training = {
        "optimizer": "adamw_torch",
        "learning_rate": 0.0002,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "gradient_checkpointing": True,
        "bf16": True,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise Week6TrainingError("Week 6 training hyperparameters do not match the accepted standard")
    effective = (
        int(training.get("per_device_train_batch_size", 0))
        * int(training.get("gradient_accumulation_steps", 0))
    )
    if effective != int(training.get("effective_global_batch_size_one_gpu", -1)) or effective != 16:
        raise Week6TrainingError("single-GPU effective global batch size must be 16")
    if set(config.get("scenarios", {})) != set(SCENARIOS):
        raise Week6TrainingError("Week 6 config must define all three scenarios")
    pilot = config.get("pilot", {})
    if int(pilot.get("max_samples", 0)) > 32 or int(pilot.get("max_steps", 0)) > 10:
        raise Week6TrainingError("Week 6 pilot exceeds the approved sample or step cap")
    refinement = config.get("refinement")
    if refinement is not None:
        if not isinstance(refinement, dict):
            raise Week6TrainingError("Week 6 refinement config must be an object")
        if refinement.get("scenario") not in SCENARIOS:
            raise Week6TrainingError("Week 6 refinement scenario is invalid")
        if not isinstance(refinement.get("repair_version"), str) or not refinement["repair_version"]:
            raise Week6TrainingError("Week 6 refinement repair_version is required")
        expected_hashes = refinement.get("expected_initial_adapter_file_sha256")
        required_adapter_hashes = {
            "adapter_config.json",
            "adapter_model.safetensors",
        }
        if (
            not isinstance(expected_hashes, dict)
            or set(expected_hashes) != required_adapter_hashes
        ):
            raise Week6TrainingError("Week 6 refinement requires initial adapter hashes")
        if any(
            not isinstance(name, str)
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for name, value in expected_hashes.items()
        ):
            raise Week6TrainingError("Week 6 refinement adapter hashes are invalid")
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


class IndexedMessageDataset:
    """Validate JSONL once and retain only byte offsets for bounded host memory."""

    def __init__(self, path: Path, *, scenario: str) -> None:
        self.path = path
        self.scenario = scenario
        self.offsets: list[int] = []
        locks: set[str] = set()
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise Week6TrainingError(
                        f"invalid training JSONL at byte offset {offset}"
                    ) from exc
                validate_training_row(row, scenario=scenario)
                self.offsets.append(offset)
                locks.add(
                    json.dumps(row["dataset_lock"], sort_keys=True, separators=(",", ":"))
                )
        if not self.offsets:
            raise Week6TrainingError("training dataset must be non-empty")
        if len(locks) != 1:
            raise Week6TrainingError("a training file contains mixed dataset locks")
        self.dataset_lock = json.loads(next(iter(locks)))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline().decode("utf-8"))


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


def _normalize_processor_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """将纯文本消息转换为 Qwen3-VL processor 要求的多模态内容列表。"""
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise Week6TrainingError(f"training message {index} must be an object")
        content = message.get("content")
        if isinstance(content, str):
            normalized_content = [{"type": "text", "text": content}]
        elif isinstance(content, list) and content and all(
            isinstance(item, dict) and isinstance(item.get("type"), str)
            for item in content
        ):
            normalized_content = [dict(item) for item in content]
        else:
            raise Week6TrainingError(
                f"training message {index} content must be text or a multimodal content list"
            )
        normalized.append({**message, "content": normalized_content})
    return normalized


def environment_report(*, require_cuda: bool = True) -> dict[str, Any]:
    minimums = {
        "torch": "2.6.0",
        "torchvision": "0.23.0",
        "transformers": "4.57.0",
        "accelerate": "1.10.0",
        "peft": "0.17.0",
        "bitsandbytes": "0.47.0",
        "kernels": "0.11.1",
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
        report["torch_cuda_version"] = torch.version.cuda
        try:
            cuda_available = bool(torch.cuda.is_available())
            report.update(
                {
                    "cuda_available": cuda_available,
                    "bf16_supported": bool(
                        cuda_available and torch.cuda.is_bf16_supported()
                    ),
                    "gpu_count": int(torch.cuda.device_count()),
                    "gpu_names": [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ],
                    "cuda_initialization_error": None,
                }
            )
        except RuntimeError as exc:
            report.update(
                {
                    "cuda_available": False,
                    "bf16_supported": False,
                    "gpu_count": 0,
                    "gpu_names": [],
                    "cuda_initialization_error": str(exc),
                }
            )
    if missing:
        report["status"] = "missing_dependencies"
    elif incompatible:
        report["status"] = "incompatible_dependencies"
    elif report.get("cuda_initialization_error"):
        report["status"] = "cuda_initialization_error"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trainable_parameter_report(model: Any) -> dict[str, Any]:
    trainable_names: list[str] = []
    trainable = 0
    total = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
    unexpected = [name for name in trainable_names if "lora_" not in name]
    if not trainable or unexpected:
        raise Week6TrainingError(
            "base-model freeze check failed; trainable parameters must be LoRA-only: "
            f"{unexpected[:10]}"
        )
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "trainable_parameter_names": trainable_names,
    }


def evaluate_pilot_gate(
    config: dict[str, Any],
    *,
    summary_path: Path,
    expected_scenario: str,
    expected_git_commit: str,
    gpu_total_memory_gb: float,
) -> dict[str, Any]:
    """Apply deterministic release gates before any full-data job can run."""
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Week6TrainingError(f"cannot read pilot summary: {summary_path}") from exc

    reasons: list[str] = []
    if summary.get("status") != "completed":
        reasons.append("pilot_status_not_completed")
    if summary.get("scenario") != expected_scenario:
        reasons.append("scenario_mismatch")
    if summary.get("git_commit") != expected_git_commit:
        reasons.append("git_commit_mismatch")
    lock = summary.get("dataset_lock") or {}
    if lock.get("dataset_version") != config["dataset"]["dataset_version"]:
        reasons.append("dataset_version_mismatch")
    if int(summary.get("global_step", -1)) != int(config["pilot"]["max_steps"]):
        reasons.append("pilot_step_count_mismatch")
    if not summary.get("checkpoints"):
        reasons.append("checkpoint_missing")
    if not summary.get("adapter_only") or not summary.get("adapter_reload_verified"):
        reasons.append("adapter_reload_not_verified")
    if not summary.get("adapter_file_sha256"):
        reasons.append("adapter_hashes_missing")

    train_loss = (summary.get("training_metrics") or {}).get("train_loss")
    if not isinstance(train_loss, (int, float)) or not math.isfinite(float(train_loss)):
        reasons.append("train_loss_not_finite")
    eval_losses = [
        item.get("eval_loss")
        for item in summary.get("log_history", [])
        if isinstance(item, dict) and "eval_loss" in item
    ]
    if not eval_losses or any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in eval_losses
    ):
        reasons.append("eval_loss_not_finite")

    peak_reserved = summary.get("peak_gpu_memory_reserved_bytes")
    capacity_bytes = int(gpu_total_memory_gb * 1024**3)
    if (
        not isinstance(peak_reserved, int)
        or peak_reserved <= 0
        or peak_reserved >= capacity_bytes
    ):
        reasons.append("gpu_memory_gate_failed")

    return {
        "status": "passed" if not reasons else "failed",
        "summary_path": str(summary_path),
        "scenario": expected_scenario,
        "expected_git_commit": expected_git_commit,
        "dataset_version": config["dataset"]["dataset_version"],
        "gpu_total_memory_gb": gpu_total_memory_gb,
        "peak_gpu_memory_reserved_bytes": peak_reserved,
        "train_loss": train_loss,
        "eval_losses": eval_losses,
        "reasons": reasons,
    }


def run_small_sample_training(
    config: dict[str, Any],
    *,
    scenario: str,
    train_path: Path,
    eval_path: Path,
    output_dir: Path,
    dataset_lock_confirmed: bool,
    resume_from_checkpoint: Path | None = None,
    init_adapter: Path | None = None,
    _run_mode: str = "pilot",
) -> dict[str, Any]:
    """Run a guarded QLoRA job; full mode is exposed only through its wrapper."""
    if _run_mode not in {"pilot", "full"}:
        raise Week6TrainingError("unsupported Week 6 training mode")
    refinement = config.get("refinement")
    if refinement is not None and scenario != refinement["scenario"]:
        raise Week6TrainingError("refinement config cannot train another scenario")
    if (
        refinement is not None
        and refinement.get("initial_adapter_required") is True
        and init_adapter is None
        and resume_from_checkpoint is None
    ):
        raise Week6TrainingError("refinement training requires an initial adapter")
    if not dataset_lock_confirmed:
        raise Week6TrainingError("formal training requires an explicit locked Week 5 dataset")
    if output_dir.exists() and resume_from_checkpoint is None:
        raise Week6TrainingError("refusing to overwrite an existing training output directory")
    if resume_from_checkpoint is not None and init_adapter is not None:
        raise Week6TrainingError("resume checkpoint and initial adapter are mutually exclusive")
    if init_adapter is not None:
        if not init_adapter.is_dir():
            raise Week6TrainingError("initial adapter directory does not exist")
        required_adapter_files = {
            "adapter_config.json",
            "adapter_model.safetensors",
        }
        if not required_adapter_files <= {path.name for path in init_adapter.iterdir()}:
            raise Week6TrainingError("initial adapter is incomplete")
    if resume_from_checkpoint is not None:
        if not output_dir.is_dir() or not resume_from_checkpoint.is_dir():
            raise Week6TrainingError("resume requires an existing output and checkpoint directory")
        try:
            resume_from_checkpoint.resolve().relative_to(output_dir.resolve())
        except ValueError as exc:
            raise Week6TrainingError("resume checkpoint must belong to the training output") from exc
    if _run_mode == "pilot":
        max_samples = int(config["pilot"]["max_samples"])
        rows = list(islice(iter_training_rows(train_path, scenario=scenario), max_samples))
        eval_rows = list(islice(iter_training_rows(eval_path, scenario=scenario), max_samples))
        if not rows or not eval_rows:
            raise Week6TrainingError("training and evaluation inputs must be non-empty")
        locks = {
            json.dumps(row["dataset_lock"], sort_keys=True, separators=(",", ":"))
            for row in [*rows, *eval_rows]
        }
        if len(locks) != 1:
            raise Week6TrainingError("training and validation inputs use different dataset locks")
        dataset_lock = rows[0]["dataset_lock"]
        train_sample_count = len(rows)
        eval_sample_count = len(eval_rows)
    else:
        train_dataset = IndexedMessageDataset(train_path, scenario=scenario)
        eval_dataset = IndexedMessageDataset(eval_path, scenario=scenario)
        if train_dataset.dataset_lock != eval_dataset.dataset_lock:
            raise Week6TrainingError("training and validation inputs use different dataset locks")
        dataset_lock = train_dataset.dataset_lock
        train_sample_count = len(train_dataset)
        eval_sample_count = len(eval_dataset)
    expected_version = config.get("dataset", {}).get("dataset_version")
    if expected_version and dataset_lock["dataset_version"] != expected_version:
        raise Week6TrainingError("training inputs do not match the configured dataset version")
    report = environment_report(require_cuda=True)
    if report["status"] != "ok":
        raise Week6TrainingError(f"training environment is not ready: {report['status']}")

    import torch
    from peft import (
        LoraConfig,
        PeftConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from peft.utils.save_and_load import load_peft_weights
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    torch.cuda.reset_peak_memory_stats()
    quant = config["quantization"]
    train = config["training"]
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
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        attn_implementation=train.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config["training"]["gradient_checkpointing"]
    )
    targets = resolve_lora_targets(model, config)
    lora = config["lora"]
    initial_adapter_hashes: dict[str, str] | None = None
    if init_adapter is None:
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
    else:
        initial_peft_config = PeftConfig.from_pretrained(str(init_adapter))
        if initial_peft_config.base_model_name_or_path != config["base_model"]:
            raise Week6TrainingError("initial adapter points to an unexpected base model")
        if (
            int(initial_peft_config.r) != int(lora["r"])
            or int(initial_peft_config.lora_alpha) != int(lora["lora_alpha"])
            or float(initial_peft_config.lora_dropout) != float(lora["lora_dropout"])
        ):
            raise Week6TrainingError("initial adapter LoRA parameters do not match the config")
        model = PeftModel.from_pretrained(
            model, str(init_adapter), is_trainable=True
        )
        initial_adapter_hashes = {
            path.name: _sha256_file(path)
            for path in sorted(init_adapter.iterdir())
            if path.is_file()
        }
        expected_hashes = (refinement or {}).get(
            "expected_initial_adapter_file_sha256", {}
        )
        if expected_hashes and any(
            initial_adapter_hashes.get(name) != expected
            for name, expected in expected_hashes.items()
        ):
            raise Week6TrainingError("initial adapter hashes do not match the config")
    parameter_report = _trainable_parameter_report(model)
    processor = AutoProcessor.from_pretrained(config["base_model"])
    scenario_config = config["scenarios"][scenario]
    format_loss_weight = float(scenario_config.get("format_constraint_loss_weight", 0.0))
    if not 0.0 <= format_loss_weight <= 1.0:
        raise Week6TrainingError("format constraint loss weight must be in [0, 1]")

    class MessageDataset:
        def __init__(self, values: list[dict[str, Any]]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.values[index]

    if _run_mode == "pilot":
        train_dataset = MessageDataset(rows)
        eval_dataset = MessageDataset(eval_rows)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) != 1:
            raise Week6TrainingError("multimodal pilot collator requires per-device batch size 1")
        messages = _normalize_processor_messages(batch[0]["messages"])
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
        if format_loss_weight:
            structural = torch.zeros_like(labels, dtype=torch.bool)
            for token_id in set(labels[labels != -100].tolist()):
                decoded = processor.tokenizer.decode([token_id], skip_special_tokens=False)
                if any(character in decoded for character in '{}[]:,"'):
                    structural |= labels == token_id
            if not structural.any():
                raise Week6TrainingError("itinerary target contains no JSON structural tokens")
            inputs["_format_mask"] = structural
        return inputs

    class WeightedTrainer(Trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.format_loss_history: list[float] = []

        def compute_loss(
            self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            weight = inputs.pop("_sample_weight")
            format_mask = inputs.pop("_format_mask", None)
            outputs = model(**inputs)
            loss = outputs.loss
            if format_mask is not None:
                import torch.nn.functional as functional

                shifted_logits = outputs.logits[:, :-1, :].contiguous()
                shifted_labels = inputs["labels"][:, 1:].contiguous()
                shifted_mask = format_mask[:, 1:].to(shifted_logits.device)
                token_loss = functional.cross_entropy(
                    shifted_logits.view(-1, shifted_logits.shape[-1]),
                    shifted_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(shifted_labels)
                constraint_loss = token_loss[shifted_mask].mean()
                self.format_loss_history.append(float(constraint_loss.detach().cpu()))
                loss = loss + format_loss_weight * constraint_loss
            loss = loss * weight.to(outputs.loss.device)
            return (loss, outputs) if return_outputs else loss

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=scenario_config["epochs"],
        max_steps=int(config["pilot"]["max_steps"]) if _run_mode == "pilot" else -1,
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
        logging_steps=1 if _run_mode == "pilot" else train["logging_steps"],
        eval_strategy="steps",
        eval_steps=(
            1 if _run_mode == "pilot" else train["evaluation_fraction_steps"]
        ),
        save_strategy="steps",
        save_steps=(
            1 if _run_mode == "pilot" else train["evaluation_fraction_steps"]
        ),
        save_total_limit=train["save_total_limit"],
        load_best_model_at_end=train["load_best_model_at_end"],
        metric_for_best_model=train["metric_for_best_model"],
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
    )
    started_at = time.time()
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    trainer.save_state()
    processor.save_pretrained(str(output_dir / "processor"))
    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    adapter_state = load_peft_weights(str(adapter_dir), device="cpu")
    if not adapter_state or not all("lora_" in key for key in adapter_state):
        raise Week6TrainingError("saved adapter weights could not be reloaded as LoRA-only")
    if peft_config.base_model_name_or_path != config["base_model"]:
        raise Week6TrainingError("saved adapter points to an unexpected base model")
    checkpoints = sorted(
        path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()
    )
    adapter_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(adapter_dir.iterdir())
        if path.is_file()
    }
    payload = {
        "status": "completed",
        "run_mode": _run_mode,
        "scenario": scenario,
        "train_samples": train_sample_count,
        "eval_samples": eval_sample_count,
        "dataset_lock": dataset_lock,
        "lora_targets": targets,
        **parameter_report,
        "adapter_only": True,
        "adapter_reload_verified": True,
        "adapter_file_sha256": adapter_hashes,
        "checkpoints": checkpoints,
        "global_step": int(trainer.state.global_step),
        "training_metrics": train_result.metrics,
        "format_constraint_loss_weight": format_loss_weight,
        "format_loss_observations": trainer.format_loss_history,
        "log_history": trainer.state.log_history,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "duration_seconds": time.time() - started_at,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "run_id": os.environ.get("TRIP_RUN_ID"),
        "git_commit": os.environ.get("TRIP_GIT_COMMIT"),
        "resumed_from_checkpoint": (
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        ),
        "initialized_from_adapter": (
            str(init_adapter) if init_adapter is not None else None
        ),
        "initial_adapter_file_sha256": initial_adapter_hashes,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def run_full_training(
    config: dict[str, Any],
    *,
    scenario: str,
    train_path: Path,
    eval_path: Path,
    output_dir: Path,
    dataset_lock_confirmed: bool,
    resume_from_checkpoint: Path | None = None,
    init_adapter: Path | None = None,
) -> dict[str, Any]:
    """Run one formally approved scenario without materializing its JSONL in memory."""
    return run_small_sample_training(
        config,
        scenario=scenario,
        train_path=train_path,
        eval_path=eval_path,
        output_dir=output_dir,
        dataset_lock_confirmed=dataset_lock_confirmed,
        resume_from_checkpoint=resume_from_checkpoint,
        init_adapter=init_adapter,
        _run_mode="full",
    )

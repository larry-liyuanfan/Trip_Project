"""Optional Week 8 product-only continuation SFT after Prompt non-selection."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.product_semantics import audit_product_references
from src.inference.system_runtime import TransformersPeftBackend
from src.training.week6_qlora import (
    _trainable_parameter_report,
    environment_report,
    resolve_lora_targets,
)
from src.training.week7_data import canonical_sha256, iter_jsonl, sha256_file
from src.training.week7_qlora import (
    IndexedWeek7Dataset,
    assistant_span_labels,
    decile_evaluation_steps,
    structure_aware_messages,
)
from src.training.week7_runtime import inference_runtime
from src.training.week8_product import (
    _render_product_messages,
    _run_one_product,
    load_week8_product_config,
    summarize_product_run,
)


class Week8ProductSFTError(ValueError):
    """Raised when continuation SFT violates its immutable Week 8 contract."""


SCHEMA_VERSION = "week8_product_continuation_sft_v1"
SILVER_LABEL_SOURCES = {"programmatic_silver", "silver"}


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_week8_product_sft_config(path: Path) -> dict[str, Any]:
    """Load and fail closed on every mentor-constrained training setting."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise Week8ProductSFTError("unsupported Week 8 product SFT config")
    if not str(payload.get("product_config_path") or "").strip():
        raise Week8ProductSFTError("product_config_path is required")
    continuation = payload.get("continuation", {})
    adapter_hash = str(continuation.get("adapter_model_sha256") or "").lower()
    if len(adapter_hash) != 64 or any(char not in "0123456789abcdef" for char in adapter_hash):
        raise Week8ProductSFTError("continuation adapter SHA-256 is invalid")
    if continuation.get("overwrite_initial_adapter") is not False:
        raise Week8ProductSFTError("the formal adapter must never be overwritten")

    lora = payload.get("lora", {})
    expected_targets = set(lora.get("expected_target_modules", []))
    configured_targets = set(lora.get("language_target_suffixes", [])) | {
        value
        for value in lora.get("visual_projection_candidates", [])
        if value != "visual_projection"
    }
    if (
        int(lora.get("r", 0)) != 16
        or int(lora.get("lora_alpha", 0)) != 32
        or float(lora.get("lora_dropout", -1)) != 0.08
        or lora.get("bias") != "none"
        or expected_targets != configured_targets
        or lora.get("save_adapter_only") is not True
    ):
        raise Week8ProductSFTError("LoRA r/alpha/target identity changed")

    quantization = payload.get("quantization", {})
    if (
        quantization.get("load_in_4bit") is not True
        or quantization.get("bnb_4bit_quant_type") != "nf4"
        or quantization.get("bnb_4bit_use_double_quant") is not True
        or quantization.get("bnb_4bit_compute_dtype") != "bfloat16"
    ):
        raise Week8ProductSFTError("continuation QLoRA quantization identity changed")

    training = payload.get("training", {})
    epochs = float(training.get("epochs", 0))
    learning_rate = float(training.get("learning_rate", 0))
    silver_weight = float(training.get("maximum_silver_sample_weight", 1))
    if not 0 < epochs <= 1:
        raise Week8ProductSFTError("continuation SFT is limited to at most one epoch")
    if not 0 < learning_rate <= 1e-5:
        raise Week8ProductSFTError("continuation SFT requires a low learning rate")
    if not 0 < silver_weight <= 0.5:
        raise Week8ProductSFTError("silver weight must not exceed 0.5")
    if float(training.get("evaluation_fraction_steps", 0)) != 0.1:
        raise Week8ProductSFTError("development evaluation must run at each 10% step")
    if training.get("metric_for_best_model") != "eval_product_composite":
        raise Week8ProductSFTError("checkpoint selection must use development product composite")
    if (
        int(training.get("per_device_train_batch_size", 0)) != 1
        or int(training.get("per_device_eval_batch_size", 0)) != 1
        or training.get("gradient_checkpointing") is not True
        or training.get("bf16") is not True
    ):
        raise Week8ProductSFTError("unsupported multimodal training runtime settings")

    development = payload.get("development", {})
    if development.get("prompt_version") != "week8_product_evidence_guard_v1":
        raise Week8ProductSFTError("SFT must keep the fixed visual-evidence Prompt")
    if development.get("selection_metric") != "product_composite":
        raise Week8ProductSFTError("unsupported development selection metric")
    if int(development.get("max_schema_retries", -1)) != 1:
        raise Week8ProductSFTError("development protocol requires exactly one Schema retry")
    if not development.get("required_slice_groups"):
        raise Week8ProductSFTError("required product error-slice groups are missing")
    return payload


def _resolve_inside_root(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Week8ProductSFTError("tracked config path escapes the repository") from exc
    return path


def _load_no_prompt_winner_evidence(
    development_dir: Path,
    *,
    product_config_path: Path,
    dataset_lock_sha256: str,
) -> dict[str, Any]:
    selection_path = Path(development_dir).resolve() / "selection.json"
    if not selection_path.is_file():
        raise Week8ProductSFTError("Prompt development selection.json is missing")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "SFT_ALLOWED_NO_PROMPT_WINNER":
        raise Week8ProductSFTError("SFT is allowed only when Prompt development has no winner")
    if selection.get("selected_role") is not None or selection.get("test_consumed") is not False:
        raise Week8ProductSFTError("Prompt evidence has an invalid pre-test state")
    if selection.get("config_sha256") != sha256_file(product_config_path):
        raise Week8ProductSFTError("Prompt evidence uses a different product config")
    if selection.get("dataset_lock_sha256") != dataset_lock_sha256:
        raise Week8ProductSFTError("Prompt evidence uses a different development lock")
    metrics_hashes = selection.get("metrics_sha256")
    if not isinstance(metrics_hashes, dict) or not metrics_hashes:
        raise Week8ProductSFTError("Prompt evidence has no metrics hashes")
    for role, expected_hash in metrics_hashes.items():
        path = selection_path.parent / str(role) / "metrics.json"
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise Week8ProductSFTError(f"Prompt metric evidence changed: {role}")
    return {
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selection_id": selection.get("selection_id"),
        "metrics_sha256": dict(sorted(metrics_hashes.items())),
    }


def _validate_product_lock_header(
    root: Path,
    product_config_path: Path,
    product_config: dict[str, Any],
) -> dict[str, Any]:
    """Verify the immutable lock without opening the final-test split."""

    lock_root = (
        root
        / product_config["dataset"]["output_root"]
        / product_config["week8"]["dataset_version"]
    )
    lock_path = lock_root / "dataset_lock.json"
    if not lock_path.is_file():
        raise Week8ProductSFTError("Week 8 product dataset lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    core = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if lock.get("lock_sha256") != canonical_sha256(core):
        raise Week8ProductSFTError("Week 8 product dataset lock hash changed")
    if lock.get("config_sha256") != sha256_file(product_config_path):
        raise Week8ProductSFTError("Week 8 product config differs from the data lock")
    if lock.get("test_status") != "LOCKED_UNCONSUMED":
        raise Week8ProductSFTError("Week 8 final test is not locked and unconsumed")
    if lock.get("dataset_version") != product_config["week8"]["dataset_version"]:
        raise Week8ProductSFTError("Week 8 product dataset version changed")
    # Only train/development content is eligible for SFT inspection.
    for split in ("train", "development"):
        relative = f"{split}/image_product_search.jsonl"
        evidence = lock.get("files", {}).get(relative)
        path = lock_root / relative
        if (
            not isinstance(evidence, dict)
            or not path.is_file()
            or sha256_file(path) != evidence.get("sha256")
        ):
            raise Week8ProductSFTError(f"Week 8 {split} artifact changed")
    return {
        "status": "PASS",
        "dataset_version": lock["dataset_version"],
        "lock_sha256": lock["lock_sha256"],
        "test_status": lock["test_status"],
    }


def _inspect_split_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    maximum_silver_weight: float,
) -> Counter[str]:
    slices: Counter[str] = Counter()
    for row in rows:
        if row.get("split") != split or row.get("scenario") != "image_product_search":
            raise Week8ProductSFTError(f"{split} contains a non-product or wrong-split row")
        if row.get("label_source") not in SILVER_LABEL_SOURCES:
            raise Week8ProductSFTError(f"{split} contains a non-silver label")
        if bool(row.get("target_provenance", {}).get("human_completed")):
            raise Week8ProductSFTError(f"{split} falsely marks silver data as human")
        weight = float(row.get("sample_weight", 1.0))
        if not 0 < weight <= maximum_silver_weight:
            raise Week8ProductSFTError(f"{split} silver sample weight exceeds the cap")
        audit = audit_product_references([row])
        if audit["metadata_proxy_samples"] or audit["issue_counts"]:
            raise Week8ProductSFTError(
                f"{split} visual SFT target contains metadata proxies or contradictions; "
                "requires a separately versioned evidence-grounded target, not more training"
            )
        slices.update(str(value) for value in row.get("error_slices", []))
    return slices


def validate_week8_product_sft_eligibility(
    root: Path,
    config_path: Path,
    prompt_development_dir: Path,
) -> dict[str, Any]:
    """Validate Prompt failure, data identity, silver weights, and error slices."""

    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = load_week8_product_sft_config(config_path)
    product_config_path = _resolve_inside_root(root, config["product_config_path"])
    product_config = load_week8_product_config(product_config_path)
    validation = _validate_product_lock_header(
        root, product_config_path, product_config
    )
    if validation["status"] != "PASS" or validation["test_status"] != "LOCKED_UNCONSUMED":
        raise Week8ProductSFTError("Week 8 product data is not eligible for SFT")
    if product_config["model"]["adapter_model_sha256"] != config["continuation"]["adapter_model_sha256"]:
        raise Week8ProductSFTError("SFT does not continue from the formal release adapter")
    if (
        product_config["model"]["base_model"] != config["model"]["base_model"]
        or product_config["model"]["base_revision"] != config["model"]["base_revision"]
        or product_config["prompts"][config["development"]["prompt_role"]]
        != config["development"]["prompt_version"]
    ):
        raise Week8ProductSFTError("SFT model or fixed Prompt differs from product development")
    prompt_evidence = _load_no_prompt_winner_evidence(
        prompt_development_dir,
        product_config_path=product_config_path,
        dataset_lock_sha256=validation["lock_sha256"],
    )

    lock_root = (
        root
        / product_config["dataset"]["output_root"]
        / product_config["week8"]["dataset_version"]
    )
    train_rows = list(iter_jsonl(lock_root / "train" / "image_product_search.jsonl"))
    development_rows = list(
        iter_jsonl(lock_root / "development" / "image_product_search.jsonl")
    )
    expected_counts = {
        "train": int(product_config["dataset"]["continuation_train_count"]),
        "development": int(product_config["dataset"]["development_count"]),
    }
    if len(train_rows) != expected_counts["train"] or len(development_rows) != expected_counts["development"]:
        raise Week8ProductSFTError("SFT split counts differ from the product lock")
    maximum_weight = float(config["training"]["maximum_silver_sample_weight"])
    train_slices = _inspect_split_rows(
        train_rows, split="train", maximum_silver_weight=maximum_weight
    )
    development_slices = _inspect_split_rows(
        development_rows,
        split="development",
        maximum_silver_weight=maximum_weight,
    )
    for group, names in config["development"]["required_slice_groups"].items():
        if sum(train_slices[str(name)] for name in names) <= 0:
            raise Week8ProductSFTError(f"training data has no {group} error-slice support")
        if sum(development_slices[str(name)] for name in names) <= 0:
            raise Week8ProductSFTError(f"development data has no {group} error-slice support")

    train_ids = {row["sample_id"] for row in train_rows}
    development_ids = {row["sample_id"] for row in development_rows}
    if train_ids & development_ids:
        raise Week8ProductSFTError("train and development sample IDs overlap")
    return {
        "status": "PASS",
        "config_sha256": sha256_file(config_path),
        "product_config_path": str(product_config_path),
        "product_config_sha256": sha256_file(product_config_path),
        "dataset_lock_sha256": validation["lock_sha256"],
        "dataset_version": validation["dataset_version"],
        "split_counts": expected_counts,
        "label_source": "programmatic_silver",
        "human_count": 0,
        "maximum_silver_sample_weight": maximum_weight,
        "train_slice_counts": dict(sorted(train_slices.items())),
        "development_slice_counts": dict(sorted(development_slices.items())),
        "prompt_development": prompt_evidence,
    }


def product_training_messages(
    root: Path,
    row: dict[str, Any],
    prompt_version: str,
) -> list[dict[str, Any]]:
    """Use the fixed evidence Prompt while supervising only compact JSON output."""

    from src.inference.system_runtime import _transformers_messages

    messages, _ = _render_product_messages(root, row, prompt_version)
    normalized = _transformers_messages(messages)
    normalized.append(
        {
            "role": "assistant",
            "content": json.dumps(
                row["target"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    )
    return normalized


def _in_memory_backend(model: Any, processor: Any, torch_module: Any) -> TransformersPeftBackend:
    """Reuse the release generation protocol without reloading checkpoint weights."""

    return TransformersPeftBackend.from_loaded(model, processor, torch_module)


def run_week8_product_continuation_sft(
    root: Path,
    config_path: Path,
    prompt_development_dir: Path,
    output_dir: Path,
    *,
    resume_from_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Train once from checkpoint-87 and select checkpoints on development only."""

    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    prompt_development_dir = Path(prompt_development_dir).resolve()
    output_dir = Path(output_dir).resolve()
    config = load_week8_product_sft_config(config_path)
    eligibility = validate_week8_product_sft_eligibility(
        root, config_path, prompt_development_dir
    )
    run_id = config["experiment_identity"]["run_id"]
    declared_run_id = os.environ.get("TRIP_RUN_ID")
    if declared_run_id is not None and declared_run_id != run_id:
        raise Week8ProductSFTError("TRIP_RUN_ID differs from the versioned SFT config")
    run_identity = {
        "run_id": run_id,
        "git_commit": _git_commit(root),
        "config_sha256": eligibility["config_sha256"],
        "product_config_sha256": eligibility["product_config_sha256"],
        "dataset_lock_sha256": eligibility["dataset_lock_sha256"],
        "prompt_selection_sha256": eligibility["prompt_development"]["selection_sha256"],
    }
    if output_dir.exists() and resume_from_checkpoint is None:
        raise Week8ProductSFTError("refusing to overwrite an existing SFT run")
    if resume_from_checkpoint is not None:
        resume_from_checkpoint = Path(resume_from_checkpoint).resolve()
        try:
            resume_from_checkpoint.relative_to(output_dir)
        except ValueError as exc:
            raise Week8ProductSFTError("resume checkpoint must be inside output_dir") from exc
        identity_path = output_dir / "run_identity.json"
        if not identity_path.is_file() or json.loads(identity_path.read_text(encoding="utf-8")) != run_identity:
            raise Week8ProductSFTError("resume identity changed")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json_new(output_dir / "run_identity.json", run_identity)

    report = environment_report(require_cuda=True)
    if report["status"] != "ok":
        raise Week8ProductSFTError(f"training environment is not ready: {report['status']}")

    import torch
    from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
    from peft.utils.save_and_load import load_peft_weights
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    adapter_env = str(config["continuation"]["adapter_path_env"])
    adapter_value = os.environ.get(adapter_env, "").strip()
    if not adapter_value:
        raise Week8ProductSFTError(f"{adapter_env} is required for continuation SFT")
    adapter_dir = Path(adapter_value).resolve()
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_config_file = adapter_dir / "adapter_config.json"
    if not adapter_file.is_file() or not adapter_config_file.is_file():
        raise Week8ProductSFTError("formal continuation adapter is incomplete")
    if sha256_file(adapter_file) != config["continuation"]["adapter_model_sha256"]:
        raise Week8ProductSFTError("formal continuation adapter SHA-256 mismatch")
    try:
        output_dir.relative_to(adapter_dir)
    except ValueError:
        pass
    else:
        raise Week8ProductSFTError("SFT output must not be written inside the formal adapter")

    train_config = config["training"]
    quant = config["quantization"]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    torch.cuda.reset_peak_memory_stats()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model"]["base_model"],
        revision=config["model"]["base_revision"],
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        attn_implementation=train_config["attn_implementation"],
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    resolved_targets = resolve_lora_targets(model, config)
    initial_config = PeftConfig.from_pretrained(str(adapter_dir))
    initial_targets = set(initial_config.target_modules or [])
    expected_targets = set(config["lora"]["expected_target_modules"])
    if (
        int(initial_config.r) != int(config["lora"]["r"])
        or int(initial_config.lora_alpha) != int(config["lora"]["lora_alpha"])
        or float(initial_config.lora_dropout) != float(config["lora"]["lora_dropout"])
        or str(initial_config.bias) != config["lora"]["bias"]
        or str(initial_config.base_model_name_or_path) != config["model"]["base_model"]
        or initial_targets != expected_targets
        or set(resolved_targets) != expected_targets
    ):
        raise Week8ProductSFTError("formal adapter LoRA identity differs from the SFT lock")
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    parameter_report = _trainable_parameter_report(model)
    processor = AutoProcessor.from_pretrained(
        config["model"]["base_model"], revision=config["model"]["base_revision"]
    )

    product_config_path = Path(eligibility["product_config_path"])
    product_config = load_week8_product_config(product_config_path)
    lock_root = (
        root
        / product_config["dataset"]["output_root"]
        / product_config["week8"]["dataset_version"]
    )
    train_dataset = IndexedWeek7Dataset(
        lock_root / "train" / "image_product_search.jsonl"
    )
    development_rows = list(
        iter_jsonl(lock_root / "development" / "image_product_search.jsonl")
    )
    eval_dataset = IndexedWeek7Dataset(
        lock_root / "development" / "image_product_search.jsonl"
    )
    prompt_version = config["development"]["prompt_version"]

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) != 1:
            raise Week8ProductSFTError("multimodal product collator requires batch size 1")
        messages = structure_aware_messages(
            processor,
            product_training_messages(root, batch[0], prompt_version),
            int(train_config["max_length"]),
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
        )
        inputs["labels"] = assistant_span_labels(
            processor, messages, inputs["input_ids"]
        )
        inputs["_sample_weight"] = torch.tensor(float(batch[0]["sample_weight"]))
        return inputs

    total_updates = math.ceil(
        len(train_dataset) / int(train_config["gradient_accumulation_steps"])
    ) * int(train_config["epochs"])
    evaluation_steps: list[int] = []

    class DecileEvaluationCallback(TrainerCallback):
        def on_train_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            evaluation_steps[:] = decile_evaluation_steps(int(state.max_steps))
            return control

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            if int(state.global_step) in evaluation_steps:
                control.should_evaluate = True
                control.should_save = True
            return control

    class Week8ProductTrainer(Trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._week8_evaluation_cache: dict[int, dict[str, float]] = {}

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            weight = inputs.pop("_sample_weight")
            outputs = model(**inputs)
            loss = outputs.loss * weight.to(outputs.loss.device)
            return (loss, outputs) if return_outputs else loss

        def evaluate(
            self,
            eval_dataset: Any = None,
            ignore_keys: Any = None,
            metric_key_prefix: str = "eval",
        ) -> dict[str, float]:
            del eval_dataset, ignore_keys
            step = int(self.state.global_step)
            if step in self._week8_evaluation_cache:
                return self._week8_evaluation_cache[step]
            started = time.perf_counter()
            backend = _in_memory_backend(self.model, processor, torch)
            with inference_runtime(self.model):
                records = []
                for row in development_rows:
                    record = _run_one_product(
                        root,
                        backend,
                        row,
                        run_id=f"{run_id}_development_step_{step:06d}",
                        prompt_version=prompt_version,
                        max_new_tokens=int(config["development"]["max_new_tokens"]),
                    )
                    record["model_name"] = f"Week8-product-SFT-step-{step}"
                    records.append(record)
            summary = summarize_product_run(root, development_rows, records)
            product = summary["scenarios"]["image_product_search"]
            summary.update(
                {
                    "status": "COMPLETED",
                    "model_role": "week8_product_continuation_checkpoint",
                    "split": "development",
                    "run_id": f"{run_id}_development_step_{step:06d}",
                    "prompt_version": prompt_version,
                    "config_sha256": eligibility["config_sha256"],
                    "product_config_sha256": eligibility["product_config_sha256"],
                    "dataset_lock_sha256": eligibility["dataset_lock_sha256"],
                    "prompt_selection_sha256": eligibility["prompt_development"]["selection_sha256"],
                    "global_step": step,
                }
            )
            evaluation_dir = output_dir / "development_evaluations" / f"step-{step:06d}"
            evaluation_dir.mkdir(parents=True, exist_ok=False)
            raw_path = evaluation_dir / "raw_outputs.jsonl"
            with raw_path.open("x", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
            summary["raw_outputs"] = {
                "path": str(raw_path.resolve()),
                "sha256": sha256_file(raw_path),
                "count": len(records),
            }
            metrics_path = evaluation_dir / "metrics.json"
            _write_json_new(metrics_path, summary)
            aggregate = product["aggregate"]
            metrics = {
                f"{metric_key_prefix}_product_composite": float(product["composite"]),
                f"{metric_key_prefix}_json_compliance": float(aggregate["json_compliance"]),
                f"{metric_key_prefix}_schema_pass": float(aggregate["schema_pass"]),
                f"{metric_key_prefix}_failure_rate": float(summary["failure_rate"]),
                f"{metric_key_prefix}_runtime": time.perf_counter() - started,
            }
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics
            )
            self._week8_evaluation_cache[step] = metrics
            return metrics

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config["epochs"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        warmup_ratio=train_config["warmup_ratio"],
        weight_decay=train_config["weight_decay"],
        max_grad_norm=train_config["max_grad_norm"],
        optim=train_config["optimizer"],
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=train_config["logging_steps"],
        eval_strategy="steps",
        eval_steps=1_000_000_000,
        save_strategy="steps",
        save_steps=1_000_000_000,
        save_total_limit=train_config["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model=train_config["metric_for_best_model"],
        greater_is_better=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Week8ProductTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
        callbacks=[
            DecileEvaluationCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=int(train_config["early_stopping_patience"])
            ),
        ],
    )
    started = time.time()
    result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint else None
        )
    )
    adapter_output = output_dir / "adapter"
    trainer.save_model(str(adapter_output))
    trainer.save_state()
    processor.save_pretrained(str(output_dir / "processor"))
    state = load_peft_weights(str(adapter_output), device="cpu")
    saved_config = PeftConfig.from_pretrained(str(adapter_output))
    if (
        not state
        or not all("lora_" in name for name in state)
        or set(saved_config.target_modules or []) != expected_targets
        or int(saved_config.r) != int(config["lora"]["r"])
        or int(saved_config.lora_alpha) != int(config["lora"]["lora_alpha"])
        or float(saved_config.lora_dropout) != float(config["lora"]["lora_dropout"])
        or str(saved_config.bias) != config["lora"]["bias"]
        or saved_config.base_model_name_or_path != config["model"]["base_model"]
    ):
        raise Week8ProductSFTError("saved adapter failed LoRA-only reload verification")

    checkpoints = sorted(path for path in output_dir.glob("checkpoint-*") if path.is_dir())
    completed_steps = [
        step for step in evaluation_steps if step <= int(trainer.state.global_step)
    ]
    development_artifacts = {}
    for step in completed_steps:
        evaluation_dir = output_dir / "development_evaluations" / f"step-{step:06d}"
        metrics_path = evaluation_dir / "metrics.json"
        raw_path = evaluation_dir / "raw_outputs.jsonl"
        if not metrics_path.is_file() or not raw_path.is_file():
            raise Week8ProductSFTError(f"development evidence is missing at step {step}")
        development_artifacts[str(step)] = {
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
            "raw_outputs_path": str(raw_path.resolve()),
            "raw_outputs_sha256": sha256_file(raw_path),
        }
    adapter_hashes = {
        path.name: sha256_file(path) for path in adapter_output.iterdir() if path.is_file()
    }
    summary = {
        "status": "COMPLETED",
        "run_id": run_id,
        "git_commit": _git_commit(root),
        "config_sha256": eligibility["config_sha256"],
        "product_config_sha256": eligibility["product_config_sha256"],
        "dataset_lock_sha256": eligibility["dataset_lock_sha256"],
        "prompt_selection_sha256": eligibility["prompt_development"]["selection_sha256"],
        "prompt_version": prompt_version,
        "train_samples": len(train_dataset),
        "development_samples": len(eval_dataset),
        "label_source": "programmatic_silver",
        "human_count": 0,
        "maximum_silver_sample_weight": train_config["maximum_silver_sample_weight"],
        "total_update_steps_planned": total_updates,
        "evaluation_steps": evaluation_steps,
        "global_step": int(trainer.state.global_step),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "checkpoints": [path.name for path in checkpoints],
        "checkpoint_hashes": {
            path.name: sha256_file(path / "adapter_model.safetensors")
            for path in checkpoints
            if (path / "adapter_model.safetensors").is_file()
        },
        "development_evaluation_artifacts": development_artifacts,
        "adapter_hashes": adapter_hashes,
        "adapter_only": True,
        "adapter_reload_verified": True,
        "lora_targets": sorted(expected_targets),
        **parameter_report,
        "training_metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "duration_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "resumed_from_checkpoint": (
            str(resume_from_checkpoint) if resume_from_checkpoint else None
        ),
        "continued_from_adapter": {
            "adapter_dir": str(adapter_dir),
            "adapter_model_sha256": sha256_file(adapter_file),
            "expected_checkpoint": config["continuation"]["expected_checkpoint"],
        },
        "test_consumed": False,
    }
    _write_json_new(output_dir / "run_summary.json", summary)
    return summary


def select_week8_product_sft_candidate(
    root: Path,
    config_path: Path,
    prompt_development_dir: Path,
    training_dir: Path,
) -> dict[str, Any]:
    """Bind the best development checkpoint and its reloaded adapter without test access."""

    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    training_dir = Path(training_dir).resolve()
    config = load_week8_product_sft_config(config_path)
    eligibility = validate_week8_product_sft_eligibility(
        root, config_path, prompt_development_dir
    )
    output_path = training_dir / "candidate_selection.json"
    if output_path.exists():
        raise Week8ProductSFTError("candidate selection already exists")
    summary_path = training_dir / "run_summary.json"
    adapter_path = training_dir / "adapter" / "adapter_model.safetensors"
    if not summary_path.is_file() or not adapter_path.is_file():
        raise Week8ProductSFTError("completed SFT summary or adapter is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_checkpoint = Path(str(summary.get("best_checkpoint") or "")).name
    if not best_checkpoint.startswith("checkpoint-"):
        raise Week8ProductSFTError("training summary has no selected checkpoint")
    try:
        step = int(best_checkpoint.removeprefix("checkpoint-"))
    except ValueError as exc:
        raise Week8ProductSFTError("selected checkpoint step is invalid") from exc
    metrics_path = (
        training_dir
        / "development_evaluations"
        / f"step-{step:06d}"
        / "metrics.json"
    )
    raw_path = metrics_path.with_name("raw_outputs.jsonl")
    if not metrics_path.is_file() or not raw_path.is_file():
        raise Week8ProductSFTError("best checkpoint development evidence is missing")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    product_composite = float(
        metrics["scenarios"]["image_product_search"]["composite"]
    )
    if (
        summary.get("status") != "COMPLETED"
        or summary.get("run_id") != config["experiment_identity"]["run_id"]
        or summary.get("config_sha256") != eligibility["config_sha256"]
        or summary.get("product_config_sha256") != eligibility["product_config_sha256"]
        or summary.get("dataset_lock_sha256") != eligibility["dataset_lock_sha256"]
        or summary.get("prompt_selection_sha256")
        != eligibility["prompt_development"]["selection_sha256"]
        or summary.get("adapter_only") is not True
        or summary.get("adapter_reload_verified") is not True
        or summary.get("continued_from_adapter", {}).get("adapter_model_sha256")
        != config["continuation"]["adapter_model_sha256"]
        or summary.get("adapter_hashes", {}).get("adapter_model.safetensors")
        != sha256_file(adapter_path)
        or metrics.get("split") != "development"
        or metrics.get("global_step") != step
        or metrics.get("dataset_lock_sha256") != eligibility["dataset_lock_sha256"]
        or abs(product_composite - float(summary.get("best_metric"))) > 1e-9
    ):
        raise Week8ProductSFTError("candidate evidence differs from the locked SFT run")
    result = {
        "status": "SFT_CANDIDATE_LOCKED",
        "selection_id": config["experiment_identity"]["candidate_selection_id"],
        "run_id": summary["run_id"],
        "selected_prompt_role": config["development"]["prompt_role"],
        "selected_prompt_version": config["development"]["prompt_version"],
        "best_checkpoint": best_checkpoint,
        "development_product_composite": product_composite,
        "adapter_dir": str((training_dir / "adapter").resolve()),
        "adapter_model_sha256": sha256_file(adapter_path),
        "development_metrics": {
            "path": str(metrics_path.resolve()),
            "sha256": sha256_file(metrics_path),
        },
        "development_raw_outputs": {
            "path": str(raw_path.resolve()),
            "sha256": sha256_file(raw_path),
        },
        "run_summary_sha256": sha256_file(summary_path),
        "dataset_lock_sha256": eligibility["dataset_lock_sha256"],
        "label_source": "programmatic_silver",
        "human_count": 0,
        "test_consumed": False,
    }
    _write_json_new(output_path, result)
    return result

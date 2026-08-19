"""Run one frozen Week 3 scenario with a verified Week 6 PEFT adapter."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from src.evaluation.config import load_evaluation_config
from src.evaluation.manifests import (
    load_configured_manifests,
    validate_exclusion_manifest,
    validate_release_provenance,
)
from src.evaluation.provenance import build_run_artifact_hashes
from src.evaluation.runner import (
    load_runtime_settings,
    run_records,
    select_inference_records,
    validate_full_run_readiness,
)
from src.training.week6_qlora import Week6TrainingError, environment_report


SCENARIOS = {
    "image_product_search",
    "after_sales",
    "itinerary_planning",
}
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def sha256_file(path: Path) -> str:
    """Hash one adapter or provenance file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_identity(adapter_dir: Path, expected_model_sha256: str) -> dict[str, Any]:
    """Require an adapter-only directory and bind all delivered files by SHA-256."""
    resolved = Path(adapter_dir)
    if not resolved.is_dir():
        raise Week6TrainingError("final evaluation adapter directory does not exist")
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(resolved.iterdir())
        if path.is_file()
    }
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if not required <= set(hashes):
        raise Week6TrainingError("final evaluation adapter directory is incomplete")
    expected = expected_model_sha256.strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise Week6TrainingError("expected adapter SHA-256 must be 64 lowercase hex characters")
    if hashes["adapter_model.safetensors"] != expected:
        raise Week6TrainingError("final evaluation adapter SHA-256 does not match")
    return {
        "adapter_dir": resolved.as_posix(),
        "adapter_file_sha256": hashes,
    }


def _decode_data_image(url: str) -> Image.Image:
    """Decode the runner's normalized image data URI for the Qwen processor."""
    if not url.startswith("data:image/") or "," not in url:
        raise Week6TrainingError("final evaluation requires normalized image data URIs")
    header, encoded = url.split(",", 1)
    if ";base64" not in header:
        raise Week6TrainingError("final evaluation image data URI must use base64")
    try:
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(payload)) as source:
            return source.convert("RGB")
    except Exception as exc:
        raise Week6TrainingError("final evaluation image data URI is unreadable") from exc


def processor_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI image blocks to the Transformers multimodal contract."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    raise Week6TrainingError("final evaluation message part must be an object")
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append({"type": "text", "text": part["text"]})
                    continue
                if part.get("type") == "image_url":
                    image_url = part.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else None
                    if not isinstance(url, str):
                        raise Week6TrainingError("final evaluation image_url is invalid")
                    parts.append({"type": "image", "image": _decode_data_image(url)})
                    continue
                raise Week6TrainingError("final evaluation message part type is unsupported")
        else:
            raise Week6TrainingError("final evaluation message content is invalid")
        normalized.append({"role": message.get("role"), "content": parts})
    return normalized


def build_transformers_transport(
    *,
    base_model: str,
    adapter_dir: Path,
    max_input_tokens: int,
) -> Callable[[str, dict[str, Any], int], str]:
    """Load one NF4 model plus adapter and return a serial inference transport."""
    report = environment_report(require_cuda=True)
    if report.get("status") != "ok":
        raise Week6TrainingError(
            f"final evaluation environment is not ready: {report.get('status')}"
        )

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    if peft_config.base_model_name_or_path != base_model:
        raise Week6TrainingError("final evaluation adapter points to an unexpected base model")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    processor = AutoProcessor.from_pretrained(base_model)

    def transport(_endpoint: str, payload: dict[str, Any], _timeout: int) -> str:
        messages = processor_messages(copy.deepcopy(payload["messages"]))
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(model.device)
        max_new_tokens = int(payload.get("max_tokens", 2048))
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
                use_cache=True,
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    return transport


def run_final_scenario_evaluation(
    *,
    root: Path,
    config_path: Path,
    run_id: str,
    scenario: str,
    prompt_version: str,
    adapter_dir: Path,
    expected_adapter_sha256: str,
    max_input_tokens: int = 8192,
) -> dict[str, Any]:
    """Evaluate exactly one frozen scenario and persist standard scoreable artifacts."""
    if scenario not in SCENARIOS:
        raise Week6TrainingError(f"unsupported final evaluation scenario: {scenario}")
    project_root = Path(root)
    resolved_config = Path(config_path)
    if not resolved_config.is_absolute():
        resolved_config = project_root / resolved_config
    config = load_evaluation_config(resolved_config)
    if config.get("dataset_version") != "week3_evaluation_v2":
        raise Week6TrainingError("final evaluation must use frozen week3_evaluation_v2")
    manifests = load_configured_manifests(config, root=project_root)
    validate_full_run_readiness(config, manifests)
    all_records = [record for rows in manifests.values() for record in rows]
    validate_exclusion_manifest(
        all_records,
        project_root / config["paths"]["exclusion_manifest"],
    )
    records = select_inference_records(manifests[scenario])
    validate_release_provenance(records)
    identity = adapter_identity(adapter_dir, expected_adapter_sha256)
    runtime = load_runtime_settings(project_root, config)
    if runtime["model_name"] != BASE_MODEL:
        raise Week6TrainingError("final evaluation runtime must use the Week 6 8B base")
    runtime["model_config"] = {
        **runtime["model_config"],
        **identity,
        "evaluation_role": "week6_locked_adapter_final",
        "scenario": scenario,
    }
    transport = build_transformers_transport(
        base_model=BASE_MODEL,
        adapter_dir=Path(adapter_dir),
        max_input_tokens=max_input_tokens,
    )
    return run_records(
        root=project_root,
        records=records,
        runs_dir=project_root / config["paths"]["runs_dir"],
        run_id=run_id,
        mode="live",
        prompt_version=prompt_version,
        run_scope="full",
        runtime=runtime,
        dataset_version=config["dataset_version"],
        artifact_hashes=build_run_artifact_hashes(
            project_root,
            config,
            prompt_version,
            [scenario],
        ),
        transport=transport,
    )

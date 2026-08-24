"""Fail-closed Qwen3-VL runtime for the packaged OTA business endpoints."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests
from pydantic import ValidationError

from src.evaluation.prompting import render_standard_prompt
from src.evaluation.schema_validation import (
    SchemaValidationError,
    load_output_schema,
    validate_output,
)
from src.inference.schemas import (
    DialogueModelOutput,
    DialogueRequest,
    DialogueResponse,
    ModelAttempt,
    TaskRequest,
    TaskResponse,
)
from src.inference.transport_utils import normalize_image_url, strip_json_fence


DEFAULT_RELEASE_CONFIG = "configs/releases/qwen3_vl_system_v1.json"


class RuntimeConfigurationError(RuntimeError):
    """Raised when release identity or local model artifacts are invalid."""


class ModelGenerationError(RuntimeError):
    """Raised when model generation fails or cannot satisfy its output contract."""


class ModelBackend(Protocol):
    """Small backend boundary used by local Transformers and test doubles."""

    def ready(self) -> tuple[bool, str]:
        """Return backend readiness without hiding a failure."""

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None,
        max_new_tokens: int,
    ) -> str:
        """Generate one raw assistant response."""


@dataclass(frozen=True)
class ReleaseSettings:
    """Validated subset of the tracked release manifest."""

    root: Path
    release_id: str
    base_model: str
    base_revision: str
    backend_name: str
    adapter_name: str
    adapter_path: Path | None
    adapter_model_sha256: str
    prompt_versions: dict[str, str]
    schema_versions: dict[str, str]
    max_new_tokens: int
    max_schema_retries: int

    @classmethod
    def load(
        cls,
        root: Path | None = None,
        config_path: Path | None = None,
    ) -> "ReleaseSettings":
        project_root = (root or Path(__file__).resolve().parents[2]).resolve()
        selected_path = config_path or Path(
            os.getenv("TRIP_RELEASE_CONFIG", DEFAULT_RELEASE_CONFIG)
        )
        if not selected_path.is_absolute():
            selected_path = project_root / selected_path
        if not selected_path.is_file():
            raise RuntimeConfigurationError(
                f"release config does not exist: {selected_path}"
            )
        try:
            payload = json.loads(selected_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeConfigurationError(
                f"release config is not valid JSON: {selected_path}"
            ) from exc

        model = payload.get("model", {})
        generation = payload.get("generation", {})
        adapter_env = model.get("adapter_path_env")
        adapter_value = os.getenv(str(adapter_env), "") if adapter_env else ""
        adapter_path = Path(adapter_value).resolve() if adapter_value else None
        settings = cls(
            root=project_root,
            release_id=_required_text(payload, "release_id"),
            base_model=_required_text(model, "base_model"),
            base_revision=_required_text(model, "base_revision"),
            backend_name=_required_text(model, "backend"),
            adapter_name=_required_text(model, "adapter_name"),
            adapter_path=adapter_path,
            adapter_model_sha256=_required_sha256(model, "adapter_model_sha256"),
            prompt_versions=_required_scenario_mapping(payload, "prompts"),
            schema_versions=_required_scenario_mapping(payload, "schemas"),
            max_new_tokens=int(generation.get("max_new_tokens", 3072)),
            max_schema_retries=int(generation.get("max_schema_retries", 1)),
        )
        if settings.backend_name not in {"transformers-peft", "openai-compatible"}:
            raise RuntimeConfigurationError(
                f"unsupported model backend: {settings.backend_name}"
            )
        if settings.max_schema_retries != 1:
            raise RuntimeConfigurationError("release must allow exactly one Schema retry")
        return settings

    def validate_adapter(self) -> tuple[bool, str]:
        """Require the exact selected adapter file before declaring readiness."""
        if self.adapter_path is None:
            return False, "TRIP_ADAPTER_DIR is not configured"
        adapter_file = self.adapter_path / "adapter_model.safetensors"
        config_file = self.adapter_path / "adapter_config.json"
        if not adapter_file.is_file() or not config_file.is_file():
            return False, "adapter_model.safetensors or adapter_config.json is missing"
        actual = _sha256_file(adapter_file)
        if actual != self.adapter_model_sha256:
            return False, (
                "adapter hash mismatch: "
                f"expected {self.adapter_model_sha256}, got {actual}"
            )
        return True, "ok"


@dataclass(frozen=True)
class GenerationResult:
    """Raw model output and exact token counts for one local generation."""

    content: str
    input_tokens: int
    output_tokens: int


class OpenAICompatibleBackend:
    """Fail-closed optional backend for a compatible Qwen3-VL server."""

    def __init__(self, settings: ReleaseSettings) -> None:
        self.settings = settings
        self.base_url = os.getenv("MODEL_API_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("MODEL_API_KEY", "").strip()
        self.timeout_seconds = int(os.getenv("MODEL_TIMEOUT_SECONDS", "180"))

    def ready(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "MODEL_API_BASE_URL is not configured"
        return self.settings.validate_adapter()

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None,
        max_new_tokens: int,
    ) -> str:
        ready, reason = self.ready()
        if not ready:
            raise RuntimeConfigurationError(reason)
        payload: dict[str, Any] = {
            "model": self.settings.adapter_name,
            "messages": _normalize_message_images(messages),
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "enable_thinking": False,
        }
        if response_format:
            payload["response_format"] = response_format
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = (
            f"{self.base_url}/chat/completions"
            if self.base_url.endswith("/v1")
            else f"{self.base_url}/v1/chat/completions"
        )
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ModelGenerationError("model request failed") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelGenerationError("model returned empty content")
        return content


class TransformersPeftBackend:
    """Lazy local Qwen3-VL loader using NF4 and the selected PEFT adapter."""

    def __init__(self, settings: ReleaseSettings) -> None:
        self.settings = settings
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    def ready(self) -> tuple[bool, str]:
        valid, reason = self.settings.validate_adapter()
        if not valid:
            return valid, reason
        try:
            self._ensure_loaded()
        except Exception as exc:
            return False, f"model backend failed to load: {exc}"
        return True, "ok"

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None,
        max_new_tokens: int,
    ) -> str:
        return self.generate_with_usage(
            messages,
            response_format=response_format,
            max_new_tokens=max_new_tokens,
        ).content

    def generate_with_usage(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None,
        max_new_tokens: int,
    ) -> GenerationResult:
        """Generate once and retain measured input/output token counts."""

        del response_format  # Schema is enforced after generation for this backend.
        self._ensure_loaded()
        try:
            normalized = _transformers_messages(messages)
            inputs = self._processor.apply_chat_template(
                normalized,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False,
            )
            device = next(self._model.parameters()).device
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            input_tokens = int(inputs["input_ids"].shape[1])
            trimmed = [
                output[len(input_ids) :]
                for input_ids, output in zip(inputs["input_ids"], generated)
            ]
            output_tokens = int(trimmed[0].shape[0])
            content = self._processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:
            raise ModelGenerationError(
                f"local model generation failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not content.strip():
            raise ModelGenerationError("model returned empty content")
        return GenerationResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        valid, reason = self.settings.validate_adapter()
        if not valid:
            raise RuntimeConfigurationError(reason)
        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen3VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeConfigurationError(
                "install requirements-training.txt for the Transformers runtime"
            ) from exc

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            self.settings.base_model,
            revision=self.settings.base_revision,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=quantization,
            trust_remote_code=False,
        )
        self._model = PeftModel.from_pretrained(
            base,
            str(self.settings.adapter_path),
            is_trainable=False,
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.settings.base_model,
            revision=self.settings.base_revision,
            trust_remote_code=False,
        )
        self._torch = torch


class ScenarioService:
    """Render, generate, validate, and retry the three business scenarios."""

    def __init__(
        self,
        settings: ReleaseSettings,
        backend: ModelBackend,
    ) -> None:
        self.settings = settings
        self.backend = backend

    def ready(self) -> dict[str, Any]:
        adapter_ready, adapter_reason = self.settings.validate_adapter()
        backend_ready, backend_reason = self.backend.ready()
        schema_errors = []
        for scenario, version in self.settings.schema_versions.items():
            try:
                load_output_schema(self.settings.root, scenario, version)
                prompt_root = (
                    self.settings.root
                    / "configs"
                    / "evaluation"
                    / "prompts"
                    / self.settings.prompt_versions[scenario]
                )
                if not (prompt_root / "common.yaml").is_file() or not (
                    prompt_root / f"{scenario}.yaml"
                ).is_file():
                    raise RuntimeConfigurationError(
                        f"prompt files are missing for {scenario}"
                    )
            except Exception as exc:
                schema_errors.append(f"{scenario}: {exc}")
        contracts_ready = not schema_errors
        ready = adapter_ready and backend_ready and contracts_ready
        return {
            "status": "ready" if ready else "not_ready",
            "release_id": self.settings.release_id,
            "model": self.settings.base_model,
            "adapter": self.settings.adapter_name,
            "backend": self.settings.backend_name,
            "checks": {
                "adapter": {"ok": adapter_ready, "detail": adapter_reason},
                "model_backend": {"ok": backend_ready, "detail": backend_reason},
                "prompt_schema_contracts": {
                    "ok": contracts_ready,
                    "detail": "ok" if contracts_ready else "; ".join(schema_errors),
                },
            },
        }

    def run_task(self, scenario: str, request: TaskRequest) -> TaskResponse:
        prompt_version = self.settings.prompt_versions[scenario]
        schema_version = self.settings.schema_versions[scenario]
        input_context = {
            "images": [{"path": path} for path in request.image_urls],
            "text_constraints": request.text_context,
        }
        rendered = render_standard_prompt(
            self.settings.root,
            scenario,
            input_context,
            prompt_version,
        )
        parsed, attempts = self._generate_validated(
            scenario=scenario,
            schema_version=schema_version,
            messages=rendered["messages"],
            response_format=rendered.get("response_format"),
        )
        return TaskResponse(
            scenario=scenario,
            result=parsed,
            schema_valid=True,
            prompt_version=prompt_version,
            model=self.settings.base_model,
            adapter=self.settings.adapter_name,
            release_id=self.settings.release_id,
            attempts=attempts,
            total_latency_ms=sum(item.latency_ms for item in attempts),
        )

    def run_dialogue(self, request: DialogueRequest) -> DialogueResponse:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是专业 OTA 多模态对话助手。承接已确认状态，不编造图片或业务事实。"
                    "仅输出 JSON：reply、state_updates、tool_calls。"
                ),
            }
        ]
        for index, turn in enumerate(request.messages):
            content: Any = turn.content
            if turn.role == "user" and index == 0 and request.image_urls:
                content = [{"type": "text", "text": turn.content}]
                content.extend(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                    for image_url in request.image_urls
                )
            messages.append({"role": turn.role, "content": content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "当前结构化状态："
                    + json.dumps(request.state, ensure_ascii=False, sort_keys=True)
                ),
            }
        )
        parsed, attempts = self._generate_dialogue(messages)
        state = dict(request.state)
        state.update(parsed.state_updates)
        return DialogueResponse(
            reply=parsed.reply,
            state=state,
            tool_calls=parsed.tool_calls,
            model=self.settings.base_model,
            adapter=self.settings.adapter_name,
            release_id=self.settings.release_id,
            attempts=attempts,
            total_latency_ms=sum(item.latency_ms for item in attempts),
        )

    def _generate_validated(
        self,
        *,
        scenario: str,
        schema_version: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], list[ModelAttempt]]:
        attempts: list[ModelAttempt] = []
        active_messages = list(messages)
        for attempt_number in range(1, self.settings.max_schema_retries + 2):
            started = time.perf_counter()
            raw = self.backend.generate(
                active_messages,
                response_format=response_format,
                max_new_tokens=self.settings.max_new_tokens,
            )
            error: str | None = None
            parsed: Any = None
            try:
                parsed = json.loads(strip_json_fence(raw))
                validate_output(
                    self.settings.root,
                    scenario,
                    parsed,
                    schema_version,
                )
            except (json.JSONDecodeError, SchemaValidationError) as exc:
                error = str(exc)
            latency_ms = (time.perf_counter() - started) * 1000
            attempts.append(
                ModelAttempt(
                    attempt=attempt_number,
                    raw_output=raw,
                    error=error,
                    latency_ms=latency_ms,
                )
            )
            if error is None and isinstance(parsed, dict):
                return parsed, attempts
            if attempt_number > self.settings.max_schema_retries:
                break
            active_messages = _correction_messages(active_messages, raw, error or "")
        raise ModelGenerationError(
            f"model output failed {scenario} Schema after {len(attempts)} attempts: "
            f"{attempts[-1].error}"
        )

    def _generate_dialogue(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[DialogueModelOutput, list[ModelAttempt]]:
        attempts: list[ModelAttempt] = []
        active_messages = list(messages)
        for attempt_number in range(1, self.settings.max_schema_retries + 2):
            started = time.perf_counter()
            raw = self.backend.generate(
                active_messages,
                response_format={"type": "json_object"},
                max_new_tokens=self.settings.max_new_tokens,
            )
            error: str | None = None
            parsed: DialogueModelOutput | None = None
            try:
                parsed = DialogueModelOutput.model_validate_json(strip_json_fence(raw))
            except ValidationError as exc:
                error = str(exc)
            attempts.append(
                ModelAttempt(
                    attempt=attempt_number,
                    raw_output=raw,
                    error=error,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            )
            if parsed is not None:
                return parsed, attempts
            if attempt_number > self.settings.max_schema_retries:
                break
            active_messages = _correction_messages(active_messages, raw, error or "")
        raise ModelGenerationError(
            f"dialogue output failed Schema after {len(attempts)} attempts: "
            f"{attempts[-1].error}"
        )


def build_service(root: Path | None = None) -> ScenarioService:
    """Build the configured production service without enabling fallback."""
    settings = ReleaseSettings.load(root=root)
    if settings.backend_name == "openai-compatible":
        backend: ModelBackend = OpenAICompatibleBackend(settings)
    else:
        backend = TransformersPeftBackend(settings)
    return ScenarioService(settings, backend)


def _correction_messages(
    messages: list[dict[str, Any]],
    raw: str,
    error: str,
) -> list[dict[str, Any]]:
    return [
        *messages,
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                "上一次输出未通过 JSON/Schema 校验。"
                f"错误：{error}。请重新读取原输入，只输出修正后的完整 JSON；"
                "不得解释、猜测缺失事实或引用本条纠错指令作为证据。"
            ),
        },
    ]


def _normalize_message_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = json.loads(json.dumps(messages, ensure_ascii=False))
    for message in normalized:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            image = part.get("image_url", {})
            if isinstance(image, dict) and isinstance(image.get("url"), str):
                image["url"] = normalize_image_url(image["url"])
    return normalized


def _transformers_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI image blocks to the Qwen3-VL processor contract."""

    normalized = _normalize_message_images(messages)
    for message in normalized:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
            continue
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if part.get("type") != "image_url":
                continue
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not isinstance(url, str) or not url:
                raise RuntimeConfigurationError(
                    "Transformers image_url block has no URL"
                )
            content[index] = {"type": "image", "image": url}
    return normalized


def _required_text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigurationError(f"release field {field} must be non-empty text")
    return value.strip()


def _required_sha256(payload: dict[str, Any], field: str) -> str:
    value = _required_text(payload, field).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeConfigurationError(f"release field {field} must be SHA-256")
    return value


def _required_scenario_mapping(
    payload: dict[str, Any],
    field: str,
) -> dict[str, str]:
    value = payload.get(field)
    scenarios = {"image_product_search", "after_sales", "itinerary_planning"}
    if not isinstance(value, dict) or set(value) != scenarios:
        raise RuntimeConfigurationError(
            f"release field {field} must define exactly {sorted(scenarios)}"
        )
    return {scenario: _required_text(value, scenario) for scenario in scenarios}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

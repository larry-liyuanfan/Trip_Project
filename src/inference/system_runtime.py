"""Fail-closed Qwen3-VL runtime for the packaged OTA business endpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
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
from src.inference.processor_cache import ProcessorInputCache, processor_signature
from src.inference.transport_utils import normalize_image_url, strip_json_fence


DEFAULT_RELEASE_CONFIG = "configs/releases/qwen3_vl_system_v1.json"
DIALOGUE_PROMPT_VERSIONS = {
    "system_repair_dialogue_v1",
    "week8_dialogue_first_turn_v1",
    "week8_dialogue_first_turn_v2",
    "week8_dialogue_first_turn_v3",
    "week8_dialogue_deterministic_v4",
}
DIALOGUE_EXECUTION_MODES = {
    "model_generated_contract",
    "deterministic_contract_v1",
}
_BUDGET_UPDATE_RE = re.compile(
    r"预算(?:改成|调整为|设为|改为|换成)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>万|千|[kK]|元)?"
)
_DAYS_UPDATE_RE = re.compile(
    r"(?:行程|天数|安排)?(?:改成|调整为|设为|改为|延长到|缩短到)\s*"
    r"(?P<value>\d+|[一二两三四五六七八九十]+)\s*天"
)
_CITY_UPDATE_RE = re.compile(
    r"(?:城市|目的地)(?:改成|调整为|设为|改为|换成)\s*"
    r"(?P<value>[\u4e00-\u9fffA-Za-z .-]{2,32}?)(?=[，。；,;]|$)"
)
_PREFERENCE_UPDATE_RE = re.compile(
    r"偏好(?:改成|调整为|设为|改为|换成)\s*"
    r"(?P<value>[^，。；,;]{1,48})"
)
_PACE_UPDATE_RE = re.compile(
    r"(?:安排|行程|节奏)(?:改得|调整得|改成|调整为|设为|改为)?\s*"
    r"(?:更)?(?P<value>松弛|轻松|悠闲|放松|紧凑|充实|快节奏)(?:一些|一点)?"
)
_POSITIVE_UPDATE_CUE_RE = re.compile(
    r"改成|调整为|设为|改为|换成|延长到|缩短到|增加|删除|取消|改得"
)
_NEGATED_POSITIVE_UPDATE_RE = re.compile(
    r"(?:不要|别|无需|不需要).{0,8}"
    r"(?:改成|调整为|设为|改为|换成|延长到|缩短到|增加|删除|取消|改得)"
)


class RuntimeConfigurationError(RuntimeError):
    """Raised when release identity or local model artifacts are invalid."""


class ModelGenerationError(RuntimeError):
    """Raised when model generation fails or cannot satisfy its output contract."""

    def __init__(self, message: str, *, attempts: list[Any] | None = None) -> None:
        super().__init__(message)
        self.attempts = list(attempts or [])


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
    max_new_tokens_by_scenario: dict[str, int]
    max_schema_retries: int
    dialogue_prompt_version: str = "system_repair_dialogue_v1"
    dialogue_max_new_tokens: int = 512
    dialogue_execution_mode: str = "model_generated_contract"
    dialogue_semantic_fallback_enabled: bool = False
    processor_cache_max_entries: int = 0
    visual_max_pixels: int | None = None

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
        dialogue = payload.get("dialogue", {})
        adapter_env = model.get("adapter_path_env")
        adapter_value = os.getenv(str(adapter_env), "") if adapter_env else ""
        adapter_path = Path(adapter_value).resolve() if adapter_value else None
        max_new_tokens = int(generation.get("max_new_tokens", 3072))
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
            max_new_tokens=max_new_tokens,
            max_new_tokens_by_scenario=_scenario_token_limits(
                generation,
                default=max_new_tokens,
            ),
            max_schema_retries=int(generation.get("max_schema_retries", 1)),
            dialogue_prompt_version=(
                _required_text(dialogue, "prompt_version")
                if dialogue
                else "system_repair_dialogue_v1"
            ),
            dialogue_max_new_tokens=int(dialogue.get("max_new_tokens", 512)),
            dialogue_execution_mode=str(
                dialogue.get("execution_mode", "model_generated_contract")
            ),
            dialogue_semantic_fallback_enabled=dialogue.get(
                "semantic_fallback_enabled", False
            ),
            processor_cache_max_entries=int(
                generation.get("processor_cache_max_entries", 0)
            ),
            visual_max_pixels=(
                int(generation["visual_max_pixels"])
                if generation.get("visual_max_pixels") is not None
                else None
            ),
        )
        if settings.backend_name not in {"transformers-peft", "openai-compatible"}:
            raise RuntimeConfigurationError(
                f"unsupported model backend: {settings.backend_name}"
            )
        if settings.max_schema_retries != 1:
            raise RuntimeConfigurationError("release must allow exactly one Schema retry")
        if settings.dialogue_prompt_version not in DIALOGUE_PROMPT_VERSIONS:
            raise RuntimeConfigurationError(
                "unsupported dialogue prompt version: "
                f"{settings.dialogue_prompt_version}"
            )
        if settings.dialogue_max_new_tokens <= 0:
            raise RuntimeConfigurationError("dialogue max_new_tokens must be positive")
        if settings.dialogue_execution_mode not in DIALOGUE_EXECUTION_MODES:
            raise RuntimeConfigurationError(
                "unsupported dialogue execution mode: "
                f"{settings.dialogue_execution_mode}"
            )
        if not isinstance(settings.dialogue_semantic_fallback_enabled, bool):
            raise RuntimeConfigurationError(
                "dialogue semantic_fallback_enabled must be boolean"
            )
        if settings.processor_cache_max_entries < 0:
            raise RuntimeConfigurationError(
                "generation processor_cache_max_entries cannot be negative"
            )
        if settings.visual_max_pixels is not None and settings.visual_max_pixels <= 0:
            raise RuntimeConfigurationError(
                "generation visual_max_pixels must be positive"
            )
        if (
            settings.dialogue_prompt_version == "week8_dialogue_deterministic_v4"
            and settings.dialogue_execution_mode != "deterministic_contract_v1"
        ):
            raise RuntimeConfigurationError(
                "week8 dialogue v4 requires deterministic_contract_v1"
            )
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
        self._processor_cache = ProcessorInputCache(
            settings.processor_cache_max_entries
        )

    def configure_processor_cache(self, max_entries: int) -> None:
        """Reset and resize the bounded CPU preprocessing cache."""

        self._processor_cache.clear(max_entries=max_entries)

    def processor_cache_snapshot(self) -> dict[str, int | float | None]:
        """Return cache observability without exposing cached tensors."""

        return self._processor_cache.snapshot()

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

        self._ensure_loaded()
        try:
            normalized = _transformers_messages(messages)
            cache_key = ProcessorInputCache.key(
                normalized,
                processor_signature(self._processor),
            )
            inputs = self._processor_cache.get(cache_key)
            if inputs is None:
                inputs = self._processor.apply_chat_template(
                    normalized,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    truncation=False,
                )
                self._processor_cache.put(cache_key, inputs)
            device = next(self._model.parameters()).device
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            generation_constraints: dict[str, Any] = {}
            if response_format and response_format.get("type") == "json_schema":
                try:
                    from lmformatenforcer import JsonSchemaParser
                    from lmformatenforcer.integrations.transformers import (
                        build_transformers_prefix_allowed_tokens_fn,
                    )
                except ImportError as exc:
                    raise RuntimeConfigurationError(
                        "install lm-format-enforcer for JSON Schema correction"
                    ) from exc
                contract = response_format.get("json_schema", {})
                schema = contract.get("schema") if isinstance(contract, dict) else None
                if not isinstance(schema, dict):
                    raise RuntimeConfigurationError(
                        "JSON Schema correction requires a schema object"
                    )
                tokenizer = getattr(self._processor, "tokenizer", None)
                if tokenizer is None:
                    raise RuntimeConfigurationError(
                        "processor does not expose a tokenizer for constrained decoding"
                    )
                generation_constraints["prefix_allowed_tokens_fn"] = (
                    build_transformers_prefix_allowed_tokens_fn(
                        tokenizer,
                        JsonSchemaParser(schema),
                    )
                )
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    **generation_constraints,
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
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is not None and self.settings.visual_max_pixels is not None:
            image_processor.max_pixels = self.settings.visual_max_pixels
        self._torch = torch


class ScenarioService:
    """Render, generate, validate, and retry the three business scenarios."""

    def __init__(
        self,
        settings: ReleaseSettings,
        backend: ModelBackend,
        *,
        dialogue_prompt_version: str | None = None,
        dialogue_max_new_tokens: int | None = None,
        dialogue_execution_mode: str | None = None,
        dialogue_semantic_fallback_enabled: bool | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.dialogue_prompt_version = (
            dialogue_prompt_version or settings.dialogue_prompt_version
        )
        self.dialogue_max_new_tokens = int(
            dialogue_max_new_tokens or settings.dialogue_max_new_tokens
        )
        self.dialogue_execution_mode = (
            dialogue_execution_mode or settings.dialogue_execution_mode
        )
        self.dialogue_semantic_fallback_enabled = (
            settings.dialogue_semantic_fallback_enabled
            if dialogue_semantic_fallback_enabled is None
            else dialogue_semantic_fallback_enabled
        )
        if self.dialogue_prompt_version not in DIALOGUE_PROMPT_VERSIONS:
            raise RuntimeConfigurationError(
                "unsupported dialogue prompt version: "
                f"{self.dialogue_prompt_version}"
            )
        if self.dialogue_max_new_tokens <= 0:
            raise RuntimeConfigurationError("dialogue max_new_tokens must be positive")
        if self.dialogue_execution_mode not in DIALOGUE_EXECUTION_MODES:
            raise RuntimeConfigurationError(
                "unsupported dialogue execution mode: "
                f"{self.dialogue_execution_mode}"
            )
        if not isinstance(self.dialogue_semantic_fallback_enabled, bool):
            raise RuntimeConfigurationError(
                "dialogue semantic fallback flag must be boolean"
            )
        if (
            self.dialogue_prompt_version == "week8_dialogue_deterministic_v4"
            and self.dialogue_execution_mode != "deterministic_contract_v1"
        ):
            raise RuntimeConfigurationError(
                "week8 dialogue deterministic prompt requires deterministic mode"
            )

    def ready(self) -> dict[str, Any]:
        adapter_ready, adapter_reason = self.settings.validate_adapter()
        backend_ready, backend_reason = self.backend.ready()
        schema_errors = []
        for scenario, version in self.settings.schema_versions.items():
            try:
                load_output_schema(self.settings.root, scenario, version)
                render_standard_prompt(
                    self.settings.root,
                    scenario,
                    {
                        "images": [{"path": "readiness-placeholder.jpg"}],
                        "text_constraints": (
                            "行程共1天" if scenario == "itinerary_planning" else None
                        ),
                    },
                    self.settings.prompt_versions[scenario],
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
        if self.dialogue_execution_mode == "deterministic_contract_v1":
            return self._run_deterministic_dialogue(request)
        messages = _dialogue_messages(
            request,
            prompt_version=self.dialogue_prompt_version,
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

    def _run_deterministic_dialogue(
        self,
        request: DialogueRequest,
    ) -> DialogueResponse:
        """Route and assemble the public contract in code, not model output."""

        updates, needs_semantic_fallback = _deterministic_state_updates(request)
        attempts: list[ModelAttempt] = []
        fallback_status = "NOT_USED"
        if needs_semantic_fallback and self.dialogue_semantic_fallback_enabled:
            try:
                fallback_updates, attempts = self._generate_state_update_fallback(
                    request
                )
                updates.update(fallback_updates)
                fallback_status = "SUCCEEDED"
            except ModelGenerationError as exc:
                attempts = list(exc.attempts)
                fallback_status = "FAILED_SAFE"
        state = dict(request.state)
        state.update(updates)
        parsed = DialogueModelOutput(
            reply=_deterministic_dialogue_reply(
                request,
                updates,
                fallback_failed=fallback_status == "FAILED_SAFE",
            ),
            state_updates=updates,
            tool_calls=[],
        )
        return DialogueResponse(
            reply=parsed.reply,
            state=state,
            tool_calls=[],
            model=self.settings.base_model,
            adapter=self.settings.adapter_name,
            release_id=self.settings.release_id,
            attempts=attempts,
            total_latency_ms=sum(item.latency_ms for item in attempts),
            execution_mode="DETERMINISTIC_CONTRACT",
            semantic_fallback_status=fallback_status,
        )

    def _generate_state_update_fallback(
        self,
        request: DialogueRequest,
    ) -> tuple[dict[str, Any], list[ModelAttempt]]:
        """Use the model only to extract ambiguous state changes."""

        attempts: list[ModelAttempt] = []
        active_messages = _state_update_fallback_messages(request)
        response_format = _state_update_fallback_response_format()
        for attempt_number in range(1, self.settings.max_schema_retries + 2):
            started = time.perf_counter()
            raw, input_tokens, output_tokens = _generate_once(
                self.backend,
                active_messages,
                response_format=response_format,
                max_new_tokens=min(self.dialogue_max_new_tokens, 128),
            )
            error: str | None = None
            updates: dict[str, Any] | None = None
            try:
                payload = json.loads(strip_json_fence(raw))
                if not isinstance(payload, dict) or set(payload) != {"state_updates"}:
                    raise ValueError(
                        "semantic fallback must contain exactly state_updates"
                    )
                candidate = payload["state_updates"]
                if not isinstance(candidate, dict) or len(candidate) > 16:
                    raise ValueError(
                        "semantic fallback state_updates must be a bounded object"
                    )
                if any(
                    not isinstance(key, str)
                    or not _is_finite_dialogue_value(value)
                    for key, value in candidate.items()
                ):
                    raise ValueError(
                        "semantic fallback values must be finite JSON scalars"
                    )
                updates = candidate
            except (json.JSONDecodeError, ValueError) as exc:
                error = str(exc)
                updates = None
            attempts.append(
                ModelAttempt(
                    attempt=attempt_number,
                    raw_output=raw,
                    error=error,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )
            if updates is not None:
                return updates, attempts
            if attempt_number > self.settings.max_schema_retries:
                break
            active_messages = [
                *active_messages,
                {
                    "role": "user",
                    "content": (
                        "上次状态抽取无效。只输出完整 JSON："
                        '{"state_updates":{}}；值仅可为字符串、数字、布尔或 null。'
                    ),
                },
            ]
        raise ModelGenerationError(
            "semantic state fallback failed after "
            f"{len(attempts)} attempts: {attempts[-1].error}",
            attempts=attempts,
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
        active_response_format = response_format
        for attempt_number in range(1, self.settings.max_schema_retries + 2):
            started = time.perf_counter()
            raw, input_tokens, output_tokens = _generate_once(
                self.backend,
                active_messages,
                response_format=active_response_format,
                max_new_tokens=self.settings.max_new_tokens_by_scenario[scenario],
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
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )
            if error is None and isinstance(parsed, dict):
                return parsed, attempts
            if attempt_number > self.settings.max_schema_retries:
                break
            active_messages = _correction_messages(
                active_messages,
                raw,
                error or "",
                scenario=scenario,
            )
            active_response_format = _json_schema_response_format(
                self.settings.root,
                scenario,
                schema_version,
            )
        raise ModelGenerationError(
            f"model output failed {scenario} Schema after {len(attempts)} attempts: "
            f"{attempts[-1].error}",
            attempts=attempts,
        )

    def _generate_dialogue(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[DialogueModelOutput, list[ModelAttempt]]:
        attempts: list[ModelAttempt] = []
        active_messages = list(messages)
        active_response_format: dict[str, Any] = (
            _dialogue_v3_response_format()
            if self.dialogue_prompt_version == "week8_dialogue_first_turn_v3"
            else {"type": "json_object"}
        )
        for attempt_number in range(1, self.settings.max_schema_retries + 2):
            started = time.perf_counter()
            raw, input_tokens, output_tokens = _generate_once(
                self.backend,
                active_messages,
                response_format=active_response_format,
                max_new_tokens=self.dialogue_max_new_tokens,
            )
            error: str | None = None
            parsed: DialogueModelOutput | None = None
            try:
                if (
                    self.dialogue_prompt_version in {
                        "week8_dialogue_first_turn_v2",
                        "week8_dialogue_first_turn_v3",
                    }
                    and not raw.startswith("{")
                ):
                    raise ValueError(
                        "week8 dialogue v2/v3 output must start with the JSON object"
                    )
                dialogue_payload = json.loads(strip_json_fence(raw))
                required_keys = {"reply", "state_updates", "tool_calls"}
                if not isinstance(dialogue_payload, dict) or set(dialogue_payload) != required_keys:
                    actual_keys = (
                        sorted(dialogue_payload)
                        if isinstance(dialogue_payload, dict)
                        else type(dialogue_payload).__name__
                    )
                    raise ValueError(
                        "dialogue output must contain exactly "
                        f"{sorted(required_keys)}; got {actual_keys}"
                    )
                parsed = DialogueModelOutput.model_validate(dialogue_payload)
                if self.dialogue_prompt_version == "week8_dialogue_first_turn_v3":
                    _validate_dialogue_v3_payload(dialogue_payload)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                error = str(exc)
                parsed = None
            attempts.append(
                ModelAttempt(
                    attempt=attempt_number,
                    raw_output=raw,
                    error=error,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )
            if parsed is not None:
                return parsed, attempts
            if attempt_number > self.settings.max_schema_retries:
                break
            active_messages = _correction_messages(
                active_messages,
                raw,
                error or "",
                scenario="dialogue",
            )
            # v1/v2 保持历史自由 JSON 纠错路径；v3 的有限值 Schema 可继续约束重试。
            active_response_format = (
                _dialogue_v3_response_format()
                if self.dialogue_prompt_version == "week8_dialogue_first_turn_v3"
                else {"type": "json_object"}
            )
        raise ModelGenerationError(
            f"dialogue output failed Schema after {len(attempts)} attempts: "
            f"{attempts[-1].error}",
            attempts=attempts,
        )


def _generate_once(
    backend: ModelBackend,
    messages: list[dict[str, Any]],
    *,
    response_format: dict[str, Any] | None,
    max_new_tokens: int,
) -> tuple[str, int | None, int | None]:
    """Use measured token counts when the backend exposes them."""

    generate_with_usage = getattr(backend, "generate_with_usage", None)
    if callable(generate_with_usage):
        generated = generate_with_usage(
            messages,
            response_format=response_format,
            max_new_tokens=max_new_tokens,
        )
        if isinstance(generated, GenerationResult):
            return generated.content, generated.input_tokens, generated.output_tokens
        raise ModelGenerationError("backend returned an invalid measured generation")
    return (
        backend.generate(
            messages,
            response_format=response_format,
            max_new_tokens=max_new_tokens,
        ),
        None,
        None,
    )


def _dialogue_v3_response_format() -> dict[str, Any]:
    """Return the bounded dialogue contract accepted by LMFE's JSON parser."""

    finite_value = {"type": ["string", "number", "boolean", "null"]}
    arguments = {
        "type": "object",
        "maxProperties": 16,
        "additionalProperties": dict(finite_value),
    }
    schema = {
        "type": "object",
        "properties": {
            "reply": {"type": "string", "minLength": 1, "maxLength": 2000},
            "state_updates": {
                "type": "object",
                "maxProperties": 16,
                "additionalProperties": dict(finite_value),
            },
            "tool_calls": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "function": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 128,
                        },
                        "arguments": arguments,
                    },
                    "required": ["function", "arguments"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["reply", "state_updates", "tool_calls"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "dialogue_week8_first_turn_v3",
            "strict": True,
            "schema": schema,
        },
    }


def _validate_dialogue_v3_payload(payload: dict[str, Any]) -> None:
    """Validate the same bounded value/tool subset even if a backend ignores LMFE."""

    if len(payload["reply"]) > 2000:
        raise ValueError("dialogue v3 reply exceeds maxLength")
    state_updates = payload["state_updates"]
    if len(state_updates) > 16:
        raise ValueError("dialogue v3 state_updates exceeds maxProperties")
    for key, value in state_updates.items():
        if not isinstance(key, str) or not _is_finite_dialogue_value(value):
            raise ValueError(
                "dialogue v3 state_updates values must be finite JSON scalars"
            )
    tool_calls = payload["tool_calls"]
    if len(tool_calls) > 4:
        raise ValueError("dialogue v3 tool_calls exceeds maxItems")
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict) or set(tool_call) != {
            "function",
            "arguments",
        }:
            raise ValueError(
                "dialogue v3 tool_calls items require function and arguments"
            )
        function = tool_call["function"]
        arguments = tool_call["arguments"]
        if not isinstance(function, str) or not 1 <= len(function) <= 128:
            raise ValueError("dialogue v3 tool function must be non-empty text")
        if not isinstance(arguments, dict) or len(arguments) > 16:
            raise ValueError("dialogue v3 tool arguments must be a bounded object")
        if any(
            not isinstance(key, str) or not _is_finite_dialogue_value(value)
            for key, value in arguments.items()
        ):
            raise ValueError(
                "dialogue v3 tool argument values must be finite JSON scalars"
            )


def _is_finite_dialogue_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return not isinstance(value, float) or math.isfinite(value)
    return False


def _deterministic_state_updates(
    request: DialogueRequest,
) -> tuple[dict[str, Any], bool]:
    """Extract bounded explicit updates and flag only ambiguous change requests."""

    latest_user = next(
        (turn.content for turn in reversed(request.messages) if turn.role == "user"),
        "",
    )
    if not latest_user:
        return {}, False
    updates: dict[str, Any] = {}

    budget_match = _BUDGET_UPDATE_RE.search(latest_user)
    if budget_match and not _match_is_negated(latest_user, budget_match.start()):
        value = float(budget_match.group("value"))
        unit = budget_match.group("unit")
        if unit in {"千", "k", "K"}:
            value *= 1000
        elif unit == "万":
            value *= 10000
        if math.isfinite(value) and 0 <= value <= 1_000_000_000:
            updates["budget"] = int(value) if value.is_integer() else value

    days_match = _DAYS_UPDATE_RE.search(latest_user)
    if days_match and not _match_is_negated(latest_user, days_match.start()):
        days = _parse_positive_days(days_match.group("value"))
        if days is not None:
            updates["days"] = days

    city_match = _CITY_UPDATE_RE.search(latest_user)
    if city_match and not _match_is_negated(latest_user, city_match.start()):
        updates["city"] = city_match.group("value").strip()

    preference_match = _PREFERENCE_UPDATE_RE.search(latest_user)
    if preference_match and not _match_is_negated(
        latest_user,
        preference_match.start(),
    ):
        updates["preference"] = preference_match.group("value").strip()

    pace_match = _PACE_UPDATE_RE.search(latest_user)
    if pace_match and not _match_is_negated(latest_user, pace_match.start()):
        pace = pace_match.group("value")
        updates["pace"] = (
            "relaxed" if pace in {"松弛", "轻松", "悠闲", "放松"} else "packed"
        )

    positive_cue = bool(_POSITIVE_UPDATE_CUE_RE.search(latest_user))
    negative_only = bool(_NEGATED_POSITIVE_UPDATE_RE.search(latest_user))
    needs_semantic_fallback = positive_cue and not negative_only and not updates
    return updates, needs_semantic_fallback


def _match_is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 8) : start]
    return any(marker in prefix for marker in ("不要", "别", "无需", "不需要"))


def _parse_positive_days(value: str) -> int | None:
    if value.isdigit():
        parsed = int(value)
        return parsed if 1 <= parsed <= 365 else None
    numerals = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        tens = numerals.get(left, 1) if left else 1
        ones = numerals.get(right, 0) if right else 0
        parsed = tens * 10 + ones
    else:
        parsed = numerals.get(value, 0)
    return parsed if 1 <= parsed <= 365 else None


def _deterministic_dialogue_reply(
    request: DialogueRequest,
    updates: dict[str, Any],
    *,
    fallback_failed: bool,
) -> str:
    if fallback_failed:
        return "未能可靠解析新的状态变化，已保留原状态，请换一种简短说法。"
    if updates:
        labels = {
            "budget": "预算",
            "days": "天数",
            "city": "城市",
            "preference": "偏好",
            "pace": "节奏",
        }
        changed = "、".join(labels.get(key, key) for key in sorted(updates))
        return f"已更新{changed}，其余已确认条件保持不变。"
    if request.image_urls:
        return "已进入对话路径并接收新图片，将承接当前状态继续处理。"
    return "已承接当前对话状态，将按你的最新要求继续处理。"


def _state_update_fallback_messages(
    request: DialogueRequest,
) -> list[dict[str, Any]]:
    latest_user = next(
        (turn.content for turn in reversed(request.messages) if turn.role == "user"),
        "",
    )
    state_json = json.dumps(request.state, ensure_ascii=False, sort_keys=True)
    return [
        {
            "role": "system",
            "content": (
                "只提取用户本轮明确新增或修改的 OTA 对话状态，不生成回复或工具调用。"
                "只输出 state_updates；无法确定时输出空对象。值只能是字符串、数字、布尔或 null。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"当前状态：{state_json}\n"
                f"本轮用户文本：{latest_user}\n"
                '仅输出 {"state_updates":{}}。'
            ),
        },
    ]


def _state_update_fallback_response_format() -> dict[str, Any]:
    finite_value = {"type": ["string", "number", "boolean", "null"]}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "dialogue_state_update_fallback_v1",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "state_updates": {
                        "type": "object",
                        "maxProperties": 16,
                        "additionalProperties": finite_value,
                    }
                },
                "required": ["state_updates"],
                "additionalProperties": False,
            },
        },
    }


def _dialogue_messages(
    request: DialogueRequest,
    *,
    prompt_version: str,
) -> list[dict[str, Any]]:
    """Render the versioned dialogue route without mutating caller history."""

    state_json = json.dumps(request.state, ensure_ascii=False, sort_keys=True)
    if prompt_version == "system_repair_dialogue_v1":
        system_content = (
            "你是专业 OTA 多模态对话助手。承接已确认状态，不编造图片或业务事实。"
            "仅输出 JSON：reply、state_updates、tool_calls。"
        )
    elif prompt_version in {
        "week8_dialogue_first_turn_v1",
        "week8_dialogue_first_turn_v2",
        "week8_dialogue_first_turn_v3",
    }:
        system_content = (
            "你正在处理 OTA 多轮对话端点，而不是商品理解、售后抽取或行程抽取端点。"
            "即使用户上传图片，也不得输出 business_category、scene_tags、itinerary、"
            "evidence、confidence 等单任务结构。只输出一个 JSON 对象，且顶层必须恰好"
            "包含 reply、state_updates、tool_calls 三个键：reply 是面向用户的简短非空回复；"
            "state_updates 只记录本轮明确新增或修改的状态，没有变化时为 {}；tool_calls 只在"
            "确需外部工具时填写对象数组，否则为 []。不得输出 Markdown、解释或额外顶层键。"
            "承接已确认状态，不编造图片、历史结果或业务事实。"
            f"当前结构化状态为：{state_json}。"
        )
    else:
        raise RuntimeConfigurationError(
            f"unsupported dialogue prompt version: {prompt_version}"
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_content}
    ]
    first_user_has_image = False
    for turn in request.messages:
        content: Any = turn.content
        if (
            turn.role == "user"
            and not first_user_has_image
            and request.image_urls
        ):
            content = [{"type": "text", "text": turn.content}]
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
                for image_url in request.image_urls
            )
            first_user_has_image = True
        messages.append({"role": turn.role, "content": content})
    if prompt_version in {
        "week8_dialogue_first_turn_v2",
        "week8_dialogue_first_turn_v3",
    }:
        messages.append(
            {
                "role": "user",
                "content": (
                    "路由控制：这是对话端点。下一个输出的首字符必须是 {，且只能输出"
                    '三键骨架 {"reply":"简短直接回复","state_updates":{},"tool_calls":[]}；'
                    "替换值即可，不得输出任务标签、Markdown 或解释。"
                ),
            }
        )
    if prompt_version == "system_repair_dialogue_v1":
        # 保留正式历史基线的尾部状态消息，确保固定样本对比只改变候选 Prompt。
        messages.append(
            {
                "role": "user",
                "content": f"当前结构化状态：{state_json}",
            }
        )
    return messages


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
    _raw: str,
    error: str,
    *,
    scenario: str | None = None,
) -> list[dict[str, Any]]:
    itinerary_contract = ""
    if scenario == "itinerary_planning":
        itinerary_contract = (
            "行程纠错时必须使用以下九键骨架，不得在任何 ] 或 } 后插入自然语言："
            '{"style_preferences":[],"hard_constraints":[],"soft_constraints":[],'
            '"required_itinerary_elements":["daily_schedule"],'
            '"itinerary":[{"day_index":1,"date":null,"summary":"简短摘要",'
            '"activities":[{"start_time":null,"end_time":null,"place_name":null,'
            '"activity":"简短活动","transport":null,"source_evidence":[]}]}],'
            '"constraint_check":[],"observed_evidence":[],"unknown_fields":[],'
            '"confidence":null}。只替换骨架中的内容值，不改变键、类型或嵌套层级；'
            "若完整多日内容可能无法闭合，先输出一个 Schema 合法的精简日程。"
        )
    elif scenario == "dialogue":
        itinerary_contract = (
            "对话纠错必须使用以下三键骨架："
            '{"reply":"简短直接回复","state_updates":{},"tool_calls":[]}。'
            "reply 必须是非空字符串；state_updates 必须是对象；tool_calls 必须是数组；"
            "不得输出场景任务标签、evidence、confidence 或骨架之外的顶层键。"
        )
    return [
        *messages,
        {
            "role": "user",
            "content": (
                "上一次输出未通过 JSON/Schema 校验。"
                f"错误：{error}。请重新读取原输入，只输出修正后的完整 JSON；"
                "不要续写或局部修补上一次输出，必须从第一个顶层键开始重新生成；"
                "保留 Schema 要求的全部字段，数组严格遵守 minItems/maxItems，"
                "删除重复证据并保持内容紧凑，确保 JSON 在生成上限内完整闭合；"
                "若错误包含 maxLength，必须缩短对应字符串；若缺少 required 字段，"
                "必须从第一个顶层键开始重写完整对象；若 JSON 未闭合，必须从头重写并闭合；"
                "不得解释、猜测缺失事实或引用本条纠错指令作为证据。"
                f"{itinerary_contract}"
            ),
        },
    ]


def _json_schema_response_format(
    root: Path,
    scenario: str,
    schema_version: str,
) -> dict[str, Any]:
    schema = load_output_schema(root, scenario, schema_version)
    if scenario == "itinerary_planning":
        itinerary = schema["properties"]["itinerary"]
        itinerary["maxItems"] = min(int(itinerary.get("maxItems", 4)), 4)
        activities = itinerary["items"]["properties"]["activities"]
        activities["maxItems"] = min(int(activities.get("maxItems", 2)), 2)
        schema["properties"]["constraint_check"]["maxItems"] = min(
            int(schema["properties"]["constraint_check"].get("maxItems", 12)),
            12,
        )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{scenario}_{schema_version}",
            "strict": True,
            "schema": schema,
        },
    }


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


def _scenario_token_limits(
    generation: dict[str, Any],
    *,
    default: int,
) -> dict[str, int]:
    scenarios = {"image_product_search", "after_sales", "itinerary_planning"}
    value = generation.get("max_new_tokens_by_scenario")
    if value is None:
        return {scenario: default for scenario in scenarios}
    if not isinstance(value, dict) or set(value) != scenarios:
        raise RuntimeConfigurationError(
            "max_new_tokens_by_scenario must define exactly the three scenarios"
        )
    limits = {scenario: int(value[scenario]) for scenario in scenarios}
    if any(limit <= 0 for limit in limits.values()):
        raise RuntimeConfigurationError("scenario token limits must be positive")
    return limits


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

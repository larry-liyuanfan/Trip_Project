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
DIALOGUE_PROMPT_VERSIONS = {
    "system_repair_dialogue_v1",
    "week8_dialogue_first_turn_v1",
}


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
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.dialogue_prompt_version = (
            dialogue_prompt_version or settings.dialogue_prompt_version
        )
        self.dialogue_max_new_tokens = int(
            dialogue_max_new_tokens or settings.dialogue_max_new_tokens
        )
        if self.dialogue_prompt_version not in DIALOGUE_PROMPT_VERSIONS:
            raise RuntimeConfigurationError(
                "unsupported dialogue prompt version: "
                f"{self.dialogue_prompt_version}"
            )
        if self.dialogue_max_new_tokens <= 0:
            raise RuntimeConfigurationError("dialogue max_new_tokens must be positive")

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
        active_response_format: dict[str, Any] = {"type": "json_object"}
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
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                error = str(exc)
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
            # Dialogue state and tool payloads intentionally allow arbitrary JSON
            # objects. lm-format-enforcer cannot handle their boolean
            # additionalProperties contract, so the retry remains model-level and
            # is validated by DialogueModelOutput after generation.
            active_response_format = {"type": "json_object"}
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
    elif prompt_version == "week8_dialogue_first_turn_v1":
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

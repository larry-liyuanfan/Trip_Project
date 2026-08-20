"""Shared, auditable Transformers generation protocol for Week 7."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


LATENCY_PROTOCOL_VERSION = "week7_transformers_latency_v4"


def _set_cache_flag(target: Any, value: bool) -> tuple[Any, Any] | None:
    if target is None or not hasattr(target, "use_cache"):
        return None
    previous = target.use_cache
    target.use_cache = value
    return target, previous


@contextmanager
def inference_runtime(model: Any) -> Iterator[None]:
    """Use the same KV-cache/eval state for standalone and in-training evaluation."""
    was_training = bool(getattr(model, "training", False))
    restored = []
    for target in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        state = _set_cache_flag(target, True)
        if state is not None and all(state[0] is not item[0] for item in restored):
            restored.append(state)
    model.eval()
    try:
        yield
    finally:
        for target, previous in restored:
            target.use_cache = previous
        if was_training:
            model.train()


def _synchronize(torch_module: Any, device: Any) -> None:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.synchronize(device)


def generate_record(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    sample_id: str,
    run_id: str,
    model_name: str,
    max_new_tokens: int,
    warmup: bool = False,
) -> dict[str, Any]:
    """Generate once with synchronized end-to-end timing and token-count evidence."""
    import torch

    device = next(model.parameters()).device
    _synchronize(torch, device)
    started = time.perf_counter()
    raw = ""
    error = None
    input_token_count = None
    generated_token_count = None
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
        )
        input_token_count = int(inputs["input_ids"].shape[1])
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        suffix = generated[:, input_token_count:]
        generated_token_count = int(suffix.shape[1])
        raw = processor.batch_decode(suffix, skip_special_tokens=True)[0].strip()
    except (RuntimeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        _synchronize(torch, device)
    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "model_name": model_name,
        "raw_output": raw,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "failed": error is not None,
        "error": error,
        "input_token_count": input_token_count,
        "generated_token_count": generated_token_count,
        "generation_max_new_tokens": max_new_tokens,
        "latency_protocol": LATENCY_PROTOCOL_VERSION,
        "warmup": warmup,
    }

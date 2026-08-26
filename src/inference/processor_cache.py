"""Bounded cache for deterministic multimodal processor outputs."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from threading import Lock
from typing import Any, Mapping


class ProcessorInputCache:
    """Reuse immutable CPU processor outputs for repeated media requests.

    The runtime only shallow-copies the cached mapping. Tensor values must therefore
    be treated as immutable and moved to the model device with non-mutating ``to``.
    """

    def __init__(self, max_entries: int = 0) -> None:
        if max_entries < 0:
            raise ValueError("processor cache max_entries cannot be negative")
        self.max_entries = int(max_entries)
        self._values: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = Lock()

    @staticmethod
    def key(messages: list[dict[str, Any]], signature: Mapping[str, Any]) -> str:
        """Build a stable key from normalized messages and processor settings."""

        serialized = json.dumps(
            {"messages": messages, "processor": dict(signature)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.max_entries:
            return None
        with self._lock:
            value = self._values.pop(key, None)
            if value is None:
                self._misses += 1
                return None
            self._values[key] = value
            self._hits += 1
            return dict(value)

    def put(self, key: str, value: Mapping[str, Any]) -> None:
        if not self.max_entries:
            return
        with self._lock:
            self._values.pop(key, None)
            self._values[key] = dict(value)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)

    def clear(self, *, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries < 0:
            raise ValueError("processor cache max_entries cannot be negative")
        with self._lock:
            self._values.clear()
            self._hits = 0
            self._misses = 0
            if max_entries is not None:
                self.max_entries = int(max_entries)

    def snapshot(self) -> dict[str, int | float | None]:
        with self._lock:
            requests = self._hits + self._misses
            return {
                "max_entries": self.max_entries,
                "entries": len(self._values),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / requests if requests else None,
            }


def processor_signature(processor: Any) -> dict[str, Any]:
    """Capture image settings that materially change processor output tensors."""

    image_processor = getattr(processor, "image_processor", None)
    return {
        "processor_class": type(processor).__name__,
        "image_processor_class": (
            type(image_processor).__name__ if image_processor is not None else None
        ),
        "max_pixels": getattr(image_processor, "max_pixels", None),
        "min_pixels": getattr(image_processor, "min_pixels", None),
        "size": getattr(image_processor, "size", None),
        "do_resize": getattr(image_processor, "do_resize", None),
    }

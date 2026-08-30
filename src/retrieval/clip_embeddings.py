"""Public CLIP image encoder used by OTA vector indexing and visual search."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Iterable


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_VECTOR_DIMENSION = 512


class ClipEmbeddingError(RuntimeError):
    """Raised when CLIP dependencies, inputs, or outputs are invalid."""


class CLIPImageEncoder:
    """Lazy, bounded CLIP encoder returning L2-normalized 512-D vectors."""

    def __init__(
        self,
        *,
        model_id: str = CLIP_MODEL_ID,
        device: str = "auto",
        batch_size: int = 20,
    ) -> None:
        if batch_size < 1:
            raise ClipEmbeddingError("batch_size must be positive")
        self.model_id = model_id
        self.requested_device = device
        self.batch_size = batch_size
        self.device = "unloaded"
        self._torch: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._execution_lock = RLock()

    def ready(self) -> tuple[bool, str]:
        try:
            self._ensure_loaded()
        except Exception as exc:
            return False, str(exc)
        return True, "ok"

    def encode(self, image_paths: Iterable[str | Path]) -> list[list[float]]:
        """Encode every readable path in caller order; never skip silently."""
        with self._execution_lock:
            return self._encode_locked(image_paths)

    def _encode_locked(self, image_paths: Iterable[str | Path]) -> list[list[float]]:
        paths = [Path(path) for path in image_paths]
        if not paths:
            raise ClipEmbeddingError("at least one image is required")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ClipEmbeddingError(f"image files do not exist: {missing[:3]}")
        self._ensure_loaded()
        from PIL import Image

        vectors: list[list[float]] = []
        for start in range(0, len(paths), self.batch_size):
            batch = paths[start : start + self.batch_size]
            images = []
            try:
                for path in batch:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
            except (OSError, ValueError) as exc:
                raise ClipEmbeddingError(f"unreadable image: {path}") from exc
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            with self._torch.inference_mode():
                encoded = self._model.get_image_features(**inputs)
                encoded = self._torch.nn.functional.normalize(encoded, p=2, dim=1)
            vectors.extend(encoded.detach().cpu().float().tolist())
        _validate_vectors(vectors, expected_count=len(paths))
        return vectors

    def _ensure_loaded(self) -> None:
        with self._execution_lock:
            self._ensure_loaded_locked()

    def _ensure_loaded_locked(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ClipEmbeddingError(
                "CLIP dependencies are unavailable; use the dedicated CLIP runtime"
            ) from exc
        if self.requested_device == "cuda" and not torch.cuda.is_available():
            raise ClipEmbeddingError("CLIP requested CUDA but CUDA is unavailable")
        device = (
            "cuda" if self.requested_device == "auto" and torch.cuda.is_available()
            else "cpu" if self.requested_device == "auto"
            else self.requested_device
        )
        model = CLIPModel.from_pretrained(self.model_id).to(device).eval()
        processor = CLIPProcessor.from_pretrained(self.model_id)
        # 依赖全部成功后再发布，处理器加载失败不能留下“已就绪”的半成品。
        self._torch, self._processor, self._model, self.device = torch, processor, model, device


def _validate_vectors(vectors: list[list[float]], *, expected_count: int) -> None:
    if len(vectors) != expected_count:
        raise ClipEmbeddingError(
            f"CLIP returned {len(vectors)} vectors for {expected_count} images"
        )
    for index, vector in enumerate(vectors):
        if len(vector) != CLIP_VECTOR_DIMENSION:
            raise ClipEmbeddingError(
                f"CLIP vector {index} has dimension {len(vector)}, expected 512"
            )
        norm = sum(value * value for value in vector) ** 0.5
        if not 0.999 <= norm <= 1.001:
            raise ClipEmbeddingError(f"CLIP vector {index} is not L2-normalized")

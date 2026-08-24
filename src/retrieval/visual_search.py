"""Fail-closed CLIP plus Milvus visual-search service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from src.retrieval.clip_embeddings import CLIP_MODEL_ID


class ImageEncoder(Protocol):
    model_id: str

    def ready(self) -> tuple[bool, str]: ...

    def encode(self, image_paths: list[str | Path]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]: ...


class VisualSearchService:
    """Encode one local query image and return normalized Milvus hits."""

    def __init__(self, encoder: ImageEncoder, store: VectorStore) -> None:
        self.encoder = encoder
        self.store = store

    def ready(self) -> tuple[bool, str]:
        return self.encoder.ready()

    def search(
        self,
        image_path: str | Path,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        vectors = self.encoder.encode([image_path])
        constrained_filters = dict(filters or {})
        constrained_filters["embedding_model"] = CLIP_MODEL_ID
        result = self.store.search(
            vectors[0],
            top_k=top_k,
            filters=constrained_filters,
        )
        hits = result[0] if result else []
        normalized = []
        for hit in hits:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            normalized.append(
                {
                    "vector_id": hit.get("id"),
                    "score": hit.get("distance"),
                    **entity,
                }
            )
        return normalized

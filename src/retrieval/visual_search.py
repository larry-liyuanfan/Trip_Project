"""Fail-closed CLIP plus Milvus visual-search service."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Protocol

from src.retrieval.clip_embeddings import CLIP_MODEL_ID
from src.retrieval.query_inputs import user_query_attributes
from src.retrieval.week8_hybrid import metadata_ranking, fuse_rankings


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

    def __init__(self, encoder: ImageEncoder, store: VectorStore, config=None) -> None:
        self.encoder = encoder
        self.store = store
        self.config = config or json.loads((Path(__file__).resolve().parents[2] / "configs/retrieval/production_hybrid_v1.json").read_text(encoding="utf-8"))

    def ready(self) -> tuple[bool, str]:
        return self.encoder.ready()

    def search(
        self,
        image_path: str | Path | None,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        query_text: str = "",
        retrieval_mode: str = "embedding",
    ) -> list[dict[str, Any]]:
        if retrieval_mode not in {"embedding", "keyword", "hybrid"}:
            raise ValueError("unsupported retrieval mode")
        if retrieval_mode == "embedding" and image_path is None:
            raise ValueError("embedding search requires an image")
        attributes = user_query_attributes(query_text, filters)
        constrained_filters = dict(filters or {})
        constrained_filters.update(attributes)
        constrained_filters["embedding_model"] = CLIP_MODEL_ID
        pool = max(top_k, int(self.config["candidate_pool_size"]))
        result = []
        if image_path is not None and retrieval_mode != "keyword":
            vectors = self.encoder.encode([image_path])
            result = self.store.search(vectors[0], top_k=top_k if retrieval_mode == "embedding" else pool,
                                       filters=constrained_filters)
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
        if retrieval_mode == "embedding":
            return normalized
        candidates = self.store.query_metadata(filters=constrained_filters, limit=pool)
        rows = {str(item["image_id"]): {"sample_id": str(item["image_id"]), "metadata": item} for item in candidates}
        for item in normalized:
            rows[str(item["image_id"])] = {"sample_id": str(item["image_id"]), "metadata": item}
        image_hits = [{"row": rows[str(item["image_id"])], "image_score": item["score"]} for item in normalized]
        # 无可解释条件的纯文本不能返回任意热门商家冒充查询结果。
        if not attributes and query_text.strip():
            rows = {key: row for key, row in rows.items() if query_text.casefold() in
                    " ".join(str(v) for v in row["metadata"].values()).casefold()}
            image_hits = [hit for hit in image_hits if hit["row"]["sample_id"] in rows]
        metadata_hits = metadata_ranking(self.config, attributes, list(rows.values()), top_k=pool)
        if retrieval_mode == "keyword" or image_path is None:
            return [{**hit["row"]["metadata"], "score": hit["metadata_score"], "source_channels": ["metadata"]} for hit in metadata_hits[:top_k]]
        hits = fuse_rankings(self.config, image_hits, metadata_hits, method="hybrid_weighted", top_k=top_k)
        return [{**hit["row"]["metadata"], "score": hit["ranking_score"], "source_channels": hit["source_channels"],
                 "component_scores": hit["component_scores"]} for hit in hits]

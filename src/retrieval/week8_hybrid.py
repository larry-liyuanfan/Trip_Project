"""Week 8 configurable Milvus/offline image channel and hybrid fusion."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from src.retrieval.milvus_vectors import (
    FILTER_FIELDS,
    OTAMilvusVectorStore,
    build_filter_expression,
    load_milvus_config,
)


class Week8HybridError(RuntimeError):
    """Raised when a hybrid backend or collection identity is invalid."""


class OfflineImageChannel:
    """Exact in-memory cosine channel used only as an explicit fallback."""

    def __init__(
        self,
        vectors: Any,
        *,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> None:
        self.vectors = vectors
        self.backend_name = "offline_cosine"
        self.fallback_used = fallback_used
        self.fallback_reason = fallback_reason
        self.last_latency_ms: float | None = None

    def search(
        self,
        query_vector: Any,
        index_rows: list[dict[str, Any]],
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        candidates = [
            row
            for row in index_rows
            if not filters or _matches_filters(row["metadata"], filters)
        ]
        hits = [
            {
                "row": row,
                "image_score": _dot(query_vector, self.vectors[row["vector_index"]]),
                "source_backend": self.backend_name,
            }
            for row in candidates
        ]
        hits.sort(key=lambda hit: (-hit["image_score"], hit["row"]["sample_id"]))
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return hits[:top_k]

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "offline_fallback": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


class MilvusImageChannel:
    """Real PyMilvus search path for an exact Week 8 index-only collection."""

    def __init__(
        self,
        config: dict[str, Any],
        index_rows: list[dict[str, Any]],
        *,
        client: Any | None = None,
    ) -> None:
        hybrid = config["hybrid"]
        self.collection = hybrid["milvus_collection"]
        self.index_rows = index_rows
        self.rows_by_image_id = {
            row["metadata"]["image_id"]: row for row in index_rows
        }
        self.fallback_used = False
        self.fallback_reason = None
        self.last_latency_ms: float | None = None
        self.ef = int(hybrid["milvus_ef"])
        self.uri = _resolved_milvus_uri(hybrid)
        self.milvus_mode = _milvus_mode(self.uri)
        self.index_type = (
            hybrid["milvus_lite_index_type"]
            if self.milvus_mode == "lite_file"
            else hybrid["milvus_remote_index_type"]
        )
        self.backend_name = (
            "milvus_lite_flat_cosine"
            if self.milvus_mode == "lite_file"
            else "milvus_remote_hnsw_cosine"
        )
        self.client = client or _create_milvus_client(hybrid)
        self._verify_collection_identity()

    def _verify_collection_identity(self) -> None:
        if not self.client.has_collection(collection_name=self.collection):
            raise Week8HybridError(f"Milvus collection is missing: {self.collection}")
        # Milvus Lite 文件由新进程打开时集合默认处于 released 状态。
        self.client.load_collection(collection_name=self.collection)
        count_rows = self.client.query(
            collection_name=self.collection,
            filter="",
            output_fields=["count(*)"],
        )
        if (
            not isinstance(count_rows, list)
            or len(count_rows) != 1
            or count_rows[0].get("count(*)") != len(self.index_rows)
        ):
            raise Week8HybridError("Milvus index row count does not match the locked index")

        expected = set(self.rows_by_image_id)
        observed: set[str] = set()
        batch_size = 100
        image_ids = sorted(expected)
        for offset in range(0, len(image_ids), batch_size):
            batch = image_ids[offset : offset + batch_size]
            rows = self.client.query(
                collection_name=self.collection,
                filter=build_filter_expression({"image_id": batch}),
                output_fields=["image_id"],
                limit=len(batch),
            )
            observed.update(
                row.get("image_id") for row in rows if isinstance(row.get("image_id"), str)
            )
        if observed != expected:
            raise Week8HybridError("Milvus image identities do not match the locked index")

    def search(
        self,
        query_vector: Any,
        index_rows: list[dict[str, Any]],
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        if index_rows is not self.index_rows and {
            row["sample_id"] for row in index_rows
        } != {row["sample_id"] for row in self.index_rows}:
            raise Week8HybridError("Milvus channel received a different locked index")
        constrained = dict(filters or {})
        constrained["embedding_model"] = "openai/clip-vit-base-patch32"
        started = time.perf_counter()
        raw = self.client.search(
            collection_name=self.collection,
            data=[_vector_list(query_vector)],
            anns_field="multimodal_vector",
            filter=build_filter_expression(constrained),
            limit=top_k,
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": self.ef} if self.index_type == "HNSW" else {},
            },
            output_fields=sorted(FILTER_FIELDS),
        )
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        hits = raw[0] if isinstance(raw, list) and raw else []
        normalized = []
        for hit in hits:
            entity = hit.get("entity") if isinstance(hit, dict) else None
            entity = entity if isinstance(entity, dict) else {}
            image_id = entity.get("image_id")
            row = self.rows_by_image_id.get(image_id)
            if row is None:
                raise Week8HybridError(f"Milvus returned an unlocked image_id: {image_id}")
            normalized.append(
                {
                    "row": row,
                    "image_score": float(hit.get("distance")),
                    "source_backend": self.backend_name,
                }
            )
        return normalized

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "collection": self.collection,
            "milvus_mode": self.milvus_mode,
            "index_type": self.index_type,
            "offline_fallback": False,
            "fallback_reason": None,
        }


def build_preferred_image_channel(
    config: dict[str, Any],
    index_rows: list[dict[str, Any]],
    vectors: Any,
    *,
    backend: str | None = None,
    milvus_factory: Callable[[], Any] | None = None,
) -> MilvusImageChannel | OfflineImageChannel:
    """Prefer real Milvus and explicitly record any allowed offline fallback."""
    hybrid = config["hybrid"]
    requested = backend or hybrid["backend_preference"]
    if requested not in {"auto", "milvus", "offline"}:
        raise Week8HybridError(f"unsupported hybrid backend: {requested}")
    if requested == "offline":
        return OfflineImageChannel(
            vectors,
            fallback_used=False,
            fallback_reason="explicit_offline_backend",
        )
    try:
        return milvus_factory() if milvus_factory else MilvusImageChannel(config, index_rows)
    except Exception as exc:
        if requested == "milvus" or not hybrid["offline_fallback_enabled"]:
            raise Week8HybridError(f"Milvus backend unavailable: {exc}") from exc
        return OfflineImageChannel(
            vectors,
            fallback_used=True,
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def load_locked_index_into_milvus(
    config: dict[str, Any],
    vectors: Any,
    index_rows: list[dict[str, Any]],
    *,
    client: Any | None = None,
    sdk: Any | None = None,
) -> dict[str, Any]:
    """Load only locked index rows; never place dev/final queries in Milvus."""
    hybrid = config["hybrid"]
    base_config = load_milvus_config(hybrid["milvus_config"])
    base_config["collection"]["name"] = config["hybrid"]["milvus_collection"]
    uri = _resolved_milvus_uri(hybrid)
    mode = _milvus_mode(uri)
    index_type = (
        hybrid["milvus_lite_index_type"]
        if mode == "lite_file"
        else hybrid["milvus_remote_index_type"]
    )
    active_client = client or _create_milvus_client(hybrid)
    store = OTAMilvusVectorStore(base_config, client=active_client, sdk=sdk)
    store.create_collection()
    stats = store.client.get_collection_stats(collection_name=store.collection)
    try:
        current = int(stats["row_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Week8HybridError("cannot read Milvus physical row count") from exc
    expected = len(index_rows)
    if current not in {0, expected}:
        raise Week8HybridError(f"Milvus collection is partial: {current}/{expected}")
    if current == 0:
        entities = []
        for row in index_rows:
            metadata = row["metadata"]
            entities.append(
                {
                    **{
                        field: metadata[field]
                        for field in FILTER_FIELDS
                    },
                    "multimodal_vector": _vector_list(vectors[row["vector_index"]]),
                }
            )
        store.batch_insert(entities)
        store.client.flush(collection_name=store.collection)
        _build_week8_indexes(store, index_type=index_type, include_scalar=mode != "lite_file")
        status = "LOADED"
    else:
        status = "ALREADY_LOADED"
    store.client.load_collection(collection_name=store.collection)
    channel = MilvusImageChannel(config, index_rows, client=store.client)
    return {
        "status": status,
        "backend": channel.backend_name,
        "milvus_mode": mode,
        "index_type": index_type,
        "collection": store.collection,
        "index_count": expected,
        "query_rows_loaded": 0,
        "identity_validation": "PASS",
    }


def fuse_rankings(
    config: dict[str, Any],
    image_hits: list[dict[str, Any]],
    metadata_hits: list[dict[str, Any]],
    *,
    method: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Fuse image and metadata rankings with traceable component scores."""
    if method not in {"hybrid_rrf", "hybrid_weighted"}:
        raise Week8HybridError(f"unsupported fusion method: {method}")
    hybrid = config["hybrid"]
    image_by_id = {hit["row"]["sample_id"]: (rank, hit) for rank, hit in enumerate(image_hits, 1)}
    metadata_by_id = {
        hit["row"]["sample_id"]: (rank, hit) for rank, hit in enumerate(metadata_hits, 1)
    }
    fused = []
    for sample_id in sorted(set(image_by_id) | set(metadata_by_id)):
        image_record = image_by_id.get(sample_id)
        metadata_record = metadata_by_id.get(sample_id)
        row = (image_record or metadata_record)[1]["row"]
        image_score = image_record[1]["image_score"] if image_record else None
        metadata_score = metadata_record[1]["metadata_score"] if metadata_record else None
        if method == "hybrid_rrf":
            rrf_k = float(hybrid["rrf_k"])
            score = sum(
                1.0 / (rrf_k + record[0])
                for record in (image_record, metadata_record)
                if record is not None
            )
        else:
            weights = hybrid["weighted_fusion"]
            normalized_image = max(0.0, min(1.0, (float(image_score) + 1.0) / 2.0)) if image_score is not None else 0.0
            normalized_metadata = float(metadata_score) if metadata_score is not None else 0.0
            score = (
                float(weights["image"]) * normalized_image
                + float(weights["metadata"]) * normalized_metadata
            )
        fused.append(
            {
                "row": row,
                "clip_score": image_score,
                "ranking_score": score,
                "source_channels": [
                    channel
                    for channel, record in (("image", image_record), ("metadata", metadata_record))
                    if record is not None
                ],
                "component_ranks": {
                    "image": image_record[0] if image_record else None,
                    "metadata": metadata_record[0] if metadata_record else None,
                },
                "component_scores": {
                    "image": image_score,
                    "metadata": metadata_score,
                },
            }
        )
    fused.sort(key=lambda hit: (-hit["ranking_score"], hit["row"]["sample_id"]))
    return fused[:top_k]


def _create_milvus_client(hybrid: dict[str, Any]) -> Any:
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise Week8HybridError("PyMilvus is unavailable") from exc
    uri = _resolved_milvus_uri(hybrid)
    if _milvus_mode(uri) == "lite_file":
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {
        "uri": uri,
        "timeout": int(hybrid["milvus_timeout_seconds"]),
    }
    token_env = hybrid.get("milvus_token_env")
    if token_env and os.getenv(token_env):
        options["token"] = os.environ[token_env]
    return MilvusClient(**options)


def _resolved_milvus_uri(hybrid: dict[str, Any]) -> str:
    return os.getenv("MILVUS_URI", str(hybrid["milvus_uri"]))


def _milvus_mode(uri: str) -> str:
    return "remote_service" if "://" in uri else "lite_file"


def _build_week8_indexes(
    store: OTAMilvusVectorStore,
    *,
    index_type: str,
    include_scalar: bool,
) -> None:
    params = store.client.prepare_index_params()
    vector_options: dict[str, Any] = {
        "field_name": "multimodal_vector",
        "index_name": "multimodal_vector_week8",
        "index_type": index_type,
        "metric_type": "COSINE",
    }
    if index_type == "HNSW":
        index = store.config["index"]
        vector_options["params"] = {
            "M": index["M"],
            "efConstruction": index["efConstruction"],
        }
    else:
        vector_options["params"] = {}
    params.add_index(**vector_options)
    if include_scalar:
        for field_name in store.config["index"]["scalar_fields"]:
            params.add_index(
                field_name=field_name,
                index_name=f"{field_name}_scalar_week8",
                index_type="INVERTED",
            )
    store.client.create_index(
        collection_name=store.collection,
        index_params=params,
        sync=True,
    )


def _metadata_score(query: dict[str, Any], candidate: dict[str, Any], weights: dict[str, Any]) -> float:
    numerator = denominator = 0.0
    for field, weight in weights.items():
        value = query.get(field)
        if value == "unknown":
            continue
        numeric_weight = float(weight)
        denominator += numeric_weight
        numerator += numeric_weight * float(value == candidate.get(field))
    return numerator / denominator if denominator else 0.0


def metadata_ranking(
    config: dict[str, Any],
    query_metadata: dict[str, Any],
    index_rows: list[dict[str, Any]],
    *,
    top_k: int,
    filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    weights = config["evaluation"]["relevance_weights"]
    hits = []
    for row in index_rows:
        if filters and not _matches_filters(row["metadata"], filters):
            continue
        hits.append(
            {
                "row": row,
                "metadata_score": _metadata_score(query_metadata, row["metadata"], weights),
            }
        )
    hits.sort(key=lambda hit: (-hit["metadata_score"], hit["row"]["sample_id"]))
    return hits[:top_k]


def _matches_filters(metadata: dict[str, Any], filters: dict[str, str]) -> bool:
    return all(metadata.get(field) == value for field, value in filters.items())


def _dot(left: Any, right: Any) -> float:
    try:
        return float(left @ right)
    except (TypeError, ValueError):
        return sum(float(a) * float(b) for a, b in zip(left, right))


def _vector_list(vector: Any) -> list[float]:
    return [float(value) for value in vector]

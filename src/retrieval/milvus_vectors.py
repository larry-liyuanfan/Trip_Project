"""Configured Milvus collection and bounded OTA image-vector operations."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from src.data.yelp_paths import parse_simple_yaml


FILTER_FIELDS = {
    "business_id",
    "image_id",
    "business_category",
    "city",
    "star_rating",
    "price_range",
    "image_type",
    "embedding_model",
}
STRING_FIELDS = FILTER_FIELDS - {"star_rating"}
REQUIRED_ENTITY_FIELDS = FILTER_FIELDS | {"multimodal_vector"}


class MilvusVectorError(ValueError):
    """Raised when collection configuration, entities, or filters are invalid."""


def load_milvus_config(path: Path | str) -> dict[str, Any]:
    """Load and validate the standalone collection configuration."""
    payload = parse_simple_yaml(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MilvusVectorError("Milvus config must be a mapping")
    collection = payload.get("collection")
    index = payload.get("index")
    connection = payload.get("connection")
    if not all(isinstance(value, dict) for value in (collection, index, connection)):
        raise MilvusVectorError("connection, collection, and index mappings are required")
    if collection.get("name") != "ota_business_image_vector":
        raise MilvusVectorError("collection name must be ota_business_image_vector")
    if collection.get("vector_dimension") != 512:
        raise MilvusVectorError("multimodal_vector dimension must be 512")
    if collection.get("embedding_model") != "openai/clip-vit-base-patch32":
        raise MilvusVectorError("embedding model must be openai/clip-vit-base-patch32")
    if index.get("metric_type") != "COSINE" or index.get("index_type") != "HNSW":
        raise MilvusVectorError("Milvus index must use HNSW with COSINE")
    for name in ("M", "efConstruction", "ef"):
        value = index.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MilvusVectorError(f"index.{name} must be a positive integer")
    scalar_fields = index.get("scalar_fields")
    if not isinstance(scalar_fields, list) or set(scalar_fields) != FILTER_FIELDS:
        raise MilvusVectorError("scalar index fields must match the fixed whitelist")
    return payload


class OTAMilvusVectorStore:
    """Five-operation SDK for the fixed OTA image collection."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        client: Any | None = None,
        sdk: Any | None = None,
    ) -> None:
        self.config = config
        self.collection = config["collection"]["name"]
        self.dimension = config["collection"]["vector_dimension"]
        self.sdk = sdk or _import_sdk()
        connection = config["connection"]
        client_options = {
            "uri": connection["uri"],
            "timeout": connection["timeout_seconds"],
        }
        token_env = connection.get("token_env")
        if token_env:
            token = os.environ.get(token_env)
            if token:
                client_options["token"] = token
        self.client = client or self.sdk.MilvusClient(**client_options)

    def create_collection(self) -> None:
        """Create the fixed collection without silently replacing an existing one."""
        if self.client.has_collection(collection_name=self.collection):
            return
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        data_type = self.sdk.DataType
        schema.add_field(
            field_name="vector_id",
            datatype=data_type.INT64,
            is_primary=True,
            auto_id=True,
        )
        for field_name, max_length in (
            ("business_id", 128),
            ("image_id", 128),
        ):
            schema.add_field(
                field_name=field_name,
                datatype=data_type.VARCHAR,
                max_length=max_length,
            )
        schema.add_field(
            field_name="multimodal_vector",
            datatype=data_type.FLOAT_VECTOR,
            dim=self.dimension,
        )
        for field_name, max_length in (
            ("business_category", 64),
            ("city", 128),
            ("price_range", 32),
            ("image_type", 64),
            ("embedding_model", 128),
        ):
            schema.add_field(
                field_name=field_name,
                datatype=data_type.VARCHAR,
                max_length=max_length,
            )
        schema.add_field(field_name="star_rating", datatype=data_type.FLOAT)
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            consistency_level=self.config["collection"]["consistency_level"],
        )

    def build_indexes(self) -> None:
        """Build configured HNSW/COSINE and scalar indexes."""
        params = self.client.prepare_index_params()
        index = self.config["index"]
        params.add_index(
            field_name="multimodal_vector",
            index_name="multimodal_vector_hnsw",
            index_type=index["index_type"],
            metric_type=index["metric_type"],
            params={"M": index["M"], "efConstruction": index["efConstruction"]},
        )
        for field_name in index["scalar_fields"]:
            params.add_index(
                field_name=field_name,
                index_name=f"{field_name}_scalar",
                index_type="INVERTED",
            )
        self.client.create_index(
            collection_name=self.collection,
            index_params=params,
            sync=True,
        )

    def batch_insert(self, entities: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Validate and insert a bounded batch."""
        rows = [self._validate_entity(entity) for entity in entities]
        if not rows:
            raise MilvusVectorError("batch insert requires at least one entity")
        return self.client.insert(collection_name=self.collection, data=rows)

    def insert_one(self, entity: dict[str, Any]) -> dict[str, Any]:
        """Insert one validated entity."""
        return self.batch_insert([entity])

    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Search by normalized vector with exact allow-listed scalar filters."""
        normalized = self._validate_vector(vector)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise MilvusVectorError("top_k must be a positive integer")
        expression = build_filter_expression(filters or {})
        return self.client.search(
            collection_name=self.collection,
            data=[normalized],
            anns_field="multimodal_vector",
            filter=expression,
            limit=top_k,
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": self.config["index"]["ef"]},
            },
            output_fields=sorted(FILTER_FIELDS),
        )

    def delete(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Delete entities selected only through the scalar-filter whitelist."""
        expression = build_filter_expression(filters)
        if not expression:
            raise MilvusVectorError("delete requires at least one filter")
        return self.client.delete(
            collection_name=self.collection,
            filter=expression,
        )

    def count_visible_entities(self) -> int:
        """查询当前逻辑可见行数，不使用包含逻辑删除行的物理统计。"""
        rows = self.client.query(
            collection_name=self.collection,
            filter="",
            output_fields=["count(*)"],
        )
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], dict)
            or isinstance(rows[0].get("count(*)"), bool)
            or not isinstance(rows[0].get("count(*)"), int)
            or rows[0]["count(*)"] < 0
        ):
            raise MilvusVectorError("Milvus count(*) returned an invalid result")
        return rows[0]["count(*)"]

    def _validate_entity(self, entity: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(entity, dict):
            raise MilvusVectorError("entity must be a mapping")
        missing = sorted(REQUIRED_ENTITY_FIELDS - entity.keys())
        extra = sorted(set(entity) - REQUIRED_ENTITY_FIELDS)
        if missing or extra:
            raise MilvusVectorError(f"entity fields invalid; missing={missing}, extra={extra}")
        row = dict(entity)
        row["multimodal_vector"] = self._validate_vector(row["multimodal_vector"])
        for field in STRING_FIELDS:
            value = row[field]
            if not isinstance(value, str) or not value.strip():
                raise MilvusVectorError(f"{field} must be non-empty text")
        rating = row["star_rating"]
        if (
            isinstance(rating, bool)
            or not isinstance(rating, (int, float))
            or not math.isfinite(rating)
            or not 0 <= rating <= 5
        ):
            raise MilvusVectorError("star_rating must be a finite number from 0 to 5")
        row["star_rating"] = float(rating)
        return row

    def _validate_vector(self, vector: Any) -> list[float]:
        if not isinstance(vector, list) or len(vector) != self.dimension:
            raise MilvusVectorError(f"vector must contain exactly {self.dimension} values")
        values = [float(value) for value in vector]
        if any(not math.isfinite(value) for value in values):
            raise MilvusVectorError("vector values must be finite")
        norm = math.sqrt(sum(value * value for value in values))
        if not 0.999 <= norm <= 1.001:
            raise MilvusVectorError("CLIP vectors must be L2-normalized")
        return values


def build_filter_expression(filters: dict[str, Any]) -> str:
    """Compile exact equality or IN filters without accepting raw expressions."""
    if not isinstance(filters, dict):
        raise MilvusVectorError("filters must be a mapping")
    unknown = sorted(set(filters) - FILTER_FIELDS)
    if unknown:
        raise MilvusVectorError(f"unsupported scalar filters: {unknown}")
    clauses = []
    for field, value in sorted(filters.items()):
        if isinstance(value, (list, tuple)):
            if not value:
                raise MilvusVectorError(f"filter list for {field} cannot be empty")
            literals = ", ".join(_literal(field, item) for item in value)
            clauses.append(f"{field} in [{literals}]")
        else:
            clauses.append(f"{field} == {_literal(field, value)}")
    return " and ".join(clauses)


def _literal(field: str, value: Any) -> str:
    if field == "star_rating":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MilvusVectorError("star_rating filters must be numeric")
        return json.dumps(float(value), allow_nan=False)
    if not isinstance(value, str) or not value:
        raise MilvusVectorError(f"{field} filters must be non-empty text")
    return json.dumps(value, ensure_ascii=False)


def _import_sdk() -> Any:
    try:
        import pymilvus
    except ImportError as exc:
        raise MilvusVectorError(
            "PyMilvus is unavailable; install requirements-milvus.txt"
        ) from exc
    return pymilvus

#!/usr/bin/env python3
"""Load the packaged 1,000-vector OTA retrieval release into Milvus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.milvus_vectors import OTAMilvusVectorStore, load_milvus_config


def load_release_vectors(
    config: dict[str, Any],
    vectors_path: Path,
    metadata_path: Path,
    *,
    store: OTAMilvusVectorStore | None = None,
) -> dict[str, Any]:
    """Load an empty collection or validate an already complete one."""

    import numpy as np

    expected = int(config["benchmark"]["vector_count"])
    vectors = np.load(vectors_path)["multimodal_vector"]
    metadata = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(vectors) != expected or len(metadata) != expected:
        raise RuntimeError("release vector and metadata counts do not match the config")
    active_store = store or OTAMilvusVectorStore(config)
    active_store.create_collection()
    stats = active_store.client.get_collection_stats(
        collection_name=active_store.collection
    )
    try:
        current = int(stats["row_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("cannot read Milvus physical row count") from exc
    if current not in {0, expected}:
        raise RuntimeError(
            f"Milvus collection has partial data: expected 0 or {expected}, got {current}"
        )
    if current == 0:
        entities = [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key != "source_image_path"
                },
                "multimodal_vector": vectors[index].astype(float).tolist(),
            }
            for index, row in enumerate(metadata)
        ]
        active_store.batch_insert(entities)
        active_store.client.flush(collection_name=active_store.collection)
        active_store.build_indexes()
        status = "LOADED"
    else:
        status = "ALREADY_LOADED"
    active_store.client.load_collection(collection_name=active_store.collection)
    visible = active_store.count_visible_entities()
    if visible != expected:
        raise RuntimeError(
            f"Milvus visible row count mismatch: expected {expected}, got {visible}"
        )
    return {
        "status": status,
        "collection": active_store.collection,
        "vector_count": visible,
        "embedding_model": config["collection"]["embedding_model"],
        "vector_dimension": config["collection"]["vector_dimension"],
        "index_type": config["index"]["index_type"],
        "metric_type": config["index"]["metric_type"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=ROOT / "docker/system/milvus_system.yaml",
        type=Path,
    )
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    result = load_release_vectors(
        load_milvus_config(args.config),
        args.vectors,
        args.metadata,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

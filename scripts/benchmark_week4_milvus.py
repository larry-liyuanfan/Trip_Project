"""Run real Milvus CRUD and bounded HNSW/COSINE performance validation."""

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.milvus_vectors import (
    OTAMilvusVectorStore,
    load_milvus_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/milvus_week4.yaml")
    args = parser.parse_args()
    config = load_milvus_config(args.config)
    benchmark = config["benchmark"]
    output = Path(benchmark["output_path"])
    if output.exists():
        raise RuntimeError(f"benchmark output already exists: {output}")

    import numpy as np

    vectors = np.load(benchmark["vectors_path"])["multimodal_vector"]
    metadata = [
        json.loads(line)
        for line in Path(benchmark["metadata_path"]).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(vectors) != len(metadata) or len(metadata) != benchmark["vector_count"]:
        raise RuntimeError("vector and metadata counts do not match benchmark config")
    entities = [
        {
            **{key: value for key, value in row.items() if key != "source_image_path"},
            "multimodal_vector": vectors[index].astype(float).tolist(),
        }
        for index, row in enumerate(metadata)
    ]
    store = OTAMilvusVectorStore(config)
    store.create_collection()
    _require_empty_collection(store)
    batch_result = store.batch_insert(entities[:-1])
    single_result = store.insert_one(entities[-1])
    store.client.flush(collection_name=store.collection)

    started = time.perf_counter()
    store.build_indexes()
    index_build_seconds = time.perf_counter() - started
    store.client.load_collection(collection_name=store.collection)
    visible_after_insert = store.count_visible_entities()
    if visible_after_insert != len(entities):
        raise RuntimeError(
            "visible row count after insert does not match the input vector count"
        )

    probe = entities[-1]
    filtered = store.search(
        probe["multimodal_vector"],
        top_k=3,
        filters={"city": probe["city"], "embedding_model": probe["embedding_model"]},
    )
    delete_result = store.delete({"image_id": probe["image_id"]})
    store.client.flush(collection_name=store.collection)
    visible_after_delete = store.count_visible_entities()
    if visible_after_delete != len(entities) - 1:
        raise RuntimeError(
            "visible row count after delete does not match the expected count"
        )
    post_delete = store.search(
        probe["multimodal_vector"],
        top_k=benchmark["top_k"],
        filters={"image_id": probe["image_id"]},
    )

    rng = random.Random(benchmark["seed"])
    query_indices = rng.sample(
        range(len(entities) - 1),
        min(benchmark["query_count"], len(entities) - 1),
    )
    latencies = []
    recall_values = []
    matrix = vectors[:-1]
    for query_index in query_indices:
        query = vectors[query_index]
        exact = np.argsort(-(matrix @ query))[: benchmark["top_k"]]
        exact_ids = {metadata[index]["image_id"] for index in exact}
        started = time.perf_counter()
        result = store.search(
            query.astype(float).tolist(),
            top_k=benchmark["top_k"],
        )
        latencies.append((time.perf_counter() - started) * 1000)
        returned_ids = {
            hit.get("entity", {}).get("image_id")
            for hit in (result[0] if result else [])
        }
        recall_values.append(
            len(exact_ids & returned_ids) / max(len(exact_ids), 1)
        )

    report = {
        "status": "completed",
        "collection": store.collection,
        "milvus_image": "milvusdb/milvus:v2.6.20",
        "pymilvus_version": store.sdk.__version__,
        "embedding_model": config["collection"]["embedding_model"],
        "vector_dimension": config["collection"]["vector_dimension"],
        "actual_vector_count_inserted": visible_after_insert,
        "actual_vector_count_after_delete": visible_after_delete,
        "index": {
            "type": config["index"]["index_type"],
            "metric": config["index"]["metric_type"],
            "M": config["index"]["M"],
            "efConstruction": config["index"]["efConstruction"],
            "ef": config["index"]["ef"],
            "build_seconds": index_build_seconds,
        },
        "search": {
            "query_count": len(query_indices),
            "top_k": benchmark["top_k"],
            "mean_latency_ms": statistics.fmean(latencies),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "recall_at_k": statistics.fmean(recall_values),
        },
        "crud": {
            "batch_insert_count": _insert_count(batch_result),
            "single_insert_count": _insert_count(single_result),
            "filtered_search_hit_count": len(filtered[0] if filtered else []),
            "delete_count": _delete_count(delete_result),
            "post_delete_hit_count": len(post_delete[0] if post_delete else []),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _insert_count(result) -> int:
    if isinstance(result, dict):
        for key in ("insert_count", "insert_cnt"):
            if isinstance(result.get(key), int):
                return result[key]
    return 0


def _delete_count(result) -> int:
    if isinstance(result, dict):
        for key in ("delete_count", "delete_cnt"):
            if isinstance(result.get(key), int):
                return result[key]
    return 0


def _require_empty_collection(store: OTAMilvusVectorStore) -> None:
    stats = store.client.get_collection_stats(collection_name=store.collection)
    try:
        physical_count = int(stats["row_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("cannot read Milvus physical row count") from exc
    if physical_count != 0:
        raise RuntimeError(
            "benchmark requires an empty collection; "
            f"physical row count is {physical_count}"
        )


def _percentile(values, quantile):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return ordered[index]


if __name__ == "__main__":
    main()

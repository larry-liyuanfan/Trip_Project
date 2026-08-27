"""Exercise production query routing against an isolated real Milvus Lite index.

Uses one identity-bound cached CLIP vector from the existing release, not a new
CLIP quality benchmark. No query reference metadata is passed to ranking.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.api import routes
from src.inference.schemas import VisualSearchRequest
from src.retrieval.milvus_vectors import OTAMilvusVectorStore, load_milvus_config
from src.retrieval.visual_search import VisualSearchService
from src.retrieval.query_inputs import user_query_attributes
from scripts.run_system_model_smoke import _sha256


def run(config_path):
    import numpy as np
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"] / "retrieval"
    output.mkdir(parents=True, exist_ok=False)
    vector_path, metadata_path = ROOT / config["retrieval_vectors"], ROOT / config["retrieval_metadata"]
    vectors = np.load(vector_path)["multimodal_vector"]
    with metadata_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != len(vectors) or len(rows) != 1000:
        raise ValueError("release vector/metadata identity count mismatch")
    image_path = (ROOT / rows[0]["source_image_path"]).resolve()
    image_sha = _sha256(image_path)

    class CachedReleaseVector:
        model_id = "openai/clip-vit-base-patch32"
        calls = 0

        def encode(self, paths):
            if [Path(path).resolve() for path in paths] != [image_path] or _sha256(image_path) != image_sha:
                raise ValueError("cached CLIP vector queried with a different image")
            self.calls += 1
            return [vectors[0].astype(float).tolist()]

    milvus = load_milvus_config(ROOT / "docker/system/milvus_system.yaml")
    milvus["connection"]["uri"] = str(output / "isolated_index.db")
    store = OTAMilvusVectorStore(milvus)
    store.create_collection()
    store.batch_insert([{**{key: value for key, value in row.items() if key != "source_image_path"},
                         "multimodal_vector": vectors[i].astype(float).tolist()} for i, row in enumerate(rows)])
    indexes = store.client.prepare_index_params()
    indexes.add_index(field_name="multimodal_vector", index_type="FLAT", metric_type="COSINE")
    store.client.create_index(collection_name=store.collection, index_params=indexes)
    store.client.load_collection(collection_name=store.collection)
    encoder = CachedReleaseVector()
    service = VisualSearchService(encoder, store)
    original_getter, original_env = routes.get_visual_search_service, os.environ.get("APP_ENV")
    routes.get_visual_search_service = lambda: service
    os.environ["APP_ENV"] = "production"
    results = []
    try:
        for query in config["retrieval_queries"]:
            request = VisualSearchRequest(**query, top_k=5,
                                          image_urls=[] if query["retrieval_mode"] == "keyword" else [str(image_path)])
            started = time.perf_counter()
            response = routes.visual_search(request)
            attrs = user_query_attributes(query["query_text"], {"city": query["city"]})
            correct = all(all(item.get(key) == value for key, value in attrs.items()) for item in response["results"])
            results.append({"request": query, "response": response, "filter_correct": correct,
                            "latency_ms": (time.perf_counter() - started) * 1000})
    finally:
        routes.get_visual_search_service = original_getter
        if original_env is None:
            os.environ.pop("APP_ENV", None)
        else:
            os.environ["APP_ENV"] = original_env
        store.client.close()
    changed = {row["business_id"] for row in results[0]["response"]["results"]} != {
        row["business_id"] for row in results[1]["response"]["results"]}
    passed = all(item["filter_correct"] for item in results) and changed and all(
        results[index]["response"]["results"] for index in (0, 1))
    summary = {"status": "PASS" if passed else "FAIL", "queries": results, "query_change_changes_results": changed,
               "cached_encoder_calls": encoder.calls, "cached_vector_source_image_sha256": image_sha,
               "vectors_sha256": _sha256(vector_path), "metadata_sha256": _sha256(metadata_path),
               "backend": "isolated_milvus_lite_flat", "production_route_executed": True,
               "reference_query_metadata_used": False, "visual_relevance_improvement_assessed": False,
               "final_test_labels_read": False}
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({"status": summary["status"], "query_change_changes_results": changed,
                      "result_counts": [len(row["response"]["results"]) for row in results]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week8/audit_repair_v1.json")
    run(parser.parse_args().config)

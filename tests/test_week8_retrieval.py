import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval.week8_relevance import (
    Week8RetrievalError,
    _vector_sha256,
    build_data_lock,
    claim_final_test,
    complete_final_test,
    evaluate_partition,
    load_config,
    select_development_method,
    sha256_file,
    validate_data_lock,
    validate_development_selection,
    write_evaluation,
)
from src.retrieval.week8_hybrid import (
    MilvusImageChannel,
    build_preferred_image_channel,
)


class Week8RetrievalTests(unittest.TestCase):
    def test_repository_config_has_fixed_release_and_silver_isolated_protocol(self):
        config = load_config("configs/week8/retrieval_relevance_v1.json")

        self.assertEqual(config["source"]["expected_count"], 1000)
        self.assertEqual(config["source"]["vector_dimension"], 512)
        self.assertEqual(
            config["source"]["embedding_model"],
            "openai/clip-vit-base-patch32",
        )
        self.assertEqual(config["selection"]["baseline_method"], "clip")
        self.assertEqual(
            config["selection"]["candidate_methods"],
            ["metadata_rerank", "hybrid_rrf", "hybrid_weighted"],
        )
        self.assertEqual(config["hybrid"]["backend_preference"], "auto")

    def test_lock_is_deterministic_and_isolates_five_dimensions(self):
        config = self._config(count=40, dimension=2)
        vectors = np.asarray(
            [[1.0, 0.0] if index % 2 == 0 else [0.0, 1.0] for index in range(40)],
            dtype="float32",
        )
        metadata = [self._metadata(index) for index in range(40)]
        # 两个不同 business 的相同图片字节必须经联合分组进入同一 partition。
        duplicate_indices = (3, 27)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, row in enumerate(metadata):
                image = root / row["source_image_path"]
                image.parent.mkdir(parents=True, exist_ok=True)
                content = b"same-image" if index in duplicate_indices else f"image-{index}".encode()
                image.write_bytes(content)

            first = root / "lock-a"
            second = root / "lock-b"
            build_data_lock(
                config,
                vectors,
                metadata,
                {"container_sha256": "a", "vectors_sha256": "b", "metadata_sha256": "c"},
                source_project_root=root,
                output_dir=first,
            )
            build_data_lock(
                config,
                vectors,
                metadata,
                {"container_sha256": "a", "vectors_sha256": "b", "metadata_sha256": "c"},
                source_project_root=root,
                output_dir=second,
            )
            manifest_a, rows_a = validate_data_lock(first)
            manifest_b, rows_b = validate_data_lock(second)

            self.assertEqual(manifest_a["five_dimension_isolation"], "PASS")
            self.assertEqual(
                {name: manifest_a["files"][name]["sha256"] for name in manifest_a["files"]},
                {name: manifest_b["files"][name]["sha256"] for name in manifest_b["files"]},
            )
            partition_by_group = {
                row["group_id"]: partition
                for partition, rows in rows_a.items()
                for row in rows
            }
            self.assertEqual(
                partition_by_group[metadata[duplicate_indices[0]]["business_id"]],
                partition_by_group[metadata[duplicate_indices[1]]["business_id"]],
            )
            for field in ("sample_id", "source_id", "image_sha256", "group_id", "template_id"):
                values = [
                    {row[field] for row in rows_a[partition]}
                    for partition in ("index", "development_query", "final_test_query")
                ]
                self.assertFalse(values[0] & values[1])
                self.assertFalse(values[0] & values[2])
                self.assertFalse(values[1] & values[2])

    def test_lock_requires_original_image_bytes(self):
        config = self._config(count=1, dimension=2)
        config["split"]["minimum_counts"] = {
            "index": 0,
            "development_query": 0,
            "final_test_query": 0,
        }
        vectors = np.asarray([[1.0, 0.0]], dtype="float32")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(Week8RetrievalError, "source image is missing"):
                build_data_lock(
                    config,
                    vectors,
                    [self._metadata(0)],
                    {"container_sha256": "a"},
                    source_project_root=Path(temporary),
                    output_dir=Path(temporary) / "lock",
                )

    def test_metadata_rerank_improves_independent_query_relevance(self):
        config = self._config(count=4, dimension=2)
        config["evaluation"]["top_k_values"] = [1]
        config["evaluation"]["minimum_relevance_grade"] = 5.0
        config["evaluation"]["candidate_pool_size"] = 3
        config["evaluation"]["rerank_weights"] = {
            "city": 0.12,
            "business_category": 0.08,
            "price_range": 0.05,
        }
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.99995, 0.01],
                [0.98, 0.19899749],
                [1.0, 0.0],
            ],
            dtype="float32",
        )
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        metadata = [
            self._metadata(0, city="B", category="restaurant", price="budget"),
            self._metadata(1, city="C", category="restaurant", price="budget"),
            self._metadata(2, city="A", category="restaurant", price="mid_range"),
            self._metadata(3, city="A", category="restaurant", price="mid_range"),
        ]
        rows = {
            "index": [self._locked_row(metadata[index], vectors, index, "index") for index in range(3)],
            "development_query": [
                self._locked_row(metadata[3], vectors, 3, "development_query")
            ],
            "final_test_query": [],
        }

        metrics, results, references = evaluate_partition(
            config,
            vectors,
            metadata,
            rows,
            "development_query",
        )

        self.assertEqual(metrics["clip"]["recall_at_1"], 0.0)
        self.assertEqual(metrics["metadata_rerank"]["recall_at_1"], 1.0)
        self.assertGreater(metrics["metadata_rerank"]["ndcg_at_1"], metrics["clip"]["ndcg_at_1"])
        self.assertEqual(metrics["metadata_rerank"]["filter_correctness"], 1.0)
        self.assertEqual(metrics["metadata_rerank"]["traceable_reference_rate"], 1.0)
        self.assertEqual(metrics["hybrid_rrf"]["source_attribution_rate"], 1.0)
        self.assertEqual(metrics["hybrid_weighted"]["retrieval_backend"], "offline_cosine")
        self.assertFalse(metrics["hybrid_weighted"]["offline_fallback"])
        self.assertIsNotNone(metrics["hybrid_weighted"]["image_channel_latency_p95_ms"])
        self.assertTrue(results)
        self.assertTrue(references)
        self.assertEqual({row["consumer_path"] for row in references}, {
            "image_product_search", "itinerary_planning"
        })

    def test_real_milvus_is_preferred_and_unavailability_is_explicit_fallback(self):
        config = self._config(count=2, dimension=2)
        config["hybrid"]["milvus_uri"] = "fixture-week8.db"
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        metadata = [self._metadata(0), self._metadata(1)]
        index_rows = [
            self._locked_row(metadata[index], vectors, index, "index")
            for index in range(2)
        ]

        class FakeMilvusClient:
            def has_collection(self, **kwargs):
                return True

            def query(self, **kwargs):
                if kwargs["output_fields"] == ["count(*)"]:
                    return [{"count(*)": 2}]
                return [{"image_id": row["metadata"]["image_id"]} for row in index_rows]

            def search(self, **kwargs):
                return [[{
                    "distance": 0.91,
                    "entity": {"image_id": metadata[0]["image_id"]},
                }]]

        channel = build_preferred_image_channel(
            config,
            index_rows,
            vectors,
            backend="auto",
            milvus_factory=lambda: MilvusImageChannel(
                config,
                index_rows,
                client=FakeMilvusClient(),
            ),
        )
        self.assertEqual(channel.describe()["backend"], "milvus_lite_flat_cosine")
        hits = channel.search(vectors[1], index_rows, top_k=1)
        self.assertEqual(hits[0]["row"]["metadata"]["image_id"], metadata[0]["image_id"])

        fallback = build_preferred_image_channel(
            config,
            index_rows,
            vectors,
            backend="auto",
            milvus_factory=lambda: (_ for _ in ()).throw(ConnectionError("service down")),
        )
        self.assertEqual(fallback.describe()["backend"], "offline_cosine")
        self.assertTrue(fallback.describe()["offline_fallback"])
        self.assertIn("service down", fallback.describe()["fallback_reason"])
        with self.assertRaisesRegex(Exception, "Milvus backend unavailable"):
            build_preferred_image_channel(
                config,
                index_rows,
                vectors,
                backend="milvus",
                milvus_factory=lambda: (_ for _ in ()).throw(ConnectionError("service down")),
            )

    def test_selection_binds_development_evidence_and_rejects_tamper(self):
        config = self._config(count=1, dimension=2)
        development_metrics = {
            "clip": {"ndcg_at_1": 0.2, "failure_rate": 0.0},
            "metadata_rerank": {"ndcg_at_1": 0.4, "failure_rate": 0.0},
        }
        selection = select_development_method(config, development_metrics)
        source_hashes = {
            "container_sha256": "a" * 64,
            "vectors_sha256": "b" * 64,
            "metadata_sha256": "c" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_dir = root / "lock"
            lock_dir.mkdir()
            lock_path = lock_dir / "dataset_lock.json"
            lock_path.write_text('{"lock":"fixed"}\n', encoding="utf-8")
            development_dir = root / "development"
            hashes = write_evaluation(
                development_dir,
                partition="development_query",
                metrics=development_metrics,
                results=[{"query_sample_id": "q1"}],
                references=[{"citation_id": "c1"}],
                selection=selection,
                data_lock_sha256=sha256_file(lock_path),
                source_hashes=source_hashes,
            )

            validated = validate_development_selection(
                config,
                development_dir / "selection.json",
                lock_dir=lock_dir,
                source_hashes=source_hashes,
            )
            self.assertEqual(validated["development_evidence"]["metrics"]["sha256"], hashes["metrics_sha256"])
            self.assertEqual(validated["data_lock_sha256"], sha256_file(lock_path))

            selection_path = development_dir / "selection.json"
            original_selection = selection_path.read_text(encoding="utf-8")
            tampered_selection = json.loads(original_selection)
            tampered_selection["selected_method"] = "clip"
            selection_path.write_text(
                json.dumps(tampered_selection, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Week8RetrievalError, "selection decision mismatch"):
                validate_development_selection(
                    config,
                    selection_path,
                    lock_dir=lock_dir,
                    source_hashes=source_hashes,
                )
            selection_path.write_text(original_selection, encoding="utf-8")

            with (development_dir / "metrics.json").open("a", encoding="utf-8") as handle:
                handle.write(" \n")
            with self.assertRaisesRegex(Week8RetrievalError, "evidence SHA-256 mismatch: metrics"):
                validate_development_selection(
                    config,
                    development_dir / "selection.json",
                    lock_dir=lock_dir,
                    source_hashes=source_hashes,
                )

    def test_selection_and_completed_final_test_marker_are_fail_closed(self):
        config = self._config(count=1, dimension=2)
        config["selection"] = {
            "baseline_method": "clip",
            "candidate_method": "metadata_rerank",
            "primary_metric": "ndcg_at_10",
            "non_regression_metrics": [
                "recall_at_10",
                "filter_correctness",
                "traceable_reference_rate",
            ],
            "required_failure_rate": 0.0,
        }
        selection = select_development_method(
            config,
            {
                "clip": {
                    "ndcg_at_10": 0.4,
                    "recall_at_10": 0.2,
                    "filter_correctness": 1.0,
                    "traceable_reference_rate": 1.0,
                    "failure_rate": 0.0,
                },
                "metadata_rerank": {
                    "ndcg_at_10": 0.6,
                    "recall_at_10": 0.2,
                    "filter_correctness": 1.0,
                    "traceable_reference_rate": 1.0,
                    "failure_rate": 0.0,
                },
            },
        )
        self.assertTrue(selection["candidate_locked"])
        self.assertEqual(selection["selected_method"], "metadata_rerank")
        selection["data_lock_sha256"] = "a" * 64
        selection["source_hashes"] = {"vectors_sha256": "b" * 64}

        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "consumed.json"
            claim_final_test(marker, selection, selection_sha256="d" * 64)
            started = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(started["status"], "STARTED")
            with self.assertRaisesRegex(Week8RetrievalError, "evidence hashes are incomplete"):
                complete_final_test(marker, {"metrics_sha256": "e" * 64})
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["status"],
                "STARTED",
            )
            completed = complete_final_test(
                marker,
                {
                    "metrics_sha256": "e" * 64,
                    "query_results_sha256": "f" * 64,
                    "business_references_sha256": "0" * 64,
                },
            )
            self.assertEqual(completed["status"], "COMPLETED")
            self.assertEqual(
                completed["final_evidence"]["metrics"]["sha256"],
                "e" * 64,
            )
            with self.assertRaisesRegex(Week8RetrievalError, "already consumed"):
                claim_final_test(marker, selection)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "COMPLETED")

    def test_multi_candidate_selection_locks_best_eligible_hybrid(self):
        config = load_config("configs/week8/retrieval_relevance_v1.json")

        def measured(ndcg):
            return {
                "ndcg_at_10": ndcg,
                "recall_at_10": 0.2,
                "filter_correctness": 1.0,
                "traceable_reference_rate": 1.0,
                "failure_rate": 0.0,
                "retrieval_backend": "milvus_remote_hnsw_cosine",
                "offline_fallback": False,
                "fallback_reason": None,
            }

        selection = select_development_method(
            config,
            {
                "clip": measured(0.40),
                "metadata_rerank": measured(0.51),
                "hybrid_rrf": measured(0.62),
                "hybrid_weighted": measured(0.58),
            },
        )

        self.assertEqual(selection["selected_method"], "hybrid_rrf")
        self.assertTrue(selection["candidate_locked"])
        self.assertTrue(selection["candidate_evaluations"]["hybrid_rrf"]["eligible"])
        self.assertEqual(selection["selected_backend"], "milvus_remote_hnsw_cosine")

    def _config(self, *, count: int, dimension: int):
        return {
            "schema_version": "week8_retrieval_relevance_config_v1",
            "experiment_id": "test-week8-retrieval",
            "dataset_version": "test-week8-retrieval-data",
            "source": {
                "release_id": "test-release",
                "embedding_model": "openai/clip-vit-base-patch32",
                "vector_dimension": dimension,
                "expected_count": count,
            },
            "split": {
                "seed": 20260826,
                "development_query_group_fraction": 0.25,
                "final_test_query_group_fraction": 0.25,
                "minimum_counts": {
                    "index": 1,
                    "development_query": 1,
                    "final_test_query": 1,
                },
                "template_ids": {
                    "index": "index-template",
                    "development_query": "dev-template",
                    "final_test_query": "test-template",
                },
            },
            "evaluation": {
                "top_k_values": [1],
                "candidate_pool_size": 10,
                "minimum_relevance_grade": 5.0,
                "relevance_weights": {
                    "city": 4.0,
                    "business_category": 2.0,
                    "price_range": 1.0,
                },
                "rerank_weights": {
                    "city": 0.12,
                    "business_category": 0.08,
                    "price_range": 0.05,
                },
                "filter_scenarios": [["city"], ["price_range"]],
                "consumer_paths": ["image_product_search", "itinerary_planning"],
            },
            "hybrid": {
                "backend_preference": "auto",
                "offline_fallback_enabled": True,
                "milvus_config": "docker/system/milvus_system.yaml",
                "milvus_uri": "http://localhost:19530",
                "milvus_token_env": "MILVUS_TOKEN",
                "milvus_timeout_seconds": 30,
                "milvus_collection": "ota_business_image_vector_week8_retrieval_v1",
                "milvus_ef": 64,
                "milvus_lite_index_type": "FLAT",
                "milvus_remote_index_type": "HNSW",
                "rrf_k": 60,
                "weighted_fusion": {"image": 0.7, "metadata": 0.3},
            },
            "selection": {
                "baseline_method": "clip",
                "candidate_method": "metadata_rerank",
                "primary_metric": "ndcg_at_1",
                "non_regression_metrics": [],
                "required_failure_rate": 0.0,
            },
        }

    @staticmethod
    def _metadata(index, *, city=None, category="restaurant", price="mid_range"):
        return {
            "business_id": f"business-{index}",
            "image_id": f"image-{index}",
            "business_category": category,
            "city": city or f"city-{index % 4}",
            "star_rating": 4.0,
            "price_range": price,
            "image_type": "business_photo",
            "embedding_model": "openai/clip-vit-base-patch32",
            "source_image_path": f"images/{index}.jpg",
        }

    @staticmethod
    def _locked_row(metadata, vectors, index, partition):
        return {
            "sample_id": f"week8-retrieval:{metadata['image_id']}",
            "source_id": f"yelp-photo:{metadata['image_id']}",
            "image_sha256": f"{'a' if partition == 'index' else 'b'}{index:063d}",
            "group_id": metadata["business_id"],
            "template_id": f"{partition}-template",
            "label_provenance": "silver",
            "vector_index": index,
            "vector_sha256": _vector_sha256(vectors[index]),
            "metadata": copy.deepcopy(metadata),
            "partition": partition,
        }


if __name__ == "__main__":
    unittest.main()

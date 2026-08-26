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
    evaluate_latency_profiles,
    evaluate_partition,
    load_config,
    select_latency_profile,
    select_development_method,
    sha256_file,
    validate_data_lock,
    validate_development_selection,
    write_evaluation,
)
from src.retrieval.week8_hybrid import (
    MetadataRankingCache,
    MilvusImageChannel,
    build_preferred_image_channel,
)


class Week8RetrievalTests(unittest.TestCase):
    def test_repository_config_has_fixed_release_and_silver_isolated_protocol(self):
        config = load_config("configs/week8/retrieval_relevance_v3.json")
        legacy = json.loads(
            Path("configs/week8/retrieval_relevance_v1.json").read_text(encoding="utf-8")
        )

        self.assertEqual(legacy["schema_version"], "week8_retrieval_relevance_config_v1")
        self.assertEqual(legacy["experiment_id"], "week8_retrieval_relevance_20260826_v1")
        self.assertEqual(legacy["dataset_version"], "week8_retrieval_query_index_20260826_v1")
        self.assertNotIn("hybrid", legacy)
        self.assertEqual(legacy["selection"]["candidate_method"], "metadata_rerank")
        self.assertEqual(
            legacy["split"]["template_ids"]["final_test_query"],
            "week8_retrieval_final_test_query_v1",
        )
        v2 = json.loads(
            Path("configs/week8/retrieval_relevance_v2.json").read_text(encoding="utf-8")
        )
        self.assertEqual(v2["schema_version"], "week8_retrieval_relevance_config_v2")
        self.assertEqual(v2["experiment_id"], "week8_retrieval_relevance_20260827_v2")
        self.assertEqual(config["schema_version"], "week8_retrieval_relevance_config_v3")
        self.assertEqual(config["dataset_version"], "week8_retrieval_query_index_20260827_v3")
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
        latency = load_config("configs/week8/retrieval_latency_v4.json")
        self.assertEqual(
            latency["schema_version"],
            "week8_retrieval_relevance_config_v4",
        )
        self.assertTrue(latency["split"]["development_only"])
        self.assertEqual(latency["split"]["final_test_query_group_fraction"], 0.0)
        self.assertEqual(
            latency["split"]["historical_query_exclusion"]["experiment_id"],
            "week8_retrieval_relevance_20260827_v3",
        )
        self.assertEqual(latency["hybrid"]["milvus_output_fields"], ["image_id"])
        self.assertFalse(latency["hybrid"]["offline_fallback_enabled"])
        latency_v5 = load_config("configs/week8/retrieval_latency_v5.json")
        self.assertEqual(
            latency_v5["schema_version"],
            "week8_retrieval_relevance_config_v5",
        )
        self.assertEqual(
            latency_v5["latency_optimization"]["metadata_cache_capacity"],
            512,
        )
        self.assertEqual(
            latency_v5["split"]["historical_query_exclusion"],
            latency["split"]["historical_query_exclusion"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            invalid_path = Path(temporary) / "invalid-v5.json"
            invalid = copy.deepcopy(latency_v5)
            invalid["latency_optimization"]["metadata_cache_capacity"] = 511
            invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(Week8RetrievalError, "locked to 512"):
                load_config(invalid_path)
        requirements = Path("requirements-milvus.txt").read_text(encoding="utf-8")
        self.assertIn("pymilvus[milvus_lite]==2.6.16", requirements)
        self.assertIn("milvus-lite==3.2.1", requirements)

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

    def test_development_only_lock_excludes_historical_query_groups_without_final_rows(self):
        config = self._config(count=80, dimension=2)
        config["dataset_version"] = "test-week8-retrieval-latency-v4"
        config["split"].update(
            {
                "seed": 20260827,
                "development_only": True,
                "development_query_group_fraction": 0.3,
                "final_test_query_group_fraction": 0.0,
                "minimum_counts": {
                    "index": 1,
                    "development_query": 1,
                    "final_test_query": 0,
                },
                "historical_query_exclusion": {
                    "experiment_id": "historical-v3",
                    "dataset_version": "historical-data-v3",
                    "seed": 20260826,
                    "development_query_group_fraction": 0.2,
                    "final_test_query_group_fraction": 0.2,
                },
            }
        )
        vectors = np.asarray(
            [[1.0, 0.0] if index % 2 == 0 else [0.0, 1.0] for index in range(80)],
            dtype="float32",
        )
        metadata = [self._metadata(index) for index in range(80)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, row in enumerate(metadata):
                image = root / row["source_image_path"]
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"latency-image-{index}".encode())
            lock = root / "lock"
            manifest = build_data_lock(
                config,
                vectors,
                metadata,
                {"container_sha256": "a", "vectors_sha256": "b", "metadata_sha256": "c"},
                source_project_root=root,
                output_dir=lock,
            )
            _, rows = validate_data_lock(lock)

        exclusion = manifest["historical_query_exclusion"]
        self.assertEqual(exclusion["status"], "PASS")
        self.assertGreater(exclusion["excluded_query_row_count"], 0)
        self.assertEqual(exclusion["eligible_row_count"], sum(map(len, rows.values())))
        self.assertEqual(rows["final_test_query"], [])
        self.assertGreater(len(rows["development_query"]), 0)

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
        config["hybrid"]["milvus_output_fields"] = ["image_id"]
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        metadata = [self._metadata(0), self._metadata(1)]
        index_rows = [
            self._locked_row(metadata[index], vectors, index, "index")
            for index in range(2)
        ]

        class FakeMilvusClient:
            loaded = False
            search_kwargs = None

            def has_collection(self, **kwargs):
                return True

            def load_collection(self, **kwargs):
                self.loaded = True

            def query(self, **kwargs):
                if not self.loaded:
                    raise RuntimeError("collection is released")
                if kwargs["output_fields"] == ["count(*)"]:
                    return [{"count(*)": 2}]
                return [{"image_id": row["metadata"]["image_id"]} for row in index_rows]

            def search(self, **kwargs):
                self.search_kwargs = kwargs
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
        self.assertEqual(channel.client.search_kwargs["output_fields"], ["image_id"])

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

    def test_metadata_cache_preserves_ranking_and_latency_selector_is_fail_closed(self):
        config = self._config(count=4, dimension=2)
        rows = [
            self._locked_row(
                self._metadata(index, city="A" if index < 2 else "B"),
                np.asarray([[1.0, 0.0]] * 4, dtype="float32"),
                index,
                "index",
            )
            for index in range(4)
        ]
        cache = MetadataRankingCache(config, rows, capacity=2)
        first = cache.search(rows[0]["metadata"], top_k=2)
        second = cache.search(rows[0]["metadata"], top_k=2)
        self.assertEqual(first, second)
        self.assertEqual(cache.entry_count, 1)
        cache.search(rows[2]["metadata"], top_k=2)
        third_signature = dict(rows[0]["metadata"])
        third_signature["price_range"] = "budget"
        cache.search(third_signature, top_k=2)
        self.assertEqual(
            cache.stats(),
            {
                "capacity": 2,
                "hits": 1,
                "misses": 3,
                "evictions": 1,
                "entry_count": 2,
            },
        )

        config["latency_optimization"] = {
            "baseline_profile": {
                "profile_id": "pool100",
                "candidate_pool_size": 100,
                "metadata_cache": False,
            },
            "candidate_profiles": [
                {
                    "profile_id": "pool50",
                    "candidate_pool_size": 50,
                    "metadata_cache": True,
                },
                {
                    "profile_id": "pool25",
                    "candidate_pool_size": 25,
                    "metadata_cache": True,
                },
            ],
            "primary_latency_metric": "latency_p95_ms",
            "quality_non_regression_metrics": ["ndcg_at_10", "recall_at_10"],
            "maximum_quality_drop": 0.0,
            "required_backend": "milvus_lite_flat_cosine",
            "required_offline_fallback": False,
            "required_failure_rate": 0.0,
        }

        def measured(ndcg, recall, latency):
            return {
                "ndcg_at_10": ndcg,
                "recall_at_10": recall,
                "latency_p95_ms": latency,
                "failure_rate": 0.0,
                "retrieval_backend": "milvus_lite_flat_cosine",
                "offline_fallback": False,
            }

        selection = select_latency_profile(
            config,
            {
                "pool100": measured(0.60, 0.20, 15.0),
                "pool50": measured(0.60, 0.20, 12.0),
                "pool25": measured(0.59, 0.20, 10.0),
            },
        )
        self.assertEqual(selection["selected_profile"], "pool50")
        self.assertTrue(selection["optimization_locked"])
        self.assertIn(
            "ndcg_at_10_regressed",
            selection["candidate_evaluations"]["pool25"]["failures"],
        )

    def test_latency_profiles_record_bounded_cache_cost_and_counters(self):
        config = self._config(count=4, dimension=2)
        config["split"]["development_only"] = True
        config["evaluation"]["top_k_values"] = [1]
        config["evaluation"]["candidate_pool_size"] = 2
        config["latency_optimization"] = {
            "metadata_cache_capacity": 4,
            "baseline_profile": {
                "profile_id": "uncached",
                "candidate_pool_size": 2,
                "metadata_cache": False,
            },
            "candidate_profiles": [
                {
                    "profile_id": "lru",
                    "candidate_pool_size": 2,
                    "metadata_cache": True,
                }
            ],
            "measurement_repeats": 2,
            "warmup_query_count": 0,
            "primary_latency_metric": "latency_p95_ms",
            "quality_non_regression_metrics": ["ndcg_at_1", "recall_at_1"],
            "maximum_quality_drop": 0.0,
            "required_backend": "milvus_lite_flat_cosine",
            "required_offline_fallback": False,
            "required_failure_rate": 0.0,
        }
        vectors = np.asarray(
            [[1.0, 0.0], [0.9, 0.4358899], [0.0, 1.0], [1.0, 0.0]],
            dtype="float32",
        )
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        metadata = [
            self._metadata(0, city="A", price="mid_range"),
            self._metadata(1, city="A", price="budget"),
            self._metadata(2, city="B", price="budget"),
            self._metadata(3, city="A", price="mid_range"),
        ]
        rows = {
            "index": [
                self._locked_row(metadata[index], vectors, index, "index")
                for index in range(3)
            ],
            "development_query": [
                self._locked_row(metadata[3], vectors, 3, "development_query")
            ],
            "final_test_query": [],
        }

        class FakeRealChannel:
            last_latency_ms = None

            def describe(self):
                return {
                    "backend": "milvus_lite_flat_cosine",
                    "offline_fallback": False,
                    "fallback_reason": None,
                }

            def search(self, query_vector, index_rows, *, top_k, filters=None):
                hits = []
                for row in index_rows:
                    if filters and any(
                        row["metadata"].get(field) != value
                        for field, value in filters.items()
                    ):
                        continue
                    hits.append(
                        {
                            "row": row,
                            "image_score": float(
                                query_vector @ vectors[row["vector_index"]]
                            ),
                            "source_backend": "milvus_lite_flat_cosine",
                        }
                    )
                hits.sort(key=lambda hit: -hit["image_score"])
                self.last_latency_ms = 0.01
                return hits[:top_k]

        metrics, _, _, _ = evaluate_latency_profiles(
            config,
            vectors,
            metadata,
            rows,
            image_channel=FakeRealChannel(),
        )
        cached = metrics["lru"]
        self.assertEqual(cached["metadata_cache_capacity"], 4)
        self.assertGreaterEqual(cached["metadata_cache_precompute_ms"], 0.0)
        self.assertGreater(cached["metadata_cache_precompute_peak_python_bytes"], 0)
        self.assertEqual(cached["metadata_cache_precompute_misses"], 3)
        self.assertEqual(cached["metadata_cache_measurement_hits"], 6)
        self.assertEqual(cached["metadata_cache_measurement_misses"], 0)
        self.assertEqual(cached["metadata_cache_evictions"], 0)
        self.assertLessEqual(cached["metadata_cache_entry_count"], 4)

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
        config = load_config("configs/week8/retrieval_relevance_v3.json")

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

import json
import hashlib
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import (
    build_release_bundle,
    run_system_model_smoke,
    tripctl,
    upload_release_oss,
)
from src.retrieval.clip_embeddings import ClipEmbeddingError, _validate_vectors
from src.retrieval.visual_search import VisualSearchService


class FakeEncoder:
    model_id = "openai/clip-vit-base-patch32"

    def ready(self):
        return True, "ok"

    def encode(self, paths):
        return [[0.0] * 511 + [1.0]]


class FakeStore:
    def __init__(self):
        self.call = None

    def search(self, vector, *, top_k=5, filters=None):
        self.call = {"vector": vector, "top_k": top_k, "filters": filters}
        return [[{"id": 7, "distance": 0.95, "entity": {"image_id": "photo-7"}}]]

    def ready(self):
        return True, "ok"


class SystemPackageTest(unittest.TestCase):
    def test_real_model_smoke_covers_three_tasks_and_dialogue(self):
        class Result:
            def __init__(self, payload):
                self.payload = payload

            def model_dump(self):
                return dict(self.payload)

        class Service:
            def __init__(self):
                self.tasks = []

            def run_task(self, scenario, request):
                self.tasks.append((scenario, request))
                return Result({"scenario": scenario, "schema_valid": True})

            def run_dialogue(self, request):
                self.dialogue = request
                return Result({"quality_tier": "DIALOGUE_BETA"})

        service = Service()
        result = run_system_model_smoke.run_model_smoke(
            service,
            Path("data/samples/images/cafe_001.jpg"),
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            {scenario for scenario, _ in service.tasks},
            {"image_product_search", "after_sales", "itinerary_planning"},
        )
        task_requests = {scenario: request for scenario, request in service.tasks}
        self.assertIsNone(task_requests["image_product_search"].text_context)
        self.assertIsNone(task_requests["after_sales"].text_context)
        self.assertTrue(task_requests["itinerary_planning"].text_context)
        self.assertEqual(len(service.dialogue.messages), 1)

    def test_tripctl_smoke_covers_four_model_scenarios_and_visual_search(self):
        class Response:
            ok = True
            status_code = 200

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        posts = []

        def post(url, *, json, timeout):
            posts.append((url, json, timeout))
            if url.endswith("image-product-search"):
                return Response({"scenario": "image_product_search", "schema_valid": True})
            if url.endswith("after-sales"):
                return Response({"scenario": "after_sales", "schema_valid": True})
            if url.endswith("itinerary-planning"):
                return Response({"scenario": "itinerary_planning", "schema_valid": True})
            if url.endswith("dialogue"):
                return Response({"quality_tier": "DIALOGUE_BETA"})
            return Response({"retrieval_mode": "clip_milvus_hnsw_cosine", "results": []})

        with TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "image.jpg"
            image.write_bytes(b"image")
            with patch.object(tripctl.requests, "get", return_value=Response({"status": "ok"})):
                with patch.object(tripctl.requests, "post", side_effect=post):
                    result = tripctl.smoke("http://service", image)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            set(result["checks"]),
            {
                "health",
                "ready",
                "image_product_search",
                "after_sales",
                "itinerary_planning",
                "dialogue",
                "visual_search",
            },
        )
        self.assertEqual(len(posts), 5)
        task_payloads = {
            url.rsplit("/", 1)[-1]: payload
            for url, payload, _ in posts
            if "/v1/tasks/" in url
        }
        self.assertIsNone(task_payloads["image-product-search"]["text_context"])
        self.assertIsNone(task_payloads["after-sales"]["text_context"])
        self.assertTrue(task_payloads["itinerary-planning"]["text_context"])

    def test_clip_vector_validation_rejects_wrong_dimension_and_norm(self):
        with self.assertRaisesRegex(ClipEmbeddingError, "dimension"):
            _validate_vectors([[1.0]], expected_count=1)
        with self.assertRaisesRegex(ClipEmbeddingError, "L2-normalized"):
            _validate_vectors([[1.0] * 512], expected_count=1)

    def test_visual_search_forces_embedding_identity_filter(self):
        store = FakeStore()
        service = VisualSearchService(FakeEncoder(), store)

        results = service.search("query.jpg", top_k=3, filters={"city": "Shanghai"})

        self.assertEqual(results[0]["image_id"], "photo-7")
        self.assertEqual(store.call["top_k"], 3)
        self.assertEqual(
            store.call["filters"],
            {
                "city": "Shanghai",
                "embedding_model": "openai/clip-vit-base-patch32",
            },
        )

    def test_milvus_system_config_requests_exactly_1000_vectors(self):
        text = Path("docker/system/milvus_system.yaml").read_text(encoding="utf-8")

        self.assertIn("vector_count: 1000", text)
        self.assertIn("index_type: HNSW", text)
        self.assertIn("metric_type: COSINE", text)

    def test_system_compose_is_fail_closed_and_has_no_literal_secrets(self):
        text = Path("docker/system/docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn('MODEL_FALLBACK_ENABLED: "false"', text)
        self.assertIn("TRIP_ADAPTER_DIR: /models/adapter", text)
        self.assertIn("milvusdb/milvus:v2.6.20", text)
        self.assertIn("${MINIO_ROOT_PASSWORD:?", text)
        self.assertNotIn("minioadmin", text.lower())

    def test_release_builder_creates_four_checksum_layers(self):
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            adapter = workspace / "adapter"
            retrieval = workspace / "retrieval"
            evidence = workspace / "report.md"
            adapter.mkdir()
            retrieval.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (retrieval / "vectors.npz").write_bytes(b"vectors")
            evidence.write_text("observed evidence", encoding="utf-8")
            release = workspace / "release_config.json"
            release.write_text(
                json.dumps(
                    {
                        "release_id": "test-release",
                        "model": {
                            "adapter_model_sha256": hashlib.sha256(b"adapter").hexdigest()
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = workspace / "release"
            with patch.object(build_release_bundle, "RUNTIME_PATHS", ["README.md"]):
                manifest = build_release_bundle.build_bundle(
                    output,
                    adapter_dir=adapter,
                    retrieval_dir=retrieval,
                    evidence_paths=[evidence],
                    release_config=release,
                )

            saved = json.loads(
                (output / "release_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(manifest["layers"]), {"runtime", "adapter", "retrieval", "evidence"})
            self.assertEqual(saved["visibility"], "private")
            self.assertEqual(saved["release"]["release_id"], "test-release")
            with tarfile.open(output / "adapter.tar.gz", "r:gz") as archive:
                self.assertIn("adapter/adapter_model.safetensors", archive.getnames())

            verified = upload_release_oss.verify_release_dir(output)
            self.assertEqual(verified, saved)
            (output / "adapter.tar.gz").write_bytes(b"tampered")
            with self.assertRaisesRegex(
                upload_release_oss.ReleaseVerificationError,
                "size mismatch|SHA-256 mismatch",
            ):
                upload_release_oss.verify_release_dir(output)

    def test_release_builder_rejects_adapter_not_bound_to_config(self):
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            adapter = workspace / "adapter"
            retrieval = workspace / "retrieval"
            adapter.mkdir()
            retrieval.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"wrong")
            release = workspace / "release.json"
            release.write_text(
                json.dumps(
                    {
                        "release_id": "test-release",
                        "model": {"adapter_model_sha256": "0" * 64},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                build_release_bundle.build_bundle(
                    workspace / "output",
                    adapter_dir=adapter,
                    retrieval_dir=retrieval,
                    evidence_paths=[],
                    release_config=release,
                )

    def test_tripctl_validate_rejects_wrong_model(self):
        with TemporaryDirectory() as tmpdir:
            release = Path(tmpdir) / "release.json"
            release.write_text(
                json.dumps({"model": {"base_model": "wrong"}, "quality": {}}),
                encoding="utf-8",
            )
            with patch.object(tripctl, "DEFAULT_RELEASE", release):
                result = tripctl.validate()

        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()

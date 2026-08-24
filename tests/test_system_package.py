import json
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import build_release_bundle, tripctl, upload_release_oss
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
            output = workspace / "release"
            with patch.object(build_release_bundle, "RUNTIME_PATHS", ["README.md"]):
                manifest = build_release_bundle.build_bundle(
                    output,
                    adapter_dir=adapter,
                    retrieval_dir=retrieval,
                    evidence_paths=[evidence],
                )

            saved = json.loads(
                (output / "release_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(manifest["layers"]), {"runtime", "adapter", "retrieval", "evidence"})
            self.assertEqual(saved["visibility"], "private")
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

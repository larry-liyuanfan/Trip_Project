"""Tests for complete v4 query/formal-index byte isolation evidence."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_exploration_pool_v4 import build_pool
from scripts.run_retrieval_query_leakage_v4 import build_evidence, finalize_report
from scripts.verify_retrieval_query_leakage_evidence_v4 import verify_evidence
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "evaluation" / "evidence_enhancement" / "exploration_pool_lock_v4.json"
SOURCE_SHA = "a" * 64
COMMIT = "b" * 40
JOB_ID = "12345"


class RetrievalQueryLeakageV4Tests(unittest.TestCase):
    def test_checked_in_protocol_binds_cross_platform_canonical_lock_contents(self) -> None:
        protocol = json.loads(
            (ROOT / "configs" / "evaluation" / "retrieval_query_leakage_v4.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            canonical_json_sha256(json.loads(LOCK.read_text(encoding="utf-8"))),
            protocol["query_pool"]["committed_lock_canonical_sha256"],
        )

    def test_complete_isolation_passes_and_registry_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pool = root / "pool"
            build_pool(pool)
            archive, overlay, config = _write_index_fixture(root)
            registry_path = root / "run" / "index_image_sha_registry.jsonl"
            report_path = root / "run" / "leakage_audit.json"
            registry_path.parent.mkdir()
            with patch.dict(os.environ, {"SLURM_JOB_ID": JOB_ID}):
                report, registry = build_evidence(
                    config_path=config,
                    retrieval_archive=archive,
                    index_overlay_root=overlay,
                    query_pool_dir=pool,
                    source_snapshot_sha256=SOURCE_SHA,
                    implementation_commit=COMMIT,
                )
            _write_jsonl(registry_path, registry)
            _write_json(report_path, finalize_report(report, registry_path))
            verified = verify_evidence(
                config_path=config,
                retrieval_archive=archive,
                index_overlay_root=overlay,
                query_pool_dir=pool,
                registry_path=registry_path,
                report_path=report_path,
                expected_job_id=JOB_ID,
                expected_source_snapshot_sha256=SOURCE_SHA,
                expected_implementation_commit=COMMIT,
            )
            self.assertEqual(verified["status"], "PASS")
            self.assertEqual(verified["formal_index_image_support"], 2)
            self.assertEqual(verified["query_image_support"], 72)
            self.assertEqual(verified["byte_collision_support"], 0)

            rows = load_jsonl(registry_path)
            rows[0]["primary_sha256"] = "f" * 64
            _write_jsonl(registry_path, rows, overwrite=True)
            with self.assertRaisesRegex(ValueError, "registry differs"):
                verify_evidence(
                    config_path=config,
                    retrieval_archive=archive,
                    index_overlay_root=overlay,
                    query_pool_dir=pool,
                    registry_path=registry_path,
                    report_path=report_path,
                    expected_job_id=JOB_ID,
                    expected_source_snapshot_sha256=SOURCE_SHA,
                    expected_implementation_commit=COMMIT,
                )


def _write_index_fixture(root: Path) -> tuple[Path, Path, Path]:
    overlay = root / "overlay"
    photos = overlay / "data" / "yelp" / "raw" / "photos"
    photos.mkdir(parents=True)
    image_bytes = {"index-a": b"formal-index-a", "index-b": b"formal-index-b"}
    metadata = []
    hashes = {}
    for image_id, payload in image_bytes.items():
        path = photos / f"{image_id}.jpg"
        path.write_bytes(payload)
        hashes[image_id] = file_sha256(path)
        metadata.append(
            {
                "image_id": image_id,
                "business_id": f"business-{image_id}",
                "source_image_path": f"data/yelp/raw/photos/{image_id}.jpg",
            }
        )
    metadata_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata).encode()
    archive = root / "retrieval.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("retrieval/clip_metadata_1000.jsonl")
        info.size = len(metadata_bytes)
        handle.addfile(info, io.BytesIO(metadata_bytes))
    identity = hashlib.sha256(
        "".join(f"{key}:{hashes[key]}\n" for key in sorted(hashes)).encode()
    ).hexdigest()
    overlay_manifest = {
        "schema_version": "week8_retrieval_photo_overlay_v1",
        "status": "COMPLETED",
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "photos_zip_size": 123,
        "requested_photo_count": 2,
        "extracted_photo_count": 2,
        "image_identity_sha256": identity,
    }
    overlay_manifest_path = overlay / "week8_retrieval_photo_overlay.json"
    _write_json(overlay_manifest_path, overlay_manifest)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    config_value = {
        "schema_version": "retrieval_query_leakage_protocol_v4",
        "formal_release_read_only": {
            "release_id": "test",
            "retrieval_archive_sha256": file_sha256(archive),
            "metadata_member": "retrieval/clip_metadata_1000.jsonl",
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "expected_index_support": 2,
        },
        "index_byte_source": {
            "overlay_schema_version": "week8_retrieval_photo_overlay_v1",
            "overlay_manifest_sha256": file_sha256(overlay_manifest_path),
            "image_identity_sha256": identity,
            "expected_extracted_photo_count": 2,
            "official_zip_size": 123,
        },
        "query_pool": {
            "committed_lock": str(LOCK.resolve()),
            "committed_lock_canonical_sha256": canonical_json_sha256(lock),
            "splits": ["training", "development", "final"],
            "expected_query_support_per_split": 24,
        },
        "acceptance": {
            "required_primary_index_byte_coverage": 1.0,
            "required_query_image_byte_coverage": 1.0,
            "maximum_byte_collision_support": 0,
            "maximum_source_identity_collision_support": 0,
        },
    }
    config = root / "retrieval_query_leakage_v4.json"
    _write_json(config, config_value)
    return archive, overlay, config


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]], *, overwrite: bool = False) -> None:
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
